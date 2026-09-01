#!/usr/bin/env python3
"""Attribute copy/cast kernels to PyTorch ops, phases, and modules. No optimization."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from megatron.core import parallel_state

from phase1_baseline import (
    TE_FUSED_ATTENTION,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
    train_step,
)
from phase3_attention_correctness import fused_backend_status
from phase3_attention_profile import baseline_model_args, model_config


COPY_CPU_OPS = (
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
    "aten::clone",
    "aten::contiguous",
    "aten::_foreach_copy",
    "aten::_foreach_copy_",
)
COPY_KERNEL_TOKENS = (
    "bfloat16_copy",
    "direct_copy",
    "copy_kernel",
    "load_withcast",
    "store_withcast",
)
ACTIVATION_KERNEL_TOKENS = (
    "gelu",
    "dropout",
    "lerp",
    "addcmul",
    "mul_functor",
    "binaryfunctor",
    "unaryfunctor",
    "silu",
    "relu",
)
PARAM_SHAPES = {
    (3072, 1024),
    (1024, 3072),
    (1024, 1024),
    (4096, 1024),
    (1024, 4096),
    (50304, 1024),
    (1024, 50304),
    (3072,),
    (1024,),
    (4096,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def classify_module_name(name: str) -> str | None:
    lowered = name.lower()
    if "linear_qkv" in lowered:
        return "qkv_projection"
    if "linear_proj" in lowered:
        return "attention_output_projection"
    if "linear_fc1" in lowered:
        return "mlp_fc1"
    if "linear_fc2" in lowered:
        return "mlp_fc2"
    if "layernorm" in lowered or "layer_norm" in lowered:
        return "normalization"
    if "core_attention" in lowered:
        return "attention"
    if "output_layer" in lowered:
        return "loss"
    if lowered.endswith("embedding") or "word_embeddings" in lowered:
        return "embedding"
    return None


def install_module_ranges(model: torch.nn.Module) -> list[Any]:
    handles: list[Any] = []
    active: list[Any] = []

    def make_pre(module_name: str):
        def pre_hook(_module: torch.nn.Module, _inputs: Any) -> None:
            recorder = record_function(f"module/{module_name}")
            recorder.__enter__()
            active.append(recorder)

        return pre_hook

    def post_hook(_module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
        if active:
            active.pop().__exit__(None, None, None)

    for name, module in model.named_modules():
        if classify_module_name(name) is None:
            continue
        handles.append(module.register_forward_pre_hook(make_pre(name)))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def attributed_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    step_index: int,
) -> float:
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
        with record_function("phase/zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with record_function("phase/forward"):
            with torch.cuda.nvtx.range("forward"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    output = model(
                        batch["tokens"],
                        batch["position_ids"],
                        batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    with record_function("module/loss"):
                        loss = masked_language_model_loss(output, batch["loss_mask"])
        with record_function("phase/backward"):
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
        with record_function("phase/optimizer"):
            with torch.cuda.nvtx.range("optimizer_step"):
                optimizer.step()
    return float(loss.detach().cpu())


class TensorCopyTracer:
    """Log Python-visible Tensor.to / copy_ / contiguous calls without changing results."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._tls = threading.local()
        self._orig_to = torch.Tensor.to
        self._orig_copy_ = torch.Tensor.copy_
        self._orig_contiguous = torch.Tensor.contiguous
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        tracer = self

        def to_wrapper(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            result = tracer._orig_to(tensor, *args, **kwargs)
            tracer._record("Tensor.to", tensor, result)
            return result

        def copy_wrapper(tensor: torch.Tensor, src: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            result = tracer._orig_copy_(tensor, src, *args, **kwargs)
            tracer._record("Tensor.copy_", src, result)
            return result

        def contiguous_wrapper(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            result = tracer._orig_contiguous(tensor, *args, **kwargs)
            if result.data_ptr() != tensor.data_ptr() or result.dtype != tensor.dtype:
                tracer._record("Tensor.contiguous", tensor, result)
            return result

        torch.Tensor.to = to_wrapper  # type: ignore[method-assign]
        torch.Tensor.copy_ = copy_wrapper  # type: ignore[method-assign]
        torch.Tensor.contiguous = contiguous_wrapper  # type: ignore[method-assign]
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        torch.Tensor.to = self._orig_to  # type: ignore[method-assign]
        torch.Tensor.copy_ = self._orig_copy_  # type: ignore[method-assign]
        torch.Tensor.contiguous = self._orig_contiguous  # type: ignore[method-assign]
        self._installed = False

    def _record(self, api: str, source: torch.Tensor, dest: torch.Tensor) -> None:
        if source.device.type != "cuda" and dest.device.type != "cuda":
            return
        if tuple(source.shape) == tuple(dest.shape) and source.dtype == dest.dtype and source.data_ptr() == dest.data_ptr():
            return
        stack = _filter_stack(threading.current_thread().__dict__.get("_not_used"))
        import traceback

        frames = [
            frame.strip()
            for frame in traceback.format_stack()[:-1]
            if _keep_stack_frame(frame)
        ]
        self.records.append(
            {
                "api": api,
                "src_dtype": str(source.dtype).replace("torch.", ""),
                "dst_dtype": str(dest.dtype).replace("torch.", ""),
                "src_shape": list(source.shape),
                "dst_shape": list(dest.shape),
                "same_storage": source.data_ptr() == dest.data_ptr(),
                "stack": frames[-8:],
            }
        )


def _keep_stack_frame(frame: str) -> bool:
    skip = (
        "phase4_copy_attribution.py",
        "torch/profiler",
        "torch/autograd",
        "site-packages/torch/profiler",
        "site-packages/torch/_ops",
        "site-packages/torch/nn/modules/module.py",
        "threading.py",
        "traceback.py",
    )
    return not any(token in frame for token in skip) and (
        "megatron" in frame
        or "transformer_engine" in frame
        or "phase1_baseline.py" in frame
        or "phase3_" in frame
        or "optim/adamw" in frame
        or "torch/nn/modules/linear" in frame
        or "functional.py" in frame
    )


def _filter_stack(stack: list[str] | None) -> list[str]:
    if not stack:
        return []
    return [frame for frame in stack if _keep_stack_frame(frame)][-8:]


def event_name(event: Any) -> str:
    return str(getattr(event, "name", "") or "")


def is_copy_cpu_op(name: str) -> bool:
    return any(name == op or name.startswith(op) for op in COPY_CPU_OPS)


def is_copy_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in COPY_KERNEL_TOKENS)


def is_activation_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ACTIVATION_KERNEL_TOKENS) and "softmax" not in lowered and "sdpa" not in lowered


def kernel_family(name: str) -> str:
    lowered = name.lower()
    if "bfloat16_copy" in lowered:
        return "bfloat16_copy"
    if "direct_copy" in lowered:
        return "direct_copy"
    if "copy_kernel" in lowered:
        return "copy_kernel"
    if "load_withcast" in lowered or "store_withcast" in lowered:
        return "cast_load_store"
    return "other_copy"


def walk_ancestors(event: Any) -> list[Any]:
    ancestors = []
    current = getattr(event, "cpu_parent", None)
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        ancestors.append(current)
        current = getattr(current, "cpu_parent", None)
    return ancestors


def infer_phase(event: Any) -> str:
    for ancestor in [event, *walk_ancestors(event)]:
        name = event_name(ancestor)
        if name.startswith("phase/"):
            return name.split("/", 1)[1]
        if name in {"forward", "backward", "optimizer_step"}:
            return name.replace("optimizer_step", "optimizer")
    return "unknown"


def infer_module(event: Any) -> str:
    for ancestor in [event, *walk_ancestors(event)]:
        name = event_name(ancestor)
        if name.startswith("module/"):
            return classify_module_name(name.split("/", 1)[1]) or name.split("/", 1)[1]
        classified = classify_module_name(name)
        if classified:
            return classified
    stack = getattr(event, "stack", None) or []
    joined = " ".join(str(frame) for frame in stack).lower()
    if "linear_qkv" in joined:
        return "qkv_projection"
    if "linear_proj" in joined:
        return "attention_output_projection"
    if "linear_fc1" in joined:
        return "mlp_fc1"
    if "linear_fc2" in joined:
        return "mlp_fc2"
    if "layernorm" in joined or "layer_norm" in joined:
        return "normalization"
    if "core_attention" in joined or "tedotproduct" in joined:
        return "attention"
    if "adam" in joined or "adamw" in joined:
        return "optimizer"
    if "masked_language_model_loss" in joined or "nll" in joined:
        return "loss"
    return "unattributed"


def useful_stack(event: Any) -> list[str]:
    frames: list[str] = []
    for frame in getattr(event, "stack", None) or []:
        text = str(frame)
        if _keep_stack_frame(text):
            frames.append(text)
    return frames[-8:]


def shape_tuple(event: Any) -> tuple[Any, ...]:
    shapes = getattr(event, "input_shapes", None) or []
    if not shapes:
        return ()
    first = shapes[0]
    if first is None:
        return ()
    return tuple(first)


def classify_kind(
    phase: str,
    module: str,
    name: str,
    shapes: tuple[Any, ...],
    stack: list[str],
    src_dtype: str | None = None,
    dst_dtype: str | None = None,
) -> str:
    joined = " ".join(stack).lower() + " " + module + " " + name.lower()
    if phase == "optimizer" or module == "optimizer" or "adam" in joined:
        return "E_optimizer_related"
    same_dtype = src_dtype is not None and src_dtype == dst_dtype
    if "contiguous" in name.lower() and same_dtype:
        return "D_layout_contiguous"
    if shapes in PARAM_SHAPES or (len(shapes) == 2 and shapes in PARAM_SHAPES):
        if phase == "backward":
            return "C_gradient_dtype"
        return "A_fp32_param_to_bf16"
    if src_dtype == "float32" and dst_dtype == "bfloat16" and len(shapes) == 2:
        return "A_fp32_param_to_bf16"
    if src_dtype == "bfloat16" and dst_dtype == "float32" and phase == "backward":
        return "C_gradient_dtype"
    if src_dtype == "bfloat16" and dst_dtype == "float32" and module == "loss":
        return "B_activation_dtype"
    if "autocasttedotproduct" in joined or "core_attention" in joined:
        return "B_activation_dtype"
    if phase == "forward" and any(dim in {8, 2048} for dim in shapes):
        return "B_activation_dtype"
    if phase == "backward":
        return "C_gradient_dtype"
    if same_dtype:
        return "D_layout_contiguous"
    return "F_other"


def us_to_ms(value: float) -> float:
    return value / 1000.0


def increment(
    table: dict[str, dict[str, float]],
    key: str,
    cuda_us: float,
    cpu_us: float = 0.0,
    count: int = 1,
) -> None:
    row = table.setdefault(key, {"cuda_us": 0.0, "cpu_us": 0.0, "count": 0})
    row["cuda_us"] += cuda_us
    row["cpu_us"] += cpu_us
    row["count"] += count


def finalize_table(table: dict[str, dict[str, float]], steps: int) -> dict[str, Any]:
    return {
        key: {
            "cuda_ms_total": us_to_ms(row["cuda_us"]),
            "cuda_ms_per_step": us_to_ms(row["cuda_us"]) / steps,
            "cpu_ms_per_step": us_to_ms(row["cpu_us"]) / steps,
            "calls_per_step": row["count"] / steps,
        }
        for key, row in sorted(table.items(), key=lambda item: item[1]["cuda_us"], reverse=True)
    }


def analyze_profiler(prof: profile, steps: int) -> dict[str, Any]:
    events = list(prof.events())
    cpu_ops: dict[str, dict[str, float]] = {}
    kernels: dict[str, dict[str, float]] = {}
    phases: dict[str, dict[str, float]] = {}
    modules: dict[str, dict[str, float]] = {}
    kinds: dict[str, dict[str, float]] = {}
    families: dict[str, dict[str, float]] = {}
    shape_groups: dict[str, dict[str, float]] = {}
    overlap = {"copy_only": 0, "activation_only": 0, "both": 0, "neither": 0}
    top_kernel_details: list[dict[str, Any]] = []
    top_op_details: list[dict[str, Any]] = []

    for event in events:
        name = event_name(event)
        cuda_us = float(getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0) or 0.0)
        self_cuda_us = float(
            getattr(event, "self_device_time_total", 0.0) or getattr(event, "self_cuda_time_total", 0.0) or 0.0
        )
        cpu_us = float(getattr(event, "cpu_time_total", 0.0) or 0.0)
        device_type = str(getattr(event, "device_type", "")).lower()
        is_cuda_kernel = "cuda" in device_type and not name.startswith("aten::") and not name.startswith("phase/") and not name.startswith("module/")

        if is_copy_kernel(name) and is_activation_kernel(name):
            overlap["both"] += 1
        elif is_copy_kernel(name):
            overlap["copy_only"] += 1
        elif is_activation_kernel(name):
            overlap["activation_only"] += 1
        elif is_cuda_kernel:
            overlap["neither"] += 1

        if not (is_copy_cpu_op(name) or is_copy_kernel(name)):
            continue

        phase = infer_phase(event)
        module = infer_module(event)
        shapes = shape_tuple(event)
        stack = useful_stack(event)
        family = kernel_family(name) if is_copy_kernel(name) else name
        kind = classify_kind(phase, module, name, shapes, stack)
        charge_us = self_cuda_us if is_cuda_kernel else cuda_us
        increment(cpu_ops, name, charge_us, cpu_us)
        increment(phases, phase, charge_us, cpu_us)
        increment(modules, module, charge_us, cpu_us)
        increment(kinds, kind, charge_us, cpu_us)
        increment(families, family, charge_us, cpu_us)
        shape_key = f"{family}|{phase}|{module}|{list(shapes)}"
        increment(shape_groups, shape_key, charge_us, cpu_us)
        increment(kernels if is_copy_kernel(name) else cpu_ops, name, 0.0, 0.0, 0)

        detail = {
            "name": name,
            "family": family,
            "phase": phase,
            "module": module,
            "kind": kind,
            "input_shapes": list(getattr(event, "input_shapes", None) or []),
            "cuda_ms": us_to_ms(charge_us),
            "stack": stack,
        }
        if is_copy_kernel(name):
            top_kernel_details.append(detail)
        else:
            top_op_details.append(detail)

    top_kernel_details.sort(key=lambda item: item["cuda_ms"], reverse=True)
    top_op_details.sort(key=lambda item: item["cuda_ms"], reverse=True)

    grouped_kernels: dict[str, dict[str, Any]] = {}
    for item in top_kernel_details:
        key = f"{item['family']}|{item['phase']}|{item['module']}|{item['kind']}|{item['input_shapes']}"
        group = grouped_kernels.setdefault(
            key,
            {
                "family": item["family"],
                "phase": item["phase"],
                "module": item["module"],
                "kind": item["kind"],
                "input_shapes": item["input_shapes"],
                "example_name": item["name"],
                "example_stack": item["stack"],
                "cuda_ms_total": 0.0,
                "count": 0,
            },
        )
        group["cuda_ms_total"] += item["cuda_ms"]
        group["count"] += 1
    kernel_groups = sorted(grouped_kernels.values(), key=lambda item: item["cuda_ms_total"], reverse=True)
    for group in kernel_groups:
        group["cuda_ms_per_step"] = group["cuda_ms_total"] / steps
        group["calls_per_step"] = group["count"] / steps

    grouped_ops: dict[str, dict[str, Any]] = {}
    for item in top_op_details:
        key = f"{item['name']}|{item['phase']}|{item['module']}|{item['kind']}|{item['input_shapes']}"
        group = grouped_ops.setdefault(
            key,
            {
                "operator": item["name"],
                "phase": item["phase"],
                "module": item["module"],
                "kind": item["kind"],
                "input_shapes": item["input_shapes"],
                "example_stack": item["stack"],
                "cuda_ms_total": 0.0,
                "count": 0,
            },
        )
        group["cuda_ms_total"] += item["cuda_ms"]
        group["count"] += 1
    op_groups = sorted(grouped_ops.values(), key=lambda item: item["cuda_ms_total"], reverse=True)
    for group in op_groups:
        group["cuda_ms_per_step"] = group["cuda_ms_total"] / steps
        group["calls_per_step"] = group["count"] / steps

    return {
        "copy_cpu_operators": finalize_table(cpu_ops, steps),
        "copy_kernels": finalize_table(
            {name: row for name, row in kernels.items() if row["count"] or row["cuda_us"]},
            steps,
        ),
        "by_phase": finalize_table(phases, steps),
        "by_module": finalize_table(modules, steps),
        "by_kind": finalize_table(kinds, steps),
        "by_kernel_family": finalize_table(families, steps),
        "top_copy_kernel_groups": kernel_groups[:25],
        "top_copy_operator_groups": op_groups[:25],
        "category_overlap": {
            "copy_cast_only_kernel_events": overlap["copy_only"],
            "activation_elementwise_only_kernel_events": overlap["activation_only"],
            "both_categories_kernel_events": overlap["both"],
            "other_cuda_kernel_events": overlap["neither"],
            "overlap_exists": overlap["both"] > 0,
            "note": (
                "Phase 3.4 matchers applied to profiler CUDA kernel names. "
                "copy_cast uses bfloat16_copy/direct_copy/copy_kernel/load_withcast/store_withcast; "
                "activation_elementwise uses gelu/dropout/lerp/addcmul/*functor/silu/relu."
            ),
        },
    }


def summarize_python_tracer(records: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = (
            f"{record['api']}|{record['src_dtype']}->{record['dst_dtype']}|"
            f"{record['src_shape']}"
        )
        group = grouped.setdefault(
            key,
            {
                "api": record["api"],
                "src_dtype": record["src_dtype"],
                "dst_dtype": record["dst_dtype"],
                "src_shape": record["src_shape"],
                "example_stack": record["stack"],
                "count": 0,
            },
        )
        group["count"] += 1
    rows = sorted(grouped.values(), key=lambda item: item["count"], reverse=True)
    for row in rows:
        row["calls_per_step"] = row["count"] / steps
        src = row["src_dtype"]
        dst = row["dst_dtype"]
        shape = tuple(row["src_shape"])
        if src == "float32" and dst == "bfloat16" and shape in PARAM_SHAPES:
            row["kind"] = "A_fp32_param_to_bf16"
        elif src == dst:
            row["kind"] = "D_layout_contiguous"
        elif src == "float32" and dst == "bfloat16":
            row["kind"] = "B_activation_dtype"
        elif src == "bfloat16" and dst == "float32":
            row["kind"] = "C_gradient_dtype_or_loss_upcast"
        else:
            row["kind"] = "F_other"
    return {
        "python_visible_copy_groups": rows[:40],
        "python_visible_copy_count_total": len(records),
        "python_visible_copy_count_per_step": len(records) / steps,
        "note": (
            "Python Tensor.to/copy_/contiguous tracer. Autocast weight casts that stay in C++ "
            "may be missing here and appear only in the profiler aten::_to_copy / kernel groups."
        ),
    }


def infer_root_causes(analysis: dict[str, Any], tracer: dict[str, Any]) -> dict[str, Any]:
    bf16_groups = [row for row in analysis["top_copy_kernel_groups"] if row["family"] == "bfloat16_copy"]
    direct_groups = [row for row in analysis["top_copy_kernel_groups"] if row["family"] == "direct_copy"]
    kind_share = analysis["by_kind"]
    dominant_kind = next(iter(kind_share), "F_other")

    def describe(groups: list[dict[str, Any]], family: str) -> dict[str, Any]:
        if not groups:
            return {"family": family, "conclusion": "not observed in profiler window"}
        top = groups[0]
        return {
            "family": family,
            "dominant_phase": top["phase"],
            "dominant_module": top["module"],
            "dominant_kind": top["kind"],
            "dominant_shapes": top["input_shapes"],
            "cuda_ms_per_step": top["cuda_ms_per_step"],
            "calls_per_step": top["calls_per_step"],
            "example_stack": top["example_stack"],
            "all_group_count": len(groups),
        }

    if dominant_kind.startswith("A"):
        recommendation = (
            "Keep FP32 parameters, but stop recasting every Linear weight/bias from FP32 to BF16 "
            "on each forward (and the matching backward casts). The next test should cache a BF16 "
            "weight view once per step or replace only the local Megatron Linear GEMM path with an "
            "autocast-aware BF16 GEMM that does not emit a full bfloat16_copy of the parameter."
        )
        target = "cached_or_fused_fp32_param_to_bf16_linear_cast"
    elif dominant_kind.startswith("B"):
        recommendation = (
            "Reduce activation dtype conversions, starting with the explicit QKV "
            "AutocastTEDotProductAttention .to(bf16) and any leftover activation up/down-casts."
        )
        target = "qkv_and_activation_cast_elimination"
    else:
        recommendation = (
            "Target the largest classified copy kind from this capture; do not change attention "
            "backend, batch size, or optimizer until that copy class is reduced."
        )
        target = dominant_kind

    return {
        "bfloat16_copy": describe(bf16_groups, "bfloat16_copy"),
        "direct_copy": describe(direct_groups, "direct_copy"),
        "dominant_copy_kind": dominant_kind,
        "recommended_phase_4_2_optimization": target,
        "recommended_phase_4_2_rationale": recommendation,
        "python_tracer_top": tracer["python_visible_copy_groups"][:5],
    }


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_single_gpu_distributed(model_args.seed)
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("TE fused attention requires Transformer Engine")
        device = torch.device(f"cuda:{local_rank}")
        model = build_model(model_args, attention_implementation=TE_FUSED_ATTENTION)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_args.learning_rate,
            foreach=False,
            fused=False,
        )
        batch = synthetic_batch(model_args, args.micro_batch_size, device)
        handles = install_module_ranges(model)
        for _ in range(args.warmup_iterations):
            train_step(model, optimizer, batch)
            torch.cuda.synchronize(device)
        backend = fused_backend_status()
        print("FusedAttention backend (sub-backend 1)")

        tracer = TensorCopyTracer()
        tracer.install()
        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                with_modules=True,
                profile_memory=False,
            ) as prof:
                for step_index in range(args.measured_iterations):
                    attributed_train_step(model, optimizer, batch, step_index)
                    torch.cuda.synchronize(device)
        finally:
            tracer.restore()
            for handle in handles:
                handle.remove()

        analysis = analyze_profiler(prof, args.measured_iterations)
        tracer_summary = summarize_python_tracer(tracer.records, args.measured_iterations)
        root = infer_root_causes(analysis, tracer_summary)
        result = {
            "status": "success",
            "experiment": "Phase 4.1 copy/cast root-cause attribution",
            "ncu_attempted": False,
            "nsys_attempted": False,
            "optimization_applied": False,
            "infrastructure": {
                "pod_id": os.environ.get("RUNPOD_POD_ID", "3ixl2btmmwghn5"),
                "gpu": "1x NVIDIA A40 48GB",
                "gpu_price_per_hour_usd": 0.44,
            },
            "configuration": {
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "model_config": model_config(TE_FUSED_ATTENTION),
                "micro_batch_size": args.micro_batch_size,
                "sequence_length": model_args.sequence_length,
                "precision": {
                    "forward_backward": "BF16 autocast",
                    "parameter_storage": "FP32",
                    "optimizer_state": "FP32",
                },
                "optimizer": {"name": "torch.optim.AdamW", "foreach": False, "fused": False},
                "parallelism": {"tensor_parallel": 1, "pipeline_parallel": 1, "data_parallel": 1},
                "warmup_iterations": args.warmup_iterations,
                "profiled_iterations": args.measured_iterations,
            },
            "transformer_engine_backend": backend,
            "profiler": {
                "activities": ["cpu", "cuda"],
                "record_shapes": True,
                "with_stack": True,
                "with_modules": True,
            },
            "attribution": analysis,
            "python_tracer": tracer_summary,
            "root_causes": root,
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE4_COPY_ATTRIBUTION_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
