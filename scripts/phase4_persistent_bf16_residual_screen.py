#!/usr/bin/env python3
"""Phase 4.4 fast screen for a persistent BF16 Transformer residual stream."""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from megatron.core import parallel_state

from phase1_baseline import (
    TE_FUSED_ATTENTION,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
    train_step,
)
from phase3_attention_correctness import fused_backend_status
from phase3_attention_profile import baseline_model_args, model_config
from phase4_bf16_residual_screen import (
    COSINE_STRONG_ALIGN,
    MICRO_BATCH_SIZE,
    PHASE41_BF16_COPY_MS,
    PHASE41_COPY_CAST_MS,
    PHASE41_DIRECT_COPY_MS,
    THROUGHPUT_PROMISING_PERCENT,
    make_model,
    run_correctness,
    run_performance,
    summarize_copy_profile,
)


COPY_REDUCTION_MATERIAL_PERCENT = 20.0
EXPECTED_LAYER_COUNT = 24
EXPECTED_DTYPE = "bfloat16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=200)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def dtype_name(value: Any) -> str | None:
    if not torch.is_tensor(value):
        return None
    return str(value.dtype).replace("torch.", "")


def install_bda_trace(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    profiler_ranges: bool,
) -> list[tuple[torch.nn.Module, str, Any]]:
    """Wrap BDA factories while retaining their JIT-fused inner functions."""
    originals: list[tuple[torch.nn.Module, str, Any]] = []

    def wrap_factory(
        original_factory: Callable[..., Any],
        layer_index: int,
        site: str,
    ) -> Callable[..., Any]:
        def traced_factory(training: bool, fused: bool):
            inner = original_factory(training, fused)

            def traced_bda(
                x_with_bias: tuple[torch.Tensor, torch.Tensor | None],
                residual: torch.Tensor,
                prob: float,
            ) -> torch.Tensor:
                x, bias = x_with_bias
                range_context = (
                    record_function(f"module/{site}")
                    if profiler_ranges
                    else torch.autograd.profiler.record_function("phase44/bda_trace")
                )
                with range_context:
                    output = inner(x_with_bias, residual, prob)
                records.append(
                    {
                        "layer_index": layer_index,
                        "site": site,
                        "x_dtype": dtype_name(x),
                        "bias_dtype": dtype_name(bias),
                        "residual_dtype": dtype_name(residual),
                        "output_dtype": dtype_name(output),
                        "x_cast_required": x.dtype != residual.dtype,
                        "bias_cast_required": bias is not None and bias.dtype != residual.dtype,
                    }
                )
                return output

            return traced_bda

        return traced_factory

    for layer_index, layer in enumerate(model.decoder.layers):
        for attribute, site in (
            ("self_attn_bda", "persistent_attention_bda"),
            ("mlp_bda", "persistent_mlp_bda"),
        ):
            original = getattr(layer, attribute)
            originals.append((layer, attribute, original))
            setattr(layer, attribute, wrap_factory(original, layer_index, site))
    return originals


def restore_bda_trace(originals: list[tuple[torch.nn.Module, str, Any]]) -> None:
    for layer, attribute, original in originals:
        setattr(layer, attribute, original)


def run_dtype_lifecycle(model_args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Gate all later work on persistent BF16 boundaries in all 24 layers."""
    model = make_model(
        model_args,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16_hidden_residual=True,
    )
    model.eval()
    batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
    layer_records: dict[int, dict[str, Any]] = {
        index: {"layer_index": index} for index in range(len(model.decoder.layers))
    }
    bda_records: list[dict[str, Any]] = []
    handles: list[Any] = []

    def make_pre_hook(index: int):
        def pre_hook(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            layer_records[index]["layer_input_dtype"] = dtype_name(hidden_states)

        return pre_hook

    def make_post_hook(index: int):
        def post_hook(_module: torch.nn.Module, _args: Any, output: Any) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            layer_records[index]["layer_output_dtype"] = dtype_name(hidden_states)

        return post_hook

    for index, layer in enumerate(model.decoder.layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_post_hook(index)))

    final_layernorm_input: dict[str, Any] = {}

    def final_norm_pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        final_layernorm_input["dtype"] = dtype_name(args[0] if args else None)

    if model.decoder.final_layernorm is not None:
        handles.append(model.decoder.final_layernorm.register_forward_pre_hook(final_norm_pre_hook))

    originals = install_bda_trace(model, bda_records, profiler_ranges=False)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            output = model(
                batch["tokens"],
                batch["position_ids"],
                batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = masked_language_model_loss(output, batch["loss_mask"])
        torch.cuda.synchronize(device)
    finally:
        restore_bda_trace(originals)
        for handle in handles:
            handle.remove()

    for record in bda_records:
        row = layer_records[record["layer_index"]]
        prefix = "attention" if record["site"] == "persistent_attention_bda" else "mlp"
        row[f"{prefix}_bda_x_dtype"] = record["x_dtype"]
        row[f"{prefix}_residual_dtype"] = record["residual_dtype"]
        row[f"{prefix}_bda_output_dtype"] = record["output_dtype"]
        row[f"{prefix}_bias_dtype_before_cast"] = record["bias_dtype"]

    required_fields = (
        "layer_input_dtype",
        "attention_residual_dtype",
        "attention_bda_output_dtype",
        "mlp_residual_dtype",
        "mlp_bda_output_dtype",
        "layer_output_dtype",
    )
    failures: list[dict[str, Any]] = []
    for index, row in layer_records.items():
        for field in required_fields:
            if row.get(field) != EXPECTED_DTYPE:
                failures.append(
                    {
                        "layer_index": index,
                        "field": field,
                        "expected": EXPECTED_DTYPE,
                        "observed": row.get(field),
                    }
                )
    if len(layer_records) != EXPECTED_LAYER_COUNT:
        failures.append(
            {
                "field": "layer_count",
                "expected": EXPECTED_LAYER_COUNT,
                "observed": len(layer_records),
            }
        )
    if final_layernorm_input.get("dtype") != EXPECTED_DTYPE:
        failures.append(
            {
                "field": "final_layernorm_input_dtype",
                "expected": EXPECTED_DTYPE,
                "observed": final_layernorm_input.get("dtype"),
            }
        )

    result = {
        "passed": not failures,
        "expected_layer_count": EXPECTED_LAYER_COUNT,
        "observed_layer_count": len(layer_records),
        "expected_boundary_dtype": EXPECTED_DTYPE,
        "all_24_layer_boundaries_bf16": not failures,
        "final_layernorm_input_dtype": final_layernorm_input.get("dtype"),
        "loss_reduction_dtype": dtype_name(loss),
        "parameter_dtypes": sorted(
            {str(parameter.dtype).replace("torch.", "") for parameter in model.parameters()}
        ),
        "layers": list(layer_records.values()),
        "failures": failures,
        "bda_calls": len(bda_records),
    }
    del model, output, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


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


class TensorToTracer:
    """Capture Python-visible CUDA Tensor.to transitions and their source stacks."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._orig_to = torch.Tensor.to
        self._tls = threading.local()

    def install(self) -> None:
        tracer = self

        def to_wrapper(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            result = tracer._orig_to(tensor, *args, **kwargs)
            if getattr(tracer._tls, "recording", False):
                return result
            if (
                tensor.device.type == "cuda"
                and result.device.type == "cuda"
                and tensor.dtype != result.dtype
            ):
                tracer._tls.recording = True
                try:
                    frames = [
                        frame.strip()
                        for frame in traceback.format_stack()[:-1]
                        if any(
                            token in frame
                            for token in (
                                "megatron",
                                "transformer_engine",
                                "phase1_baseline.py",
                                "phase4_",
                            )
                        )
                        and "to_wrapper" not in frame
                    ]
                    tracer.records.append(
                        {
                            "src_dtype": dtype_name(tensor),
                            "dst_dtype": dtype_name(result),
                            "shape": list(tensor.shape),
                            "stack": frames[-8:],
                        }
                    )
                finally:
                    tracer._tls.recording = False
            return result

        torch.Tensor.to = to_wrapper  # type: ignore[method-assign]

    def restore(self) -> None:
        torch.Tensor.to = self._orig_to  # type: ignore[method-assign]


def walk_ancestors(event: Any) -> list[Any]:
    ancestors: list[Any] = []
    current = getattr(event, "cpu_parent", None)
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        ancestors.append(current)
        current = getattr(current, "cpu_parent", None)
    return ancestors


def event_name(event: Any) -> str:
    return str(getattr(event, "name", "") or "")


def infer_module(event: Any) -> str:
    for ancestor in [event, *walk_ancestors(event)]:
        name = event_name(ancestor)
        if name.startswith("module/"):
            raw = name.split("/", 1)[1]
            if raw in {"persistent_attention_bda", "persistent_mlp_bda"}:
                return raw
            return classify_module_name(raw) or raw
    return "unattributed"


def shape_tuple(event: Any) -> tuple[Any, ...]:
    shapes = getattr(event, "input_shapes", None) or []
    if not shapes or shapes[0] is None:
        return ()
    return tuple(shapes[0])


def classify_remaining_source(module: str, shape: tuple[Any, ...]) -> str:
    if module in {"persistent_attention_bda", "persistent_mlp_bda"}:
        return "persistent_bda_bias_cast"
    if module in {"qkv_projection", "mlp_fc1"}:
        return "layernorm_to_linear_autocast"
    if module == "mlp_fc2":
        return "fc1_bias_gelu_promotion_then_fc2_autocast"
    if module == "attention" and shape == (2048, 8, 16, 64):
        return "qkv_adapter"
    return "other"


def summarize_aten_copy_sources(prof: profile, steps: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str, tuple[Any, ...]], dict[str, Any]] = {}
    source_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"cuda_ms": 0.0, "calls": 0.0}
    )
    for event in prof.events():
        if event_name(event) != "aten::copy_":
            continue
        module = infer_module(event)
        shape = shape_tuple(event)
        source = classify_remaining_source(module, shape)
        cuda_ms = float(
            getattr(event, "device_time_total", 0.0)
            or getattr(event, "cuda_time_total", 0.0)
            or 0.0
        ) / 1000.0
        key = (source, module, shape)
        row = grouped.setdefault(
            key,
            {
                "source": source,
                "module": module,
                "shape": list(shape),
                "cuda_ms_total": 0.0,
                "calls": 0,
            },
        )
        row["cuda_ms_total"] += cuda_ms
        row["calls"] += 1
        source_totals[source]["cuda_ms"] += cuda_ms
        source_totals[source]["calls"] += 1

    groups = sorted(grouped.values(), key=lambda row: row["cuda_ms_total"], reverse=True)
    for row in groups:
        row["cuda_ms_per_step"] = row.pop("cuda_ms_total") / steps
        row["calls_per_step"] = row.pop("calls") / steps
    return {
        "by_source": {
            source: {
                "cuda_ms_per_step": values["cuda_ms"] / steps,
                "calls_per_step": values["calls"] / steps,
            }
            for source, values in sorted(
                source_totals.items(),
                key=lambda item: item[1]["cuda_ms"],
                reverse=True,
            )
        },
        "top_groups": groups[:30],
        "classification_note": (
            "Uses unique aten::copy_ events. Forward module ranges identify QKV/FC1 "
            "LayerNorm-to-Linear autocast, FC1-bias/GELU-to-FC2 autocast, QKV adapter, "
            "and the corrected BDA bias casts. Unparented backward/layout work is 'other'."
        ),
    }


def summarize_tensor_to(records: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str, tuple[int, ...], str], dict[str, Any]] = {}
    for record in records:
        joined = " ".join(record["stack"])
        if "AutocastTEDotProductAttention" in joined or "phase1_baseline.py" in joined:
            source = "qkv_adapter"
        elif "tensor_parallel/layers.py" in joined:
            source = "linear_backward"
        elif "layer_norm.py" in joined:
            source = "te_final_layernorm"
        elif "bf16_hidden_residual.py" in joined:
            source = "persistent_bda"
        else:
            source = "other_python_visible"
        key = (
            record["src_dtype"],
            record["dst_dtype"],
            tuple(record["shape"]),
            source,
        )
        row = grouped.setdefault(
            key,
            {
                "src_dtype": record["src_dtype"],
                "dst_dtype": record["dst_dtype"],
                "shape": record["shape"],
                "source": source,
                "example_stack": record["stack"],
                "count": 0,
            },
        )
        row["count"] += 1
    rows = sorted(grouped.values(), key=lambda row: row["count"], reverse=True)
    for row in rows:
        row["calls_per_step"] = row["count"] / steps
    return {"groups": rows[:40], "calls_per_step": len(records) / steps}


def run_variant_profiler(
    model_args: argparse.Namespace,
    device: torch.device,
    warmup_iterations: int,
    profile_iterations: int,
) -> dict[str, Any]:
    model = make_model(
        model_args,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        bf16_hidden_residual=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_args.learning_rate,
        foreach=False,
        fused=False,
    )
    batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
    for _ in range(warmup_iterations):
        train_step(model, optimizer, batch)
        torch.cuda.synchronize(device)

    handles = install_module_ranges(model)
    bda_records: list[dict[str, Any]] = []
    originals = install_bda_trace(model, bda_records, profiler_ranges=True)
    tracer = TensorToTracer()
    tracer.install()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            profile_memory=False,
        ) as prof:
            for step_index in range(profile_iterations):
                with record_function("phase/train_step"):
                    train_step(model, optimizer, batch)
                torch.cuda.synchronize(device)
                print(f"PHASE44_PROFILE_STEP={step_index + 1}")
    finally:
        tracer.restore()
        restore_bda_trace(originals)
        for handle in handles:
            handle.remove()

    aggregate = summarize_copy_profile(prof, profile_iterations)
    aten_sources = summarize_aten_copy_sources(prof, profile_iterations)
    tensor_to = summarize_tensor_to(tracer.records, profile_iterations)
    x_casts = [record for record in bda_records if record["x_cast_required"]]
    bias_casts = [record for record in bda_records if record["bias_cast_required"]]
    residual_bf16_to_fp32 = [
        record
        for record in x_casts
        if record["x_dtype"] == "bfloat16" and record["residual_dtype"] == "float32"
    ]
    result = {
        **aggregate,
        "aten_copy_source_attribution": aten_sources,
        "python_tensor_to": tensor_to,
        "bda_operand_trace": {
            "calls_per_step": len(bda_records) / profile_iterations,
            "x_casts_required_per_step": len(x_casts) / profile_iterations,
            "bias_casts_required_per_step": len(bias_casts) / profile_iterations,
            "residual_bf16_to_fp32_casts_per_step": len(residual_bf16_to_fp32)
            / profile_iterations,
            "residual_upcast_eliminated": not residual_bf16_to_fp32,
            "sample_records": bda_records[:8],
        },
    }
    del model, optimizer, prof
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(
    lifecycle: dict[str, Any],
    correctness: dict[str, Any],
    variant_a: dict[str, Any],
    variant_b: dict[str, Any],
    profile_b: dict[str, Any],
) -> dict[str, Any]:
    no_nonfinite = not (
        correctness["loss"]["baseline_nonfinite"]
        or correctness["loss"]["bf16_residual_nonfinite"]
        or correctness["forward"]["baseline_nonfinite"]
        or correctness["forward"]["bf16_residual_nonfinite"]
        or correctness["gradients"]["any_nonfinite"]
    )
    gradients_aligned = (
        correctness["gradients"]["min_cosine_similarity"] >= COSINE_STRONG_ALIGN
    )
    throughput_delta_percent = (
        (variant_b["tokens_per_second"] - variant_a["tokens_per_second"])
        / variant_a["tokens_per_second"]
        * 100.0
    )
    copy_ms = profile_b["copy_cast_kernel_ms_per_step"]
    copy_reduction_percent = (PHASE41_COPY_CAST_MS - copy_ms) / PHASE41_COPY_CAST_MS * 100.0
    material_copy_reduction = copy_reduction_percent >= COPY_REDUCTION_MATERIAL_PERCENT
    throughput_gate = throughput_delta_percent >= THROUGHPUT_PROMISING_PERCENT
    promising = (
        lifecycle["passed"]
        and no_nonfinite
        and gradients_aligned
        and material_copy_reduction
        and throughput_gate
    )
    return {
        "promising": promising,
        "hypothesis_supported": promising,
        "full_20_plus_100_benchmark_run": False,
        "gates": {
            "all_layer_boundaries_bf16": lifecycle["passed"],
            "no_nan_inf": no_nonfinite,
            "gradients_strongly_aligned": gradients_aligned,
            "min_cosine_similarity": correctness["gradients"]["min_cosine_similarity"],
            "copy_cast_decreased_materially": material_copy_reduction,
            "copy_reduction_threshold_percent": COPY_REDUCTION_MATERIAL_PERCENT,
            "copy_cast_reduction_percent_vs_phase41": copy_reduction_percent,
            "throughput_improved_at_least_5_percent": throughput_gate,
            "throughput_delta_percent": throughput_delta_percent,
        },
        "copy_cast_ms_per_step": {
            "phase41": PHASE41_COPY_CAST_MS,
            "variant_b": copy_ms,
            "reduction_ms": PHASE41_COPY_CAST_MS - copy_ms,
            "reduction_percent": copy_reduction_percent,
        },
        "bfloat16_copy_ms_per_step": {
            "phase41": PHASE41_BF16_COPY_MS,
            "variant_b": profile_b["families"]
            .get("bfloat16_copy", {})
            .get("cuda_ms_per_step", 0.0),
        },
        "direct_copy_ms_per_step": {
            "phase41": PHASE41_DIRECT_COPY_MS,
            "variant_b": profile_b["families"]
            .get("direct_copy", {})
            .get("cuda_ms_per_step", 0.0),
        },
        "rationale": (
            "Persistent BF16 residuals passed all gates, materially reduced copies, and "
            "improved throughput by at least 5%."
            if promising
            else "Stop after the fast screen: at least one lifecycle, correctness, copy, "
            "or throughput gate failed. Do not run 20+100."
        ),
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_single_gpu_distributed(model_args.seed)
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("TE fused attention requires TRANSFORMER_ENGINE_DISABLE to be unset")
        device = torch.device(f"cuda:{local_rank}")

        lifecycle = run_dtype_lifecycle(model_args, device)
        backend = fused_backend_status()
        print("FusedAttention backend (sub-backend 1)")
        print("PHASE44_LIFECYCLE=" + json.dumps(lifecycle, sort_keys=True))
        if not lifecycle["passed"]:
            result = {
                "status": "lifecycle_gate_failed",
                "experiment": "Phase 4.4 persistent BF16 residual stream screen",
                "lifecycle": lifecycle,
                "transformer_engine_backend": backend,
                "benchmark_attempted": False,
                "profile_attempted": False,
                "environment": collect_environment(),
            }
            write_result(args.output_json, result)
            return

        correctness = run_correctness(model_args, device)
        print("PHASE44_CORRECTNESS_DONE")
        variant_a = run_performance(
            model_args,
            device,
            bf16_hidden_residual=False,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
            gpu_sample_interval_ms=args.gpu_sample_interval_ms,
        )
        print("PHASE44_VARIANT_A_DONE")
        variant_b = run_performance(
            model_args,
            device,
            bf16_hidden_residual=True,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
            gpu_sample_interval_ms=args.gpu_sample_interval_ms,
        )
        print("PHASE44_VARIANT_B_DONE")
        profile_b = run_variant_profiler(
            model_args,
            device,
            warmup_iterations=args.warmup_iterations,
            profile_iterations=args.profile_iterations,
        )
        print("PHASE44_PROFILE_B_DONE")
        decision = decide(lifecycle, correctness, variant_a, variant_b, profile_b)
        bfloat16_ms = (
            profile_b["families"].get("bfloat16_copy", {}).get("cuda_ms_per_step", 0.0)
        )
        direct_ms = (
            profile_b["families"].get("direct_copy", {}).get("cuda_ms_per_step", 0.0)
        )
        result = {
            "status": "success",
            "experiment": "Phase 4.4 persistent BF16 residual stream screen",
            "optimization_applied": True,
            "full_benchmark_run": False,
            "ncu_attempted": False,
            "nsys_attempted": False,
            "infrastructure": {
                "pod_id": os.environ.get("RUNPOD_POD_ID"),
                "gpu": "1x NVIDIA A40 48GB",
                "gpu_price_per_hour_usd": 0.44,
            },
            "configuration": {
                "parameter_count": variant_a["parameter_count"],
                "model_config": model_config(TE_FUSED_ATTENTION),
                "micro_batch_size": MICRO_BATCH_SIZE,
                "sequence_length": model_args.sequence_length,
                "precision": {
                    "forward_backward": "BF16 autocast",
                    "parameter_storage": "FP32",
                    "optimizer_state": "FP32",
                    "layernorm": "unchanged",
                    "loss_reduction": "FP32",
                    "variant_a_residual": "FP32",
                    "variant_b_residual": "persistent BF16",
                },
                "optimizer": {"name": "torch.optim.AdamW", "foreach": False, "fused": False},
                "parallelism": {"tensor_parallel": 1, "pipeline_parallel": 1, "data_parallel": 1},
                "warmup_iterations": args.warmup_iterations,
                "measured_iterations": args.measured_iterations,
                "profile_iterations": args.profile_iterations,
            },
            "dtype_change": {
                "repository": "lab only; upstream Megatron-LM unchanged",
                "file": "scripts/bf16_hidden_residual.py",
                "operations": [
                    "one-time decoder-entry hidden_states.to(bfloat16)",
                    "x.to(residual.dtype) only when x differs",
                    "bias.to(residual.dtype) independently when bias differs",
                ],
            },
            "transformer_engine_backend": backend,
            "lifecycle": lifecycle,
            "correctness": correctness,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "profile_b": profile_b,
            "comparison": {
                "step_time_ms": {
                    "A": variant_a["average_step_time_ms"],
                    "B": variant_b["average_step_time_ms"],
                    "delta_ms": variant_b["average_step_time_ms"]
                    - variant_a["average_step_time_ms"],
                    "improvement_percent": (
                        (variant_a["average_step_time_ms"] - variant_b["average_step_time_ms"])
                        / variant_a["average_step_time_ms"]
                        * 100.0
                    ),
                },
                "tokens_per_second": {
                    "A": variant_a["tokens_per_second"],
                    "B": variant_b["tokens_per_second"],
                    "delta_percent": (
                        (variant_b["tokens_per_second"] - variant_a["tokens_per_second"])
                        / variant_a["tokens_per_second"]
                        * 100.0
                    ),
                },
                "mfu_percent": {
                    "A": variant_a["mfu"]["mfu_percent"],
                    "B": variant_b["mfu"]["mfu_percent"],
                    "delta_points": variant_b["mfu"]["mfu_percent"]
                    - variant_a["mfu"]["mfu_percent"],
                },
                "peak_allocated_memory_mib": {
                    "A": variant_a["peak_allocated_memory_mib"],
                    "B": variant_b["peak_allocated_memory_mib"],
                    "delta_mib": variant_b["peak_allocated_memory_mib"]
                    - variant_a["peak_allocated_memory_mib"],
                },
                "copy_cast_ms_per_step": {
                    "phase41": PHASE41_COPY_CAST_MS,
                    "B": profile_b["copy_cast_kernel_ms_per_step"],
                    "reduction_ms": PHASE41_COPY_CAST_MS
                    - profile_b["copy_cast_kernel_ms_per_step"],
                    "reduction_percent": (
                        (PHASE41_COPY_CAST_MS - profile_b["copy_cast_kernel_ms_per_step"])
                        / PHASE41_COPY_CAST_MS
                        * 100.0
                    ),
                },
                "bfloat16_copy_ms_per_step": {
                    "phase41": PHASE41_BF16_COPY_MS,
                    "B": bfloat16_ms,
                    "reduction_ms": PHASE41_BF16_COPY_MS - bfloat16_ms,
                },
                "direct_copy_ms_per_step": {
                    "phase41": PHASE41_DIRECT_COPY_MS,
                    "B": direct_ms,
                    "reduction_ms": PHASE41_DIRECT_COPY_MS - direct_ms,
                },
            },
            "decision": decision,
            "environment": collect_environment(),
        }
        write_result(args.output_json, result)
        print("PHASE44_DECISION=" + json.dumps(decision, sort_keys=True))
        print("PHASE44_SCREEN_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
