#!/usr/bin/env python3
"""Run one Phase 7 MCore DDP tensor-parallel training variant."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:  # pragma: no cover - TE is required on the GPU pod
    TEColumnParallelLinear = None
    TELayerNormColumnParallelLinear = None
    TERowParallelLinear = None
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from phase1_baseline import (
    A40_DENSE_BF16_PEAK_TFLOPS,
    TE_FUSED_ATTENTION,
    build_model,
    masked_language_model_loss,
    synthetic_batch,
    training_flops_per_iteration,
)
from phase3_attention_correctness import fused_backend_status
from phase3_attention_profile import baseline_model_args
from phase6_megatron_ddp_lifecycle import (
    DDPOptimizerBundle,
    assert_main_grad_pointers,
    assert_optimizer_consumed_main_grad,
    assert_zeroed_lifecycle,
    build_ddp_optimizer_bundle,
    finalize_gradients,
    lifecycle_metadata,
    main_grad_pointers,
    named_trainable_parameters,
    unwrap_model,
    zero_gradients,
)


MICRO_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--te-linear", action="store_true")
    parser.add_argument("--tp-comm-overlap", action="store_true")
    parser.add_argument("--smoke-iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=100)
    parser.add_argument("--profile-mode", action="store_true")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    for field in ("smoke_iterations", "warmup_iterations", "measured_iterations"):
        if getattr(args, field) < 1:
            parser.error(f"{field.replace('_', ' ')} must be positive")
    return args


def run_command(command: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def initialize_distributed(seed: int, tensor_parallel_size: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    if dist.get_world_size() != tensor_parallel_size:
        raise RuntimeError(
            f"world size {dist.get_world_size()} != TP size {tensor_parallel_size}"
        )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    return local_rank


class MultiGpuNvidiaSmiSampler:
    """Sample utilization and memory independently for selected GPUs."""

    def __init__(self, device_indices: list[int], interval_ms: int) -> None:
        self.device_indices = device_indices
        self.interval_ms = interval_ms
        self.utilization: dict[int, list[float]] = defaultdict(list)
        self.memory_mib: dict[int, list[float]] = defaultdict(list)
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-lms",
                str(self.interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        selected = set(self.device_indices)
        for line in self._process.stdout:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                device = int(fields[0])
                if device not in selected:
                    continue
                self.utilization[device].append(float(fields[1]))
                self.memory_mib[device].append(float(fields[2]))
            except ValueError:
                continue

    def stop(self) -> dict[str, Any]:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        return {
            str(device): {
                "sample_count": len(self.utilization[device]),
                "average_utilization_percent": (
                    statistics.fmean(self.utilization[device])
                    if self.utilization[device]
                    else None
                ),
                "median_utilization_percent": (
                    statistics.median(self.utilization[device])
                    if self.utilization[device]
                    else None
                ),
                "peak_memory_mib": (
                    max(self.memory_mib[device]) if self.memory_mib[device] else None
                ),
            }
            for device in self.device_indices
        }


def module_metadata(name: str, module: torch.nn.Module) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    return {
        "name": name,
        "type": f"{type(module).__module__}.{type(module).__name__}",
        "weight_shape": list(weight.shape) if isinstance(weight, torch.Tensor) else None,
        "weight_numel": weight.numel() if isinstance(weight, torch.Tensor) else None,
    }


def sharding_metadata(model: torch.nn.Module, tensor_parallel_size: int) -> dict[str, Any]:
    raw_model = unwrap_model(model)
    if raw_model.config.tensor_model_parallel_size != tensor_parallel_size:
        raise RuntimeError(
            "TransformerConfig TP size does not match the initialized TP group: "
            f"{raw_model.config.tensor_model_parallel_size} != {tensor_parallel_size}"
        )
    layer = raw_model.decoder.layers[0]
    modules = {
        "embedding": raw_model.embedding.word_embeddings,
        "attention_qkv": layer.self_attention.linear_qkv,
        "attention_projection": layer.self_attention.linear_proj,
        "mlp_fc1": layer.mlp.linear_fc1,
        "mlp_fc2": layer.mlp.linear_fc2,
        "output_layer": raw_model.output_layer,
    }
    te_column = tuple(
        cls
        for cls in (TEColumnParallelLinear, TELayerNormColumnParallelLinear)
        if cls is not None
    )
    te_row = tuple(cls for cls in (TERowParallelLinear,) if cls is not None)
    use_te_linear = bool(te_column) and any(
        isinstance(modules[name], te_column)
        for name in ("attention_qkv", "mlp_fc1")
    )
    if use_te_linear:
        expected_types: dict[str, Any] = {
            "embedding": VocabParallelEmbedding,
            "attention_qkv": te_column,
            "attention_projection": te_row,
            "mlp_fc1": te_column,
            "mlp_fc2": te_row,
            "output_layer": (ColumnParallelLinear, *te_column),
        }
    else:
        expected_types = {
            "embedding": VocabParallelEmbedding,
            "attention_qkv": ColumnParallelLinear,
            "attention_projection": RowParallelLinear,
            "mlp_fc1": ColumnParallelLinear,
            "mlp_fc2": RowParallelLinear,
            "output_layer": ColumnParallelLinear,
        }
    type_checks = {
        name: isinstance(module, expected_types[name]) for name, module in modules.items()
    }
    if not all(type_checks.values()):
        raise RuntimeError(f"Model did not select expected TP modules: {type_checks}")

    metadata = {
        name: module_metadata(name, module) for name, module in modules.items()
    }
    output_weight = getattr(modules["output_layer"], "weight", None)
    embedding_weight = modules["embedding"].weight
    metadata["output_layer"]["effective_weight_shape"] = list(
        (output_weight if output_weight is not None else embedding_weight).shape
    )
    metadata["output_layer"]["tied_to_embedding"] = (
        output_weight is None
        or output_weight.data_ptr() == embedding_weight.data_ptr()
    )
    metadata["type_checks"] = type_checks
    metadata["tensor_parallel_size"] = tensor_parallel_size
    metadata["transformer_config_tensor_parallel_size"] = (
        raw_model.config.tensor_model_parallel_size
    )
    metadata["transformer_config_sequence_parallel"] = bool(
        raw_model.config.sequence_parallel
    )
    for name, module in modules.items():
        metadata[name]["sequence_parallel"] = bool(
            getattr(module, "sequence_parallel", False)
        )
        metadata[name]["allreduce_dgrad"] = getattr(module, "allreduce_dgrad", None)
        metadata[name]["input_is_parallel"] = getattr(module, "input_is_parallel", None)
        metadata[name]["reduce_scatter_embeddings"] = getattr(
            module, "reduce_scatter_embeddings", None
        )
    return metadata


def sequence_parallel_runtime_checks(
    model: torch.nn.Module,
    tensor_parallel_size: int,
    sequence_parallel: bool,
    sequence_length: int,
) -> dict[str, Any]:
    raw_model = unwrap_model(model)
    layer = raw_model.decoder.layers[0]
    qkv = layer.self_attention.linear_qkv
    proj = layer.self_attention.linear_proj
    fc1 = layer.mlp.linear_fc1
    fc2 = layer.mlp.linear_fc2
    if raw_model.config.sequence_parallel != sequence_parallel:
        raise RuntimeError(
            "TransformerConfig.sequence_parallel does not match the requested flag: "
            f"{raw_model.config.sequence_parallel} != {sequence_parallel}"
        )
    if sequence_parallel and sequence_length % tensor_parallel_size != 0:
        raise RuntimeError(
            "Sequence length must be divisible by TP size when sequence "
            f"parallel is enabled: {sequence_length} % {tensor_parallel_size} != 0"
        )
    module_flags = {
        "attention_qkv": {
            "type": type(qkv).__name__,
            "sequence_parallel": bool(getattr(qkv, "sequence_parallel", False)),
            "allreduce_dgrad": bool(getattr(qkv, "allreduce_dgrad", False)),
            "ub_name": getattr(qkv, "ub_name", None),
            "ub_overlap_ag_fprop": bool(getattr(qkv, "ub_overlap_ag_fprop", False)),
            "ub_overlap_rs_fprop": bool(getattr(qkv, "ub_overlap_rs_fprop", False)),
        },
        "attention_projection": {
            "type": type(proj).__name__,
            "sequence_parallel": bool(getattr(proj, "sequence_parallel", False)),
            "input_is_parallel": bool(getattr(proj, "input_is_parallel", False)),
            "ub_name": getattr(proj, "ub_name", None),
            "ub_overlap_ag_fprop": bool(getattr(proj, "ub_overlap_ag_fprop", False)),
            "ub_overlap_rs_fprop": bool(getattr(proj, "ub_overlap_rs_fprop", False)),
        },
        "mlp_fc1": {
            "type": type(fc1).__name__,
            "sequence_parallel": bool(getattr(fc1, "sequence_parallel", False)),
            "allreduce_dgrad": bool(getattr(fc1, "allreduce_dgrad", False)),
            "ub_name": getattr(fc1, "ub_name", None),
            "ub_overlap_ag_fprop": bool(getattr(fc1, "ub_overlap_ag_fprop", False)),
            "ub_overlap_rs_fprop": bool(getattr(fc1, "ub_overlap_rs_fprop", False)),
        },
        "mlp_fc2": {
            "type": type(fc2).__name__,
            "sequence_parallel": bool(getattr(fc2, "sequence_parallel", False)),
            "input_is_parallel": bool(getattr(fc2, "input_is_parallel", False)),
            "ub_name": getattr(fc2, "ub_name", None),
            "ub_overlap_ag_fprop": bool(getattr(fc2, "ub_overlap_ag_fprop", False)),
            "ub_overlap_rs_fprop": bool(getattr(fc2, "ub_overlap_rs_fprop", False)),
        },
        "output_layer": {
            "type": type(raw_model.output_layer).__name__,
            "sequence_parallel": bool(raw_model.output_layer.sequence_parallel),
            "allreduce_dgrad": getattr(raw_model.output_layer, "allreduce_dgrad", None),
        },
    }
    te_linear_active = all(
        "TE" in module_flags[name]["type"]
        for name in (
            "attention_qkv",
            "attention_projection",
            "mlp_fc1",
            "mlp_fc2",
        )
    )
    if sequence_parallel:
        if tensor_parallel_size <= 1:
            raise RuntimeError("Sequence parallel requires tensor parallel size > 1")
        if not all(
            module_flags[name]["sequence_parallel"]
            for name in (
                "attention_qkv",
                "attention_projection",
                "mlp_fc1",
                "mlp_fc2",
            )
        ):
            raise RuntimeError(
                "Sequence parallel was requested but TP linear modules did not "
                f"activate it: {module_flags}"
            )
        if module_flags["attention_qkv"]["allreduce_dgrad"] or module_flags["mlp_fc1"][
            "allreduce_dgrad"
        ]:
            raise RuntimeError(
                "Column-parallel dgrad All-Reduce must be disabled when sequence "
                f"parallel is active: {module_flags}"
            )
        if not te_linear_active and not (
            module_flags["attention_projection"]["input_is_parallel"]
            and module_flags["mlp_fc2"]["input_is_parallel"]
        ):
            raise RuntimeError(
                "Row-parallel sequence parallel requires input_is_parallel=True: "
                f"{module_flags}"
            )
    elif tensor_parallel_size > 1:
        if any(
            module_flags[name]["sequence_parallel"]
            for name in (
                "attention_qkv",
                "attention_projection",
                "mlp_fc1",
                "mlp_fc2",
            )
        ):
            raise RuntimeError(
                "Sequence parallel leaked into the SP=False baseline: "
                f"{module_flags}"
            )
        if not te_linear_active and not (
            module_flags["attention_qkv"]["allreduce_dgrad"]
            and module_flags["mlp_fc1"]["allreduce_dgrad"]
        ):
            raise RuntimeError(
                "TP=2 SP=False must keep column-parallel dgrad All-Reduce: "
                f"{module_flags}"
            )
    return {
        "requested": sequence_parallel,
        "transformer_config_sequence_parallel": bool(
            raw_model.config.sequence_parallel
        ),
        "active": bool(raw_model.config.sequence_parallel) and tensor_parallel_size > 1,
        "te_linear_active": te_linear_active,
        "tp_comm_overlap_config": bool(getattr(raw_model.config, "tp_comm_overlap", False)),
        "module_flags": module_flags,
        "sequence_length_divisible_by_tp": (
            sequence_length % tensor_parallel_size == 0
        ),
    }


def initialize_userbuffers_if_needed(
    enabled: bool,
    sequence_length: int,
    micro_batch_size: int,
    hidden_size: int,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested": enabled,
        "ub_skipmc": os.environ.get("UB_SKIPMC"),
        "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"),
        "initialized": False,
        "communicator_names": [],
        "shape": [sequence_length * micro_batch_size, hidden_size],
        "bootstrap_backend": "nccl",
    }
    if os.environ.get("NCCL_P2P_DISABLE") == "1":
        raise RuntimeError(
            "NCCL P2P is disabled. Phase 7.4 must abort instead of continuing "
            "without P2P."
        )
    if not enabled:
        return metadata
    if os.environ.get("UB_SKIPMC") != "1":
        raise RuntimeError("UB_SKIPMC=1 is required for A40 Userbuffers IPC")
    from transformer_engine.pytorch import module as te_module

    existing = getattr(te_module.base, "_ub_communicators", None)
    if existing:
        raise RuntimeError("Userbuffers communicators were already initialized")
    te_module.base.initialize_ub(
        shape=metadata["shape"],
        tp_size=tensor_parallel_size,
        use_fp8=False,
        ub_cfgs={},
        bootstrap_backend="nccl",
    )
    comms = te_module.base._ub_communicators or {}
    names = sorted(
        key[0] if isinstance(key, tuple) else str(key) for key in comms.keys()
    )
    if not names:
        raise RuntimeError(
            "initialize_ub returned no communicators; refusing silent fallback"
        )
    metadata["initialized"] = True
    metadata["communicator_names"] = names
    metadata["communicator_count"] = len(names)
    return metadata


def inspect_userbuffers(
    model: torch.nn.Module,
    requested_overlap: bool,
    init_metadata: dict[str, Any],
) -> dict[str, Any]:
    raw_model = unwrap_model(model)
    try:
        from transformer_engine.pytorch import module as te_module

        comms = getattr(te_module.base, "_ub_communicators", None) or {}
    except ImportError:
        comms = {}
    names = sorted(
        key[0] if isinstance(key, tuple) else str(key) for key in comms.keys()
    )
    def linear_ub_state(module: torch.nn.Module) -> dict[str, Any]:
        return {
            "type": type(module).__name__,
            "ub_name": getattr(module, "ub_name", None),
            "ub_overlap_ag": bool(getattr(module, "ub_overlap_ag", False)),
            "ub_overlap_rs": bool(getattr(module, "ub_overlap_rs", False)),
            "ub_overlap_ag_fprop": bool(getattr(module, "ub_overlap_ag_fprop", False)),
            "ub_overlap_rs_fprop": bool(getattr(module, "ub_overlap_rs_fprop", False)),
            "ub_overlap_ag_dgrad": bool(getattr(module, "ub_overlap_ag_dgrad", False)),
            "ub_overlap_rs_dgrad": bool(getattr(module, "ub_overlap_rs_dgrad", False)),
            "ub_bulk_dgrad": bool(getattr(module, "ub_bulk_dgrad", False)),
            "ub_bulk_wgrad": bool(getattr(module, "ub_bulk_wgrad", False)),
        }

    layer = raw_model.decoder.layers[0]
    linear_state = {
        "attention_qkv": linear_ub_state(layer.self_attention.linear_qkv),
        "attention_projection": linear_ub_state(layer.self_attention.linear_proj),
        "mlp_fc1": linear_ub_state(layer.mlp.linear_fc1),
        "mlp_fc2": linear_ub_state(layer.mlp.linear_fc2),
    }
    expected_names = {"qkv_fprop", "proj_fprop", "fc1_fprop", "fc2_fprop"}
    te_types = all("TE" in item["type"] for item in linear_state.values())
    has_expected_communicators = expected_names.issubset(set(names))
    ub_names = {
        item["ub_name"]
        for item in linear_state.values()
        if item.get("ub_name") is not None
    }
    column_overlap = (
        linear_state["attention_qkv"]["ub_overlap_ag_fprop"]
        and linear_state["mlp_fc1"]["ub_overlap_ag_fprop"]
    )
    row_overlap = (
        linear_state["attention_projection"]["ub_overlap_rs_fprop"]
        and linear_state["mlp_fc2"]["ub_overlap_rs_fprop"]
    )
    active = bool(
        requested_overlap
        and bool(raw_model.config.tp_comm_overlap)
        and init_metadata.get("initialized")
        and has_expected_communicators
        and te_types
        and {"qkv", "proj", "fc1", "fc2"}.issubset(ub_names)
        and column_overlap
        and row_overlap
    )
    if requested_overlap and not active:
        raise RuntimeError(
            "Userbuffers overlap was requested but is not active; refusing "
            f"silent fallback: init={init_metadata} linears={linear_state} "
            f"communicators={names}"
        )
    if not requested_overlap and (init_metadata.get("initialized") or names):
        raise RuntimeError(
            "Userbuffers communicators exist on a non-overlap variant: "
            f"init={init_metadata} communicators={names}"
        )
    return {
        "requested": requested_overlap,
        "active": active,
        "silent_fallback": False,
        "config_tp_comm_overlap": bool(raw_model.config.tp_comm_overlap),
        "communicator_names": names,
        "communicator_count": len(names),
        "linear_userbuffers": linear_state,
        "initialization": init_metadata,
    }


def capture_linear_input_shapes(model: torch.nn.Module) -> tuple[dict[str, list[int]], list[Any]]:
    raw_model = unwrap_model(model)
    layer = raw_model.decoder.layers[0]
    captured: dict[str, list[int]] = {}
    handles = []
    targets = {
        "attention_qkv": layer.self_attention.linear_qkv,
        "attention_projection": layer.self_attention.linear_proj,
        "mlp_fc1": layer.mlp.linear_fc1,
        "mlp_fc2": layer.mlp.linear_fc2,
    }

    def make_hook(key: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any = None) -> None:
            if key in captured or not inputs:
                return
            tensor = inputs[0]
            if isinstance(tensor, torch.Tensor):
                captured[key] = [int(dim) for dim in tensor.shape]

        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    return captured, handles


def assert_sequence_parallel_activation_shapes(
    shapes: dict[str, list[int]],
    sequence_parallel: bool,
    sequence_length: int,
) -> None:
    qkv_shape = shapes.get("attention_qkv")
    fc1_shape = shapes.get("mlp_fc1")
    if qkv_shape is None or fc1_shape is None:
        raise RuntimeError(f"Failed to capture QKV/FC1 input shapes: {shapes}")
    qkv_has_full_sequence = sequence_length in qkv_shape
    fc1_has_full_sequence = sequence_length in fc1_shape
    if sequence_parallel:
        if qkv_has_full_sequence or fc1_has_full_sequence:
            raise RuntimeError(
                "Sequence parallel is not sharding activations before column-parallel "
                f"GEMMs; captured shapes={shapes}"
            )
    elif not qkv_has_full_sequence or not fc1_has_full_sequence:
        raise RuntimeError(
            "TP baseline unexpectedly sharded the sequence dimension on QKV/FC1 "
            f"inputs; captured shapes={shapes}"
        )


def instrument_tp_modules(model: torch.nn.Module) -> None:
    """Add forward attribution ranges for the short Nsight run."""

    raw_model = unwrap_model(model)
    modules: list[tuple[str, torch.nn.Module]] = [
        ("tp::embedding", raw_model.embedding.word_embeddings),
        ("tp::output_layer", raw_model.output_layer),
    ]
    for layer_index, layer in enumerate(raw_model.decoder.layers):
        modules.extend(
            [
                (f"tp::layer{layer_index:02d}::attention_qkv", layer.self_attention.linear_qkv),
                (
                    f"tp::layer{layer_index:02d}::attention_projection",
                    layer.self_attention.linear_proj,
                ),
                (f"tp::layer{layer_index:02d}::mlp_fc1", layer.mlp.linear_fc1),
                (f"tp::layer{layer_index:02d}::mlp_fc2", layer.mlp.linear_fc2),
            ]
        )
    for label, module in modules:
        original_forward = module.forward

        def wrapped_forward(
            *args: Any,
            _label: str = label,
            _forward: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            with torch.cuda.nvtx.range(_label):
                return _forward(*args, **kwargs)

        module.forward = wrapped_forward


def forward_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        return masked_language_model_loss(output, batch["loss_mask"])


def train_step(
    bundle: DDPOptimizerBundle,
    batch: dict[str, torch.Tensor],
    profile_step: int | None = None,
) -> float:
    step_context = (
        torch.cuda.nvtx.range(f"train_step_{profile_step:03d}")
        if profile_step is not None
        else torch.cuda.nvtx.range("train_step")
    )
    with step_context:
        with torch.cuda.nvtx.range("optimizer_zero_grad"):
            zero_gradients(bundle)
        with torch.cuda.nvtx.range("forward"):
            loss = forward_loss(bundle.model, batch)
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
        with torch.cuda.nvtx.range("finalize_model_grads"):
            finalize_gradients(bundle.model)
        with torch.cuda.nvtx.range("optimizer_step"):
            update_successful, _, _ = bundle.optimizer.step()
            if not update_successful:
                raise RuntimeError("FP32Optimizer unexpectedly rejected an update")
    return float(loss.detach().cpu())


def main_grads_finite(bundle: DDPOptimizerBundle) -> bool:
    return all(
        bool(torch.isfinite(parameter.main_grad).all().item())
        for _, parameter in named_trainable_parameters(bundle.model)
    )


def parameters_changed(
    before: dict[str, torch.Tensor],
    bundle: DDPOptimizerBundle,
) -> bool:
    current = dict(named_trainable_parameters(bundle.model))
    return any(
        not torch.equal(value, current[name].detach().cpu())
        for name, value in before.items()
    )


def collect_environment(tensor_parallel_size: int) -> dict[str, Any]:
    gpu_lines = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    gpus = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 5:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "driver": fields[2],
                    "memory_total_mib": float(fields[3]),
                    "pci_bus_id": fields[4],
                }
            )
    try:
        te_version = importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError:
        te_version = None
    nccl_version = torch.cuda.nccl.version()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": ".".join(str(part) for part in nccl_version),
        "cudnn": torch.backends.cudnn.version(),
        "megatron_lm_commit": run_command(
            ["git", "rev-parse", "HEAD"],
            cwd="/workspace/Megatron-LM",
        ),
        "project_commit": run_command(
            ["git", "rev-parse", "HEAD"],
            cwd="/workspace/megatron-performance-lab",
        ),
        "transformer_engine_installed": (
            importlib.util.find_spec("transformer_engine") is not None
        ),
        "transformer_engine_version": te_version,
        "gpus": gpus[:tensor_parallel_size],
        "cuda_graph_enabled": False,
        "ub_skipmc": os.environ.get("UB_SKIPMC"),
        "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"),
    }


def gather_objects(value: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_distributed(model_args.seed, args.tensor_parallel_size)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("Transformer Engine must remain enabled")
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
            raise RuntimeError(
                "CUDA_DEVICE_MAX_CONNECTIONS=1 is required for TP and sequence parallel"
            )
        if args.sequence_parallel and args.tensor_parallel_size <= 1:
            raise RuntimeError("Sequence parallel requires tensor_model_parallel_size > 1")
        if args.tp_comm_overlap and not args.te_linear:
            raise RuntimeError("tp_comm_overlap requires the TE Linear path")
        if args.tp_comm_overlap and not args.sequence_parallel:
            raise RuntimeError("tp_comm_overlap requires sequence_parallel=True")
        if args.te_linear and args.tensor_parallel_size <= 1:
            raise RuntimeError("Phase 7.4 TE Linear variants require TP=2")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required")

        ub_init = initialize_userbuffers_if_needed(
            enabled=args.tp_comm_overlap,
            sequence_length=model_args.sequence_length,
            micro_batch_size=MICRO_BATCH_SIZE,
            hidden_size=model_args.hidden_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            bias_dropout_fusion=True,
            cuda_graph_impl="none",
            tensor_model_parallel_size=args.tensor_parallel_size,
            sequence_parallel=args.sequence_parallel,
            use_te_layernorm=True,
            use_te_linear=args.te_linear,
            tp_comm_overlap=args.tp_comm_overlap,
        )
        if model.config.cuda_graph_impl != "none":
            raise RuntimeError("CUDA Graph must remain disabled")
        if bool(model.config.sequence_parallel) != args.sequence_parallel:
            raise RuntimeError(
                "build_model did not honor sequence_parallel="
                f"{args.sequence_parallel}"
            )
        if bool(model.config.tp_comm_overlap) != args.tp_comm_overlap:
            raise RuntimeError(
                "build_model did not honor tp_comm_overlap="
                f"{args.tp_comm_overlap}"
            )
        bundle = build_ddp_optimizer_bundle(model, model_args.learning_rate)
        if args.profile_mode:
            instrument_tp_modules(bundle.model)
        batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
        pointers = main_grad_pointers(bundle.model)
        shards = sharding_metadata(bundle.model, args.tensor_parallel_size)
        sp_runtime = sequence_parallel_runtime_checks(
            bundle.model,
            args.tensor_parallel_size,
            args.sequence_parallel,
            model_args.sequence_length,
        )
        if args.te_linear and not sp_runtime.get("te_linear_active"):
            raise RuntimeError(
                "TE Linear path was requested but decoder linears are not TE modules: "
                f"{sp_runtime.get('module_flags')}"
            )
        if not args.te_linear and sp_runtime.get("te_linear_active"):
            raise RuntimeError("TE Linear leaked into the local-spec TP baseline")
        userbuffers_runtime = inspect_userbuffers(
            bundle.model,
            args.tp_comm_overlap,
            ub_init,
        )
        captured_shapes, shape_handles = capture_linear_input_shapes(bundle.model)
        rank_shards = gather_objects({"rank": rank, **shards})

        tracked_name, tracked_parameter = next(
            (name, parameter)
            for name, parameter in named_trainable_parameters(bundle.model)
            if name.endswith("decoder.layers.0.self_attention.linear_qkv.weight")
        )
        parameters_before = {
            tracked_name: tracked_parameter.detach().cpu().clone()
        }
        smoke_losses = []
        lifecycle_checks = []
        for smoke_index in range(args.smoke_iterations):
            zero_gradients(bundle)
            torch.cuda.synchronize(device)
            assert_zeroed_lifecycle(bundle)
            assert_main_grad_pointers(bundle.model, pointers)
            loss = forward_loss(bundle.model, batch)
            loss.backward()
            finalize_gradients(bundle.model)
            torch.cuda.synchronize(device)
            gradients_finite = main_grads_finite(bundle)
            update_successful, _, _ = bundle.optimizer.step()
            torch.cuda.synchronize(device)
            assert_optimizer_consumed_main_grad(bundle)
            assert_main_grad_pointers(bundle.model, pointers)
            if not update_successful or not gradients_finite:
                raise RuntimeError("Smoke-step lifecycle failed")
            smoke_losses.append(float(loss.detach().cpu()))
            lifecycle_checks.append(
                {
                    "step": smoke_index,
                    "loss_finite": math.isfinite(smoke_losses[-1]),
                    "main_grads_finite": gradients_finite,
                    "optimizer_consumed_main_grad": True,
                    "main_grad_addresses_stable": True,
                }
            )
        for handle in shape_handles:
            handle.remove()
        assert_sequence_parallel_activation_shapes(
            captured_shapes,
            args.sequence_parallel,
            model_args.sequence_length,
        )
        sp_runtime["captured_linear_input_shapes"] = captured_shapes
        rank_sp_runtime = gather_objects({"rank": rank, **sp_runtime})
        rank_userbuffers = gather_objects({"rank": rank, **userbuffers_runtime})
        changed = parameters_changed(parameters_before, bundle)
        if not changed:
            raise RuntimeError("No model parameter changed during smoke steps")

        warmup_losses = []
        for _ in range(args.warmup_iterations):
            warmup_losses.append(train_step(bundle, batch))
            torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        sampler = (
            MultiGpuNvidiaSmiSampler(
                list(range(args.tensor_parallel_size)),
                args.gpu_sample_interval_ms,
            )
            if rank == 0
            else None
        )
        if sampler is not None:
            sampler.start()
        dist.barrier()
        if args.profile_mode and rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
        dist.barrier()

        measured_losses = []
        local_step_times_ms = []
        try:
            with torch.cuda.nvtx.range("profile_window"):
                emit_context = (
                    torch.autograd.profiler.emit_nvtx(record_shapes=True)
                    if args.profile_mode
                    else torch.autograd.profiler.emit_nvtx(enabled=False)
                )
                with emit_context:
                    for step_index in range(args.measured_iterations):
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        measured_losses.append(
                            train_step(
                                bundle,
                                batch,
                                step_index if args.profile_mode else None,
                            )
                        )
                        torch.cuda.synchronize(device)
                        local_step_times_ms.append(
                            (time.perf_counter() - start) * 1000.0
                        )
        finally:
            dist.barrier()
            if args.profile_mode and rank == 0:
                torch.cuda.cudart().cudaProfilerStop()
            dist.barrier()
            gpu_monitoring = sampler.stop() if sampler is not None else None

        rank_times = gather_objects(
            {
                "rank": rank,
                "step_times_ms": local_step_times_ms,
                "peak_allocated_memory_mib": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
                "peak_reserved_memory_mib": (
                    torch.cuda.max_memory_reserved(device) / 1024**2
                ),
            }
        )
        rank_losses = gather_objects({"rank": rank, "losses": measured_losses})
        if rank != 0:
            return

        global_step_times_ms = [
            max(rank_time["step_times_ms"][index] for rank_time in rank_times)
            for index in range(args.measured_iterations)
        ]
        average_step_time_ms = statistics.fmean(global_step_times_ms)
        tokens_per_step = MICRO_BATCH_SIZE * model_args.sequence_length
        tokens_per_second = tokens_per_step / (average_step_time_ms / 1000.0)
        flops_per_iteration = training_flops_per_iteration(
            MICRO_BATCH_SIZE,
            model_args.sequence_length,
            model_args.num_layers,
            model_args.hidden_size,
            model_args.vocab_size,
        )
        aggregate_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1e12
        mfu_percent = (
            aggregate_tflops
            / (args.tensor_parallel_size * A40_DENSE_BF16_PEAK_TFLOPS)
            * 100.0
        )
        all_losses = [
            value
            for rank_loss in rank_losses
            for value in rank_loss["losses"]
        ]
        if not all(math.isfinite(value) for value in all_losses):
            raise RuntimeError("Non-finite measured loss")
        max_rank_loss_difference = max(
            abs(rank_losses[0]["losses"][index] - rank_loss["losses"][index])
            for rank_loss in rank_losses
            for index in range(args.measured_iterations)
        )

        result = {
            "status": "success",
            "experiment": "Phase 7.4 TE Userbuffers TP communication overlap",
            "run_label": args.run_label,
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "model_config": {
                "architecture": "Megatron Core GPTModel",
                "num_layers": model_args.num_layers,
                "hidden_size": model_args.hidden_size,
                "ffn_hidden_size": model_args.ffn_hidden_size,
                "num_attention_heads": model_args.num_attention_heads,
                "head_dimension": model_args.hidden_size // model_args.num_attention_heads,
                "vocab_size": model_args.vocab_size,
                "sequence_length": model_args.sequence_length,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "attention_implementation": TE_FUSED_ATTENTION,
                "bias_dropout_fusion": True,
                "bias_activation_fusion": False,
                "cuda_graph_impl": "none",
                "sequence_parallel": args.sequence_parallel,
                "te_linear": args.te_linear,
                "tp_comm_overlap": args.tp_comm_overlap,
                "layernorm_implementation": "TENorm",
            },
            "parallelism": {
                "tensor_parallel": args.tensor_parallel_size,
                "sequence_parallel": args.sequence_parallel,
                "tp_comm_overlap": args.tp_comm_overlap,
                "te_linear": args.te_linear,
                "pipeline_parallel": 1,
                "data_parallel": 1,
                "world_size": args.tensor_parallel_size,
            },
            "sharding_verification": rank_shards,
            "sequence_parallel_runtime": rank_sp_runtime,
            "userbuffers_runtime": rank_userbuffers,
            "lifecycle": lifecycle_metadata(bundle),
            "correctness_smoke": {
                "steps": args.smoke_iterations,
                "rank0_losses": smoke_losses,
                "per_step_checks": lifecycle_checks,
                "parameters_updated": changed,
                "all_ranks_initialized": len(rank_shards) == args.tensor_parallel_size,
                "forward_succeeded": True,
                "backward_succeeded": True,
                "sequence_parallel_active": bool(
                    args.sequence_parallel and sp_runtime["active"]
                ),
                "te_linear_active": bool(sp_runtime.get("te_linear_active")),
                "userbuffers_active": bool(userbuffers_runtime["active"]),
                "userbuffers_silent_fallback": False,
                "max_measured_loss_difference_between_ranks": (
                    max_rank_loss_difference
                ),
                "nccl_errors": False,
                "deadlock": False,
            },
            "precision": {
                "forward_backward": "BF16 autocast",
                "parameter_storage": "FP32",
                "main_grad": "FP32",
                "optimizer_state": "FP32",
            },
            "optimizer": {
                "wrapper": "MCore FP32Optimizer",
                "base": "torch.optim.AdamW",
                "learning_rate": model_args.learning_rate,
                "foreach": False,
                "fused": False,
                "distributed_optimizer": False,
            },
            "data": {
                "type": "fixed synthetic random token IDs",
                "seed": model_args.seed + 1,
                "same_batch_on_all_tp_ranks": True,
            },
            "tokens_per_step": tokens_per_step,
            "smoke_iterations": args.smoke_iterations,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "warmup_final_loss": warmup_losses[-1],
            "measured_losses_rank0": measured_losses,
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": statistics.median(global_step_times_ms),
            "step_time_standard_deviation_ms": (
                statistics.stdev(global_step_times_ms)
                if len(global_step_times_ms) > 1
                else 0.0
            ),
            "global_step_times_ms": global_step_times_ms,
            "rank_timing_and_memory": rank_times,
            "gpu_monitoring": gpu_monitoring,
            "tokens_per_second": tokens_per_second,
            "mfu": {
                "training_flops_per_iteration": flops_per_iteration,
                "aggregate_achieved_tflops": aggregate_tflops,
                "achieved_tflops_per_gpu": (
                    aggregate_tflops / args.tensor_parallel_size
                ),
                "a40_dense_bf16_peak_tflops_per_gpu": (
                    A40_DENSE_BF16_PEAK_TFLOPS
                ),
                "aggregate_mfu_percent": mfu_percent,
            },
            "transformer_engine_backend": fused_backend_status(),
            "instrumentation": {
                "cuda_profiler_capture_range": args.profile_mode,
                "module_nvtx_attribution": args.profile_mode,
                "cuda_graph_enabled": False,
            },
            "environment": collect_environment(args.tensor_parallel_size),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print("PHASE7_TP_RUN_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
