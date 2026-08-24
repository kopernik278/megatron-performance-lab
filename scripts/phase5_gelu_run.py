#!/usr/bin/env python3
"""Run one controlled Phase 5.3 bias-plus-GELU fusion variant."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import (
    bias_dropout_add_fused_train,
    get_bias_dropout_add,
)
from megatron.core.utils import configure_nvtx_profiling

from phase1_baseline import (
    A40_DENSE_BF16_PEAK_TFLOPS,
    NvidiaSmiSampler,
    TE_FUSED_ATTENTION,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    synthetic_batch,
    train_step,
    training_flops_per_iteration,
)
from phase3_attention_correctness import fused_backend_status
from phase3_attention_profile import baseline_model_args
from phase5_bda_run import run_measured_steps


MICRO_BATCH_SIZE = 8


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-gelu-fusion", type=parse_bool, required=True)
    parser.add_argument("--warmup-iterations", type=int, required=True)
    parser.add_argument("--measured-iterations", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--profile-mode", action="store_true")
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_iterations < 1 or args.measured_iterations < 1:
        parser.error("iteration counts must be positive")
    return args


def model_config(bias_gelu_fusion: bool) -> dict[str, Any]:
    return {
        "architecture": "Megatron Core GPTModel",
        "num_layers": 24,
        "hidden_size": 1024,
        "ffn_hidden_size": 4096,
        "num_attention_heads": 16,
        "head_dimension": 64,
        "vocab_size": 50304,
        "max_position_embeddings": 2048,
        "position_embedding_type": "learned_absolute",
        "share_embeddings_and_output_weights": True,
        "hidden_dropout": 0.1,
        "attention_dropout": 0.1,
        "layernorm_epsilon": 1.0e-5,
        "attention_backend": "fused",
        "attention_implementation": TE_FUSED_ATTENTION,
        "core_attention": "TEDotProductAttention",
        "transformer_layer_spec": "get_gpt_layer_local_spec",
        "bias_gelu_fusion": bias_gelu_fusion,
        "bias_activation_fusion": bias_gelu_fusion,
        "bias_dropout_fusion": True,
        "masked_softmax_fusion": False,
        "cross_entropy_loss_fusion": False,
        "cuda_graph_enabled": False,
    }


def optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> list[str]:
    return sorted(
        {
            str(value.dtype)
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
    )


def verify_static_controls(
    model: torch.nn.Module,
    bias_gelu_fusion: bool,
) -> dict[str, Any]:
    config = model.config
    expected = {
        "bias_activation_fusion": bias_gelu_fusion,
        "bias_dropout_fusion": True,
        "masked_softmax_fusion": False,
        "cross_entropy_loss_fusion": False,
    }
    actual = {name: bool(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected fusion configuration: {actual}")
    if bool(config.gated_linear_unit):
        raise RuntimeError("Gated linear units would select a different activation path")
    if not bool(config.add_bias_linear):
        raise RuntimeError("Bias-plus-GELU fusion requires add_bias_linear=True")
    if bool(config.use_te_activation_func):
        raise RuntimeError("TE activation must remain disabled for this experiment")

    selected_bda = get_bias_dropout_add(training=True, fused=True)
    if selected_bda is not bias_dropout_add_fused_train:
        raise RuntimeError("The retained BDA fusion did not select the fused train path")

    layers = list(model.decoder.layers)
    if len(layers) != config.num_layers:
        raise RuntimeError("Unexpected transformer layer count")
    if any(layer.mlp.activation_func is not F.gelu for layer in layers):
        raise RuntimeError("The model activation is not torch.nn.functional.gelu")

    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"Parameter dtype changed: {parameter_dtypes}")
    return {
        "transformer_config": actual,
        "activation_function": "torch.nn.functional.gelu",
        "activation_is_f_gelu_in_all_layers": True,
        "add_bias_linear": bool(config.add_bias_linear),
        "gated_linear_unit": bool(config.gated_linear_unit),
        "use_te_activation_func": bool(config.use_te_activation_func),
        "bda_function": f"{selected_bda.__module__}.{selected_bda.__name__}",
        "parameter_dtypes": parameter_dtypes,
        "params_dtype": str(config.params_dtype),
        "pipeline_dtype": str(config.pipeline_dtype),
        "fp32_residual_connection": bool(config.fp32_residual_connection),
    }


def probe_runtime_gelu_path(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    bias_gelu_fusion: bool,
) -> dict[str, Any]:
    """Observe whether MLP.forward invokes bias_gelu_impl once per layer."""
    import megatron.core.transformer.mlp as mlp_module

    original = mlp_module.bias_gelu_impl
    calls = 0

    def counted_bias_gelu(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    mlp_module.bias_gelu_impl = counted_bias_gelu
    try:
        torch.manual_seed(987654)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    batch["tokens"],
                    batch["position_ids"],
                    batch["attention_mask"],
                )
        torch.cuda.synchronize()
        del output
    finally:
        mlp_module.bias_gelu_impl = original

    expected_calls = len(model.decoder.layers) if bias_gelu_fusion else 0
    if calls != expected_calls:
        raise RuntimeError(
            f"bias_gelu_impl runtime calls were {calls}; expected {expected_calls}"
        )
    return {
        "verified": True,
        "observed_bias_gelu_impl_calls": calls,
        "expected_bias_gelu_impl_calls": expected_calls,
        "selected_path": (
            "megatron.core.fusions.fused_bias_gelu.bias_gelu_impl"
            if bias_gelu_fusion
            else "bias add followed by torch.nn.functional.gelu"
        ),
        "source_branch_conditions": {
            "bias_activation_fusion": bias_gelu_fusion,
            "activation_func_is_f_gelu": True,
            "gated_linear_unit": False,
            "add_bias_linear": True,
        },
    }


@contextlib.contextmanager
def fused_gelu_nvtx_instrumentation(enabled: bool) -> Iterator[None]:
    """Add exact forward/backward ranges around the fused GELU primitives."""
    if not enabled:
        yield
        return

    import megatron.core.fusions.fused_bias_gelu as fused_module

    original_forward = fused_module.bias_gelu
    original_backward = fused_module.bias_gelu_back

    def instrumented_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        with torch.cuda.nvtx.range("gelu::forward_fused"):
            return original_forward(*args, **kwargs)

    def instrumented_backward(*args: Any, **kwargs: Any) -> torch.Tensor:
        with torch.cuda.nvtx.range("gelu::backward_fused"):
            return original_backward(*args, **kwargs)

    fused_module.bias_gelu = instrumented_forward
    fused_module.bias_gelu_back = instrumented_backward
    try:
        yield
    finally:
        fused_module.bias_gelu = original_forward
        fused_module.bias_gelu_back = original_backward


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_single_gpu_distributed(model_args.seed)
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("TE fused attention requires Transformer Engine")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")

        device = torch.device(f"cuda:{local_rank}")
        model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            bias_activation_fusion=args.bias_gelu_fusion,
            bias_dropout_fusion=True,
        )
        static_controls = verify_static_controls(model, args.bias_gelu_fusion)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_args.learning_rate,
            foreach=False,
            fused=False,
        )
        batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
        runtime_path = probe_runtime_gelu_path(
            model,
            batch,
            args.bias_gelu_fusion,
        )
        torch.cuda.empty_cache()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        configure_nvtx_profiling(args.profile_mode)
        torch.manual_seed(model_args.seed + 2)
        with fused_gelu_nvtx_instrumentation(args.profile_mode):
            warmup_losses: list[float] = []
            for _ in range(args.warmup_iterations):
                warmup_losses.append(train_step(model, optimizer, batch))
                torch.cuda.synchronize(device)

            backend = fused_backend_status()
            print("FusedAttention backend (sub-backend 1)")

            torch.cuda.reset_peak_memory_stats(device)
            sampler = NvidiaSmiSampler(args.gpu_sample_interval_ms)
            sampler.start()
            try:
                measured_losses, step_times_ms = run_measured_steps(
                    model,
                    optimizer,
                    batch,
                    args.measured_iterations,
                    device,
                    args.profile_mode,
                )
            finally:
                gpu_monitoring = sampler.stop()
        configure_nvtx_profiling(False)

        if not all(math.isfinite(value) for value in warmup_losses + measured_losses):
            raise RuntimeError("A non-finite training loss was observed")

        average_step_time_ms = statistics.fmean(step_times_ms)
        tokens_per_step = MICRO_BATCH_SIZE * model_args.sequence_length
        flops_per_iteration = training_flops_per_iteration(
            MICRO_BATCH_SIZE,
            model_args.sequence_length,
            model_args.num_layers,
            model_args.hidden_size,
            model_args.vocab_size,
        )
        achieved_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1.0e12
        result = {
            "status": "success",
            "experiment": "Phase 5.3 bias-plus-GELU fusion A/B",
            "run_label": args.run_label,
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "variant": "B" if args.bias_gelu_fusion else "A",
            "bias_gelu_fusion": args.bias_gelu_fusion,
            "bias_activation_fusion": args.bias_gelu_fusion,
            "bias_dropout_fusion": True,
            "parameter_count": parameter_count,
            "model_config": model_config(args.bias_gelu_fusion),
            "control_verification": {
                **static_controls,
                "runtime_gelu_path": runtime_path,
                "optimizer_state_dtypes": optimizer_state_dtypes(optimizer),
            },
            "parallelism": {
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": 1,
            },
            "precision": {
                "forward_backward": "BF16 autocast",
                "parameter_storage": "FP32",
                "residual_stream": "FP32",
                "optimizer_state": "FP32",
                "bf16_enabled": True,
            },
            "optimizer": {
                "name": "torch.optim.AdamW",
                "learning_rate": model_args.learning_rate,
                "foreach": False,
                "fused": False,
            },
            "data": {
                "type": "fixed synthetic random token IDs",
                "seed": model_args.seed + 1,
                "model_seed": model_args.seed,
                "training_rng_seed": model_args.seed + 2,
            },
            "micro_batch_size": MICRO_BATCH_SIZE,
            "global_batch_size": MICRO_BATCH_SIZE,
            "sequence_length": model_args.sequence_length,
            "tokens_per_step": tokens_per_step,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "warmup_final_loss": warmup_losses[-1],
            "measured_losses": measured_losses,
            "final_loss": measured_losses[-1],
            "losses_finite": True,
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": statistics.median(step_times_ms),
            "step_time_standard_deviation_ms": (
                statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0
            ),
            "step_times_ms": step_times_ms,
            "tokens_per_second": tokens_per_step / (average_step_time_ms / 1000.0),
            "milliseconds_per_token": average_step_time_ms / tokens_per_step,
            "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            "gpu_monitoring": gpu_monitoring,
            "mfu": {
                "training_flops_per_iteration": flops_per_iteration,
                "achieved_tflops": achieved_tflops,
                "gpu_dense_bf16_peak_tflops": A40_DENSE_BF16_PEAK_TFLOPS,
                "mfu_percent": achieved_tflops / A40_DENSE_BF16_PEAK_TFLOPS * 100.0,
                "formula": (
                    "F_iter = 72*B*S*L*H^2 + 6*B*L*S^2*H + 6*B*S*H*V; "
                    "MFU = (F_iter / step_seconds) / 149.7e12"
                ),
            },
            "transformer_engine_backend": backend,
            "instrumentation": {
                "cuda_profiler_capture_range": args.profile_mode,
                "megatron_forward_activation_nvtx": args.profile_mode,
                "manual_fused_gelu_forward_backward_nvtx": args.profile_mode,
                "manual_step_nvtx": True,
                "cuda_graph_enabled": False,
                "unrelated_kernel_fusion_added": False,
            },
            "invocation": {
                "argv": sys.argv,
                "cuda_device_max_connections": os.environ.get(
                    "CUDA_DEVICE_MAX_CONNECTIONS"
                ),
            },
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE5_GELU_RUN_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        configure_nvtx_profiling(False)
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
