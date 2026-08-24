#!/usr/bin/env python3
"""Run one controlled Phase 5.2 bias-dropout-add fusion variant."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import (
    bias_dropout_add_fused_train,
    get_bias_dropout_add,
)

from phase1_baseline import (
    A40_DENSE_BF16_PEAK_TFLOPS,
    NvidiaSmiSampler,
    TE_FUSED_ATTENTION,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
    train_step,
    training_flops_per_iteration,
)
from phase3_attention_correctness import fused_backend_status
from phase3_attention_profile import baseline_model_args


MICRO_BATCH_SIZE = 8
BDA_SITES_PER_STEP = 48


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-dropout-fusion", type=parse_bool, required=True)
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


def instrumented_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    step_index: int,
) -> float:
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
        with torch.cuda.nvtx.range("optimizer_zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.nvtx.range("forward"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    batch["tokens"],
                    batch["position_ids"],
                    batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = masked_language_model_loss(output, batch["loss_mask"])
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
        with torch.cuda.nvtx.range("optimizer_step"):
            optimizer.step()
    return float(loss.detach().cpu())


def run_measured_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    measured_iterations: int,
    device: torch.device,
    profile_mode: bool,
) -> tuple[list[float], list[float]]:
    losses: list[float] = []
    step_times_ms: list[float] = []
    if profile_mode:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        with torch.cuda.nvtx.range("profile_window"):
            for step_index in range(measured_iterations):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                losses.append(
                    instrumented_train_step(
                        model,
                        optimizer,
                        batch,
                        step_index,
                    )
                )
                torch.cuda.synchronize(device)
                step_times_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        torch.cuda.synchronize(device)
        if profile_mode:
            torch.cuda.cudart().cudaProfilerStop()
    return losses, step_times_ms


def model_config(bias_dropout_fusion: bool) -> dict[str, Any]:
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
        "bias_dropout_fusion": bias_dropout_fusion,
        "bias_activation_fusion": False,
        "masked_softmax_fusion": False,
        "cross_entropy_loss_fusion": False,
        "cuda_graph_enabled": False,
    }


def optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> list[str]:
    dtypes = {
        str(value.dtype)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    return sorted(dtypes)


def verify_controls(model: torch.nn.Module, bias_dropout_fusion: bool) -> dict[str, Any]:
    config = model.config
    expected = {
        "bias_dropout_fusion": bias_dropout_fusion,
        "bias_activation_fusion": False,
        "masked_softmax_fusion": False,
        "cross_entropy_loss_fusion": False,
    }
    actual = {name: bool(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected fusion configuration: {actual}")

    selected = get_bias_dropout_add(training=True, fused=bias_dropout_fusion)
    if bias_dropout_fusion and selected is not bias_dropout_add_fused_train:
        raise RuntimeError("Fused BDA configuration did not select the fused train function")
    if not bias_dropout_fusion and selected is bias_dropout_add_fused_train:
        raise RuntimeError("Baseline BDA configuration selected the fused train function")

    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"Parameter dtype changed: {parameter_dtypes}")
    return {
        "transformer_config": actual,
        "selected_bda_function": f"{selected.__module__}.{selected.__name__}",
        "bda_sites_per_step": BDA_SITES_PER_STEP,
        "parameter_dtypes": parameter_dtypes,
        "params_dtype": str(config.params_dtype),
        "pipeline_dtype": str(config.pipeline_dtype),
        "fp32_residual_connection": bool(config.fp32_residual_connection),
    }


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
            bias_dropout_fusion=args.bias_dropout_fusion,
            instrument_bda=args.profile_mode,
        )
        controls = verify_controls(model, args.bias_dropout_fusion)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_args.learning_rate,
            foreach=False,
            fused=False,
        )
        batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

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
            "experiment": "Phase 5.2 bias-dropout-add fusion A/B",
            "run_label": args.run_label,
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "variant": "B" if args.bias_dropout_fusion else "A",
            "bias_dropout_fusion": args.bias_dropout_fusion,
            "parameter_count": parameter_count,
            "model_config": model_config(args.bias_dropout_fusion),
            "control_verification": {
                **controls,
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
                "manual_bda_forward_nvtx": args.profile_mode,
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
        print("PHASE5_BDA_RUN_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
