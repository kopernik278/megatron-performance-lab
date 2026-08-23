#!/usr/bin/env python3
"""Run one fused-attention micro-batch scaling point."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state

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
from phase3_attention_profile import (
    baseline_model_args,
    instrumented_train_step,
    model_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=100)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=200)
    parser.add_argument(
        "--profile-mode",
        action="store_true",
        help="Bracket measured steps with the CUDA profiler API for Nsight capture.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.micro_batch_size < 1:
        parser.error("--micro-batch-size must be positive")
    if args.warmup_iterations < 1 or args.measured_iterations < 1:
        parser.error("iteration counts must be positive")
    return args


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
            with torch.autograd.profiler.emit_nvtx(record_shapes=False):
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
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_args.learning_rate,
            foreach=False,
            fused=False,
        )
        batch = synthetic_batch(model_args, args.micro_batch_size, device)
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

        average_step_time_ms = statistics.fmean(step_times_ms)
        tokens_per_step = args.micro_batch_size * model_args.sequence_length
        flops_per_iteration = training_flops_per_iteration(
            args.micro_batch_size,
            model_args.sequence_length,
            model_args.num_layers,
            model_args.hidden_size,
            model_args.vocab_size,
        )
        achieved_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1.0e12
        result = {
            "status": "success",
            "experiment": "Phase 3.3 fused-attention micro-batch scaling",
            "run_mode": "nsight_profile" if args.profile_mode else "full_benchmark",
            "parameter_count": parameter_count,
            "model_config": model_config(TE_FUSED_ATTENTION),
            "parallelism": {
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": 1,
            },
            "precision": {
                "forward_backward": "BF16 autocast",
                "parameter_storage": "FP32",
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
            },
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.micro_batch_size,
            "sequence_length": model_args.sequence_length,
            "tokens_per_step": tokens_per_step,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "warmup_final_loss": warmup_losses[-1],
            "measured_losses": measured_losses,
            "final_loss": measured_losses[-1],
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": statistics.median(step_times_ms),
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
                "autograd_nvtx": True,
                "cuda_graph_enabled": False,
                "unrelated_kernel_fusion_added": False,
            },
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE3_MICROBATCH_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
