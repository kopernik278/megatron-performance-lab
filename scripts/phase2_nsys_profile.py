#!/usr/bin/env python3
"""Profile the unchanged Phase 1.2 baseline with Nsight Systems."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state

from phase1_baseline import (
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
    training_flops_per_iteration,
)


MICRO_BATCH_SIZE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--profiled-iterations", type=int, default=15)
    parser.add_argument("--output-json", type=Path, default=Path("results/phase2_nsys_run_metrics.json"))
    return parser.parse_args()


def baseline_model_args() -> argparse.Namespace:
    return argparse.Namespace(
        sequence_length=2048,
        vocab_size=50304,
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        learning_rate=1.0e-4,
        seed=1234,
    )


def instrumented_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    step_index: int,
) -> float:
    with torch.cuda.nvtx.range(f"train_step_{step_index:02d}"):
        with torch.cuda.nvtx.range("optimizer_zero_grad"):
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.nvtx.range("forward"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output_tensor = model(
                    batch["tokens"],
                    batch["position_ids"],
                    batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = masked_language_model_loss(output_tensor, batch["loss_mask"])

        with torch.cuda.nvtx.range("backward"):
            loss.backward()

        with torch.cuda.nvtx.range("optimizer_step"):
            optimizer.step()

    return float(loss.detach().cpu())


def model_config() -> dict[str, Any]:
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
        "attention_backend": "unfused",
        "transformer_layer_spec": "get_gpt_layer_local_spec",
    }


def main() -> None:
    args = parse_args()
    baseline_args = baseline_model_args()
    local_rank = initialize_single_gpu_distributed(baseline_args.seed)
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") != "1":
            raise RuntimeError("TRANSFORMER_ENGINE_DISABLE=1 is required")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")

        device = torch.device(f"cuda:{local_rank}")
        model = build_model(baseline_args)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=baseline_args.learning_rate,
            foreach=False,
            fused=False,
        )
        batch = synthetic_batch(baseline_args, MICRO_BATCH_SIZE, device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        warmup_losses: list[float] = []
        from phase1_baseline import train_step

        for _ in range(args.warmup_iterations):
            warmup_losses.append(train_step(model, optimizer, batch))
            torch.cuda.synchronize(device)

        profiled_losses: list[float] = []
        profiled_step_times_ms: list[float] = []
        torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range("profile_window"):
                with torch.autograd.profiler.emit_nvtx(record_shapes=False):
                    for step_index in range(args.profiled_iterations):
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        profiled_losses.append(
                            instrumented_train_step(model, optimizer, batch, step_index)
                        )
                        torch.cuda.synchronize(device)
                        profiled_step_times_ms.append((time.perf_counter() - start) * 1000.0)
        finally:
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStop()

        average_step_time_ms = statistics.fmean(profiled_step_times_ms)
        flops_per_iteration = training_flops_per_iteration(
            MICRO_BATCH_SIZE,
            baseline_args.sequence_length,
            baseline_args.num_layers,
            baseline_args.hidden_size,
            baseline_args.vocab_size,
        )
        achieved_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1.0e12
        result = {
            "status": "success",
            "parameter_count": parameter_count,
            "model_config": model_config(),
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
                "learning_rate": baseline_args.learning_rate,
                "foreach": False,
                "fused": False,
            },
            "micro_batch_size": MICRO_BATCH_SIZE,
            "global_batch_size": MICRO_BATCH_SIZE,
            "sequence_length": baseline_args.sequence_length,
            "warmup_iterations": args.warmup_iterations,
            "profiled_iterations": args.profiled_iterations,
            "warmup_final_loss": warmup_losses[-1],
            "profiled_losses": profiled_losses,
            "final_loss": profiled_losses[-1],
            "profiled_step_times_ms": profiled_step_times_ms,
            "average_profiled_step_time_ms": average_step_time_ms,
            "median_profiled_step_time_ms": statistics.median(profiled_step_times_ms),
            "mfu": {
                "training_flops_per_iteration": flops_per_iteration,
                "achieved_tflops_from_profiled_timing": achieved_tflops,
                "gpu_dense_bf16_peak_tflops": 149.7,
                "mfu_percent_from_profiled_timing": achieved_tflops / 149.7 * 100.0,
                "formula": (
                    "F_iter = 72*B*S*L*H^2 + 6*B*L*S^2*H + 6*B*S*H*V; "
                    "MFU = (F_iter / step_seconds) / 149.7e12"
                ),
            },
            "instrumentation": {
                "cuda_profiler_capture_range": True,
                "autograd_nvtx": True,
                "manual_nvtx_ranges": [
                    "profile_window",
                    "train_step_NN",
                    "optimizer_zero_grad",
                    "forward",
                    "backward",
                    "optimizer_step",
                ],
                "transformer_engine_enabled": False,
                "cuda_graph_enabled": False,
                "kernel_fusion_added": False,
            },
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE2_NSYS_RUN_METRICS_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
