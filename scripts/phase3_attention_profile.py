#!/usr/bin/env python3
"""Run one Phase 3.2 attention variant under a steady-state Nsight capture."""

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
    A40_DENSE_BF16_PEAK_TFLOPS,
    ATTENTION_IMPLEMENTATIONS,
    LOCAL_UNFUSED_ATTENTION,
    TE_FUSED_ATTENTION,
    NvidiaSmiSampler,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
    train_step,
    training_flops_per_iteration,
)
from phase3_attention_correctness import fused_backend_status


MICRO_BATCH_SIZE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-implementation",
        choices=ATTENTION_IMPLEMENTATIONS,
        required=True,
    )
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=100)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=200)
    parser.add_argument("--output-json", type=Path, required=True)
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


def model_config(attention_implementation: str) -> dict[str, Any]:
    fused = attention_implementation == TE_FUSED_ATTENTION
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
        "attention_backend": "fused" if fused else "unfused",
        "attention_implementation": attention_implementation,
        "core_attention": "TEDotProductAttention" if fused else "DotProductAttention",
        "transformer_layer_spec": "get_gpt_layer_local_spec",
    }


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_single_gpu_distributed(model_args.seed)
    try:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
        if args.attention_implementation == LOCAL_UNFUSED_ATTENTION:
            if os.environ.get("TRANSFORMER_ENGINE_DISABLE") != "1":
                raise RuntimeError("The local baseline requires TRANSFORMER_ENGINE_DISABLE=1")
        elif os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("TE fused attention requires TRANSFORMER_ENGINE_DISABLE to be unset")

        device = torch.device(f"cuda:{local_rank}")
        model = build_model(
            model_args,
            attention_implementation=args.attention_implementation,
        )
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

        te_backend = (
            fused_backend_status()
            if args.attention_implementation == TE_FUSED_ATTENTION
            else None
        )
        if te_backend is not None:
            print("FusedAttention backend (sub-backend 1)")

        torch.cuda.reset_peak_memory_stats(device)
        measured_losses: list[float] = []
        step_times_ms: list[float] = []
        sampler = NvidiaSmiSampler(args.gpu_sample_interval_ms)
        sampler.start()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range("profile_window"):
                with torch.autograd.profiler.emit_nvtx(record_shapes=False):
                    for step_index in range(args.measured_iterations):
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        measured_losses.append(
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
            torch.cuda.cudart().cudaProfilerStop()
            gpu_monitoring = sampler.stop()

        average_step_time_ms = statistics.fmean(step_times_ms)
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
            "attention_implementation": args.attention_implementation,
            "parameter_count": parameter_count,
            "model_config": model_config(args.attention_implementation),
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
            "micro_batch_size": MICRO_BATCH_SIZE,
            "global_batch_size": MICRO_BATCH_SIZE,
            "sequence_length": model_args.sequence_length,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "warmup_final_loss": warmup_losses[-1],
            "measured_losses": measured_losses,
            "final_loss": measured_losses[-1],
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": statistics.median(step_times_ms),
            "step_times_ms": step_times_ms,
            "tokens_per_second": (
                MICRO_BATCH_SIZE
                * model_args.sequence_length
                / (average_step_time_ms / 1000.0)
            ),
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
            "transformer_engine_backend": te_backend,
            "instrumentation": {
                "cuda_profiler_capture_range": True,
                "autograd_nvtx": True,
                "manual_nvtx_ranges": [
                    "profile_window",
                    "train_step_NNN",
                    "optimizer_zero_grad",
                    "forward",
                    "backward",
                    "optimizer_step",
                ],
                "cuda_graph_enabled": False,
                "unrelated_kernel_fusion_added": False,
            },
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE3_PROFILE_METRICS_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
