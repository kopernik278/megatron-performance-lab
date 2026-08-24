#!/usr/bin/env python3
"""Run one Phase 6.3 MCore DDP CUDA Graph timing variant."""

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

from phase1_baseline import (
    A40_DENSE_BF16_PEAK_TFLOPS,
    NvidiaSmiSampler,
    TE_FUSED_ATTENTION,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
    training_flops_per_iteration,
)
from phase3_attention_correctness import fused_backend_status
from phase3_attention_profile import baseline_model_args
from phase6_cuda_graph_run import CUDA_GRAPH_IMPL, model_config, parse_bool, verify_graph_state
from phase6_megatron_ddp_lifecycle import (
    DDPOptimizerBundle,
    assert_main_grad_pointers,
    build_ddp_optimizer_bundle,
    create_local_cudagraphs_preserving_gradients,
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
    parser.add_argument("--cuda-graph", type=parse_bool, required=True)
    parser.add_argument("--graph-warmup-iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, required=True)
    parser.add_argument("--measured-iterations", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--profile-mode", action="store_true")
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.graph_warmup_iterations < 1:
        parser.error("graph warmup iterations must be positive")
    if args.warmup_iterations < 1 or args.measured_iterations < 1:
        parser.error("benchmark iteration counts must be positive")
    return args


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


def lifecycle_step(
    bundle: DDPOptimizerBundle,
    batch: dict[str, torch.Tensor],
    create_graphs: bool = False,
) -> tuple[float, dict[str, float] | None]:
    zero_gradients(bundle)
    loss = forward_loss(bundle.model, batch)
    loss.backward()
    finalize_gradients(bundle.model)

    capture_stats = None
    if create_graphs:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
            cache_enabled=False,
        ):
            capture_stats = create_local_cudagraphs_preserving_gradients(
                bundle.model
            )

    update_successful, _, _ = bundle.optimizer.step()
    if not update_successful:
        raise RuntimeError("FP32Optimizer unexpectedly rejected an update")
    return float(loss.detach().cpu()), capture_stats


def instrumented_step(
    bundle: DDPOptimizerBundle,
    batch: dict[str, torch.Tensor],
    step_index: int,
) -> float:
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
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


def run_measured_steps(
    bundle: DDPOptimizerBundle,
    batch: dict[str, torch.Tensor],
    measured_iterations: int,
    device: torch.device,
    profile_mode: bool,
) -> tuple[list[float], list[float], list[float]]:
    losses: list[float] = []
    step_times_ms: list[float] = []
    cpu_process_times_ms: list[float] = []
    if profile_mode:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        with torch.cuda.nvtx.range("profile_window"):
            for step_index in range(measured_iterations):
                torch.cuda.synchronize(device)
                wall_start = time.perf_counter()
                cpu_start = time.process_time()
                losses.append(
                    instrumented_step(
                        bundle,
                        batch,
                        step_index,
                    )
                )
                torch.cuda.synchronize(device)
                cpu_process_times_ms.append(
                    (time.process_time() - cpu_start) * 1000.0
                )
                step_times_ms.append((time.perf_counter() - wall_start) * 1000.0)
    finally:
        torch.cuda.synchronize(device)
        if profile_mode:
            torch.cuda.cudart().cudaProfilerStop()
    return losses, step_times_ms, cpu_process_times_ms


def verify_controls(
    bundle: DDPOptimizerBundle,
    cuda_graph_enabled: bool,
) -> dict[str, Any]:
    model = unwrap_model(bundle.model)
    config = model.config
    expected = {
        "bias_dropout_fusion": True,
        "bias_activation_fusion": False,
        "masked_softmax_fusion": False,
        "cross_entropy_loss_fusion": False,
    }
    actual = {name: bool(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected optimization configuration: {actual}")
    expected_impl = CUDA_GRAPH_IMPL if cuda_graph_enabled else "none"
    if config.cuda_graph_impl != expected_impl:
        raise RuntimeError(
            f"Expected cuda_graph_impl={expected_impl}, got {config.cuda_graph_impl}"
        )
    if config.cuda_graph_modules:
        raise RuntimeError("Phase 6.3 must capture the whole TransformerLayer")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"Parameter dtype changed: {parameter_dtypes}")
    lifecycle = lifecycle_metadata(bundle)
    if lifecycle["use_distributed_optimizer"]:
        raise RuntimeError("Distributed optimizer must remain disabled")
    return {
        "transformer_config": {
            **actual,
            "cuda_graph_impl": config.cuda_graph_impl,
            "cuda_graph_modules": [],
            "cuda_graph_warmup_steps": config.cuda_graph_warmup_steps,
        },
        "parameter_dtypes": parameter_dtypes,
        "params_dtype": str(config.params_dtype),
        "pipeline_dtype": str(config.pipeline_dtype),
        "fp32_residual_connection": bool(config.fp32_residual_connection),
        "graph_safe_rng_tracker_for_both_variants": "Transformer Engine",
        "ddp_lifecycle": lifecycle,
    }


def gradients_finite(bundle: DDPOptimizerBundle) -> bool:
    return all(
        bool(torch.isfinite(parameter.main_grad).all().item())
        for _, parameter in named_trainable_parameters(bundle.model)
    )


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_single_gpu_distributed(
        model_args.seed,
        use_te_rng_tracker=True,
    )
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("TE fused attention requires Transformer Engine")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")

        device = torch.device(f"cuda:{local_rank}")
        raw_model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            bias_dropout_fusion=True,
            cuda_graph_impl=CUDA_GRAPH_IMPL if args.cuda_graph else "none",
            cuda_graph_warmup_steps=args.graph_warmup_iterations,
        )
        raw_model.train()
        bundle = build_ddp_optimizer_bundle(
            raw_model,
            model_args.learning_rate,
        )
        controls = verify_controls(bundle, args.cuda_graph)
        gradient_pointers = main_grad_pointers(bundle.model)
        batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
        parameter_count = sum(
            parameter.numel() for parameter in unwrap_model(bundle.model).parameters()
        )

        priming_loss, capture_stats = lifecycle_step(
            bundle,
            batch,
            create_graphs=args.cuda_graph,
        )
        graph_state = verify_graph_state(
            unwrap_model(bundle.model),
            args.cuda_graph,
        )

        warmup_losses = []
        for _ in range(args.warmup_iterations):
            warmup_loss, _ = lifecycle_step(bundle, batch)
            warmup_losses.append(warmup_loss)
            torch.cuda.synchronize(device)

        backend = fused_backend_status()
        print("FusedAttention backend (sub-backend 1)")

        torch.cuda.reset_peak_memory_stats(device)
        sampler = NvidiaSmiSampler(args.gpu_sample_interval_ms)
        sampler.start()
        try:
            measured_losses, step_times_ms, cpu_process_times_ms = run_measured_steps(
                bundle,
                batch,
                args.measured_iterations,
                device,
                args.profile_mode,
            )
        finally:
            gpu_monitoring = sampler.stop()

        losses_finite = all(
            math.isfinite(value)
            for value in [priming_loss, *warmup_losses, *measured_losses]
        )
        if not losses_finite:
            raise RuntimeError("A non-finite training loss was observed")
        if not gradients_finite(bundle):
            raise RuntimeError("A non-finite DDP main_grad was observed")
        assert_main_grad_pointers(bundle.model, gradient_pointers)

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
        environment = collect_environment()
        environment["cuda_graph_enabled"] = args.cuda_graph
        result = {
            "status": "success",
            "experiment": "Phase 6.3 MCore DDP CUDA Graph fast A/B",
            "run_label": args.run_label,
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "variant": "B" if args.cuda_graph else "A",
            "cuda_graph_enabled": args.cuda_graph,
            "parameter_count": parameter_count,
            "model_config": model_config(args.cuda_graph),
            "control_verification": controls,
            "cuda_graph": {
                "mechanism": (
                    "Megatron Core cuda_graph_impl=local with empty "
                    "cuda_graph_modules (whole TransformerLayer)"
                ),
                "optimizer_inside_graph": False,
                "zero_grad_inside_graph": False,
                "raw_adamw_main_grad_bridge": False,
                "ddp_owned_main_grad": True,
                "captured_parameter_count": (
                    sum(
                        parameter.numel() > 0
                        for layer in unwrap_model(bundle.model).decoder.layers
                        for parameter in layer.parameters()
                        if parameter.requires_grad
                    )
                    if args.cuda_graph
                    else 0
                ),
                "capture_warmup_iterations": (
                    args.graph_warmup_iterations if args.cuda_graph else 0
                ),
                "capture_stats": capture_stats,
                "state": graph_state,
            },
            "parallelism": {
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": 1,
            },
            "precision": {
                "forward_backward": "BF16 autocast",
                "parameter_storage": "FP32",
                "main_grad": "FP32",
                "residual_stream": "FP32",
                "optimizer_state": "FP32",
                "bf16_enabled": True,
            },
            "optimizer": {
                "wrapper": "megatron.core.optimizer.optimizer.FP32Optimizer",
                "name": "torch.optim.AdamW",
                "learning_rate": model_args.learning_rate,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "foreach": False,
                "fused": False,
                "clip_grad": 0.0,
                "distributed_optimizer": False,
                "zero_grad_set_to_none": True,
            },
            "data": {
                "type": "fixed synthetic random token IDs",
                "seed": model_args.seed + 1,
                "model_seed": model_args.seed,
                "static_tensor_addresses": True,
            },
            "micro_batch_size": MICRO_BATCH_SIZE,
            "global_batch_size": MICRO_BATCH_SIZE,
            "sequence_length": model_args.sequence_length,
            "tokens_per_step": tokens_per_step,
            "priming_iterations": 1,
            "benchmark_warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "priming_loss": priming_loss,
            "warmup_final_loss": warmup_losses[-1],
            "measured_losses": measured_losses,
            "final_loss": measured_losses[-1],
            "losses_finite": losses_finite,
            "main_grads_finite": True,
            "main_grad_addresses_stable": True,
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": statistics.median(step_times_ms),
            "step_time_standard_deviation_ms": (
                statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0
            ),
            "step_times_ms": step_times_ms,
            "cpu_process_time": {
                "average_ms_per_step": statistics.fmean(cpu_process_times_ms),
                "median_ms_per_step": statistics.median(cpu_process_times_ms),
                "standard_deviation_ms": (
                    statistics.stdev(cpu_process_times_ms)
                    if len(cpu_process_times_ms) > 1
                    else 0.0
                ),
                "samples_ms": cpu_process_times_ms,
                "definition": (
                    "time.process_time around each synchronized end-to-end step; "
                    "excludes time the process is descheduled"
                ),
            },
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
                "manual_step_nvtx": True,
                "nsys_cuda_graph_node_trace_required": args.cuda_graph,
                "unrelated_optimization_added": False,
            },
            "invocation": {
                "argv": sys.argv,
                "cuda_device_max_connections": os.environ.get(
                    "CUDA_DEVICE_MAX_CONNECTIONS"
                ),
            },
            "environment": environment,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print("PHASE6_DDP_CUDA_GRAPH_RUN_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
