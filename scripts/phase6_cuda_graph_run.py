#!/usr/bin/env python3
"""Run one controlled Phase 6.1 CUDA Graph training variant."""

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
from megatron.core.transformer.cuda_graphs import create_cudagraphs

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


MICRO_BATCH_SIZE = 8
CUDA_GRAPH_IMPL = "local"
CUDA_GRAPH_MODULES: list[str] = []


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


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


def graph_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return parameters owned by the TransformerLayers captured by MCore."""

    parameters: dict[int, torch.nn.Parameter] = {}
    for layer in model.decoder.layers:
        for parameter in layer.parameters():
            if parameter.requires_grad:
                parameters[id(parameter)] = parameter
    return list(parameters.values())


def prepare_main_grad_buffers(parameters: list[torch.nn.Parameter]) -> None:
    """Add the persistent main_grad buffers required by MCore local graphs."""

    for parameter in parameters:
        if hasattr(parameter, "main_grad"):
            raise RuntimeError("Unexpected pre-existing main_grad on raw AdamW model")
        parameter.main_grad = torch.zeros_like(parameter)


def zero_gradients(
    optimizer: torch.optim.Optimizer,
    captured_parameters: list[torch.nn.Parameter],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for parameter in captured_parameters:
        parameter.main_grad.zero_()


def expose_main_grads_to_optimizer(
    captured_parameters: list[torch.nn.Parameter],
) -> None:
    """Temporarily bridge graph-owned main_grad buffers to torch AdamW."""

    for parameter in captured_parameters:
        if parameter.grad is not None:
            raise RuntimeError(
                "A captured TransformerLayer parameter unexpectedly received param.grad"
            )
        parameter.grad = parameter.main_grad


def clear_main_grad_aliases(captured_parameters: list[torch.nn.Parameter]) -> None:
    for parameter in captured_parameters:
        parameter.grad = None


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


def eager_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(model, batch)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def graphed_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    captured_parameters: list[torch.nn.Parameter],
) -> float:
    zero_gradients(optimizer, captured_parameters)
    loss = forward_loss(model, batch)
    loss.backward()
    expose_main_grads_to_optimizer(captured_parameters)
    try:
        optimizer.step()
    finally:
        clear_main_grad_aliases(captured_parameters)
    return float(loss.detach().cpu())


def record_and_create_graphs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    captured_parameters: list[torch.nn.Parameter],
) -> tuple[float, dict[str, float]]:
    """Record one eager iteration, then build the existing MCore layer graphs."""

    # MCore normally creates local graphs at the end of the first schedule, before
    # the outer optimizer step.  This harness uses raw AdamW, so finish an equivalent
    # eager priming step first, then clear param.grad before backward graph capture.
    priming_loss = eager_train_step(model, optimizer, batch)
    optimizer.zero_grad(set_to_none=True)
    for parameter in captured_parameters:
        parameter.main_grad.zero_()

    # The accepted workload uses torch autocast around model execution.  Capture
    # under the same BF16 autocast mode, with its weight cache disabled so graph
    # nodes do not retain casts allocated by a pre-capture warmup.
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
        cache_enabled=False,
    ):
        capture_stats = create_cudagraphs()
    torch.cuda.synchronize()
    if capture_stats is None:
        raise RuntimeError("MCore create_cudagraphs did not create any graphs")
    return priming_loss, {
        "time_seconds": float(capture_stats["time"]),
        "allocated_bytes": float(capture_stats["allocated_bytes"]),
        "reserved_bytes": float(capture_stats["reserved_bytes"]),
    }


def verify_graph_state(
    model: torch.nn.Module,
    cuda_graph_enabled: bool,
) -> dict[str, Any]:
    from megatron.core.transformer import cuda_graphs

    layers = list(model.decoder.layers)
    managers = [
        getattr(layer, "cudagraph_manager", None)
        for layer in layers
        if hasattr(layer, "cudagraph_manager")
    ]
    runners = [
        runner
        for manager in managers
        for runner in manager.cudagraph_runners
    ]
    state = {
        "implementation": (
            "MCore CudaGraphManager per-TransformerLayer"
            if cuda_graph_enabled
            else "eager"
        ),
        "cuda_graph_impl": model.config.cuda_graph_impl,
        "cuda_graph_modules": [
            getattr(module, "name", str(module))
            for module in model.config.cuda_graph_modules
        ],
        "transformer_layer_count": len(layers),
        "manager_count": len(managers),
        "runner_count": len(runners),
        "forward_graph_count": sum(runner.fwd_graph is not None for runner in runners),
        "backward_graph_count": sum(runner.bwd_graph is not None for runner in runners),
        "global_graphs_created": bool(
            cuda_graphs._CudagraphGlobalRecord.cudagraph_created
        ),
        "replay_ready": bool(runners)
        and all(
            runner.cudagraph_created
            and runner.fwd_graph is not None
            and runner.bwd_graph is not None
            for runner in runners
        ),
    }
    if cuda_graph_enabled:
        expected = len(layers)
        if (
            state["manager_count"] != expected
            or state["runner_count"] != expected
            or state["forward_graph_count"] != expected
            or state["backward_graph_count"] != expected
            or not state["global_graphs_created"]
            or not state["replay_ready"]
        ):
            raise RuntimeError(f"CUDA Graph creation verification failed: {state}")
    else:
        if managers or runners or state["global_graphs_created"]:
            raise RuntimeError(f"Eager variant unexpectedly owns CUDA graphs: {state}")
    return state


def instrumented_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    captured_parameters: list[torch.nn.Parameter],
    cuda_graph_enabled: bool,
    step_index: int,
) -> float:
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
        with torch.cuda.nvtx.range("optimizer_zero_grad"):
            if cuda_graph_enabled:
                zero_gradients(optimizer, captured_parameters)
            else:
                optimizer.zero_grad(set_to_none=True)
        with torch.cuda.nvtx.range("forward"):
            loss = forward_loss(model, batch)
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
        if cuda_graph_enabled:
            with torch.cuda.nvtx.range("optimizer_grad_bridge"):
                expose_main_grads_to_optimizer(captured_parameters)
        with torch.cuda.nvtx.range("optimizer_step"):
            try:
                optimizer.step()
            finally:
                if cuda_graph_enabled:
                    clear_main_grad_aliases(captured_parameters)
    return float(loss.detach().cpu())


def run_measured_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    captured_parameters: list[torch.nn.Parameter],
    cuda_graph_enabled: bool,
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
                    instrumented_train_step(
                        model,
                        optimizer,
                        batch,
                        captured_parameters,
                        cuda_graph_enabled,
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


def model_config(cuda_graph_enabled: bool) -> dict[str, Any]:
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
        "bias_dropout_fusion": True,
        "bias_activation_fusion": False,
        "masked_softmax_fusion": False,
        "cross_entropy_loss_fusion": False,
        "cuda_graph_enabled": cuda_graph_enabled,
        "cuda_graph_impl": CUDA_GRAPH_IMPL if cuda_graph_enabled else "none",
        "cuda_graph_modules": CUDA_GRAPH_MODULES,
    }


def optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> list[str]:
    dtypes = {
        str(value.dtype)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    return sorted(dtypes)


def verify_controls(
    model: torch.nn.Module,
    cuda_graph_enabled: bool,
) -> dict[str, Any]:
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
        raise RuntimeError("Phase 6.1 must capture the whole TransformerLayer")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"Parameter dtype changed: {parameter_dtypes}")
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
    }


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
        model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            bias_dropout_fusion=True,
            cuda_graph_impl=CUDA_GRAPH_IMPL if args.cuda_graph else "none",
            cuda_graph_warmup_steps=args.graph_warmup_iterations,
        )
        controls = verify_controls(model, args.cuda_graph)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_args.learning_rate,
            foreach=False,
            fused=False,
        )
        batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        captured_parameters = graph_parameters(model) if args.cuda_graph else []
        if args.cuda_graph:
            prepare_main_grad_buffers(captured_parameters)

        capture_stats: dict[str, float] | None = None
        if args.cuda_graph:
            priming_loss, capture_stats = record_and_create_graphs(
                model,
                optimizer,
                batch,
                captured_parameters,
            )
        else:
            priming_loss = eager_train_step(model, optimizer, batch)
        graph_state = verify_graph_state(model, args.cuda_graph)

        warmup_losses: list[float] = []
        step_function = graphed_train_step if args.cuda_graph else eager_train_step
        for _ in range(args.warmup_iterations):
            if args.cuda_graph:
                warmup_losses.append(
                    step_function(
                        model,
                        optimizer,
                        batch,
                        captured_parameters,
                    )
                )
            else:
                warmup_losses.append(step_function(model, optimizer, batch))
            torch.cuda.synchronize(device)

        backend = fused_backend_status()
        print("FusedAttention backend (sub-backend 1)")

        torch.cuda.reset_peak_memory_stats(device)
        sampler = NvidiaSmiSampler(args.gpu_sample_interval_ms)
        sampler.start()
        try:
            measured_losses, step_times_ms, cpu_process_times_ms = run_measured_steps(
                model,
                optimizer,
                batch,
                captured_parameters,
                args.cuda_graph,
                args.measured_iterations,
                device,
                args.profile_mode,
            )
        finally:
            gpu_monitoring = sampler.stop()

        if not all(
            math.isfinite(value)
            for value in [priming_loss, *warmup_losses, *measured_losses]
        ):
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
        environment = collect_environment()
        environment["cuda_graph_enabled"] = args.cuda_graph
        result = {
            "status": "success",
            "experiment": "Phase 6.1 CUDA Graph feasibility and fast A/B",
            "run_label": args.run_label,
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "variant": "B" if args.cuda_graph else "A",
            "cuda_graph_enabled": args.cuda_graph,
            "parameter_count": parameter_count,
            "model_config": model_config(args.cuda_graph),
            "control_verification": {
                **controls,
                "optimizer_state_dtypes": optimizer_state_dtypes(optimizer),
            },
            "cuda_graph": {
                "mechanism": (
                    "Megatron Core cuda_graph_impl=local with empty "
                    "cuda_graph_modules (whole TransformerLayer)"
                ),
                "optimizer_inside_graph": False,
                "zero_grad_inside_graph": False,
                "raw_adamw_main_grad_bridge": args.cuda_graph,
                "captured_parameter_count": len(captured_parameters),
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
                "residual_stream": "FP32",
                "optimizer_state": "FP32",
                "bf16_enabled": True,
            },
            "optimizer": {
                "name": "torch.optim.AdamW",
                "learning_rate": model_args.learning_rate,
                "foreach": False,
                "fused": False,
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
            "losses_finite": True,
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
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE6_CUDA_GRAPH_RUN_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
