#!/usr/bin/env python3
"""Phase 4.2 fast screen: BF16 hidden/residual stream vs current fused MB=8."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from megatron.core import parallel_state

from bf16_hidden_residual import (
    assert_fp32_master_weights,
    enable_bf16_hidden_residual_stream,
)
from phase1_baseline import (
    A40_DENSE_BF16_PEAK_TFLOPS,
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
from phase3_attention_correctness import fused_backend_status, tensor_error
from phase3_attention_profile import baseline_model_args, model_config


PHASE41_COPY_CAST_MS = 255.9110933999933
PHASE41_BF16_COPY_MS = 174.722054399994
PHASE41_DIRECT_COPY_MS = 81.18903899999928
COPY_KERNEL_TOKENS = (
    "bfloat16_copy",
    "direct_copy",
    "copy_kernel",
    "load_withcast",
    "store_withcast",
)
MICRO_BATCH_SIZE = 8
COSINE_STRONG_ALIGN = 0.99
THROUGHPUT_PROMISING_PERCENT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=200)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def event_name(event: Any) -> str:
    return str(getattr(event, "name", "") or "")


def event_cuda_self_ms(event: Any) -> float:
    value = float(
        getattr(event, "self_device_time_total", 0.0)
        or getattr(event, "self_cuda_time_total", 0.0)
        or 0.0
    )
    return value / 1000.0


def event_cuda_total_ms(event: Any) -> float:
    value = float(
        getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0) or 0.0
    )
    return value / 1000.0


def is_copy_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in COPY_KERNEL_TOKENS)


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


def summarize_copy_profile(prof: profile, steps: int) -> dict[str, Any]:
    families: dict[str, dict[str, float]] = {}
    copy_self_ms = 0.0
    copy_calls = 0
    aten_copy_ms = 0.0
    aten_copy_calls = 0
    for event in prof.events():
        name = event_name(event)
        if name == "aten::copy_":
            aten_copy_ms += event_cuda_total_ms(event)
            aten_copy_calls += 1
        device_type = str(getattr(event, "device_type", "")).lower()
        is_cuda_kernel = "cuda" in device_type and not name.startswith("aten::")
        if not (is_cuda_kernel and is_copy_kernel(name)):
            continue
        family = kernel_family(name)
        row = families.setdefault(family, {"cuda_ms": 0.0, "calls": 0})
        cuda_ms = event_cuda_self_ms(event)
        row["cuda_ms"] += cuda_ms
        row["calls"] += 1
        copy_self_ms += cuda_ms
        copy_calls += 1
    return {
        "profiled_steps": steps,
        "aten_copy_cuda_ms_per_step": aten_copy_ms / steps,
        "aten_copy_calls_per_step": aten_copy_calls / steps,
        "copy_cast_kernel_ms_per_step": copy_self_ms / steps,
        "copy_cast_kernel_calls_per_step": copy_calls / steps,
        "families": {
            name: {
                "cuda_ms_per_step": row["cuda_ms"] / steps,
                "calls_per_step": row["calls"] / steps,
            }
            for name, row in sorted(families.items(), key=lambda item: item[1]["cuda_ms"], reverse=True)
        },
    }


def cosine_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    left = reference.detach().float().reshape(-1)
    right = candidate.detach().float().reshape(-1)
    denom = float(left.norm() * right.norm())
    if denom == 0.0:
        return 1.0 if float(left.norm()) == 0.0 and float(right.norm()) == 0.0 else 0.0
    return float(torch.dot(left, right) / denom)


def has_nonfinite(tensor: torch.Tensor) -> bool:
    return bool((~torch.isfinite(tensor)).any().item())


def make_model(
    model_args: argparse.Namespace,
    *,
    hidden_dropout: float,
    attention_dropout: float,
    bf16_hidden_residual: bool,
) -> torch.nn.Module:
    torch.manual_seed(model_args.seed)
    model = build_model(
        model_args,
        attention_implementation=TE_FUSED_ATTENTION,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
    )
    wrap_info = None
    if bf16_hidden_residual:
        wrap_info = enable_bf16_hidden_residual_stream(model)
    non_fp32 = assert_fp32_master_weights(model)
    if non_fp32:
        raise RuntimeError(f"Master weights are not all FP32: {non_fp32[:8]}")
    model._bf16_wrap_info = wrap_info
    return model


def forward_backward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = masked_language_model_loss(output, batch["loss_mask"])
    loss.backward()
    torch.cuda.synchronize()
    return output, loss


def collect_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def probe_activation_dtypes(model: torch.nn.Module) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def describe(value: Any) -> dict[str, Any] | None:
        if not torch.is_tensor(value) or not value.is_floating_point():
            return None
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
        }

    def make_hook(label: str):
        def hook(_module: torch.nn.Module, inputs: Any, output: Any) -> None:
            first_input = inputs[0] if inputs else None
            first_output = output[0] if isinstance(output, tuple) else output
            records.append(
                {
                    "site": label,
                    "input": describe(first_input),
                    "output": describe(first_output),
                }
            )

        return hook

    handles = []
    named = dict(model.named_modules())
    for label, key in (
        ("embedding", "embedding"),
        ("decoder", "decoder"),
        ("final_layernorm", "decoder.final_layernorm"),
        ("output_layer", "output_layer"),
    ):
        module = named.get(key)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(label)))
    return records, handles


def run_correctness(model_args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    batch = synthetic_batch(model_args, MICRO_BATCH_SIZE, device)
    baseline = make_model(
        model_args,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16_hidden_residual=False,
    )
    baseline.eval()
    state_dict = {
        name: value.detach().cpu().clone()
        if isinstance(value, torch.Tensor)
        else value
        for name, value in baseline.state_dict().items()
    }
    baseline_output, baseline_loss = forward_backward(baseline, batch)
    baseline_grads = collect_gradients(baseline)
    baseline_output_cpu = baseline_output.detach().cpu()
    baseline_loss_value = float(baseline_loss.detach().cpu())
    del baseline, baseline_output, baseline_loss
    gc.collect()
    torch.cuda.empty_cache()

    candidate = make_model(
        model_args,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16_hidden_residual=True,
    )
    candidate.load_state_dict(state_dict, strict=True)
    candidate.eval()
    probe_records, handles = probe_activation_dtypes(candidate)
    candidate_output, candidate_loss = forward_backward(candidate, batch)
    for handle in handles:
        handle.remove()
    candidate_grads = collect_gradients(candidate)
    candidate_output_cpu = candidate_output.detach().cpu()
    candidate_loss_value = float(candidate_loss.detach().cpu())
    wrap_info = getattr(candidate, "_bf16_wrap_info", None)
    del candidate, candidate_output, candidate_loss
    gc.collect()
    torch.cuda.empty_cache()

    if baseline_grads.keys() != candidate_grads.keys():
        raise RuntimeError("Baseline and BF16-residual variants produced different gradient sets")

    gradient_rows: list[dict[str, Any]] = []
    for name in baseline_grads:
        reference = baseline_grads[name]
        candidate_grad = candidate_grads[name]
        row = tensor_error(reference, candidate_grad)
        row["name"] = name
        row["cosine_similarity"] = cosine_similarity(reference, candidate_grad)
        row["has_nonfinite"] = has_nonfinite(reference) or has_nonfinite(candidate_grad)
        gradient_rows.append(row)
    gradient_rows.sort(key=lambda item: item["max_absolute_error"], reverse=True)
    cosines = [row["cosine_similarity"] for row in gradient_rows]
    worst = gradient_rows[0]
    return {
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "seed": model_args.seed,
        "bitwise_equality_required": False,
        "wrap_info": wrap_info,
        "dtype_boundaries_observed": probe_records,
        "loss": {
            "baseline": baseline_loss_value,
            "bf16_residual": candidate_loss_value,
            "absolute_difference": abs(candidate_loss_value - baseline_loss_value),
            "relative_difference": abs(candidate_loss_value - baseline_loss_value)
            / max(abs(baseline_loss_value), 1.0e-6),
            "baseline_nonfinite": not torch.isfinite(torch.tensor(baseline_loss_value)),
            "bf16_residual_nonfinite": not torch.isfinite(torch.tensor(candidate_loss_value)),
        },
        "forward": {
            **tensor_error(baseline_output_cpu, candidate_output_cpu),
            "baseline_nonfinite": has_nonfinite(baseline_output_cpu),
            "bf16_residual_nonfinite": has_nonfinite(candidate_output_cpu),
        },
        "gradients": {
            "compared_parameter_count": len(gradient_rows),
            "mean_cosine_similarity": statistics.fmean(cosines),
            "min_cosine_similarity": min(cosines),
            "worst_parameter": worst["name"],
            "worst_error": {key: worst[key] for key in worst if key != "name"},
            "representative": [
                {key: row[key] for key in row}
                for row in gradient_rows[:8]
            ],
            "any_nonfinite": any(row["has_nonfinite"] for row in gradient_rows),
        },
    }


def summarize_timing(
    model_args: argparse.Namespace,
    step_times_ms: list[float],
    losses: list[float],
) -> dict[str, Any]:
    average_step_time_ms = statistics.fmean(step_times_ms)
    tokens_per_iteration = MICRO_BATCH_SIZE * model_args.sequence_length
    tokens_per_second = tokens_per_iteration / (average_step_time_ms / 1000.0)
    flops_per_iteration = training_flops_per_iteration(
        MICRO_BATCH_SIZE,
        model_args.sequence_length,
        model_args.num_layers,
        model_args.hidden_size,
        model_args.vocab_size,
    )
    achieved_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1.0e12
    return {
        "average_step_time_ms": average_step_time_ms,
        "median_step_time_ms": statistics.median(step_times_ms),
        "step_times_ms": step_times_ms,
        "tokens_per_second": tokens_per_second,
        "final_loss": losses[-1],
        "measured_losses": losses,
        "mfu": {
            "training_flops_per_iteration": flops_per_iteration,
            "achieved_tflops": achieved_tflops,
            "gpu_dense_bf16_peak_tflops": A40_DENSE_BF16_PEAK_TFLOPS,
            "mfu_percent": achieved_tflops / A40_DENSE_BF16_PEAK_TFLOPS * 100.0,
        },
    }


def run_performance(
    model_args: argparse.Namespace,
    device: torch.device,
    *,
    bf16_hidden_residual: bool,
    warmup_iterations: int,
    measured_iterations: int,
    gpu_sample_interval_ms: int,
) -> dict[str, Any]:
    model = make_model(
        model_args,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        bf16_hidden_residual=bf16_hidden_residual,
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
    opt_state = optimizer.state[next(iter(model.parameters()))]
    optimizer_state_dtypes = sorted(
        {str(value.dtype).replace("torch.", "") for value in opt_state.values() if torch.is_tensor(value)}
    )
    torch.cuda.reset_peak_memory_stats(device)
    sampler = NvidiaSmiSampler(gpu_sample_interval_ms)
    losses: list[float] = []
    step_times_ms: list[float] = []
    sampler.start()
    try:
        for _ in range(measured_iterations):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            losses.append(train_step(model, optimizer, batch))
            torch.cuda.synchronize(device)
            step_times_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        gpu_monitoring = sampler.stop()
    timing = summarize_timing(model_args, step_times_ms, losses)
    result = {
        "variant": "B_bf16_hidden_residual" if bf16_hidden_residual else "A_current_fp32_residual",
        "bf16_hidden_residual": bf16_hidden_residual,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_dtypes": sorted(
            {str(parameter.dtype).replace("torch.", "") for parameter in model.parameters()}
        ),
        "optimizer_state_dtypes": optimizer_state_dtypes,
        "wrap_info": getattr(model, "_bf16_wrap_info", None),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "gpu_monitoring": gpu_monitoring,
        **timing,
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_variant_profiler(
    model_args: argparse.Namespace,
    device: torch.device,
    *,
    bf16_hidden_residual: bool,
    warmup_iterations: int,
    profile_iterations: int,
) -> dict[str, Any]:
    model = make_model(
        model_args,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        bf16_hidden_residual=bf16_hidden_residual,
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
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(profile_iterations):
            train_step(model, optimizer, batch)
            torch.cuda.synchronize(device)
    summary = summarize_copy_profile(prof, profile_iterations)
    del model, optimizer, prof
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def decide(correctness: dict[str, Any], variant_a: dict[str, Any], variant_b: dict[str, Any], profile_b: dict[str, Any]) -> dict[str, Any]:
    no_nan = not (
        correctness["loss"]["baseline_nonfinite"]
        or correctness["loss"]["bf16_residual_nonfinite"]
        or correctness["forward"]["baseline_nonfinite"]
        or correctness["forward"]["bf16_residual_nonfinite"]
        or correctness["gradients"]["any_nonfinite"]
    )
    grads_aligned = correctness["gradients"]["min_cosine_similarity"] >= COSINE_STRONG_ALIGN
    copy_cast_ms = profile_b["copy_cast_kernel_ms_per_step"]
    bfloat16_copy_ms = profile_b["families"].get("bfloat16_copy", {}).get("cuda_ms_per_step", 0.0)
    direct_copy_ms = profile_b["families"].get("direct_copy", {}).get("cuda_ms_per_step", 0.0)
    copy_reduction_percent = (PHASE41_COPY_CAST_MS - copy_cast_ms) / PHASE41_COPY_CAST_MS * 100.0
    throughput_delta_percent = (
        (variant_b["tokens_per_second"] - variant_a["tokens_per_second"])
        / variant_a["tokens_per_second"]
        * 100.0
    )
    copy_down_substantially = copy_reduction_percent >= 20.0
    throughput_up = throughput_delta_percent >= THROUGHPUT_PROMISING_PERCENT
    promising = no_nan and grads_aligned and copy_down_substantially and throughput_up
    return {
        "promising": promising,
        "full_20_plus_100_benchmark_run": False,
        "gates": {
            "no_nan_inf": no_nan,
            "gradients_strongly_aligned": grads_aligned,
            "min_cosine_similarity": correctness["gradients"]["min_cosine_similarity"],
            "cosine_threshold": COSINE_STRONG_ALIGN,
            "copy_cast_decreased_substantially": copy_down_substantially,
            "copy_cast_reduction_percent_vs_phase41": copy_reduction_percent,
            "throughput_improved_at_least_5_percent": throughput_up,
            "throughput_delta_percent": throughput_delta_percent,
        },
        "copy_cast_ms_per_step": {
            "phase41": PHASE41_COPY_CAST_MS,
            "variant_b": copy_cast_ms,
            "delta": copy_cast_ms - PHASE41_COPY_CAST_MS,
        },
        "bfloat16_copy_ms_per_step": {
            "phase41": PHASE41_BF16_COPY_MS,
            "variant_b": bfloat16_copy_ms,
            "delta": bfloat16_copy_ms - PHASE41_BF16_COPY_MS,
        },
        "direct_copy_ms_per_step": {
            "phase41": PHASE41_DIRECT_COPY_MS,
            "variant_b": direct_copy_ms,
            "delta": direct_copy_ms - PHASE41_DIRECT_COPY_MS,
        },
        "hypothesis_supported": promising,
        "rationale": (
            "BF16 hidden/residual stream is promising: no NaN/Inf, gradients stay "
            "aligned, copy/cast fell substantially, and throughput improved by at least 5%."
            if promising
            else "Stop after the screen: numerical behavior, copy/cast reduction, or "
            "throughput did not meet the Phase 4.2 promising rule. No 20+100 benchmark."
        ),
    }


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_single_gpu_distributed(model_args.seed)
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("TE fused attention requires TRANSFORMER_ENGINE_DISABLE to be unset")
        device = torch.device(f"cuda:{local_rank}")
        correctness = run_correctness(model_args, device)
        backend = fused_backend_status()
        print("FusedAttention backend (sub-backend 1)")
        print("PHASE42_CORRECTNESS_DONE")
        variant_a = run_performance(
            model_args,
            device,
            bf16_hidden_residual=False,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
            gpu_sample_interval_ms=args.gpu_sample_interval_ms,
        )
        print("PHASE42_VARIANT_A_DONE")
        variant_b = run_performance(
            model_args,
            device,
            bf16_hidden_residual=True,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
            gpu_sample_interval_ms=args.gpu_sample_interval_ms,
        )
        print("PHASE42_VARIANT_B_DONE")
        profile_b = run_variant_profiler(
            model_args,
            device,
            bf16_hidden_residual=True,
            warmup_iterations=args.warmup_iterations,
            profile_iterations=args.profile_iterations,
        )
        print("PHASE42_PROFILE_B_DONE")
        decision = decide(correctness, variant_a, variant_b, profile_b)
        result = {
            "status": "success",
            "experiment": "Phase 4.2 BF16 hidden/residual stream screen",
            "optimization_applied": True,
            "full_benchmark_run": False,
            "ncu_attempted": False,
            "nsys_attempted": False,
            "infrastructure": {
                "pod_id": os.environ.get("RUNPOD_POD_ID"),
                "gpu": "1x NVIDIA A40 48GB",
                "gpu_price_per_hour_usd": 0.44,
                "replaced_unstartable_pod": "3ixl2btmmwghn5",
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
                    "hidden_residual_stream_A": "FP32",
                    "hidden_residual_stream_B": "BF16",
                },
                "optimizer": {"name": "torch.optim.AdamW", "foreach": False, "fused": False},
                "parallelism": {"tensor_parallel": 1, "pipeline_parallel": 1, "data_parallel": 1},
                "warmup_iterations": args.warmup_iterations,
                "measured_iterations": args.measured_iterations,
                "profile_iterations": args.profile_iterations,
            },
            "dtype_change": {
                "repository": "lab only; Megatron-LM was not modified",
                "file": "scripts/bf16_hidden_residual.py",
                "mechanism": (
                    "Wrap model.decoder.forward and cast hidden_states to BF16 once at "
                    "TransformerBlock entry. Residual connections then stay BF16, so "
                    "fused_bias_dropout x.to(residual.dtype) does not promote to FP32. "
                    "Loss still uses masked_language_model_loss(...).float()."
                ),
                "unchanged": [
                    "params_dtype=float32",
                    "pipeline_dtype=float32",
                    "TransformerConfig.bf16=False",
                    "AdamW foreach=False fused=False",
                    "TE cuDNN FusedAttention sub-backend 1",
                    "micro-batch 8",
                    "sequence length 2048",
                ],
            },
            "transformer_engine_backend": backend,
            "correctness": correctness,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "profile_b": profile_b,
            "comparison": {
                "step_time_ms": {
                    "A": variant_a["average_step_time_ms"],
                    "B": variant_b["average_step_time_ms"],
                    "delta_ms": variant_b["average_step_time_ms"] - variant_a["average_step_time_ms"],
                    "delta_percent": (
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
                    "delta_points": variant_b["mfu"]["mfu_percent"] - variant_a["mfu"]["mfu_percent"],
                },
                "peak_allocated_memory_mib": {
                    "A": variant_a["peak_allocated_memory_mib"],
                    "B": variant_b["peak_allocated_memory_mib"],
                    "delta_mib": variant_b["peak_allocated_memory_mib"] - variant_a["peak_allocated_memory_mib"],
                },
            },
            "decision": decision,
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE42_DECISION=" + json.dumps(decision, sort_keys=True))
        print("PHASE42_SCREEN_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
