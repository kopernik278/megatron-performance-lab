#!/usr/bin/env python3
"""Analyze the Phase 3.2 local-versus-fused Nsight Systems traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_nsys import sha256


NS_PER_MS = 1_000_000.0
ATTENTION_TOKENS = (
    "attention",
    "attn",
    "bmm",
    "softmax",
    "masked_fill",
    "masked_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--fused-sqlite", type=Path, required=True)
    parser.add_argument("--fused-metrics", type=Path, required=True)
    parser.add_argument("--fused-trace", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nvtx_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    return row["text"] or strings.get(row["textId"], "")


def attention_nvtx_names_by_correlation(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> dict[int, set[str]]:
    ranges_by_thread: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    query = """
        SELECT start, end, globalTid, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
    """
    for row in connection.execute(query, (window_start, window_end)):
        name = nvtx_name(row, strings)
        lowered = name.lower()
        if any(token in lowered for token in ATTENTION_TOKENS):
            ranges_by_thread[row["globalTid"]].append((row["start"], row["end"], name))

    calls_by_thread: dict[int, list[tuple[int, int]]] = defaultdict(list)
    query = """
        SELECT start, globalTid, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start >= ? AND start < ? AND correlationId IS NOT NULL
    """
    for row in connection.execute(query, (window_start, window_end)):
        calls_by_thread[row["globalTid"]].append((row["start"], row["correlationId"]))

    result: dict[int, set[str]] = defaultdict(set)
    for thread_id, calls in calls_by_thread.items():
        events: list[tuple[int, int, int, str | int]] = []
        for index, (start, end, name) in enumerate(ranges_by_thread.get(thread_id, [])):
            events.append((start, 0, index, name))
            events.append((end, 2, index, name))
        for start, correlation_id in calls:
            events.append((start, 1, correlation_id, correlation_id))
        events.sort(key=lambda item: (item[0], item[1]))

        active: dict[int, str] = {}
        for _, event_kind, identifier, payload in events:
            if event_kind == 0:
                active[identifier] = str(payload)
            elif event_kind == 1:
                result[int(payload)].update(active.values())
            else:
                active.pop(identifier, None)
    return result


def analyze_trace(
    sqlite_path: Path,
    trace_path: Path,
    run_metrics: dict[str, Any],
) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    strings = {
        row["id"]: row["value"]
        for row in connection.execute("SELECT id, value FROM StringIds")
    }
    window = connection.execute(
        """
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE (text = 'profile_window' OR textId IN (
            SELECT id FROM StringIds WHERE value = 'profile_window'
        )) AND end IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    if window is None:
        raise RuntimeError(f"profile_window NVTX range not found in {sqlite_path}")
    window_start, window_end = window["start"], window["end"]

    kernels = list(
        connection.execute(
            """
            SELECT start, end, correlationId, demangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )
    memcopies = list(
        connection.execute(
            """
            SELECT start, end, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )
    correlation_ranges = attention_nvtx_names_by_correlation(
        connection,
        strings,
        window_start,
        window_end,
    )

    kernel_total_ns = 0
    attention_total_ns = 0
    gemm_total_ns = 0
    under_50_us = 0
    kernel_totals: dict[str, int] = defaultdict(int)
    kernel_counts: dict[str, int] = defaultdict(int)
    for row in kernels:
        duration = row["end"] - row["start"]
        name = strings.get(row["demangledName"], f"StringId({row['demangledName']})")
        lowered = name.lower()
        active_ranges = " ".join(
            correlation_ranges.get(row["correlationId"], set())
        ).lower()
        kernel_total_ns += duration
        kernel_totals[name] += duration
        kernel_counts[name] += 1
        if duration < 50_000:
            under_50_us += 1
        if "gemm" in lowered:
            gemm_total_ns += duration
        if any(token in lowered or token in active_ranges for token in ATTENTION_TOKENS):
            attention_total_ns += duration

    top_kernels = [
        {
            "name": name,
            "count": kernel_counts[name],
            "total_ms": total_ns / NS_PER_MS,
            "kernel_time_share_percent": total_ns / kernel_total_ns * 100.0,
        }
        for name, total_ns in sorted(
            kernel_totals.items(), key=lambda item: item[1], reverse=True
        )[:15]
    ]
    measured_iterations = run_metrics["measured_iterations"]
    memcpy_total_ns = sum(row["end"] - row["start"] for row in memcopies)
    summary = {
        "profile_window_ms": (window_end - window_start) / NS_PER_MS,
        "kernel_count": len(kernels),
        "kernel_count_per_step": len(kernels) / measured_iterations,
        "kernel_total_ms": kernel_total_ns / NS_PER_MS,
        "kernels_under_50_us": under_50_us,
        "kernels_under_50_us_percent": under_50_us / len(kernels) * 100.0,
        "attention_gpu_time_ms": attention_total_ns / NS_PER_MS,
        "attention_gpu_time_per_step_ms": attention_total_ns / measured_iterations / NS_PER_MS,
        "attention_gpu_time_share_percent": attention_total_ns / kernel_total_ns * 100.0,
        "gemm_gpu_time_ms": gemm_total_ns / NS_PER_MS,
        "gemm_gpu_time_per_step_ms": gemm_total_ns / measured_iterations / NS_PER_MS,
        "gemm_gpu_time_share_percent": gemm_total_ns / kernel_total_ns * 100.0,
        "memory_copy_count": len(memcopies),
        "memory_copy_time_ms": memcpy_total_ns / NS_PER_MS,
        "memory_copy_time_per_step_ms": memcpy_total_ns / measured_iterations / NS_PER_MS,
        "memory_copy_profile_window_share_percent": (
            memcpy_total_ns / (window_end - window_start) * 100.0
        ),
        "top_15_cuda_kernels_by_total_gpu_time": top_kernels,
        "trace": {
            "path": str(trace_path),
            "size_bytes": trace_path.stat().st_size,
            "sha256": sha256(trace_path),
            "sqlite_path": str(sqlite_path),
            "sqlite_size_bytes": sqlite_path.stat().st_size,
            "sqlite_sha256": sha256(sqlite_path),
        },
        "category_note": (
            "Attention and GEMM categories are non-exclusive and use total CUDA kernel "
            "execution time as the denominator. Attention uses kernel symbols and "
            "correlated autograd NVTX ranges."
        ),
    }
    connection.close()
    return summary


def percent_change(baseline: float, fused: float) -> float:
    return (fused - baseline) / baseline * 100.0


def improvement_when_lower(baseline: float, fused: float) -> float:
    return (baseline - fused) / baseline * 100.0


def compare_controls(baseline: dict[str, Any], fused: dict[str, Any]) -> None:
    model_exclusions = {"attention_backend", "attention_implementation", "core_attention"}
    baseline_model = {
        key: value for key, value in baseline["model_config"].items() if key not in model_exclusions
    }
    fused_model = {
        key: value for key, value in fused["model_config"].items() if key not in model_exclusions
    }
    if baseline_model != fused_model:
        raise RuntimeError("A/B model configurations differ beyond core attention")

    for field in (
        "parameter_count",
        "parallelism",
        "precision",
        "optimizer",
        "data",
        "micro_batch_size",
        "global_batch_size",
        "sequence_length",
        "warmup_iterations",
        "measured_iterations",
    ):
        if baseline[field] != fused[field]:
            raise RuntimeError(f"A/B control field differs: {field}")

    environment_exclusions = {
        "timestamp_utc",
        "transformer_engine_disabled",
    }
    baseline_environment = {
        key: value
        for key, value in baseline["environment"].items()
        if key not in environment_exclusions
    }
    fused_environment = {
        key: value
        for key, value in fused["environment"].items()
        if key not in environment_exclusions
    }
    if baseline_environment != fused_environment:
        raise RuntimeError("A/B software or hardware environments differ")


def variant_summary(run: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "attention_implementation": run["attention_implementation"],
        "average_step_time_ms": run["average_step_time_ms"],
        "median_step_time_ms": run["median_step_time_ms"],
        "tokens_per_second": run["tokens_per_second"],
        "mfu_percent": run["mfu"]["mfu_percent"],
        "peak_allocated_memory_mib": run["peak_allocated_memory_mib"],
        "peak_reserved_memory_mib": run["peak_reserved_memory_mib"],
        "peak_nvidia_smi_memory_mib": run["gpu_monitoring"][
            "peak_nvidia_smi_memory_mib"
        ],
        "average_gpu_utilization_percent": run["gpu_monitoring"][
            "average_gpu_utilization_percent"
        ],
        "final_loss": run["final_loss"],
        "profile": profile,
    }


def main() -> None:
    args = parse_args()
    baseline_run = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    fused_run = json.loads(args.fused_metrics.read_text(encoding="utf-8"))
    correctness = json.loads(args.correctness.read_text(encoding="utf-8"))
    compare_controls(baseline_run, fused_run)

    baseline_profile = analyze_trace(
        args.baseline_sqlite,
        args.baseline_trace,
        baseline_run,
    )
    fused_profile = analyze_trace(
        args.fused_sqlite,
        args.fused_trace,
        fused_run,
    )
    baseline = variant_summary(baseline_run, baseline_profile)
    fused = variant_summary(fused_run, fused_profile)

    b_step = baseline["average_step_time_ms"]
    f_step = fused["average_step_time_ms"]
    b_attention = baseline_profile["attention_gpu_time_per_step_ms"]
    f_attention = fused_profile["attention_gpu_time_per_step_ms"]
    hypothesis_supported = (
        f_step < b_step
        and f_attention < b_attention
        and fused_profile["kernel_count_per_step"] < baseline_profile["kernel_count_per_step"]
    )
    result = {
        "status": "success",
        "experiment": "Phase 3.2 controlled fused-attention A/B benchmark",
        "correctness": correctness,
        "infrastructure": {
            "pod_id": args.pod_id,
            "cloud": "Secure Cloud",
            "gpu": "1x NVIDIA A40 48GB",
            "gpu_price_per_hour_usd": 0.44,
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        },
        "controls": {
            "only_intended_difference": (
                "Megatron local DotProductAttention versus TEDotProductAttention "
                "forced to cuDNN FusedAttention sub-backend 1"
            ),
            "model_config": baseline_run["model_config"],
            "parameter_count": baseline_run["parameter_count"],
            "parallelism": baseline_run["parallelism"],
            "precision": baseline_run["precision"],
            "optimizer": baseline_run["optimizer"],
            "data": baseline_run["data"],
            "warmup_iterations": baseline_run["warmup_iterations"],
            "measured_iterations": baseline_run["measured_iterations"],
            "environment": fused_run["environment"],
        },
        "baseline": baseline,
        "fused": fused,
        "delta": {
            "average_step_time_ms": f_step - b_step,
            "step_time_improvement_percent": improvement_when_lower(b_step, f_step),
            "speedup_ratio": b_step / f_step,
            "tokens_per_second": fused["tokens_per_second"] - baseline["tokens_per_second"],
            "tokens_per_second_percent": percent_change(
                baseline["tokens_per_second"], fused["tokens_per_second"]
            ),
            "mfu_percentage_points": fused["mfu_percent"] - baseline["mfu_percent"],
            "mfu_percent": percent_change(baseline["mfu_percent"], fused["mfu_percent"]),
            "peak_nvidia_smi_memory_mib": (
                fused["peak_nvidia_smi_memory_mib"]
                - baseline["peak_nvidia_smi_memory_mib"]
            ),
            "kernel_count_per_step": (
                fused_profile["kernel_count_per_step"]
                - baseline_profile["kernel_count_per_step"]
            ),
            "kernel_count_improvement_percent": improvement_when_lower(
                baseline_profile["kernel_count_per_step"],
                fused_profile["kernel_count_per_step"],
            ),
            "kernels_under_50_us_percentage_points": (
                fused_profile["kernels_under_50_us_percent"]
                - baseline_profile["kernels_under_50_us_percent"]
            ),
            "attention_share_percentage_points": (
                fused_profile["attention_gpu_time_share_percent"]
                - baseline_profile["attention_gpu_time_share_percent"]
            ),
            "attention_time_per_step_ms": f_attention - b_attention,
            "attention_time_improvement_percent": improvement_when_lower(
                b_attention, f_attention
            ),
            "gemm_share_percentage_points": (
                fused_profile["gemm_gpu_time_share_percent"]
                - baseline_profile["gemm_gpu_time_share_percent"]
            ),
            "memory_copy_time_per_step_ms": (
                fused_profile["memory_copy_time_per_step_ms"]
                - baseline_profile["memory_copy_time_per_step_ms"]
            ),
        },
        "hypothesis": {
            "phase3_1_hypothesis_supported": hypothesis_supported,
            "criterion": (
                "Fused attention must reduce average step time, attention GPU time per step, "
                "and kernel count per step while all controls remain equal."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
