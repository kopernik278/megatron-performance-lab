#!/usr/bin/env python3
"""Analyze Phase 3.3 full runs and short Nsight Systems profiles."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_nsys import (
    category_matches,
    relevant_nvtx_names_by_correlation,
    sha256,
)


NS_PER_MS = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-batches", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("profiles"))
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.micro_batches = [int(value) for value in args.micro_batches.split(",")]
    if args.micro_batches[:3] != [1, 2, 4]:
        parser.error("--micro-batches must begin with 1,2,4")
    return args


def profile_summary(
    sqlite_path: Path,
    trace_path: Path,
    profile_run: dict[str, Any],
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
        raise RuntimeError(f"profile_window not found in {sqlite_path}")
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
    correlations = relevant_nvtx_names_by_correlation(
        connection,
        strings,
        window_start,
        window_end,
    )

    kernel_total_ns = 0
    category_totals_ns: dict[str, int] = defaultdict(int)
    kernel_totals_ns: dict[str, int] = defaultdict(int)
    kernel_counts: dict[str, int] = defaultdict(int)
    under_50_us = 0
    for row in kernels:
        duration = row["end"] - row["start"]
        name = strings.get(row["demangledName"], f"StringId({row['demangledName']})")
        kernel_total_ns += duration
        kernel_totals_ns[name] += duration
        kernel_counts[name] += 1
        if duration < 50_000:
            under_50_us += 1
        matches = category_matches(
            name,
            correlations.get(row["correlationId"], set()),
        )
        for category, matched in matches.items():
            if matched:
                category_totals_ns[category] += duration

    iterations = profile_run["measured_iterations"]
    tokens_per_step = profile_run["tokens_per_step"]
    top_kernels = [
        {
            "name": name,
            "count": kernel_counts[name],
            "total_ms": total_ns / NS_PER_MS,
            "kernel_time_share_percent": total_ns / kernel_total_ns * 100.0,
        }
        for name, total_ns in sorted(
            kernel_totals_ns.items(), key=lambda item: item[1], reverse=True
        )[:10]
    ]
    result = {
        "profiled_iterations": iterations,
        "profile_window_ms": (window_end - window_start) / NS_PER_MS,
        "kernel_count": len(kernels),
        "kernel_count_per_step": len(kernels) / iterations,
        "kernels_per_token": len(kernels) / iterations / tokens_per_step,
        "kernels_under_50_us_percent": under_50_us / len(kernels) * 100.0,
        "kernel_total_ms": kernel_total_ns / NS_PER_MS,
        "attention_time_share_percent": (
            category_totals_ns["attention"] / kernel_total_ns * 100.0
        ),
        "gemm_time_share_percent": (
            category_totals_ns["gemm_matmul"] / kernel_total_ns * 100.0
        ),
        "optimizer_time_share_percent": (
            category_totals_ns["optimizer"] / kernel_total_ns * 100.0
        ),
        "attention_time_per_step_ms": (
            category_totals_ns["attention"] / iterations / NS_PER_MS
        ),
        "gemm_time_per_step_ms": (
            category_totals_ns["gemm_matmul"] / iterations / NS_PER_MS
        ),
        "optimizer_time_per_step_ms": (
            category_totals_ns["optimizer"] / iterations / NS_PER_MS
        ),
        "top_10_cuda_kernels_by_total_gpu_time": top_kernels,
        "category_note": (
            "Categories are non-exclusive and use total CUDA kernel execution time "
            "as the denominator. Attention uses correlated NVTX ranges and demangled "
            "fused-attention symbols; optimizer uses correlated NVTX ranges; GEMM "
            "uses demangled kernel symbols."
        ),
        "trace": {
            "path": str(trace_path),
            "size_bytes": trace_path.stat().st_size,
            "sha256": sha256(trace_path),
            "sqlite_path": str(sqlite_path),
            "sqlite_size_bytes": sqlite_path.stat().st_size,
            "sqlite_sha256": sha256(sqlite_path),
        },
    }
    connection.close()
    return result


def compare_controls(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    for field in (
        "parameter_count",
        "model_config",
        "parallelism",
        "precision",
        "optimizer",
        "data",
        "sequence_length",
        "warmup_iterations",
    ):
        if candidate[field] != reference[field]:
            raise RuntimeError(f"Control field differs across micro-batches: {field}")

    excluded_environment = {"timestamp_utc", "hostname"}
    reference_environment = {
        key: value
        for key, value in reference["environment"].items()
        if key not in excluded_environment
    }
    candidate_environment = {
        key: value
        for key, value in candidate["environment"].items()
        if key not in excluded_environment
    }
    if candidate_environment != reference_environment:
        raise RuntimeError("Hardware or software environment changed across runs")


def linear_fit(points: list[tuple[float, float]], x: float) -> tuple[float, float, float]:
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    slope = sum((px - mean_x) * (py - mean_y) for px, py in points) / denominator
    intercept = mean_y - slope * mean_x
    return intercept + slope * x, intercept, slope


def main() -> None:
    args = parse_args()
    runs: dict[int, dict[str, Any]] = {}
    profiles: dict[int, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for micro_batch in args.micro_batches:
        run_path = args.results_dir / f"phase3_mb{micro_batch}_run.json"
        profile_run_path = args.results_dir / f"phase3_mb{micro_batch}_profile_run.json"
        sqlite_path = args.profiles_dir / f"phase3_mb{micro_batch}.sqlite"
        trace_path = args.profiles_dir / f"phase3_mb{micro_batch}.nsys-rep"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        profile_run = json.loads(profile_run_path.read_text(encoding="utf-8"))
        if reference is None:
            reference = run
        else:
            compare_controls(reference, run)
        compare_controls(run, profile_run)
        if profile_run["micro_batch_size"] != micro_batch:
            raise RuntimeError("Profile micro-batch does not match its filename")
        runs[micro_batch] = run
        profiles[micro_batch] = profile_summary(sqlite_path, trace_path, profile_run)

    baseline = runs[1]
    configurations: dict[str, Any] = {}
    for micro_batch in args.micro_batches:
        run = runs[micro_batch]
        profile = profiles[micro_batch]
        throughput_ratio = run["tokens_per_second"] / baseline["tokens_per_second"]
        batch_ratio = micro_batch / baseline["micro_batch_size"]
        configurations[str(micro_batch)] = {
            "micro_batch_size": micro_batch,
            "tokens_per_step": run["tokens_per_step"],
            "average_step_time_ms": run["average_step_time_ms"],
            "median_step_time_ms": run["median_step_time_ms"],
            "tokens_per_second": run["tokens_per_second"],
            "milliseconds_per_token": run["milliseconds_per_token"],
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
            "scaling_relative_to_mb1": {
                "batch_ratio": batch_ratio,
                "step_time_ratio": (
                    run["average_step_time_ms"] / baseline["average_step_time_ms"]
                ),
                "throughput_ratio": throughput_ratio,
                "throughput_gain_percent": (throughput_ratio - 1.0) * 100.0,
                "ideal_constant_step_scaling_efficiency_percent": (
                    throughput_ratio / batch_ratio * 100.0
                ),
                "mfu_ratio": run["mfu"]["mfu_percent"] / baseline["mfu"]["mfu_percent"],
            },
            "profile": profile,
        }

    fit_points = [
        (float(micro_batch), runs[micro_batch]["gpu_monitoring"]["peak_nvidia_smi_memory_mib"])
        for micro_batch in (1, 2, 4)
    ]
    predicted_mb8, memory_intercept, memory_slope = linear_fit(fit_points, 8.0)
    total_memory_mib = baseline["environment"]["gpu_memory_total_mib"]
    safety_limit_mib = total_memory_mib * 0.90
    mb8_attempted = 8 in args.micro_batches
    mb8_decision = {
        "method": "least-squares linear fit to MB1/MB2/MB4 peak nvidia-smi memory",
        "intercept_mib": memory_intercept,
        "slope_mib_per_micro_batch": memory_slope,
        "predicted_mb8_peak_mib": predicted_mb8,
        "gpu_total_memory_mib": total_memory_mib,
        "safety_fraction": 0.90,
        "safety_limit_mib": safety_limit_mib,
        "prediction_safe": predicted_mb8 <= safety_limit_mib,
        "mb8_attempted": mb8_attempted,
        "mb8_completed": mb8_attempted,
    }

    best_throughput = max(args.micro_batches, key=lambda mb: runs[mb]["tokens_per_second"])
    best_mfu = max(args.micro_batches, key=lambda mb: runs[mb]["mfu"]["mfu_percent"])
    result = {
        "status": "success",
        "experiment": "Phase 3.3 fused-attention micro-batch scaling",
        "infrastructure": {
            "pod_id": args.pod_id,
            "cloud": "Secure Cloud",
            "gpu": "1x NVIDIA A40 48GB",
            "gpu_price_per_hour_usd": 0.44,
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        },
        "controls": {
            "model_config": baseline["model_config"],
            "parameter_count": baseline["parameter_count"],
            "parallelism": baseline["parallelism"],
            "precision": baseline["precision"],
            "optimizer": baseline["optimizer"],
            "data": baseline["data"],
            "sequence_length": baseline["sequence_length"],
            "warmup_iterations": baseline["warmup_iterations"],
            "measured_iterations": baseline["measured_iterations"],
            "environment": baseline["environment"],
            "only_intended_variable": "micro_batch_size",
        },
        "scaling_efficiency_formula": (
            "efficiency = (throughput_MB / throughput_MB1) / (MB / 1) * 100; "
            "100% means the larger batch completed in the MB1 step time"
        ),
        "configurations": configurations,
        "mb8_decision": mb8_decision,
        "best_throughput_micro_batch": best_throughput,
        "best_mfu_micro_batch": best_mfu,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
