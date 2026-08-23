#!/usr/bin/env python3
"""Summarize a Phase 2 Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NS_PER_MS = 1_000_000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--nvtx-projection-csv", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def duration_summary(durations_ns: list[int]) -> dict[str, float | int]:
    values_ms = [duration / NS_PER_MS for duration in durations_ns]
    return {
        "count": len(values_ms),
        "total_ms": sum(values_ms),
        "average_ms": statistics.fmean(values_ms) if values_ms else 0.0,
        "median_ms": statistics.median(values_ms) if values_ms else 0.0,
        "p95_ms": percentile(values_ms, 0.95),
        "maximum_ms": max(values_ms, default=0.0),
    }


def merge_intervals(
    intervals: Iterable[tuple[int, int]], window_start: int, window_end: int
) -> tuple[list[tuple[int, int]], list[int]]:
    clipped = sorted(
        (max(start, window_start), min(end, window_end))
        for start, end in intervals
        if end > window_start and start < window_end
    )
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    gaps: list[int] = []
    cursor = window_start
    for start, end in merged:
        if start > cursor:
            gaps.append(start - cursor)
        cursor = max(cursor, end)
    if cursor < window_end:
        gaps.append(window_end - cursor)
    return merged, gaps


def load_nvtx_projections(path: Path) -> dict[str, dict[str, float | int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Range,"))
    reader = csv.DictReader(lines[header_index:])
    selected: dict[str, dict[str, float | int]] = {}
    for row in reader:
        name = row["Range"].lstrip(":")
        if name in {"profile_window", "forward", "optimizer_step"} or name.startswith(
            "train_step_"
        ):
            selected[name] = {
                "projected_time_ns": int(row["Total Proj Time (ns)"]),
                "range_time_ns": int(row["Total Range Time (ns)"]),
                "gpu_operations": int(row["Total GPU Ops"]),
            }
    return selected


def nvtx_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    return row["text"] or strings.get(row["textId"], "")


def relevant_nvtx_names_by_correlation(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> dict[int, set[str]]:
    keywords = (
        "optimizer_step",
        "attention",
        "bmm",
        "softmax",
        "masked_fill",
        "masked_scale",
        "layer_norm",
        "layernorm",
        "native_layer_norm",
    )
    ranges_by_thread: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    query = """
        SELECT start, end, globalTid, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
    """
    for row in connection.execute(query, (window_start, window_end)):
        name = nvtx_name(row, strings)
        if any(keyword in name.lower() for keyword in keywords):
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


def category_matches(
    kernel_name: str, active_nvtx_names: set[str]
) -> dict[str, bool]:
    name = kernel_name.lower()
    ranges = " ".join(active_nvtx_names).lower()
    return {
        "gemm_matmul": "gemm" in name,
        "attention": any(
            token in name or token in ranges
            for token in (
                "attention",
                "bmm",
                "softmax",
                "masked_fill",
                "masked_scale",
                "sdpa",
                "fused_attn",
            )
        ),
        "normalization": any(
            token in name or token in ranges
            for token in ("layer_norm", "layernorm", "native_layer_norm")
        ),
        "optimizer": "optimizer_step" in ranges,
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    run_metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    projections = load_nvtx_projections(args.nvtx_projection_csv)

    exact_match_fields = (
        "model_config",
        "parallelism",
        "precision",
        "optimizer",
        "micro_batch_size",
        "global_batch_size",
        "sequence_length",
        "warmup_iterations",
    )
    for field in exact_match_fields:
        if run_metrics[field] != baseline[field]:
            raise RuntimeError(f"Phase 2 profile changed baseline field: {field}")

    connection = sqlite3.connect(args.sqlite)
    connection.row_factory = sqlite3.Row
    strings = {
        row["id"]: row["value"] for row in connection.execute("SELECT id, value FROM StringIds")
    }

    window_row = connection.execute(
        """
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE (text = 'profile_window' OR textId IN (
            SELECT id FROM StringIds WHERE value = 'profile_window'
        )) AND end IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    if window_row is None:
        raise RuntimeError("profile_window NVTX range not found")
    window_start, window_end = window_row["start"], window_row["end"]
    window_ns = window_end - window_start

    kernel_rows = list(
        connection.execute(
            """
            SELECT start, end, correlationId, demangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )
    memcpy_rows = list(
        connection.execute(
            """
            SELECT m.start, m.end, m.correlationId, m.bytes, e.label AS copyLabel
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
            LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS e ON e.id = m.copyKind
            WHERE m.end > ? AND m.start < ?
            """,
            (window_start, window_end),
        )
    )
    memset_rows = list(
        connection.execute(
            """
            SELECT start, end, correlationId, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMSET
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )

    all_gpu_intervals = [
        (row["start"], row["end"]) for row in kernel_rows + memcpy_rows + memset_rows
    ]
    merged, idle_gaps_ns = merge_intervals(all_gpu_intervals, window_start, window_end)
    active_ns = sum(end - start for start, end in merged)

    correlation_nvtx = relevant_nvtx_names_by_correlation(
        connection, strings, window_start, window_end
    )
    kernel_totals_ns: dict[str, int] = defaultdict(int)
    kernel_counts: dict[str, int] = defaultdict(int)
    category_totals_ns: dict[str, int] = defaultdict(int)
    kernel_durations_ns: list[int] = []
    for row in kernel_rows:
        duration = row["end"] - row["start"]
        name = strings.get(row["demangledName"], f"StringId({row['demangledName']})")
        kernel_totals_ns[name] += duration
        kernel_counts[name] += 1
        kernel_durations_ns.append(duration)
        matches = category_matches(name, correlation_nvtx.get(row["correlationId"], set()))
        for category, matched in matches.items():
            if matched:
                category_totals_ns[category] += duration

    kernel_total_ns = sum(kernel_durations_ns)
    top_kernels = []
    for name, total_ns in sorted(kernel_totals_ns.items(), key=lambda item: item[1], reverse=True)[:15]:
        top_kernels.append(
            {
                "name": name,
                "count": kernel_counts[name],
                "total_ms": total_ns / NS_PER_MS,
                "average_us": total_ns / kernel_counts[name] / 1_000.0,
                "kernel_time_share_percent": total_ns / kernel_total_ns * 100.0,
            }
        )

    categories = {
        name: {
            "total_ms": total_ns / NS_PER_MS,
            "kernel_time_share_percent": total_ns / kernel_total_ns * 100.0,
        }
        for name, total_ns in category_totals_ns.items()
    }

    small_kernels = {}
    for threshold_us in (10, 50, 100):
        selected = [duration for duration in kernel_durations_ns if duration < threshold_us * 1_000]
        small_kernels[f"under_{threshold_us}_us"] = {
            "count": len(selected),
            "count_share_percent": len(selected) / len(kernel_durations_ns) * 100.0,
            "total_ms": sum(selected) / NS_PER_MS,
            "kernel_time_share_percent": sum(selected) / kernel_total_ns * 100.0,
        }

    memcpy_by_kind: dict[str, dict[str, float | int]] = {}
    for kind in sorted({row["copyLabel"] or "Unknown" for row in memcpy_rows}):
        selected = [row for row in memcpy_rows if (row["copyLabel"] or "Unknown") == kind]
        memcpy_by_kind[kind] = {
            "count": len(selected),
            "total_ms": sum(row["end"] - row["start"] for row in selected) / NS_PER_MS,
            "bytes": sum(row["bytes"] for row in selected),
        }
    memcpy_total_ns = sum(row["end"] - row["start"] for row in memcpy_rows)

    runtime_rows = list(
        connection.execute(
            """
            SELECT start, end, globalTid, nameId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )
    api_durations: dict[str, list[int]] = defaultdict(list)
    launches_by_thread: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in runtime_rows:
        name = strings.get(row["nameId"], f"StringId({row['nameId']})")
        api_durations[name].append(row["end"] - row["start"])
        if "launchkernel" in name.lower():
            launches_by_thread[row["globalTid"]].append((row["start"], row["end"]))

    launch_gaps_ns: list[int] = []
    for launches in launches_by_thread.values():
        launches.sort()
        for previous, current in zip(launches, launches[1:]):
            launch_gaps_ns.append(max(0, current[0] - previous[1]))
    launch_api_ns = [
        duration
        for name, durations in api_durations.items()
        if "launchkernel" in name.lower()
        for duration in durations
    ]

    sync_apis = {
        name: duration_summary(durations)
        for name, durations in sorted(api_durations.items())
        if "synchronize" in name.lower()
    }
    top_cuda_apis = [
        {"name": name, **duration_summary(durations)}
        for name, durations in sorted(
            api_durations.items(), key=lambda item: sum(item[1]), reverse=True
        )[:10]
    ]

    step_projections = [
        value for name, value in projections.items() if name.startswith("train_step_")
    ]
    step_projected_ns = sum(int(value["projected_time_ns"]) for value in step_projections)
    profile_projection = projections["profile_window"]
    forward_projection = projections["forward"]
    optimizer_projection = projections["optimizer_step"]

    baseline_mfu = baseline["mfu"]
    expected_flops = (
        72 * 1 * 2048 * 24 * 1024**2
        + 6 * 1 * 24 * 2048**2 * 1024
        + 6 * 1 * 2048 * 1024 * 50304
    )
    if expected_flops != baseline_mfu["training_flops_per_iteration"]:
        raise RuntimeError("Phase 1.2 MFU FLOP count does not match the validated formula")

    summary: dict[str, Any] = {
        "status": "success",
        "experiment": "Phase 2.1 Nsight Systems baseline profile",
        "baseline_source_commit": baseline["environment"]["project_commit"],
        "profile_source_commit": run_metrics["environment"]["project_commit"],
        "infrastructure": {
            "pod_id": "nrk1bdpgmo1ej3",
            "cloud": "Secure Cloud",
            "gpu": "1x NVIDIA A40 48GB",
            "gpu_price_per_hour_usd": 0.44,
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "trace_preserved_on_pod": True,
        },
        "configuration": {
            "parameter_count": run_metrics["parameter_count"],
            "model": run_metrics["model_config"],
            "parallelism": run_metrics["parallelism"],
            "precision": run_metrics["precision"],
            "optimizer": run_metrics["optimizer"],
            "micro_batch_size": run_metrics["micro_batch_size"],
            "global_batch_size": run_metrics["global_batch_size"],
            "sequence_length": run_metrics["sequence_length"],
            "warmup_iterations": run_metrics["warmup_iterations"],
            "profiled_iterations": run_metrics["profiled_iterations"],
        },
        "mfu_validation": {
            "formula": "F_iter = 72*B*S*L*H^2 + 6*B*L*S^2*H + 6*B*S*H*V",
            "assumptions": [
                "dense GPT training FLOPs with forward plus backward",
                "causal attention uses half of the full S-by-S attention matrix",
                "embedding lookup, normalization, activation, optimizer, and other elementwise FLOPs are excluded",
                "A40 dense BF16 Tensor Core peak is 149.7 TFLOP/s; sparsity peak is not used",
            ],
            "training_flops_per_iteration": expected_flops,
            "baseline_average_step_time_ms": baseline["average_step_time_ms"],
            "baseline_achieved_tflops": baseline_mfu["achieved_tflops"],
            "a40_dense_bf16_peak_tflops": baseline_mfu["gpu_dense_bf16_peak_tflops"],
            "baseline_mfu_percent": baseline_mfu["mfu_percent"],
            "profiled_timing_mfu_percent": run_metrics["mfu"][
                "mfu_percent_from_profiled_timing"
            ],
        },
        "profile": {
            "exact_command": (
                "PYTHONPATH=/workspace/Megatron-LM TRANSFORMER_ENGINE_DISABLE=1 "
                "CUDA_DEVICE_MAX_CONNECTIONS=1 "
                "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile "
                "--trace=cuda,nvtx,osrt,cublas,cudnn --sample=process-tree "
                "--cpuctxsw=process-tree --capture-range=cudaProfilerApi "
                "--capture-range-end=stop --cuda-memory-usage=true --force-overwrite=true "
                "--output=profiles/phase2_nsys_baseline .venv/bin/python -m "
                "torch.distributed.run --standalone --nproc_per_node=1 "
                "scripts/phase2_nsys_profile.py --warmup-iterations 20 "
                "--profiled-iterations 15 "
                "--output-json=results/phase2_nsys_run_metrics.json"
            ),
            "trace_path": str(args.trace),
            "trace_size_bytes": args.trace.stat().st_size,
            "trace_sha256": sha256(args.trace),
            "sqlite_path": str(args.sqlite),
            "sqlite_size_bytes": args.sqlite.stat().st_size,
            "sqlite_sha256": sha256(args.sqlite),
            "nsight_systems_version": "2025.1.1.0",
            "capture": ["CUDA kernels", "CUDA API", "NVTX", "OS runtime", "cuBLAS", "cuDNN"],
            "profile_window_ms": window_ns / NS_PER_MS,
            "profile_window_gpu_projection_ms": int(profile_projection["projected_time_ns"])
            / NS_PER_MS,
            "total_profiled_step_projected_time_ms": step_projected_ns / NS_PER_MS,
            "average_profiled_step_projected_time_ms": step_projected_ns
            / len(step_projections)
            / NS_PER_MS,
            "instrumented_average_step_time_ms": run_metrics["average_profiled_step_time_ms"],
            "instrumented_median_step_time_ms": run_metrics["median_profiled_step_time_ms"],
            "final_loss": run_metrics["final_loss"],
            "nvtx_projected_gpu_operations": int(profile_projection["gpu_operations"]),
            "nvtx_projected_gpu_operations_per_step": int(profile_projection["gpu_operations"])
            / run_metrics["profiled_iterations"],
            "cuda_gpu_activity_count": len(kernel_rows) + len(memcpy_rows) + len(memset_rows),
            "cuda_gpu_activity_count_per_step": (
                len(kernel_rows) + len(memcpy_rows) + len(memset_rows)
            )
            / run_metrics["profiled_iterations"],
        },
        "gpu_timeline": {
            "active_union_ms": active_ns / NS_PER_MS,
            "idle_ms": sum(idle_gaps_ns) / NS_PER_MS,
            "idle_percent": sum(idle_gaps_ns) / window_ns * 100.0,
            "idle_gap_summary": duration_summary(idle_gaps_ns),
            "kernel_count": len(kernel_rows),
            "kernel_count_per_step": len(kernel_rows) / run_metrics["profiled_iterations"],
            "kernel_total_ms": kernel_total_ns / NS_PER_MS,
            "memcpy_count": len(memcpy_rows),
            "memset_count": len(memset_rows),
        },
        "top_15_cuda_kernels_by_total_gpu_time": top_kernels,
        "kernel_categories": {
            **categories,
            "note": "Categories are non-exclusive and use total CUDA kernel execution time as denominator.",
        },
        "optimizer_gpu_projection": {
            "projected_time_ms": int(optimizer_projection["projected_time_ns"]) / NS_PER_MS,
            "profiled_step_time_share_percent": int(optimizer_projection["projected_time_ns"])
            / step_projected_ns
            * 100.0,
            "gpu_operations": int(optimizer_projection["gpu_operations"]),
        },
        "forward_gpu_projection": {
            "projected_time_ms": int(forward_projection["projected_time_ns"]) / NS_PER_MS,
            "profiled_step_time_share_percent": int(forward_projection["projected_time_ns"])
            / step_projected_ns
            * 100.0,
            "gpu_operations": int(forward_projection["gpu_operations"]),
        },
        "memory_copies": {
            "total_ms": memcpy_total_ns / NS_PER_MS,
            "profile_window_time_share_percent": memcpy_total_ns / window_ns * 100.0,
            "by_kind": memcpy_by_kind,
        },
        "small_kernels": {
            **small_kernels,
            "many_small_kernels_present": small_kernels["under_50_us"][
                "count_share_percent"
            ]
            > 50.0,
        },
        "cpu_launch": {
            "kernel_launch_api": duration_summary(launch_api_ns),
            "inter_launch_host_gap": duration_summary(launch_gaps_ns),
            "gpu_visible_idle_gap": duration_summary(idle_gaps_ns),
            "note": (
                "Inter-launch gaps are aggregated independently per CPU thread and can overlap; "
                "use their median and p95 rather than their total as a wall-time measure."
            ),
        },
        "top_10_cuda_apis_by_total_cpu_time": top_cuda_apis,
        "cuda_api_note": (
            "CUDA API time is asynchronous host-side API duration and is not additive with GPU wall time."
        ),
        "synchronization": {
            "cuda_synchronization_apis": sync_apis,
            "obvious_long_gpu_starvation": sum(idle_gaps_ns) / window_ns > 0.05,
            "note": (
                "Explicit synchronizations delimit measured steps and copy each loss to the host; "
                "interpret their counts as benchmark instrumentation, not an optimization result."
            ),
        },
        "environment": run_metrics["environment"],
    }
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
