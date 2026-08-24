#!/usr/bin/env python3
"""Analyze Phase 5.2 timing runs and Nsight Systems BDA traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NS_PER_MS = 1_000_000.0
EXPECTED_BDA_SITES_PER_STEP = 48.0
EXPECTED_REMOVED_FORWARD_LAUNCHES_PER_STEP = 96.0
EVIDENCE_MINIMUM_REMOVED_LAUNCHES = 72.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-screen", type=Path, required=True)
    parser.add_argument("--fused-screen", type=Path, required=True)
    parser.add_argument("--baseline-profile-metrics", type=Path, required=True)
    parser.add_argument("--fused-profile-metrics", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--fused-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--fused-trace", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--stability-baseline", type=Path)
    parser.add_argument("--stability-fused", type=Path)
    parser.add_argument("--formal-baseline", type=Path)
    parser.add_argument("--formal-fused", type=Path)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for pair in (
        ("stability", args.stability_baseline, args.stability_fused),
        ("formal", args.formal_baseline, args.formal_fused),
    ):
        if (pair[1] is None) != (pair[2] is None):
            parser.error(f"{pair[0]} baseline and fused metrics must be supplied together")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
    window_start: int,
    window_end: int,
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


def nvtx_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    return row["text"] or strings.get(row["textId"], "")


def bda_names_by_correlation(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> tuple[dict[int, set[str]], dict[str, int]]:
    ranges_by_thread: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    range_counts: dict[str, int] = defaultdict(int)
    query = """
        SELECT start, end, globalTid, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
    """
    for row in connection.execute(query, (window_start, window_end)):
        name = nvtx_name(row, strings)
        if name.startswith("bda::"):
            ranges_by_thread[row["globalTid"]].append((row["start"], row["end"], name))
            range_counts[name] += 1

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
    return result, dict(range_counts)


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def gpu_intervals(
    connection: sqlite3.Connection,
    table: str,
    window_start: int,
    window_end: int,
) -> list[tuple[int, int]]:
    if not table_exists(connection, table):
        return []
    return [
        (row["start"], row["end"])
        for row in connection.execute(
            f"SELECT start, end FROM {table} WHERE end > ? AND start < ?",
            (window_start, window_end),
        )
    ]


def kernel_breakdown(
    totals_ns: dict[str, int],
    counts: dict[str, int],
    total_ns: int,
    limit: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "count": counts[name],
            "total_ms": value / NS_PER_MS,
            "average_us": value / counts[name] / 1_000.0,
            "kernel_time_share_percent": value / total_ns * 100.0 if total_ns else 0.0,
        }
        for name, value in sorted(
            totals_ns.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    ]


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
    window_ns = window_end - window_start

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
    bda_correlations, bda_range_counts = bda_names_by_correlation(
        connection,
        strings,
        window_start,
        window_end,
    )

    kernel_totals_ns: dict[str, int] = defaultdict(int)
    kernel_counts: dict[str, int] = defaultdict(int)
    bda_totals_ns: dict[str, int] = defaultdict(int)
    bda_counts: dict[str, int] = defaultdict(int)
    bda_by_site: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "total_ns": 0}
    )
    kernel_durations_ns: list[int] = []
    bda_kernel_durations_ns: list[int] = []
    for row in kernels:
        duration = row["end"] - row["start"]
        name = strings.get(row["demangledName"], f"StringId({row['demangledName']})")
        kernel_totals_ns[name] += duration
        kernel_counts[name] += 1
        kernel_durations_ns.append(duration)
        active_bda = bda_correlations.get(row["correlationId"], set())
        if active_bda:
            bda_totals_ns[name] += duration
            bda_counts[name] += 1
            bda_kernel_durations_ns.append(duration)
            for site in active_bda:
                bda_by_site[site]["count"] += 1
                bda_by_site[site]["total_ns"] += duration

    kernel_total_ns = sum(kernel_durations_ns)
    bda_total_ns = sum(bda_kernel_durations_ns)
    iterations = run_metrics["measured_iterations"]
    all_gpu_intervals = [
        (row["start"], row["end"]) for row in kernels
    ]
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        all_gpu_intervals.extend(
            gpu_intervals(connection, table, window_start, window_end)
        )
    merged, idle_gaps_ns = merge_intervals(
        all_gpu_intervals,
        window_start,
        window_end,
    )
    active_ns = sum(end - start for start, end in merged)
    idle_ns = sum(idle_gaps_ns)
    under_50_us = [
        duration for duration in kernel_durations_ns if duration < 50_000
    ]
    total_bda_ranges = sum(bda_range_counts.values())

    result = {
        "profile_window_ms": window_ns / NS_PER_MS,
        "profiled_iterations": iterations,
        "kernel_count": len(kernels),
        "kernel_count_per_step": len(kernels) / iterations,
        "kernel_total_ms": kernel_total_ns / NS_PER_MS,
        "kernel_time_per_step_ms": kernel_total_ns / iterations / NS_PER_MS,
        "kernels_under_50_us": {
            "count": len(under_50_us),
            "count_per_step": len(under_50_us) / iterations,
            "count_share_percent": (
                len(under_50_us) / len(kernels) * 100.0 if kernels else 0.0
            ),
            "total_ms": sum(under_50_us) / NS_PER_MS,
            "time_per_step_ms": sum(under_50_us) / iterations / NS_PER_MS,
        },
        "gpu_timeline": {
            "active_union_ms": active_ns / NS_PER_MS,
            "idle_ms": idle_ns / NS_PER_MS,
            "idle_ms_per_step": idle_ns / iterations / NS_PER_MS,
            "idle_percent": idle_ns / window_ns * 100.0,
            "idle_gap_count": len(idle_gaps_ns),
        },
        "bda_forward": {
            "attribution": (
                "CUDA kernels whose launch correlation ID occurs inside manual "
                "bda::self_attention or bda::mlp NVTX ranges; forward only"
            ),
            "range_count": total_bda_ranges,
            "ranges_per_step": total_bda_ranges / iterations,
            "expected_ranges_per_step": EXPECTED_BDA_SITES_PER_STEP,
            "range_counts": bda_range_counts,
            "kernel_count": len(bda_kernel_durations_ns),
            "kernel_count_per_step": len(bda_kernel_durations_ns) / iterations,
            "gpu_time_ms": bda_total_ns / NS_PER_MS,
            "gpu_time_per_step_ms": bda_total_ns / iterations / NS_PER_MS,
            "by_site": {
                site: {
                    "kernel_count": values["count"],
                    "kernel_count_per_step": values["count"] / iterations,
                    "gpu_time_ms": values["total_ns"] / NS_PER_MS,
                    "gpu_time_per_step_ms": values["total_ns"]
                    / iterations
                    / NS_PER_MS,
                }
                for site, values in sorted(bda_by_site.items())
            },
            "top_kernel_families": kernel_breakdown(
                bda_totals_ns,
                bda_counts,
                bda_total_ns,
                15,
            ),
        },
        "top_15_cuda_kernels_by_total_gpu_time": kernel_breakdown(
            kernel_totals_ns,
            kernel_counts,
            kernel_total_ns,
            15,
        ),
        "trace": {
            "path": str(trace_path),
            "size_bytes": trace_path.stat().st_size,
            "sha256": sha256(trace_path),
            "sqlite_path": str(sqlite_path),
            "sqlite_size_bytes": sqlite_path.stat().st_size,
            "sqlite_sha256": sha256(sqlite_path),
            "preserved_on_pod": True,
        },
        "measurement_note": (
            "Kernel categories are not inferred from the old overlapping "
            "activation/elementwise subtotal. BDA forward attribution comes from "
            "explicit identical NVTX instrumentation in both variants."
        ),
    }
    connection.close()
    return result


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def environment_without_run_identity(environment: dict[str, Any]) -> dict[str, Any]:
    excluded = {"timestamp_utc"}
    return {key: value for key, value in environment.items() if key not in excluded}


def compare_controls(baseline: dict[str, Any], fused: dict[str, Any]) -> None:
    baseline_model = {
        key: value
        for key, value in baseline["model_config"].items()
        if key != "bias_dropout_fusion"
    }
    fused_model = {
        key: value
        for key, value in fused["model_config"].items()
        if key != "bias_dropout_fusion"
    }
    if baseline_model != fused_model:
        raise RuntimeError("A/B model configurations differ beyond BDA fusion")
    if baseline["model_config"]["bias_dropout_fusion"] is not False:
        raise RuntimeError("A does not have bias_dropout_fusion=False")
    if fused["model_config"]["bias_dropout_fusion"] is not True:
        raise RuntimeError("B does not have bias_dropout_fusion=True")

    for field in (
        "parameter_count",
        "parallelism",
        "precision",
        "optimizer",
        "data",
        "micro_batch_size",
        "global_batch_size",
        "sequence_length",
        "tokens_per_step",
        "warmup_iterations",
        "measured_iterations",
    ):
        if baseline[field] != fused[field]:
            raise RuntimeError(f"A/B control field differs: {field}")
    if environment_without_run_identity(
        baseline["environment"]
    ) != environment_without_run_identity(fused["environment"]):
        raise RuntimeError("A/B software or hardware environments differ")


def percent_change(baseline: float, fused: float) -> float:
    return (fused - baseline) / baseline * 100.0


def improvement_when_lower(baseline: float, fused: float) -> float:
    return (baseline - fused) / baseline * 100.0


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    monitoring = run["gpu_monitoring"]
    return {
        "variant": run["variant"],
        "bias_dropout_fusion": run["bias_dropout_fusion"],
        "average_step_time_ms": run["average_step_time_ms"],
        "median_step_time_ms": run["median_step_time_ms"],
        "step_time_standard_deviation_ms": run["step_time_standard_deviation_ms"],
        "step_times_ms": run["step_times_ms"],
        "tokens_per_second": run["tokens_per_second"],
        "mfu_percent": run["mfu"]["mfu_percent"],
        "peak_allocated_memory_mib": run["peak_allocated_memory_mib"],
        "peak_reserved_memory_mib": run["peak_reserved_memory_mib"],
        "peak_nvidia_smi_memory_mib": monitoring["peak_nvidia_smi_memory_mib"],
        "average_gpu_utilization_percent": monitoring[
            "average_gpu_utilization_percent"
        ],
        "final_loss": run["final_loss"],
        "losses_finite": run["losses_finite"],
        "run_label": run["run_label"],
        "environment": run["environment"],
    }


def timing_delta(
    baseline: dict[str, Any],
    fused: dict[str, Any],
) -> dict[str, float]:
    baseline_step = baseline["average_step_time_ms"]
    fused_step = fused["average_step_time_ms"]
    return {
        "average_step_time_ms": fused_step - baseline_step,
        "median_step_time_ms": (
            fused["median_step_time_ms"] - baseline["median_step_time_ms"]
        ),
        "step_time_improvement_percent": improvement_when_lower(
            baseline_step,
            fused_step,
        ),
        "speedup_ratio": baseline_step / fused_step,
        "tokens_per_second": (
            fused["tokens_per_second"] - baseline["tokens_per_second"]
        ),
        "tokens_per_second_percent": percent_change(
            baseline["tokens_per_second"],
            fused["tokens_per_second"],
        ),
        "mfu_percentage_points": (
            fused["mfu"]["mfu_percent"] - baseline["mfu"]["mfu_percent"]
        ),
        "mfu_percent": percent_change(
            baseline["mfu"]["mfu_percent"],
            fused["mfu"]["mfu_percent"],
        ),
        "peak_allocated_memory_mib": (
            fused["peak_allocated_memory_mib"]
            - baseline["peak_allocated_memory_mib"]
        ),
        "peak_reserved_memory_mib": (
            fused["peak_reserved_memory_mib"] - baseline["peak_reserved_memory_mib"]
        ),
        "peak_nvidia_smi_memory_mib": (
            fused["gpu_monitoring"]["peak_nvidia_smi_memory_mib"]
            - baseline["gpu_monitoring"]["peak_nvidia_smi_memory_mib"]
        ),
    }


def profile_delta(
    baseline: dict[str, Any],
    fused: dict[str, Any],
) -> dict[str, Any]:
    baseline_bda = baseline["bda_forward"]
    fused_bda = fused["bda_forward"]
    baseline_small = baseline["kernels_under_50_us"]
    fused_small = fused["kernels_under_50_us"]
    baseline_idle = baseline["gpu_timeline"]
    fused_idle = fused["gpu_timeline"]
    bda_removed = (
        baseline_bda["kernel_count_per_step"] - fused_bda["kernel_count_per_step"]
    )
    total_removed = baseline["kernel_count_per_step"] - fused["kernel_count_per_step"]
    bda_ranges_valid = (
        baseline_bda["ranges_per_step"] == EXPECTED_BDA_SITES_PER_STEP
        and fused_bda["ranges_per_step"] == EXPECTED_BDA_SITES_PER_STEP
    )
    evidence_confirmed = (
        bda_ranges_valid
        and bda_removed >= EVIDENCE_MINIMUM_REMOVED_LAUNCHES
        and total_removed >= EVIDENCE_MINIMUM_REMOVED_LAUNCHES
        and fused_bda["gpu_time_per_step_ms"]
        < baseline_bda["gpu_time_per_step_ms"]
    )
    return {
        "kernel_count_per_step": (
            fused["kernel_count_per_step"] - baseline["kernel_count_per_step"]
        ),
        "kernel_launches_removed_per_step": total_removed,
        "kernel_count_improvement_percent": improvement_when_lower(
            baseline["kernel_count_per_step"],
            fused["kernel_count_per_step"],
        ),
        "bda_forward_launches_removed_per_step": bda_removed,
        "bda_forward_gpu_time_per_step_ms": (
            fused_bda["gpu_time_per_step_ms"]
            - baseline_bda["gpu_time_per_step_ms"]
        ),
        "bda_forward_gpu_time_reduction_per_step_ms": (
            baseline_bda["gpu_time_per_step_ms"]
            - fused_bda["gpu_time_per_step_ms"]
        ),
        "bda_forward_gpu_time_improvement_percent": improvement_when_lower(
            baseline_bda["gpu_time_per_step_ms"],
            fused_bda["gpu_time_per_step_ms"],
        ),
        "kernels_under_50_us_count_per_step": (
            fused_small["count_per_step"] - baseline_small["count_per_step"]
        ),
        "kernels_under_50_us_removed_per_step": (
            baseline_small["count_per_step"] - fused_small["count_per_step"]
        ),
        "kernels_under_50_us_percentage_points": (
            fused_small["count_share_percent"]
            - baseline_small["count_share_percent"]
        ),
        "gpu_idle_ms_per_step": (
            fused_idle["idle_ms_per_step"] - baseline_idle["idle_ms_per_step"]
        ),
        "gpu_idle_percent_percentage_points": (
            fused_idle["idle_percent"] - baseline_idle["idle_percent"]
        ),
        "expected_removed_forward_launches_per_step": (
            EXPECTED_REMOVED_FORWARD_LAUNCHES_PER_STEP
        ),
        "bda_ranges_valid": bda_ranges_valid,
        "fusion_evidence_confirmed": evidence_confirmed,
        "evidence_criterion": (
            "Both traces contain exactly 48 BDA forward ranges/step; B removes at "
            "least 72 BDA-attributed and total kernels/step; BDA forward GPU time falls."
        ),
    }


def decide_validation(
    correctness: dict[str, Any],
    screen_delta: dict[str, float],
    profiler_delta: dict[str, Any],
    stability_delta: dict[str, float] | None,
) -> dict[str, Any]:
    throughput_gain = screen_delta["tokens_per_second_percent"]
    evidence = profiler_delta["fusion_evidence_confirmed"]
    if not correctness.get("passed", False):
        return {
            "formal_validation_required": False,
            "stability_repeat_required": False,
            "reason": "Correctness failed; performance validation is not allowed.",
        }
    if throughput_gain < 1.0:
        return {
            "formal_validation_required": False,
            "stability_repeat_required": False,
            "reason": "Screen throughput gain is below 1%; long validation is skipped.",
        }
    if throughput_gain >= 2.0:
        return {
            "formal_validation_required": bool(evidence),
            "stability_repeat_required": False,
            "reason": (
                "Screen gain is at least 2% and profiler evidence confirms BDA fusion."
                if evidence
                else "Screen gain is at least 2%, but profiler evidence does not confirm fusion."
            ),
        }
    if stability_delta is None:
        return {
            "formal_validation_required": False,
            "stability_repeat_required": True,
            "reason": (
                "Screen gain is between 1% and 2%; one reversed-order 3+15 repeat "
                "is required before deciding."
            ),
        }

    repeat_gain = stability_delta["tokens_per_second_percent"]
    stable = (
        repeat_gain >= 0.5
        and abs(repeat_gain - throughput_gain) <= 1.0
        and (repeat_gain + throughput_gain) / 2.0 >= 1.0
    )
    return {
        "formal_validation_required": bool(evidence and stable),
        "stability_repeat_required": False,
        "stability_confirmed": stable,
        "reason": (
            "The 1-2% screen is supported by fusion evidence and a stable repeat."
            if evidence and stable
            else "The 1-2% screen lacks stable repeat and/or fusion evidence."
        ),
        "stability_criterion": (
            "Repeat gain >=0.5%, gain spread <=1.0 percentage point, and mean gain >=1%."
        ),
    }


def main() -> None:
    args = parse_args()
    baseline_screen = load(args.baseline_screen)
    fused_screen = load(args.fused_screen)
    baseline_profile_run = load(args.baseline_profile_metrics)
    fused_profile_run = load(args.fused_profile_metrics)
    correctness = load(args.correctness)
    compare_controls(baseline_screen, fused_screen)
    compare_controls(baseline_profile_run, fused_profile_run)

    baseline_profile = analyze_trace(
        args.baseline_sqlite,
        args.baseline_trace,
        baseline_profile_run,
    )
    fused_profile = analyze_trace(
        args.fused_sqlite,
        args.fused_trace,
        fused_profile_run,
    )
    screen_delta = timing_delta(baseline_screen, fused_screen)
    profiler_delta = profile_delta(baseline_profile, fused_profile)

    stability: dict[str, Any] | None = None
    stability_delta: dict[str, float] | None = None
    if args.stability_baseline is not None and args.stability_fused is not None:
        stability_baseline = load(args.stability_baseline)
        stability_fused = load(args.stability_fused)
        compare_controls(stability_baseline, stability_fused)
        stability_delta = timing_delta(stability_baseline, stability_fused)
        stability = {
            "baseline": run_summary(stability_baseline),
            "fused": run_summary(stability_fused),
            "delta": stability_delta,
            "run_order": ["B", "A"],
        }

    decision = decide_validation(
        correctness,
        screen_delta,
        profiler_delta,
        stability_delta,
    )

    formal_validation: dict[str, Any]
    if args.formal_baseline is not None and args.formal_fused is not None:
        formal_baseline = load(args.formal_baseline)
        formal_fused = load(args.formal_fused)
        compare_controls(formal_baseline, formal_fused)
        formal_validation = {
            "run": True,
            "baseline": run_summary(formal_baseline),
            "fused": run_summary(formal_fused),
            "delta": timing_delta(formal_baseline, formal_fused),
            "warmup_iterations": formal_baseline["warmup_iterations"],
            "measured_iterations": formal_baseline["measured_iterations"],
        }
    else:
        formal_validation = {
            "run": False,
            "reason": decision["reason"],
        }

    result = {
        "status": "success",
        "experiment": "Phase 5.2 Bias-Dropout-Add fusion A/B",
        "correctness": correctness,
        "infrastructure": {
            "pod_id": args.pod_id,
            "cloud": "Secure Cloud",
            "gpu": "1x NVIDIA A40 48GB",
            "gpu_price_per_hour_usd": 0.44,
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "trace_preserved_on_stopped_pod": True,
        },
        "controls": {
            "only_intended_variable": "bias_dropout_fusion",
            "A": {"bias_dropout_fusion": False},
            "B": {"bias_dropout_fusion": True},
            "model_config_except_intended_variable": {
                key: value
                for key, value in baseline_screen["model_config"].items()
                if key != "bias_dropout_fusion"
            },
            "parameter_count": baseline_screen["parameter_count"],
            "parallelism": baseline_screen["parallelism"],
            "precision": baseline_screen["precision"],
            "optimizer": baseline_screen["optimizer"],
            "data": baseline_screen["data"],
            "environment": baseline_screen["environment"],
            "explicitly_disabled": [
                "bias_activation_fusion",
                "masked_softmax_fusion",
                "cross_entropy_loss_fusion",
                "CUDA Graph",
                "optimizer fusion",
                "dtype changes",
            ],
        },
        "fast_screening": {
            "warmup_iterations": baseline_screen["warmup_iterations"],
            "measured_iterations": baseline_screen["measured_iterations"],
            "run_order": ["A", "B"],
            "baseline": run_summary(baseline_screen),
            "fused": run_summary(fused_screen),
            "delta": screen_delta,
        },
        "nsight_systems": {
            "profile_run_warmup_iterations": baseline_profile_run[
                "warmup_iterations"
            ],
            "profiled_iterations": baseline_profile_run["measured_iterations"],
            "baseline": baseline_profile,
            "fused": fused_profile,
            "delta": profiler_delta,
            "scope_note": (
                "BDA-attributed counts and time are forward-only. Overall kernel and "
                "GPU-idle deltas include the full forward, backward, and optimizer step."
            ),
        },
        "stability_repeat": stability,
        "decision": decision,
        "formal_validation": formal_validation,
        "accounting_note": (
            "The Phase 3.4 295.82 ms activation/elementwise category is not used as "
            "pure elementwise time or as the basis of this result because it has "
            "known GEMM and optimizer overlap."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
