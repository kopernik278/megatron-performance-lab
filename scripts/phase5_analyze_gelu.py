#!/usr/bin/env python3
"""Analyze Phase 5.3 timing runs and Nsight Systems GELU traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from phase5_analyze_bda import (
    NS_PER_MS,
    environment_without_run_identity,
    gpu_intervals,
    kernel_breakdown,
    load,
    merge_intervals,
    sha256,
    timing_delta,
)


EXPECTED_ACTIVATION_SITES_PER_STEP = 24.0
EXPECTED_REMOVED_FORWARD_LAUNCHES_PER_STEP = 24.0
EVIDENCE_MINIMUM_REMOVED_LAUNCHES = 18.0


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
    for label, baseline, fused in (
        ("stability", args.stability_baseline, args.stability_fused),
        ("formal", args.formal_baseline, args.formal_fused),
    ):
        if (baseline is None) != (fused is None):
            parser.error(f"{label} baseline and fused metrics must be supplied together")
    return args


def nvtx_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    return row["text"] or strings.get(row["textId"], "")


def correlations_in_ranges(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
    predicate: Callable[[str], bool],
) -> tuple[set[int], dict[str, int]]:
    ranges_by_thread: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for row in connection.execute(
        """
        SELECT start, end, globalTid, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
        """,
        (window_start, window_end),
    ):
        name = nvtx_name(row, strings)
        if predicate(name):
            ranges_by_thread[row["globalTid"]].append((row["start"], row["end"], name))
            counts[name] += 1

    correlations: set[int] = set()
    for row in connection.execute(
        """
        SELECT start, globalTid, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start >= ? AND start < ? AND correlationId IS NOT NULL
        """,
        (window_start, window_end),
    ):
        if any(
            start <= row["start"] < end
            for start, end, _ in ranges_by_thread.get(row["globalTid"], [])
        ):
            correlations.add(row["correlationId"])
    return correlations, dict(counts)


def is_activation_range(name: str) -> bool:
    return name.endswith(".forward.activation")


def is_fused_backward_range(name: str) -> bool:
    return name == "gelu::backward_fused"


def is_eager_gelu_backward_kernel(name: str) -> bool:
    normalized = name.lower().replace("_", "")
    return "gelubackward" in normalized


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
    activation_correlations, activation_ranges = correlations_in_ranges(
        connection,
        strings,
        window_start,
        window_end,
        is_activation_range,
    )
    fused_backward_correlations, fused_backward_ranges = correlations_in_ranges(
        connection,
        strings,
        window_start,
        window_end,
        is_fused_backward_range,
    )

    all_totals_ns: dict[str, int] = defaultdict(int)
    all_counts: dict[str, int] = defaultdict(int)
    forward_totals_ns: dict[str, int] = defaultdict(int)
    forward_counts: dict[str, int] = defaultdict(int)
    backward_totals_ns: dict[str, int] = defaultdict(int)
    backward_counts: dict[str, int] = defaultdict(int)
    kernel_durations_ns: list[int] = []
    forward_indices: set[int] = set()
    backward_indices: set[int] = set()
    fused_variant = bool(run_metrics["bias_gelu_fusion"])

    for index, row in enumerate(kernels):
        duration = row["end"] - row["start"]
        name = strings.get(row["demangledName"], f"StringId({row['demangledName']})")
        all_totals_ns[name] += duration
        all_counts[name] += 1
        kernel_durations_ns.append(duration)
        if row["correlationId"] in activation_correlations:
            forward_indices.add(index)
            forward_totals_ns[name] += duration
            forward_counts[name] += 1
        backward_match = (
            row["correlationId"] in fused_backward_correlations
            if fused_variant
            else is_eager_gelu_backward_kernel(name)
        )
        if backward_match:
            backward_indices.add(index)
            backward_totals_ns[name] += duration
            backward_counts[name] += 1

    iterations = run_metrics["measured_iterations"]
    forward_total_ns = sum(
        kernels[index]["end"] - kernels[index]["start"] for index in forward_indices
    )
    backward_total_ns = sum(
        kernels[index]["end"] - kernels[index]["start"] for index in backward_indices
    )
    gelu_indices = forward_indices | backward_indices
    gelu_total_ns = sum(
        kernels[index]["end"] - kernels[index]["start"] for index in gelu_indices
    )
    under_50_us = [duration for duration in kernel_durations_ns if duration < 50_000]

    all_gpu_intervals = [(row["start"], row["end"]) for row in kernels]
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
    activation_range_count = sum(activation_ranges.values())
    fused_backward_range_count = sum(fused_backward_ranges.values())
    kernel_total_ns = sum(kernel_durations_ns)

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
        "bias_gelu_forward": {
            "attribution": (
                "CUDA kernels launched inside Megatron MLP.forward.activation NVTX "
                "ranges; includes the separate bias add in A"
            ),
            "range_count": activation_range_count,
            "ranges_per_step": activation_range_count / iterations,
            "expected_ranges_per_step": EXPECTED_ACTIVATION_SITES_PER_STEP,
            "range_counts": activation_ranges,
            "kernel_count": len(forward_indices),
            "kernel_count_per_step": len(forward_indices) / iterations,
            "gpu_time_ms": forward_total_ns / NS_PER_MS,
            "gpu_time_per_step_ms": forward_total_ns / iterations / NS_PER_MS,
            "top_kernel_families": kernel_breakdown(
                forward_totals_ns,
                forward_counts,
                forward_total_ns,
                15,
            ),
        },
        "gelu_backward": {
            "attribution": (
                "gelu::backward_fused NVTX correlation"
                if fused_variant
                else "CUDA kernel names containing GeluBackward"
            ),
            "range_count": fused_backward_range_count,
            "ranges_per_step": fused_backward_range_count / iterations,
            "kernel_count": len(backward_indices),
            "kernel_count_per_step": len(backward_indices) / iterations,
            "gpu_time_ms": backward_total_ns / NS_PER_MS,
            "gpu_time_per_step_ms": backward_total_ns / iterations / NS_PER_MS,
            "top_kernel_families": kernel_breakdown(
                backward_totals_ns,
                backward_counts,
                backward_total_ns,
                15,
            ),
        },
        "gelu_related": {
            "definition": "Union of attributed bias-plus-GELU forward and GELU backward kernels",
            "kernel_count": len(gelu_indices),
            "kernel_count_per_step": len(gelu_indices) / iterations,
            "gpu_time_ms": gelu_total_ns / NS_PER_MS,
            "gpu_time_per_step_ms": gelu_total_ns / iterations / NS_PER_MS,
        },
        "top_15_cuda_kernels_by_total_gpu_time": kernel_breakdown(
            all_totals_ns,
            all_counts,
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
            "Forward attribution uses identical Megatron activation NVTX ranges in "
            "both variants. The baseline backward kernel has an explicit "
            "GeluBackward name; the compiled fused backward is manually ranged."
        ),
    }
    connection.close()
    return result


def compare_controls(baseline: dict[str, Any], fused: dict[str, Any]) -> None:
    excluded = {"bias_gelu_fusion", "bias_activation_fusion"}
    baseline_model = {
        key: value for key, value in baseline["model_config"].items() if key not in excluded
    }
    fused_model = {
        key: value for key, value in fused["model_config"].items() if key not in excluded
    }
    if baseline_model != fused_model:
        raise RuntimeError("A/B model configurations differ beyond bias-plus-GELU fusion")
    if baseline["model_config"]["bias_gelu_fusion"] is not False:
        raise RuntimeError("A does not have bias_gelu_fusion=False")
    if fused["model_config"]["bias_gelu_fusion"] is not True:
        raise RuntimeError("B does not have bias_gelu_fusion=True")
    if baseline["bias_dropout_fusion"] is not True or fused["bias_dropout_fusion"] is not True:
        raise RuntimeError("BDA fusion was not retained in both variants")

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

    baseline_runtime = baseline["control_verification"]["runtime_gelu_path"]
    fused_runtime = fused["control_verification"]["runtime_gelu_path"]
    if (
        not baseline_runtime["verified"]
        or not fused_runtime["verified"]
        or baseline_runtime["observed_bias_gelu_impl_calls"] != 0
        or fused_runtime["observed_bias_gelu_impl_calls"] != 24
    ):
        raise RuntimeError("Runtime selection of the fused bias-plus-GELU path failed")


def improvement_when_lower(baseline: float, fused: float) -> float:
    return (baseline - fused) / baseline * 100.0 if baseline else 0.0


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    monitoring = run["gpu_monitoring"]
    return {
        "variant": run["variant"],
        "bias_gelu_fusion": run["bias_gelu_fusion"],
        "bias_activation_fusion": run["bias_activation_fusion"],
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
        "runtime_gelu_path": run["control_verification"]["runtime_gelu_path"],
        "environment": run["environment"],
    }


def profile_delta(
    baseline: dict[str, Any],
    fused: dict[str, Any],
    baseline_run: dict[str, Any],
    fused_run: dict[str, Any],
) -> dict[str, Any]:
    baseline_forward = baseline["bias_gelu_forward"]
    fused_forward = fused["bias_gelu_forward"]
    baseline_backward = baseline["gelu_backward"]
    fused_backward = fused["gelu_backward"]
    baseline_related = baseline["gelu_related"]
    fused_related = fused["gelu_related"]
    baseline_small = baseline["kernels_under_50_us"]
    fused_small = fused["kernels_under_50_us"]

    forward_removed = (
        baseline_forward["kernel_count_per_step"]
        - fused_forward["kernel_count_per_step"]
    )
    total_removed = baseline["kernel_count_per_step"] - fused["kernel_count_per_step"]
    ranges_valid = (
        baseline_forward["ranges_per_step"] == EXPECTED_ACTIVATION_SITES_PER_STEP
        and fused_forward["ranges_per_step"] == EXPECTED_ACTIVATION_SITES_PER_STEP
        and fused_backward["ranges_per_step"] == EXPECTED_ACTIVATION_SITES_PER_STEP
    )
    runtime_verified = (
        baseline_run["control_verification"]["runtime_gelu_path"]["verified"]
        and fused_run["control_verification"]["runtime_gelu_path"]["verified"]
    )
    evidence = (
        ranges_valid
        and runtime_verified
        and forward_removed >= EVIDENCE_MINIMUM_REMOVED_LAUNCHES
        and total_removed >= EVIDENCE_MINIMUM_REMOVED_LAUNCHES
        and fused_forward["gpu_time_per_step_ms"]
        < baseline_forward["gpu_time_per_step_ms"]
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
        "bias_gelu_forward_launches_removed_per_step": forward_removed,
        "bias_gelu_forward_gpu_time_reduction_per_step_ms": (
            baseline_forward["gpu_time_per_step_ms"]
            - fused_forward["gpu_time_per_step_ms"]
        ),
        "bias_gelu_forward_gpu_time_improvement_percent": improvement_when_lower(
            baseline_forward["gpu_time_per_step_ms"],
            fused_forward["gpu_time_per_step_ms"],
        ),
        "gelu_backward_launches_removed_per_step": (
            baseline_backward["kernel_count_per_step"]
            - fused_backward["kernel_count_per_step"]
        ),
        "gelu_backward_gpu_time_reduction_per_step_ms": (
            baseline_backward["gpu_time_per_step_ms"]
            - fused_backward["gpu_time_per_step_ms"]
        ),
        "gelu_backward_gpu_time_improvement_percent": improvement_when_lower(
            baseline_backward["gpu_time_per_step_ms"],
            fused_backward["gpu_time_per_step_ms"],
        ),
        "gelu_related_launches_removed_per_step": (
            baseline_related["kernel_count_per_step"]
            - fused_related["kernel_count_per_step"]
        ),
        "gelu_related_gpu_time_reduction_per_step_ms": (
            baseline_related["gpu_time_per_step_ms"]
            - fused_related["gpu_time_per_step_ms"]
        ),
        "kernels_under_50_us_count_per_step": (
            fused_small["count_per_step"] - baseline_small["count_per_step"]
        ),
        "kernels_under_50_us_removed_per_step": (
            baseline_small["count_per_step"] - fused_small["count_per_step"]
        ),
        "kernels_under_50_us_percentage_points": (
            fused_small["count_share_percent"] - baseline_small["count_share_percent"]
        ),
        "expected_removed_forward_launches_per_step": (
            EXPECTED_REMOVED_FORWARD_LAUNCHES_PER_STEP
        ),
        "activation_ranges_valid": ranges_valid,
        "runtime_path_verified": runtime_verified,
        "fusion_evidence_confirmed": evidence,
        "evidence_criterion": (
            "Both traces contain exactly 24 MLP activation ranges/step, the fused "
            "trace contains 24 fused-backward ranges/step, at least 18 forward and "
            "total kernels/step are removed, forward activation time falls, and "
            "runtime probing observes bias_gelu_impl only in B."
        ),
    }


def decide_validation(
    correctness: dict[str, Any],
    screen_delta: dict[str, float],
    profiler_delta: dict[str, Any],
    stability_delta: dict[str, float] | None,
) -> dict[str, Any]:
    if not correctness.get("passed", False):
        return {
            "formal_validation_required": False,
            "stability_repeat_required": False,
            "reason": (
                "Correctness did not satisfy the strict A/B tolerance; formal "
                "performance validation is skipped."
            ),
        }
    throughput_gain = screen_delta["tokens_per_second_percent"]
    evidence = profiler_delta["fusion_evidence_confirmed"]
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
                "Screen gain is at least 2% and profiler evidence confirms fusion."
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
    profiler_delta = profile_delta(
        baseline_profile,
        fused_profile,
        baseline_profile_run,
        fused_profile_run,
    )

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
        "experiment": "Phase 5.3 Bias-plus-GELU fusion A/B",
        "correctness": correctness,
        "infrastructure": {
            "pod_id": args.pod_id,
            "cloud": "Secure Cloud",
            "gpu": "1x NVIDIA A40 48GB",
            "gpu_price_per_hour_usd": 0.44,
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "driver": baseline_screen["environment"]["driver"],
            "trace_preserved_on_stopped_pod": True,
        },
        "constraint_deviations": [
            {
                "field": "NVIDIA driver",
                "phase5_2_value": "570.195.03",
                "phase5_3_value": baseline_screen["environment"]["driver"],
                "reason": (
                    "The Phase 5.2 Pod could not restart because its host had no "
                    "free A40. The replacement A40 host supplied driver 580.159.03. "
                    "A and B still ran on the same replacement Pod."
                ),
                "affects_internal_ab_control": False,
                "affects_cross_phase_comparison": True,
            }
        ]
        if baseline_screen["environment"]["driver"] != "570.195.03"
        else [],
        "controls": {
            "only_intended_variable": "bias_activation_fusion (bias_gelu_fusion)",
            "A": {"bias_gelu_fusion": False, "bias_activation_fusion": False},
            "B": {"bias_gelu_fusion": True, "bias_activation_fusion": True},
            "model_config_except_intended_variable": {
                key: value
                for key, value in baseline_screen["model_config"].items()
                if key not in {"bias_gelu_fusion", "bias_activation_fusion"}
            },
            "parameter_count": baseline_screen["parameter_count"],
            "parallelism": baseline_screen["parallelism"],
            "precision": baseline_screen["precision"],
            "optimizer": baseline_screen["optimizer"],
            "data": baseline_screen["data"],
            "environment": baseline_screen["environment"],
            "retained": ["cuDNN FusedAttention", "micro-batch 8", "bias_dropout_fusion"],
            "explicitly_disabled": [
                "masked_softmax_fusion",
                "cross_entropy_loss_fusion",
                "CUDA Graph",
                "optimizer fusion",
                "dtype changes",
            ],
        },
        "runtime_path_verification": {
            "baseline": baseline_screen["control_verification"]["runtime_gelu_path"],
            "fused": fused_screen["control_verification"]["runtime_gelu_path"],
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
                "Bias-plus-GELU forward attribution includes the separate bias-add "
                "kernel in A. Backward attribution covers the GELU derivative "
                "kernel. Overall kernel deltas cover the full training step."
            ),
        },
        "stability_repeat": stability,
        "decision": decision,
        "formal_validation": formal_validation,
        "numerical_note": (
            "The pinned Megatron fused path uses tanh-approximate GELU, whereas A "
            "uses torch.nn.functional.gelu's exact/erf path. This formula difference "
            "is an inherent consequence of the requested fusion flag."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
