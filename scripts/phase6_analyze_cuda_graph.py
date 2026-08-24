#!/usr/bin/env python3
"""Analyze Phase 6.3 DDP CUDA Graph timing runs and Nsight Systems traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NS_PER_MS = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-screen", type=Path, required=True)
    parser.add_argument("--graphed-screen", type=Path, required=True)
    parser.add_argument("--baseline-profile-metrics", type=Path, required=True)
    parser.add_argument("--graphed-profile-metrics", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--graphed-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--graphed-trace", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--formal-baseline", type=Path)
    parser.add_argument("--formal-graphed", type=Path)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.formal_baseline is None) != (args.formal_graphed is None):
        parser.error("formal baseline and graphed metrics must be supplied together")
    return args


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        row["name"]
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


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


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def string_value(value: Any, strings: dict[int, str]) -> str:
    if isinstance(value, int):
        return strings.get(value, f"StringId({value})")
    return str(value or "")


def runtime_name(
    row: sqlite3.Row,
    columns: set[str],
    strings: dict[int, str],
) -> str:
    for candidate in ("nameId", "name", "demangledName"):
        if candidate in columns and row[candidate] is not None:
            return string_value(row[candidate], strings)
    return "unknown"


def is_launch_api(name: str) -> bool:
    normalized = name.lower()
    return any(
        token in normalized
        for token in (
            "cudalaunchkernel",
            "cudalaunchkernelexc",
            "cudagraphlaunch",
            "culaunchkernel",
        )
    )


def is_graph_launch_api(name: str) -> bool:
    return "cudagraphlaunch" in name.lower()


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
            f'SELECT start, end FROM "{table}" WHERE end > ? AND start < ?',
            (window_start, window_end),
        )
    ]


def kernel_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    value = row["demangledName"]
    return string_value(value, strings)


def analyze_launch_apis(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
    iterations: int,
) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if not table_exists(connection, table):
        raise RuntimeError(f"{table} missing from Nsight SQLite export")
    columns = table_columns(connection, table)
    required = {"start", "end", "globalTid"}
    if not required.issubset(columns):
        raise RuntimeError(f"{table} lacks required columns: {required - columns}")

    rows = list(
        connection.execute(
            f'SELECT * FROM "{table}" WHERE end > ? AND start < ? ORDER BY start',
            (window_start, window_end),
        )
    )
    named = [
        (
            runtime_name(row, columns, strings),
            int(row["start"]),
            int(row["end"]),
            int(row["globalTid"]),
        )
        for row in rows
    ]
    launches = [event for event in named if is_launch_api(event[0])]
    graph_launches = [event for event in launches if is_graph_launch_api(event[0])]
    launch_counts = Counter(event[0] for event in launches)
    runtime_counts = Counter(event[0] for event in named)
    by_thread: dict[int, list[tuple[str, int, int, int]]] = defaultdict(list)
    for event in launches:
        by_thread[event[3]].append(event)
    main_thread = max(by_thread, key=lambda thread: len(by_thread[thread])) if by_thread else None
    main_launches = sorted(by_thread.get(main_thread, []), key=lambda event: event[1])
    gap_us = [
        max(0, current[1] - previous[2]) / 1_000.0
        for previous, current in zip(main_launches, main_launches[1:])
    ]
    launch_api_time_ms = sum(end - start for _, start, end, _ in launches) / NS_PER_MS
    return {
        "runtime_api_call_count": len(named),
        "runtime_api_calls_per_step": len(named) / iterations,
        "launch_api_call_count": len(launches),
        "launch_api_calls_per_step": len(launches) / iterations,
        "graph_launch_call_count": len(graph_launches),
        "graph_launch_calls_per_step": len(graph_launches) / iterations,
        "ordinary_launch_calls_per_step": (
            (len(launches) - len(graph_launches)) / iterations
        ),
        "launch_api_time_ms_per_step": launch_api_time_ms / iterations,
        "launch_api_breakdown": [
            {"name": name, "count": count, "count_per_step": count / iterations}
            for name, count in launch_counts.most_common()
        ],
        "top_runtime_api_calls": [
            {"name": name, "count": count, "count_per_step": count / iterations}
            for name, count in runtime_counts.most_common(15)
        ],
        "main_launch_thread_global_tid": main_thread,
        "cpu_launch_gaps": {
            "definition": (
                "non-negative interval from one launch API return to the next "
                "launch API entry on the thread issuing the most launches"
            ),
            "sample_count": len(gap_us),
            "mean_us": statistics.fmean(gap_us) if gap_us else 0.0,
            "median_us": statistics.median(gap_us) if gap_us else 0.0,
            "p95_us": percentile(gap_us, 0.95),
            "max_us": max(gap_us, default=0.0),
            "total_ms_per_step": sum(gap_us) / 1000.0 / iterations,
        },
    }


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
    window_start, window_end = int(window["start"]), int(window["end"])
    window_ns = window_end - window_start
    iterations = int(run_metrics["measured_iterations"])

    kernels = list(
        connection.execute(
            """
            SELECT start, end, demangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )
    durations_ns = [int(row["end"] - row["start"]) for row in kernels]
    names = [kernel_name(row, strings) for row in kernels]
    totals_ns: dict[str, int] = defaultdict(int)
    counts = Counter(names)
    for name, duration in zip(names, durations_ns):
        totals_ns[name] += duration

    all_gpu_intervals = [
        (int(row["start"]), int(row["end"])) for row in kernels
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
    under_50_us = [duration for duration in durations_ns if duration < 50_000]

    step_ranges = []
    for row in connection.execute(
        """
        SELECT start, end, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
        """,
        (window_start, window_end),
    ):
        name = row["text"] or strings.get(row["textId"], "")
        if name.startswith("train_step_"):
            step_ranges.append((row["end"] - row["start"]) / NS_PER_MS)

    launch_metrics = analyze_launch_apis(
        connection,
        strings,
        window_start,
        window_end,
        iterations,
    )
    kernel_total_ns = sum(durations_ns)
    result = {
        "profile_window_ms": window_ns / NS_PER_MS,
        "profiled_iterations": iterations,
        "kernel_count": len(kernels),
        "kernel_count_per_step": len(kernels) / iterations,
        "kernel_time_ms_per_step": kernel_total_ns / NS_PER_MS / iterations,
        "kernels_under_50_us": {
            "count": len(under_50_us),
            "count_per_step": len(under_50_us) / iterations,
            "count_share_percent": (
                len(under_50_us) / len(kernels) * 100.0 if kernels else 0.0
            ),
            "time_ms_per_step": sum(under_50_us) / NS_PER_MS / iterations,
        },
        "gpu_timeline": {
            "active_union_ms": active_ns / NS_PER_MS,
            "idle_ms": idle_ns / NS_PER_MS,
            "idle_ms_per_step": idle_ns / NS_PER_MS / iterations,
            "idle_percent": idle_ns / window_ns * 100.0 if window_ns else 0.0,
            "idle_gap_count": len(idle_gaps_ns),
        },
        "cuda_api": launch_metrics,
        "cpu_step_nvtx": {
            "sample_count": len(step_ranges),
            "average_wall_ms": (
                statistics.fmean(step_ranges) if step_ranges else None
            ),
            "median_wall_ms": statistics.median(step_ranges) if step_ranges else None,
        },
        "top_15_cuda_kernels_by_total_gpu_time": [
            {
                "name": name,
                "count": counts[name],
                "total_ms": total / NS_PER_MS,
                "average_us": total / counts[name] / 1_000.0,
            }
            for name, total in sorted(
                totals_ns.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:15]
        ],
        "trace": {
            "path": str(trace_path),
            "size_bytes": trace_path.stat().st_size,
            "sha256": sha256(trace_path),
            "sqlite_path": str(sqlite_path),
            "sqlite_size_bytes": sqlite_path.stat().st_size,
            "sqlite_sha256": sha256(sqlite_path),
            "preserved_on_pod": True,
            "cuda_graph_trace_mode": "node",
        },
    }
    connection.close()
    return result


def environment_without_run_identity(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in environment.items()
        if key not in {"timestamp_utc", "cuda_graph_enabled"}
    }


def compare_controls(baseline: dict[str, Any], graphed: dict[str, Any]) -> None:
    differing_model_fields = {
        key
        for key in baseline["model_config"]
        if baseline["model_config"].get(key) != graphed["model_config"].get(key)
    }
    allowed = {"cuda_graph_enabled", "cuda_graph_impl"}
    if differing_model_fields != allowed:
        raise RuntimeError(
            "A/B model configurations differ beyond CUDA Graph: "
            f"{sorted(differing_model_fields)}"
        )
    if baseline["cuda_graph_enabled"] is not False:
        raise RuntimeError("A unexpectedly enabled CUDA Graph")
    if graphed["cuda_graph_enabled"] is not True:
        raise RuntimeError("B did not enable CUDA Graph")
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
        "priming_iterations",
        "benchmark_warmup_iterations",
        "measured_iterations",
    ):
        if baseline[field] != graphed[field]:
            raise RuntimeError(f"A/B control field differs: {field}")
    if environment_without_run_identity(
        baseline["environment"]
    ) != environment_without_run_identity(graphed["environment"]):
        raise RuntimeError("A/B software or hardware environments differ")


def percent_change(baseline: float, candidate: float) -> float:
    return (candidate - baseline) / baseline * 100.0


def improvement_when_lower(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": run["variant"],
        "cuda_graph_enabled": run["cuda_graph_enabled"],
        "average_step_time_ms": run["average_step_time_ms"],
        "median_step_time_ms": run["median_step_time_ms"],
        "step_time_standard_deviation_ms": run[
            "step_time_standard_deviation_ms"
        ],
        "step_times_ms": run["step_times_ms"],
        "average_cpu_process_time_ms": run["cpu_process_time"][
            "average_ms_per_step"
        ],
        "median_cpu_process_time_ms": run["cpu_process_time"][
            "median_ms_per_step"
        ],
        "tokens_per_second": run["tokens_per_second"],
        "mfu_percent": run["mfu"]["mfu_percent"],
        "peak_allocated_memory_mib": run["peak_allocated_memory_mib"],
        "peak_reserved_memory_mib": run["peak_reserved_memory_mib"],
        "peak_nvidia_smi_memory_mib": run["gpu_monitoring"][
            "peak_nvidia_smi_memory_mib"
        ],
        "final_loss": run["final_loss"],
        "losses_finite": run["losses_finite"],
        "cuda_graph": run["cuda_graph"],
        "environment": run["environment"],
    }


def timing_delta(
    baseline: dict[str, Any],
    graphed: dict[str, Any],
) -> dict[str, float]:
    baseline_step = baseline["average_step_time_ms"]
    graphed_step = graphed["average_step_time_ms"]
    baseline_cpu = baseline["cpu_process_time"]["average_ms_per_step"]
    graphed_cpu = graphed["cpu_process_time"]["average_ms_per_step"]
    return {
        "average_step_time_ms": graphed_step - baseline_step,
        "median_step_time_ms": (
            graphed["median_step_time_ms"] - baseline["median_step_time_ms"]
        ),
        "step_time_improvement_percent": improvement_when_lower(
            baseline_step,
            graphed_step,
        ),
        "speedup_ratio": baseline_step / graphed_step,
        "tokens_per_second": (
            graphed["tokens_per_second"] - baseline["tokens_per_second"]
        ),
        "tokens_per_second_percent": percent_change(
            baseline["tokens_per_second"],
            graphed["tokens_per_second"],
        ),
        "mfu_percentage_points": (
            graphed["mfu"]["mfu_percent"] - baseline["mfu"]["mfu_percent"]
        ),
        "cpu_process_time_ms": graphed_cpu - baseline_cpu,
        "cpu_process_time_reduction_percent": improvement_when_lower(
            baseline_cpu,
            graphed_cpu,
        ),
        "peak_allocated_memory_mib": (
            graphed["peak_allocated_memory_mib"]
            - baseline["peak_allocated_memory_mib"]
        ),
        "peak_reserved_memory_mib": (
            graphed["peak_reserved_memory_mib"]
            - baseline["peak_reserved_memory_mib"]
        ),
    }


def profile_delta(
    baseline: dict[str, Any],
    graphed: dict[str, Any],
) -> dict[str, Any]:
    baseline_api = baseline["cuda_api"]
    graphed_api = graphed["cuda_api"]
    baseline_idle = baseline["gpu_timeline"]
    graphed_idle = graphed["gpu_timeline"]
    baseline_launches = baseline_api["launch_api_calls_per_step"]
    graphed_launches = graphed_api["launch_api_calls_per_step"]
    return {
        "kernel_count_per_step": (
            graphed["kernel_count_per_step"] - baseline["kernel_count_per_step"]
        ),
        "kernel_count_reduction_percent": improvement_when_lower(
            baseline["kernel_count_per_step"],
            graphed["kernel_count_per_step"],
        ),
        "cuda_api_launches_per_step": graphed_launches - baseline_launches,
        "cuda_api_launch_reduction_per_step": baseline_launches - graphed_launches,
        "cuda_api_launch_reduction_percent": improvement_when_lower(
            baseline_launches,
            graphed_launches,
        ),
        "graph_launches_per_step": graphed_api["graph_launch_calls_per_step"],
        "ordinary_launch_reduction_per_step": (
            baseline_api["ordinary_launch_calls_per_step"]
            - graphed_api["ordinary_launch_calls_per_step"]
        ),
        "launch_api_time_ms_per_step": (
            graphed_api["launch_api_time_ms_per_step"]
            - baseline_api["launch_api_time_ms_per_step"]
        ),
        "cpu_launch_gap_total_ms_per_step": (
            graphed_api["cpu_launch_gaps"]["total_ms_per_step"]
            - baseline_api["cpu_launch_gaps"]["total_ms_per_step"]
        ),
        "cpu_launch_gap_total_reduction_percent": improvement_when_lower(
            baseline_api["cpu_launch_gaps"]["total_ms_per_step"],
            graphed_api["cpu_launch_gaps"]["total_ms_per_step"],
        ),
        "gpu_idle_ms_per_step": (
            graphed_idle["idle_ms_per_step"] - baseline_idle["idle_ms_per_step"]
        ),
        "gpu_idle_reduction_ms_per_step": (
            baseline_idle["idle_ms_per_step"] - graphed_idle["idle_ms_per_step"]
        ),
        "gpu_idle_reduction_percent": improvement_when_lower(
            baseline_idle["idle_ms_per_step"],
            graphed_idle["idle_ms_per_step"],
        ),
        "replay_profiler_verified": (
            baseline_api["graph_launch_calls_per_step"] == 0
            and graphed_api["graph_launch_calls_per_step"] > 0
        ),
    }


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline_screen)
    graphed = load(args.graphed_screen)
    correctness = load(args.correctness)
    compare_controls(baseline, graphed)
    baseline_profile_run = load(args.baseline_profile_metrics)
    graphed_profile_run = load(args.graphed_profile_metrics)
    compare_controls(baseline_profile_run, graphed_profile_run)
    baseline_profile = analyze_trace(
        args.baseline_sqlite,
        args.baseline_trace,
        baseline_profile_run,
    )
    graphed_profile = analyze_trace(
        args.graphed_sqlite,
        args.graphed_trace,
        graphed_profile_run,
    )
    timing = timing_delta(baseline, graphed)
    profiling = profile_delta(baseline_profile, graphed_profile)
    qualified = (
        correctness["passed"]
        and graphed["cuda_graph"]["state"]["replay_ready"]
        and timing["tokens_per_second_percent"] >= 2.0
    )

    formal = None
    if args.formal_baseline is not None:
        formal_baseline = load(args.formal_baseline)
        formal_graphed = load(args.formal_graphed)
        compare_controls(formal_baseline, formal_graphed)
        formal = {
            "baseline": run_summary(formal_baseline),
            "graphed": run_summary(formal_graphed),
            "delta": timing_delta(formal_baseline, formal_graphed),
        }
    if qualified != (formal is not None):
        raise RuntimeError(
            "Formal validation presence does not match the >=2% correctness/replay gate"
        )

    result = {
        "status": "success",
        "experiment": "Phase 6.3 MCore DDP CUDA Graph fast A/B",
        "mechanism": {
            "selected": (
                "Megatron Core cuda_graph_impl=local with cuda_graph_modules=[]; "
                "one forward and one backward graph per TransformerLayer"
            ),
            "selection_reason": (
                "MCore DDP owns persistent main_grad buffers while the local graph "
                "path records weight-gradient accumulation into those buffers. "
                "FP32Optimizer leaves the unchanged AdamW step outside capture."
            ),
            "alternatives_rejected": {
                "transformer_engine": (
                    "Would change the selected MCore local graph mechanism and "
                    "requires separate higher-level capture integration."
                ),
                "full_iteration": (
                    "FullCudaGraphWrapper requires Megatron's forward_backward_func "
                    "and static data-iterator contract; it is outside this "
                    "per-TransformerLayer experiment."
                ),
                "custom_torch_cudagraph": (
                    "Rejected in favor of the pinned MCore implementation."
                ),
            },
            "compatibility_findings": {
                "fixed_shapes": "satisfied: batch=8 and seq=2048 are constant",
                "input_addresses": (
                    "MCore copies runtime layer inputs to graph-owned static buffers"
                ),
                "parameter_addresses": (
                    "AdamW updates parameters in place, preserving parameter storage"
                ),
                "gradient_lifecycle": (
                    "MCore DDP allocates all main_grad views, performs in-place "
                    "buffer reset, and finalizes gradients before optimizer access"
                ),
                "optimizer": (
                    "MCore FP32Optimizer exposes DDP main_grad to the unchanged "
                    "torch.optim.AdamW, outside all graphs"
                ),
                "zero_grad": (
                    "DDP.zero_grad_buffer runs before FP32Optimizer.zero_grad and "
                    "preserves all graph-visible buffer addresses"
                ),
                "dropout_rng": (
                    "MCore CUDAGraph objects register graph-safe Megatron RNG states; "
                    "the graph-safe tracker is used in both A and B"
                ),
                "dynamic_allocations": (
                    "captured layer allocations use MCore's shared CUDA graph pool; "
                    "embedding, loss, final norm, optimizer remain eager"
                ),
                "numerics": (
                    "BF16 autocast, FP32 parameters/residuals/AdamW, dropout, and all "
                    "fusion flags are unchanged"
                ),
                "distributed_optimizer": "disabled",
            },
            "pinned_source": {
                "megatron_lm_commit": baseline["environment"]["megatron_lm_commit"],
                "pytorch": baseline["environment"]["pytorch"],
                "transformer_engine": baseline["environment"][
                    "transformer_engine_version"
                ],
            },
        },
        "correctness": correctness,
        "infrastructure": {
            "pod_id": args.pod_id,
            "gpu": baseline["environment"]["gpu"],
            "driver": baseline["environment"]["driver"],
            "same_pod_for_a_b": (
                baseline["environment"]["hostname"]
                == graphed["environment"]["hostname"]
            ),
        },
        "fast_screen": {
            "protocol": {
                "graph_capture_warmup_iterations": 5,
                "benchmark_warmup_iterations": 3,
                "measured_iterations": 15,
                "priming_iterations": 1,
            },
            "baseline": run_summary(baseline),
            "graphed": run_summary(graphed),
            "delta": timing,
        },
        "nsight_systems": {
            "protocol": {
                "same_pod": True,
                "cuda_graph_trace_mode": "node",
                "profiled_iterations": baseline_profile["profiled_iterations"],
            },
            "baseline": baseline_profile,
            "graphed": graphed_profile,
            "delta": profiling,
        },
        "decision": {
            "formal_validation_threshold_percent": 2.0,
            "correctness_passed": correctness["passed"],
            "capture_replay_succeeded": bool(
                graphed["cuda_graph"]["state"]["replay_ready"]
                and profiling["replay_profiler_verified"]
            ),
            "fast_throughput_gain_percent": timing[
                "tokens_per_second_percent"
            ],
            "qualified_for_formal_validation": qualified,
            "formal_validation_run": formal is not None,
            "outcome": (
                "formal_validation_completed"
                if formal is not None
                else "fast_screen_only_below_2_percent_gate"
            ),
        },
        "formal_validation": formal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PHASE6_CUDA_GRAPH_ANALYSIS_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
