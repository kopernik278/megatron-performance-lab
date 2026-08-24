#!/usr/bin/env python3
"""Analyze Phase 7.1 TP1/TP2 timing and communication traces."""

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
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--tp1", type=Path, required=True)
    parser.add_argument("--tp2", type=Path, required=True)
    parser.add_argument("--tp2-profile", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def string_value(value: Any, strings: dict[int, str]) -> str:
    if isinstance(value, int):
        return strings.get(value, f"StringId({value})")
    return str(value or "")


def kernel_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    for field in ("demangledName", "shortName", "mangledName"):
        if field in row.keys() and row[field] is not None:
            return string_value(row[field], strings)
    return "unknown"


def nvtx_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    return str(row["text"] or strings.get(row["textId"], ""))


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    clipped = sorted(
        (max(item_start, start), min(item_end, end))
        for item_start, item_end in intervals
        if item_end > start and item_start < end
    )
    merged: list[tuple[int, int]] = []
    for item_start, item_end in clipped:
        if not merged or item_start > merged[-1][1]:
            merged.append((item_start, item_end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], item_end))
    return merged


def interval_total(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_total(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> int:
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            total += end - start
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def is_nccl_kernel(name: str) -> bool:
    return "nccl" in name.lower()


def collective_type(name: str) -> str:
    normalized = name.lower().replace("_", "")
    if "allreduce" in normalized:
        return "All-Reduce"
    if "allgather" in normalized:
        return "All-Gather"
    if "reducescatter" in normalized:
        return "Reduce-Scatter"
    if "broadcast" in normalized:
        return "Broadcast"
    if "reduce" in normalized:
        return "Reduce"
    return "Unresolved NCCL kernel"


def correlation_nvtx_names(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> dict[int, set[str]]:
    ranges_by_thread: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT start, end, globalTid, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
        """,
        (window_start, window_end),
    ):
        ranges_by_thread[int(row["globalTid"])].append(
            (int(row["start"]), int(row["end"]), nvtx_name(row, strings))
        )
    calls_by_thread: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT start, globalTid, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start >= ? AND start < ? AND correlationId IS NOT NULL
        """,
        (window_start, window_end),
    ):
        calls_by_thread[int(row["globalTid"])].append(
            (int(row["start"]), int(row["correlationId"]))
        )

    result: dict[int, set[str]] = defaultdict(set)
    for thread, calls in calls_by_thread.items():
        events: list[tuple[int, int, int, str | int]] = []
        for index, (start, end, name) in enumerate(ranges_by_thread.get(thread, [])):
            events.append((start, 0, index, name))
            events.append((end, 2, index, name))
        for start, correlation_id in calls:
            events.append((start, 1, correlation_id, correlation_id))
        events.sort(key=lambda event: (event[0], event[1]))
        active: dict[int, str] = {}
        for _, kind, identifier, payload in events:
            if kind == 0:
                active[identifier] = str(payload)
            elif kind == 1:
                result[int(payload)].update(active.values())
            else:
                active.pop(identifier, None)
    return result


def attribution(names: set[str]) -> str:
    combined = " ".join(names).lower()
    if "attention_projection" in combined:
        return "attention TP forward"
    if "mlp_fc2" in combined:
        return "MLP TP forward"
    if "embedding" in combined:
        return "vocabulary embedding forward"
    if "output_layer" in combined or "vocab" in combined:
        return "vocabulary/output forward"
    if "backward" in combined:
        return "backward TP"
    if "finalize_model_grads" in combined:
        return "TP gradient finalization"
    if "forward" in combined:
        return "other forward"
    return "unattributed"


def analyze_trace(
    sqlite_path: Path,
    trace_path: Path,
    run: dict[str, Any],
) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    strings = {
        int(row["id"]): str(row["value"])
        for row in connection.execute("SELECT id, value FROM StringIds")
    }
    windows = list(
        connection.execute(
            """
            SELECT start, end
            FROM NVTX_EVENTS
            WHERE (text = 'profile_window' OR textId IN (
                SELECT id FROM StringIds WHERE value = 'profile_window'
            )) AND end IS NOT NULL
            """
        )
    )
    if not windows:
        raise RuntimeError("No profile_window NVTX ranges found")
    window_start = min(int(row["start"]) for row in windows)
    window_end = max(int(row["end"]) for row in windows)
    iterations = int(run["measured_iterations"])

    kernel_columns = table_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
    selected_columns = ["start", "end", "correlationId"]
    selected_columns.extend(
        field
        for field in ("demangledName", "shortName", "mangledName", "deviceId")
        if field in kernel_columns
    )
    kernels = list(
        connection.execute(
            f"""
            SELECT {', '.join(selected_columns)}
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )
    correlation_names = correlation_nvtx_names(
        connection,
        strings,
        window_start,
        window_end,
    )

    nccl_rows = []
    compute_rows = []
    for row in kernels:
        name = kernel_name(row, strings)
        entry = {
            "name": name,
            "start": int(row["start"]),
            "end": int(row["end"]),
            "duration_ns": int(row["end"] - row["start"]),
            "correlation_id": int(row["correlationId"]),
            "device_id": int(row["deviceId"]) if "deviceId" in row.keys() else 0,
            "attribution": attribution(
                correlation_names.get(int(row["correlationId"]), set())
            ),
        }
        if is_nccl_kernel(name):
            nccl_rows.append(entry)
        else:
            compute_rows.append(entry)

    devices = sorted({row["device_id"] for row in nccl_rows + compute_rows})
    per_device = {}
    for device in devices:
        comm = merge_intervals(
            (
                (row["start"], row["end"])
                for row in nccl_rows
                if row["device_id"] == device
            ),
            window_start,
            window_end,
        )
        compute = merge_intervals(
            (
                (row["start"], row["end"])
                for row in compute_rows
                if row["device_id"] == device
            ),
            window_start,
            window_end,
        )
        comm_ns = interval_total(comm)
        compute_ns = interval_total(compute)
        overlap_ns = intersection_total(comm, compute)
        per_device[str(device)] = {
            "communication_union_ms_per_step": comm_ns / NS_PER_MS / iterations,
            "compute_union_ms_per_step": compute_ns / NS_PER_MS / iterations,
            "communication_compute_overlap_ms_per_step": (
                overlap_ns / NS_PER_MS / iterations
            ),
            "communication_overlap_percent": (
                overlap_ns / comm_ns * 100.0 if comm_ns else 0.0
            ),
            "exposed_communication_ms_per_step": (
                (comm_ns - overlap_ns) / NS_PER_MS / iterations
            ),
        }

    type_counts = Counter(collective_type(row["name"]) for row in nccl_rows)
    type_times_ns: dict[str, int] = defaultdict(int)
    attribution_counts = Counter(row["attribution"] for row in nccl_rows)
    attribution_times_ns: dict[str, int] = defaultdict(int)
    name_counts = Counter(row["name"] for row in nccl_rows)
    name_times_ns: dict[str, int] = defaultdict(int)
    for row in nccl_rows:
        kind = collective_type(row["name"])
        type_times_ns[kind] += row["duration_ns"]
        attribution_times_ns[row["attribution"]] += row["duration_ns"]
        name_times_ns[row["name"]] += row["duration_ns"]

    nvtx_collectives = Counter()
    for row in connection.execute(
        """
        SELECT text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
        """,
        (window_start, window_end),
    ):
        name = nvtx_name(row, strings)
        if any(
            token in name.lower().replace("_", "")
            for token in ("allreduce", "allgather", "reducescatter", "nccl")
        ):
            nvtx_collectives[name] += 1

    tp = int(run["parallelism"]["tensor_parallel"])
    average_nccl_kernel_time_ms_per_step_per_gpu = (
        sum(row["duration_ns"] for row in nccl_rows)
        / NS_PER_MS
        / iterations
        / tp
    )
    result = {
        "profile_window_ms": (window_end - window_start) / NS_PER_MS,
        "profiled_iterations": iterations,
        "profile_window_rank_count": len(windows),
        "nccl_kernel_launches": len(nccl_rows),
        "nccl_kernel_launches_per_step_all_gpus": len(nccl_rows) / iterations,
        "estimated_logical_collectives_per_step": (
            len(nccl_rows) / iterations / tp
        ),
        "logical_count_estimation": (
            "one NCCL kernel per rank per collective on this two-rank run; "
            "kernel launches divided by TP world size"
        ),
        "average_nccl_kernel_time_ms_per_step_per_gpu": (
            average_nccl_kernel_time_ms_per_step_per_gpu
        ),
        "collective_types": {
            kind: {
                "kernel_launch_count": type_counts.get(kind, 0),
                "estimated_logical_count_per_step": (
                    type_counts.get(kind, 0) / iterations / tp
                ),
                "average_kernel_time_ms_per_step_per_gpu": (
                    type_times_ns.get(kind, 0) / NS_PER_MS / iterations / tp
                ),
            }
            for kind in (
                "All-Reduce",
                "All-Gather",
                "Reduce-Scatter",
                "Broadcast",
                "Reduce",
                "Unresolved NCCL kernel",
            )
        },
        "nvtx_collective_events": [
            {"name": name, "count": count}
            for name, count in nvtx_collectives.most_common(30)
        ],
        "attribution": {
            name: {
                "kernel_launch_count": attribution_counts[name],
                "average_kernel_time_ms_per_step_per_gpu": (
                    attribution_times_ns[name] / NS_PER_MS / iterations / tp
                ),
            }
            for name in sorted(attribution_counts)
        },
        "per_device_overlap": per_device,
        "top_nccl_kernels": [
            {
                "name": name,
                "count": name_counts[name],
                "total_ms_all_gpus": name_times_ns[name] / NS_PER_MS,
            }
            for name in sorted(
                name_counts,
                key=lambda item: name_times_ns[item],
                reverse=True,
            )[:20]
        ],
        "tensor_sizes": {
            "source": (
                "derived from fixed tensor shapes and MCore TP semantics because "
                "Nsight Systems SQLite does not export NCCL API payload sizes"
            ),
            "transformer_activation_elements": (
                run["model_config"]["micro_batch_size"]
                * run["model_config"]["sequence_length"]
                * run["model_config"]["hidden_size"]
            ),
            "large_bf16_activation_collective_bytes": (
                run["model_config"]["micro_batch_size"]
                * run["model_config"]["sequence_length"]
                * run["model_config"]["hidden_size"]
                * 2
            ),
            "embedding_fp32_collective_bytes": (
                run["model_config"]["micro_batch_size"]
                * run["model_config"]["sequence_length"]
                * run["model_config"]["hidden_size"]
                * 4
            ),
            "expected_large_all_reduces_per_step": {
                "attention_projection_forward": run["model_config"]["num_layers"],
                "mlp_fc2_forward": run["model_config"]["num_layers"],
                "qkv_dgrad_backward": run["model_config"]["num_layers"],
                "fc1_dgrad_backward": run["model_config"]["num_layers"],
                "total": run["model_config"]["num_layers"] * 4,
            },
            "expected_other_all_reduces": {
                "embedding_forward": 1,
                "vocab_parallel_cross_entropy": 3,
                "tensor_parallel_layernorm_gradient_finalization": 1,
            },
            "all_gather_expected": False,
            "reduce_scatter_expected": False,
            "reason": (
                "sequence parallelism and distributed optimizer are disabled"
            ),
        },
        "trace": {
            "path": str(trace_path),
            "size_bytes": trace_path.stat().st_size,
            "sha256": sha256(trace_path),
            "sqlite_path": str(sqlite_path),
            "sqlite_size_bytes": sqlite_path.stat().st_size,
            "sqlite_sha256": sha256(sqlite_path),
            "preserved_on_pod": True,
        },
    }
    connection.close()
    return result


def compare_controls(tp1: dict[str, Any], tp2: dict[str, Any]) -> None:
    if tp1["model_config"] != tp2["model_config"]:
        raise RuntimeError("TP1 and TP2 model controls differ")
    for field in (
        "precision",
        "optimizer",
        "data",
        "tokens_per_step",
        "smoke_iterations",
        "warmup_iterations",
        "measured_iterations",
    ):
        if tp1[field] != tp2[field]:
            raise RuntimeError(f"TP1 and TP2 control differs: {field}")
    if tp1["parallelism"]["tensor_parallel"] != 1:
        raise RuntimeError("TP1 result is not TP=1")
    if tp2["parallelism"]["tensor_parallel"] != 2:
        raise RuntimeError("TP2 result is not TP=2")


def timing_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "tensor_parallel": run["parallelism"]["tensor_parallel"],
        "average_step_time_ms": run["average_step_time_ms"],
        "median_step_time_ms": run["median_step_time_ms"],
        "tokens_per_second": run["tokens_per_second"],
        "aggregate_mfu_percent": run["mfu"]["aggregate_mfu_percent"],
        "rank_timing_and_memory": run["rank_timing_and_memory"],
        "gpu_monitoring": run["gpu_monitoring"],
    }


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    tp1 = load(args.tp1)
    tp2 = load(args.tp2)
    tp2_profile = load(args.tp2_profile)
    compare_controls(tp1, tp2)
    if tp2_profile["model_config"] != tp2["model_config"]:
        raise RuntimeError("Profile changed TP2 model controls")

    speedup = tp2["tokens_per_second"] / tp1["tokens_per_second"]
    scaling_efficiency = speedup / 2.0
    smoke_loss_errors = [
        abs(left - right)
        for left, right in zip(
            tp1["correctness_smoke"]["rank0_losses"],
            tp2["correctness_smoke"]["rank0_losses"],
        )
    ]
    communication = analyze_trace(args.sqlite, args.trace, tp2_profile)
    per_device_overlap = list(communication["per_device_overlap"].values())
    average_overlap_percent = statistics.fmean(
        item["communication_overlap_percent"] for item in per_device_overlap
    )
    average_exposed_ms = statistics.fmean(
        item["exposed_communication_ms_per_step"] for item in per_device_overlap
    )

    result = {
        "status": "success",
        "experiment": "Phase 7.1 two-GPU tensor-parallel baseline",
        "infrastructure": topology["infrastructure"],
        "topology": topology,
        "configuration": {
            "model": tp2["model_config"],
            "precision": tp2["precision"],
            "optimizer": tp2["optimizer"],
            "tp1_parallelism": tp1["parallelism"],
            "tp2_parallelism": tp2["parallelism"],
        },
        "tp_implementation": {
            "layer_spec": "get_gpt_layer_local_spec",
            "rank_shards": tp2["sharding_verification"],
            "all_required_modules_are_megatron_tp_modules": all(
                all(rank["type_checks"].values())
                for rank in tp2["sharding_verification"]
            ),
        },
        "correctness": {
            "tp1": tp1["correctness_smoke"],
            "tp2": tp2["correctness_smoke"],
            "tp1_vs_tp2_smoke_loss_absolute_errors": smoke_loss_errors,
            "tp1_vs_tp2_smoke_loss_max_absolute_error": max(smoke_loss_errors),
            "passed": bool(
                topology["nccl_all_reduce_sanity"]["passed"]
                and tp1["correctness_smoke"]["parameters_updated"]
                and tp2["correctness_smoke"]["parameters_updated"]
                and all(
                    item["loss_finite"] and item["main_grads_finite"]
                    for item in tp2["correctness_smoke"]["per_step_checks"]
                )
            ),
        },
        "fast_benchmark": {
            "protocol": {
                "warmup_iterations": tp2["warmup_iterations"],
                "measured_iterations": tp2["measured_iterations"],
                "micro_batch_size": tp2["model_config"]["micro_batch_size"],
                "tokens_per_step": tp2["tokens_per_step"],
            },
            "tp1": timing_summary(tp1),
            "tp2": timing_summary(tp2),
            "speedup": speedup,
            "throughput_gain_percent": (speedup - 1.0) * 100.0,
            "scaling_efficiency": scaling_efficiency,
            "scaling_efficiency_percent": scaling_efficiency * 100.0,
        },
        "communication_profile": communication,
        "communication_summary": {
            "nccl_ms_per_step_per_gpu": communication[
                "average_nccl_kernel_time_ms_per_step_per_gpu"
            ],
            "collectives_per_step": communication[
                "estimated_logical_collectives_per_step"
            ],
            "collective_types": communication["collective_types"],
            "average_communication_compute_overlap_percent": (
                average_overlap_percent
            ),
            "average_exposed_communication_ms_per_step": average_exposed_ms,
            "dominant_distributed_bottleneck": (
                "large activation All-Reduce traffic from row-parallel forward "
                "projections and column-parallel backward dgrad"
            ),
        },
        "decision": {
            "cuda_graph_enabled": False,
            "communication_optimization_performed": False,
            "baseline_established": True,
        },
        "environment": tp2["environment"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PHASE7_TP_ANALYSIS_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
