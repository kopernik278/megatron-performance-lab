#!/usr/bin/env python3
"""Analyze Phase 9.1 DP=2 gradient All-Reduce overlap A/B traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase7_analyze_tp import (
    NS_PER_MS,
    intersection_total,
    interval_total,
    is_gemm_kernel,
    is_nccl_kernel,
    kernel_name,
    merge_intervals,
    nvtx_name,
    sha256,
    table_columns,
    table_exists,
)


ACCEPTED_DP1_TOKENS_PER_SECOND = 15801.942
ACCEPTED_DP1_SOURCE = "Phase 5.2 formal B (bias_dropout_fusion=True, MB=8, DP=1)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--dp1", type=Path, default=None)
    parser.add_argument("--variant-a", type=Path, required=True)
    parser.add_argument("--variant-b", type=Path, required=True)
    parser.add_argument("--sqlite-a", type=Path, required=True)
    parser.add_argument("--sqlite-b", type=Path, required=True)
    parser.add_argument("--trace-a", type=Path, required=True)
    parser.add_argument("--trace-b", type=Path, required=True)
    parser.add_argument("--formal-a", type=Path, default=None)
    parser.add_argument("--formal-b", type=Path, default=None)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_allreduce_kernel(name: str) -> bool:
    compact = name.lower().replace("-", "").replace("_", "")
    if "allreduce" in compact:
        return True
    return False


def is_dp_comm_kernel(name: str) -> bool:
    """DP=2 NCCL All-Reduce is often implemented as SendRecv on two GPUs."""

    lower = name.lower()
    compact = lower.replace("-", "").replace("_", "")
    if "nccl" not in compact:
        return False
    if any(token in compact for token in ("allgather", "reducescatter", "broadcast")):
        return False
    return (
        "allreduce" in compact
        or "sendrecv" in compact
        or "ncclsend" in compact
        or "ncclrecv" in compact
    )


def profile_window(connection: sqlite3.Connection) -> tuple[int, int]:
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
    return min(int(row[0]) for row in windows), max(int(row[1]) for row in windows)


def collect_named_ranges(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT start, end, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
        """,
        (window_start, window_end),
    ):
        name = nvtx_name(row, strings)
        start = max(int(row["start"]), window_start)
        end = min(int(row["end"]), window_end)
        if end <= start:
            continue
        if name.startswith("train_step_"):
            ranges["train_step"].append((start, end))
        elif name in {
            "backward",
            "forward",
            "finalize_model_grads",
            "optimizer_step",
            "dp_start_grad_sync",
            "dp_finish_grad_sync",
        }:
            ranges[name].append((start, end))
    return ranges


def runtime_sync_ms(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
    iterations: int,
) -> dict[str, float]:
    if not table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return {
            "cuda_synchronize_ms_per_step": 0.0,
            "cuda_synchronize_count_per_step": 0.0,
        }
    columns = table_columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME")
    name_field = "nameId" if "nameId" in columns else ("name" if "name" in columns else None)
    if name_field is None:
        return {
            "cuda_synchronize_ms_per_step": 0.0,
            "cuda_synchronize_count_per_step": 0.0,
        }
    total_ns = 0
    count = 0
    query = f"""
        SELECT start, end, {name_field}
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE end > ? AND start < ?
    """
    for row in connection.execute(query, (window_start, window_end)):
        raw = row[name_field]
        name = strings.get(int(raw), str(raw)) if isinstance(raw, int) else str(raw or "")
        compact = name.lower().replace("_", "")
        if "synchronize" not in compact:
            continue
        start = max(int(row["start"]), window_start)
        end = min(int(row["end"]), window_end)
        if end <= start:
            continue
        total_ns += end - start
        count += 1
    return {
        "cuda_synchronize_ms_per_step": total_ns / NS_PER_MS / max(iterations, 1),
        "cuda_synchronize_count_per_step": count / max(iterations, 1),
    }


def analyze_sqlite(sqlite_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    strings = {
        int(row["id"]): str(row["value"])
        for row in connection.execute("SELECT id, value FROM StringIds")
    }
    window_start, window_end = profile_window(connection)
    window_ns = window_end - window_start
    named = collect_named_ranges(connection, strings, window_start, window_end)
    iterations = len(named.get("train_step", [])) or int(run.get("measured_iterations") or 1)

    kernel_columns = table_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
    selected = ["start", "end"]
    selected.extend(
        field
        for field in ("demangledName", "shortName", "mangledName", "deviceId")
        if field in kernel_columns
    )
    kernels = list(
        connection.execute(
            f"""
            SELECT {', '.join(selected)}
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )

    by_device: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {
            "all": [],
            "compute": [],
            "nccl": [],
            "allreduce": [],
            "dp_comm": [],
            "gemm": [],
        }
    )
    kernel_counts: dict[str, int] = defaultdict(int)
    kernel_ms: dict[str, float] = defaultdict(float)
    for row in kernels:
        name = kernel_name(row, strings)
        start = max(int(row["start"]), window_start)
        end = min(int(row["end"]), window_end)
        if end <= start:
            continue
        device = (
            int(row["deviceId"])
            if "deviceId" in row.keys() and row["deviceId"] is not None
            else 0
        )
        interval = (start, end)
        by_device[device]["all"].append(interval)
        if is_dp_comm_kernel(name) or is_nccl_kernel(name):
            by_device[device]["nccl"].append(interval)
            if is_dp_comm_kernel(name):
                by_device[device]["dp_comm"].append(interval)
            if is_allreduce_kernel(name):
                by_device[device]["allreduce"].append(interval)
            kernel_counts[name] += 1
            kernel_ms[name] += (end - start) / NS_PER_MS
        else:
            by_device[device]["compute"].append(interval)
            if is_gemm_kernel(name):
                by_device[device]["gemm"].append(interval)

    backward = merge_intervals(named.get("backward", []), window_start, window_end)
    finalize = merge_intervals(named.get("finalize_model_grads", []), window_start, window_end)
    start_sync = merge_intervals(named.get("dp_start_grad_sync", []), window_start, window_end)
    finish_sync = merge_intervals(named.get("dp_finish_grad_sync", []), window_start, window_end)

    per_device: dict[str, Any] = {}
    overlap_percents: list[float] = []
    exposed_ms: list[float] = []
    comm_ms: list[float] = []
    backward_comm_ms: list[float] = []
    finalize_comm_ms: list[float] = []
    for device, groups in sorted(by_device.items()):
        compute = merge_intervals(groups["compute"], window_start, window_end)
        dp_comm = merge_intervals(groups["dp_comm"] or groups["nccl"], window_start, window_end)
        busy = merge_intervals(groups["all"], window_start, window_end)
        overlap_ns = intersection_total(dp_comm, compute)
        comm_ns = interval_total(dp_comm)
        exposed_ns = comm_ns - overlap_ns
        overlap_percent = (100.0 * overlap_ns / comm_ns) if comm_ns else 0.0
        overlap_percents.append(overlap_percent)
        exposed_ms.append(exposed_ns / NS_PER_MS / iterations)
        comm_ms.append(comm_ns / NS_PER_MS / iterations)
        backward_comm_ms.append(
            intersection_total(dp_comm, backward) / NS_PER_MS / iterations
        )
        finalize_comm_ms.append(
            intersection_total(dp_comm, finalize) / NS_PER_MS / iterations
        )
        idle_ns = window_ns - interval_total(busy)
        per_device[str(device)] = {
            "busy_ms_per_step": interval_total(busy) / NS_PER_MS / iterations,
            "idle_ms_per_step": idle_ns / NS_PER_MS / iterations,
            "idle_fraction": idle_ns / window_ns if window_ns else 0.0,
            "compute_ms_per_step": interval_total(compute) / NS_PER_MS / iterations,
            "gemm_ms_per_step": interval_total(
                merge_intervals(groups["gemm"], window_start, window_end)
            )
            / NS_PER_MS
            / iterations,
            "dp_comm_ms_per_step": comm_ns / NS_PER_MS / iterations,
            "allreduce_named_ms_per_step": interval_total(
                merge_intervals(groups["allreduce"], window_start, window_end)
            )
            / NS_PER_MS
            / iterations,
            "overlap_ms_per_step": overlap_ns / NS_PER_MS / iterations,
            "overlap_percent": overlap_percent,
            "exposed_comm_ms_per_step": exposed_ns / NS_PER_MS / iterations,
            "backward_comm_ms_per_step": backward_comm_ms[-1],
            "finalize_comm_ms_per_step": finalize_comm_ms[-1],
        }

    top_kernels = sorted(
        (
            {"name": name, "launches": count, "total_ms": kernel_ms[name]}
            for name, count in kernel_counts.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )[:8]
    named_allreduce_launches = sum(
        count for name, count in kernel_counts.items() if is_allreduce_kernel(name)
    )
    dp_comm_launches = sum(kernel_counts.values())
    sync = runtime_sync_ms(connection, strings, window_start, window_end, iterations)
    connection.close()
    return {
        "sqlite": str(sqlite_path),
        "window_ms": window_ns / NS_PER_MS,
        "measured_iterations": iterations,
        "per_device": per_device,
        "mean_dp_comm_ms_per_step": statistics.fmean(comm_ms) if comm_ms else 0.0,
        "mean_overlap_percent": statistics.fmean(overlap_percents) if overlap_percents else 0.0,
        "mean_exposed_comm_ms_per_step": statistics.fmean(exposed_ms) if exposed_ms else 0.0,
        "mean_backward_comm_ms_per_step": (
            statistics.fmean(backward_comm_ms) if backward_comm_ms else 0.0
        ),
        "mean_finalize_comm_ms_per_step": (
            statistics.fmean(finalize_comm_ms) if finalize_comm_ms else 0.0
        ),
        "nvtx_ms_per_step": {
            "backward": interval_total(backward) / NS_PER_MS / iterations,
            "finalize_model_grads": interval_total(finalize) / NS_PER_MS / iterations,
            "dp_start_grad_sync": interval_total(start_sync) / NS_PER_MS / iterations,
            "dp_finish_grad_sync": interval_total(finish_sync) / NS_PER_MS / iterations,
            "optimizer_step": interval_total(
                merge_intervals(named.get("optimizer_step", []), window_start, window_end)
            )
            / NS_PER_MS
            / iterations,
        },
        "allreduce_named_launch_count": named_allreduce_launches,
        "allreduce_named_launches_per_step": named_allreduce_launches / iterations,
        "dp_comm_launch_count": dp_comm_launches,
        "dp_comm_launches_per_step": dp_comm_launches / iterations,
        "top_dp_comm_kernels": top_kernels,
        "cpu_gpu_sync": sync,
        "note": (
            "On 2-GPU PCIe, NCCL All-Reduce often appears as ncclDevKernel_SendRecv. "
            "dp_comm includes named AllReduce plus NCCL SendRecv used for gradient sync."
        ),
    }


def pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a = payload["variant_a"]
    b = payload["variant_b"]
    comparison = payload["comparison"]
    traces = payload["traces"]
    dp1 = payload.get("same_host_dp1")
    accepted = payload["accepted_dp1"]
    infra = payload["infrastructure"]
    topo = payload["topology"]
    formal = payload.get("formal")
    lines = [
        "# Phase 9.1: DP=2 gradient All-Reduce overlap",
        "",
        "FAST ITERATION MODE (5 warmup + 20 measured) unless a formal 20+100 cell is noted.",
        "CUDA Graph stayed off. Distributed optimizer stayed off. TP=1, PP=1. Bucket size was",
        "the MCore default (`max(40000000, 1000000 * dp_size)`); buckets were not tuned.",
        "",
        "Pinned API (Megatron-LM `09fde85`): `DistributedDataParallelConfig.overlap_grad_reduce`",
        "and CLI `--overlap-grad-reduce`. With `use_distributed_optimizer=False` this issues",
        "gradient **All-Reduce**, not ReduceScatter. `overlap_param_gather` stayed false.",
        "",
        "A→B isolates gradient-communication overlap. Weak scaling vs DP=1 is reported separately.",
        "",
        "## Outcome",
        "",
        f"- Status: **{payload['status']}**",
        f"- Formal 20+100: **{'yes' if payload.get('formal_validation_ran') else 'no'}**",
        f"- A→B throughput: **{comparison['throughput_change_percent']:+.2f}%** "
        f"({fmt(a['tokens_per_second'])} → {fmt(b['tokens_per_second'])} tok/s)",
        f"- Overlap (B, DP comm ∩ compute): **{traces['B']['mean_overlap_percent']:.1f}%**",
        f"- Exposed DP comm: {fmt(traces['A']['mean_exposed_comm_ms_per_step'])} → "
        f"{fmt(traces['B']['mean_exposed_comm_ms_per_step'])} ms/step/GPU",
        f"- Dominant remaining bottleneck: {comparison['dominant_remaining_bottleneck']}",
        "",
        "## Infrastructure",
        "",
        f"- RunPod Pod: `{infra['pod_id']}` ({infra.get('pod_status', 'unknown')})",
        f"- Data center / public IP: {infra.get('data_center')}, `{infra.get('public_ip')}`",
        f"- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, ${infra['price_per_hour_usd']:.2f}/h",
        f"- Topology path: **{topo.get('gpu0_gpu1_path')}**, same-NUMA={topo.get('same_numa')}",
        f"- CUDA peer access bidirectional: {topo.get('p2p_bidirectional')}",
        f"- NCCL All-Reduce sanity: {topo.get('nccl_all_reduce_passed')}",
        "- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`",
        "- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`",
        f"- Lab commit on the Pod: `{infra.get('project_commit')}`",
        f"- `CUDA_DEVICE_MAX_CONNECTIONS={infra.get('cuda_device_max_connections', '8')}`. "
        "`NCCL_P2P_DISABLE` unset.",
        "",
        "## Correctness (variant B)",
        "",
    ]
    b_corr = b.get("correctness") or {}
    lines.extend(
        [
            f"- Both ranks initialized: {b_corr.get('both_ranks_initialized')}",
            f"- Different data per DP rank: {b_corr.get('different_data_per_dp_rank')}",
            f"- Forward/backward/`main_grad`/optimizer: "
            f"{b_corr.get('forward')}/{b_corr.get('backward')}/"
            f"{b_corr.get('main_grad')}/{b_corr.get('optimizer')}",
            f"- Gradients synchronized: {b_corr.get('gradients_synchronized')}",
            f"- Parameters identical after optimizer: "
            f"{b_corr.get('parameters_identical_after_optimizer')}",
            f"- Finite loss / no NaN-Inf / no deadlock: "
            f"{b_corr.get('finite_loss')}/{b_corr.get('no_nan_inf')}/{b_corr.get('no_deadlock')}",
            "",
            "## DP=1 vs DP=2 throughput",
            "",
            f"- Accepted DP=1 (published, {accepted['source']}): "
            f"**{fmt(accepted['tokens_per_second'])} tok/s**",
        ]
    )
    if dp1 is not None:
        lines.append(
            f"- Same-host DP=1 FAST: **{fmt(dp1['tokens_per_second'])} tok/s**, "
            f"{fmt(dp1['average_step_time_ms'])} ms, MFU {dp1['mfu_percent']:.2f}%"
        )
    lines.extend(
        [
            f"- DP=2 A (overlap off): **{fmt(a['tokens_per_second'])} tok/s**",
            f"- DP=2 B (overlap on): **{fmt(b['tokens_per_second'])} tok/s**",
            f"- Weak-scaling efficiency vs accepted DP=1 (A): "
            f"{pct(comparison['weak_scaling_efficiency_vs_accepted_a_percent'])}",
            f"- Weak-scaling efficiency vs accepted DP=1 (B): "
            f"{pct(comparison['weak_scaling_efficiency_vs_accepted_b_percent'])}",
        ]
    )
    if comparison.get("weak_scaling_efficiency_vs_same_host_a_percent") is not None:
        lines.extend(
            [
                f"- Weak-scaling efficiency vs same-host DP=1 (A): "
                f"{pct(comparison['weak_scaling_efficiency_vs_same_host_a_percent'])}",
                f"- Weak-scaling efficiency vs same-host DP=1 (B): "
                f"{pct(comparison['weak_scaling_efficiency_vs_same_host_b_percent'])}",
            ]
        )
    lines.extend(
        [
            "",
            "Weak-scaling efficiency = DP2 global tok/s / (2 × DP1 tok/s).",
            "",
            "## A/B performance",
            "",
            "| Variant | overlap_grad_reduce | tok/s | step ms | per-GPU MFU | VRAM smi |",
            "|---|---|---:|---:|---:|---:|",
            f"| A DP=2 overlap off | False | {fmt(a['tokens_per_second'])} | "
            f"{fmt(a['average_step_time_ms'])} | {a['per_gpu_mfu_percent']:.2f}% | "
            f"{fmt((a.get('smi_peak_memory_mib') or [None])[0], 0)} |",
            f"| B DP=2 overlap on | True | {fmt(b['tokens_per_second'])} | "
            f"{fmt(b['average_step_time_ms'])} | {b['per_gpu_mfu_percent']:.2f}% | "
            f"{fmt((b.get('smi_peak_memory_mib') or [None])[0], 0)} |",
            "",
            "## Gradient communication",
            "",
            f"- Named All-Reduce launches/step: "
            f"{traces['A']['allreduce_named_launches_per_step']:.2f} → "
            f"{traces['B']['allreduce_named_launches_per_step']:.2f}",
            f"- DP comm launches/step (AllReduce+SendRecv): "
            f"{traces['A']['dp_comm_launches_per_step']:.2f} → "
            f"{traces['B']['dp_comm_launches_per_step']:.2f}",
            f"- DP comm time: {fmt(traces['A']['mean_dp_comm_ms_per_step'])} → "
            f"{fmt(traces['B']['mean_dp_comm_ms_per_step'])} ms/step/GPU",
            f"- Comm during backward: {fmt(traces['A']['mean_backward_comm_ms_per_step'])} → "
            f"{fmt(traces['B']['mean_backward_comm_ms_per_step'])} ms/step/GPU",
            f"- Comm during finalize: {fmt(traces['A']['mean_finalize_comm_ms_per_step'])} → "
            f"{fmt(traces['B']['mean_finalize_comm_ms_per_step'])} ms/step/GPU",
            f"- Overlap %: {traces['A']['mean_overlap_percent']:.1f}% → "
            f"{traces['B']['mean_overlap_percent']:.1f}%",
            f"- Exposed comm: {fmt(traces['A']['mean_exposed_comm_ms_per_step'])} → "
            f"{fmt(traces['B']['mean_exposed_comm_ms_per_step'])} ms/step/GPU",
            f"- Bucket count A/B: {a.get('buckets', {}).get('bucket_count')} / "
            f"{b.get('buckets', {}).get('bucket_count')}",
            f"- Effective bucket size A/B: {a.get('buckets', {}).get('effective_bucket_size')} / "
            f"{b.get('buckets', {}).get('effective_bucket_size')}",
            "",
            "### Buckets (B)",
            "",
        ]
    )
    for bucket in (b.get("buckets") or {}).get("buckets") or []:
        lines.append(
            f"- buffer {bucket.get('buffer_index')} bucket {bucket.get('bucket_index')}: "
            f"{bucket.get('numel_unpadded')} unpadded elems, "
            f"{bucket.get('padded_numel')} padded, "
            f"{bucket.get('bytes')} bytes, "
            f"{bucket.get('param_count')} params, dtype={bucket.get('grad_dtype')}"
        )
    if formal:
        lines.extend(
            [
                "",
                "## Formal 20+100",
                "",
                f"- A: {fmt(formal['A']['tokens_per_second'])} tok/s, "
                f"{fmt(formal['A']['average_step_time_ms'])} ms",
                f"- B: {fmt(formal['B']['tokens_per_second'])} tok/s, "
                f"{fmt(formal['B']['average_step_time_ms'])} ms",
                f"- A→B: {formal['throughput_change_percent']:+.2f}%",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision_text"],
            "",
            "## Commands",
            "",
            "```bash",
            f"bash scripts/phase9_dp_pod.sh {infra['pod_id']} {infra['price_per_hour_usd']}",
            "```",
            "",
            "Raw outputs: `results/phase91_work/`. Summary: `results/phase9_dp2_grad_overlap.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dominant_bottleneck(traces: dict[str, Any], comparison: dict[str, Any]) -> str:
    b = traces["B"]
    exposed = b["mean_exposed_comm_ms_per_step"]
    overlap = b["mean_overlap_percent"]
    if comparison["throughput_change_percent"] < 2.0 and overlap < 20.0:
        return (
            "gradient All-Reduce still mostly exposed; overlap_grad_reduce did not hide "
            "communication behind backward compute on this host"
        )
    if exposed > 20.0:
        return (
            "exposed gradient All-Reduce still large after overlap; default 40M-parameter "
            "buckets were not tuned"
        )
    return "non-communication step time (compute / optimizer); remaining exposed DP All-Reduce"


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    variant_a = load(args.variant_a)
    variant_b = load(args.variant_b)
    dp1 = load(args.dp1) if args.dp1 and args.dp1.exists() else None
    traces = {
        "A": analyze_sqlite(args.sqlite_a, variant_a),
        "B": analyze_sqlite(args.sqlite_b, variant_b),
    }
    traces["A"]["nsys"] = str(args.trace_a)
    traces["B"]["nsys"] = str(args.trace_b)
    traces["A"]["sha256"] = sha256(args.sqlite_a) if args.sqlite_a.exists() else None
    traces["B"]["sha256"] = sha256(args.sqlite_b) if args.sqlite_b.exists() else None

    a_tps = float(variant_a["tokens_per_second"])
    b_tps = float(variant_b["tokens_per_second"])
    throughput_change_percent = (b_tps - a_tps) / a_tps * 100.0 if a_tps else 0.0
    accepted_dp1 = ACCEPTED_DP1_TOKENS_PER_SECOND
    weak_a_accepted = 100.0 * a_tps / (2.0 * accepted_dp1)
    weak_b_accepted = 100.0 * b_tps / (2.0 * accepted_dp1)
    same_host_a = None
    same_host_b = None
    if dp1 is not None:
        dp1_tps = float(dp1["tokens_per_second"])
        same_host_a = 100.0 * a_tps / (2.0 * dp1_tps)
        same_host_b = 100.0 * b_tps / (2.0 * dp1_tps)

    comparison = {
        "throughput_change_percent": throughput_change_percent,
        "step_time_change_ms": float(variant_b["average_step_time_ms"])
        - float(variant_a["average_step_time_ms"]),
        "overlap_percent_a": traces["A"]["mean_overlap_percent"],
        "overlap_percent_b": traces["B"]["mean_overlap_percent"],
        "exposed_comm_reduction_ms": traces["A"]["mean_exposed_comm_ms_per_step"]
        - traces["B"]["mean_exposed_comm_ms_per_step"],
        "weak_scaling_efficiency_vs_accepted_a_percent": weak_a_accepted,
        "weak_scaling_efficiency_vs_accepted_b_percent": weak_b_accepted,
        "weak_scaling_efficiency_vs_same_host_a_percent": same_host_a,
        "weak_scaling_efficiency_vs_same_host_b_percent": same_host_b,
    }
    comparison["dominant_remaining_bottleneck"] = dominant_bottleneck(traces, comparison)

    b_corr = variant_b.get("correctness") or {}
    correctness_ok = all(
        bool(b_corr.get(key))
        for key in (
            "both_ranks_initialized",
            "different_data_per_dp_rank",
            "backward",
            "main_grad",
            "gradients_synchronized",
            "parameters_identical_after_optimizer",
            "finite_loss",
            "no_deadlock",
        )
    )
    formal_ran = args.formal_a is not None and args.formal_b is not None
    if correctness_ok and throughput_change_percent >= 2.0 and not formal_ran:
        decision = (
            f"Correctness passed and FAST gain {throughput_change_percent:+.2f}% is >=2%, "
            "but formal 20+100 files were not provided."
        )
        status = "success"
    elif correctness_ok and throughput_change_percent >= 2.0:
        decision = (
            f"Correctness passed. FAST gain {throughput_change_percent:+.2f}% met the 2% gate; "
            "formal 20+100 ran. Overlap_grad_reduce is the isolated A→B variable."
        )
        status = "success"
    elif correctness_ok:
        decision = (
            f"Correctness passed but FAST gain {throughput_change_percent:+.2f}% is below 2%, "
            "so formal 20+100 did not run. Keep the FAST screen. "
            "overlap_grad_reduce is not accepted as a throughput win on this host."
        )
        status = "success"
    else:
        decision = "Correctness failed; do not interpret throughput."
        status = "failed"

    env = variant_b.get("environment") or {}
    formal_payload = None
    if formal_ran:
        formal_a = load(args.formal_a)
        formal_b = load(args.formal_b)
        formal_gain = (
            (formal_b["tokens_per_second"] - formal_a["tokens_per_second"])
            / formal_a["tokens_per_second"]
            * 100.0
        )
        formal_payload = {
            "A": {
                "tokens_per_second": formal_a["tokens_per_second"],
                "average_step_time_ms": formal_a["average_step_time_ms"],
            },
            "B": {
                "tokens_per_second": formal_b["tokens_per_second"],
                "average_step_time_ms": formal_b["average_step_time_ms"],
            },
            "throughput_change_percent": formal_gain,
        }

    payload = {
        "status": status,
        "experiment": "Phase 9.1 DP=2 gradient-communication overlap",
        "iteration_mode": "FAST ITERATION MODE",
        "mechanism": "MCore DistributedDataParallelConfig.overlap_grad_reduce",
        "isolation_caveat": (
            "A→B changes only overlap_grad_reduce. Weak scaling vs DP=1 is a separate comparison."
        ),
        "formal_validation_ran": formal_ran,
        "accepted_dp1": {
            "tokens_per_second": accepted_dp1,
            "source": ACCEPTED_DP1_SOURCE,
        },
        "same_host_dp1": dp1,
        "infrastructure": {
            "pod_id": args.pod_id,
            "price_per_hour_usd": args.price_per_hour_usd,
            "pod_status": "deleted",
            "public_ip": topology.get("public_ip") or topology.get("host_public_ip"),
            "data_center": topology.get("data_center") or topology.get("data_center_id"),
            "project_commit": env.get("project_commit"),
            "megatron_lm_commit": env.get("megatron_lm_commit"),
            "pytorch": env.get("pytorch"),
            "cuda_runtime": env.get("cuda_runtime"),
            "nccl": env.get("nccl"),
            "cuda_device_max_connections": env.get("cuda_device_max_connections"),
        },
        "topology": {
            "gpu0_gpu1_path": topology.get("gpu0_gpu1_path"),
            "same_numa": topology.get("same_numa_acceptable"),
            "p2p_bidirectional": (topology.get("p2p_accessibility") or {}).get(
                "bidirectional_gpu0_gpu1"
            ),
            "nccl_all_reduce_passed": (topology.get("nccl_all_reduce_sanity") or {}).get(
                "passed"
            ),
        },
        "variant_a": variant_a,
        "variant_b": variant_b,
        "traces": traces,
        "comparison": comparison,
        "formal": formal_payload,
        "decision_text": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown, payload)


if __name__ == "__main__":
    main()
