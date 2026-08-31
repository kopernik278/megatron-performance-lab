#!/usr/bin/env python3
"""Analyze Phase 9.2 DP=2 distributed optimizer A/B/C traces."""

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


ACCEPTED_DP2_BASELINE_TOKENS_PER_SECOND = 28936.38
ACCEPTED_DP2_BASELINE_SOURCE = (
    "Phase 9.1 FAST B (overlap_grad_reduce=True, distributed optimizer off)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--variant-a", type=Path, required=True)
    parser.add_argument("--variant-b", type=Path, required=True)
    parser.add_argument("--variant-c", type=Path, required=True)
    parser.add_argument("--sqlite-a", type=Path, required=True)
    parser.add_argument("--sqlite-b", type=Path, required=True)
    parser.add_argument("--sqlite-c", type=Path, required=True)
    parser.add_argument("--trace-a", type=Path, required=True)
    parser.add_argument("--trace-b", type=Path, required=True)
    parser.add_argument("--trace-c", type=Path, required=True)
    parser.add_argument("--formal-b", type=Path, default=None)
    parser.add_argument("--formal-c", type=Path, default=None)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def is_allreduce_kernel(name: str) -> bool:
    return "allreduce" in compact_name(name)


def is_reducescatter_kernel(name: str) -> bool:
    compact = compact_name(name)
    return "reducescatter" in compact or "reduce_scatter" in compact


def is_allgather_kernel(name: str) -> bool:
    compact = compact_name(name)
    return "allgather" in compact or "all_gather" in compact


def is_sendrecv_kernel(name: str) -> bool:
    compact = compact_name(name)
    return (
        "sendrecv" in compact
        or "ncclsend" in compact
        or "ncclrecv" in compact
    )


def classify_nccl_kernel(name: str) -> str | None:
    if "nccl" not in compact_name(name) and not is_nccl_kernel(name):
        return None
    if is_allgather_kernel(name):
        return "allgather"
    if is_reducescatter_kernel(name):
        return "reducescatter"
    if is_allreduce_kernel(name):
        return "allreduce"
    if is_sendrecv_kernel(name):
        return "sendrecv"
    return "other_nccl"


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
            "dp_start_param_sync",
        }:
            ranges[name].append((start, end))
    return ranges


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
            "reducescatter": [],
            "allgather": [],
            "grad_comm": [],
            "param_comm": [],
            "gemm": [],
        }
    )
    kernel_counts: dict[str, int] = defaultdict(int)
    kernel_ms: dict[str, float] = defaultdict(float)
    comm_class_counts: dict[str, int] = defaultdict(int)
    comm_class_ms: dict[str, float] = defaultdict(float)

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
        comm_class = classify_nccl_kernel(name)
        if comm_class is not None:
            by_device[device]["nccl"].append(interval)
            if comm_class == "allreduce":
                by_device[device]["allreduce"].append(interval)
                by_device[device]["grad_comm"].append(interval)
            elif comm_class == "reducescatter":
                by_device[device]["reducescatter"].append(interval)
                by_device[device]["grad_comm"].append(interval)
            elif comm_class == "allgather":
                by_device[device]["allgather"].append(interval)
                by_device[device]["param_comm"].append(interval)
            elif comm_class == "sendrecv":
                use_dist = bool(run.get("use_distributed_optimizer"))
                target = "grad_comm" if not use_dist else "grad_comm"
                by_device[device][target].append(interval)
            kernel_counts[name] += 1
            kernel_ms[name] += (end - start) / NS_PER_MS
            comm_class_counts[comm_class] += 1
            comm_class_ms[comm_class] += (end - start) / NS_PER_MS
        else:
            by_device[device]["compute"].append(interval)
            if is_gemm_kernel(name):
                by_device[device]["gemm"].append(interval)

    backward = merge_intervals(named.get("backward", []), window_start, window_end)
    forward = merge_intervals(named.get("forward", []), window_start, window_end)
    finalize = merge_intervals(named.get("finalize_model_grads", []), window_start, window_end)
    optimizer = merge_intervals(named.get("optimizer_step", []), window_start, window_end)
    start_grad = merge_intervals(named.get("dp_start_grad_sync", []), window_start, window_end)
    finish_grad = merge_intervals(named.get("dp_finish_grad_sync", []), window_start, window_end)
    start_param = merge_intervals(named.get("dp_start_param_sync", []), window_start, window_end)

    per_device: dict[str, Any] = {}
    overlap_percents: list[float] = []
    exposed_grad_ms: list[float] = []
    exposed_param_ms: list[float] = []
    grad_comm_ms: list[float] = []
    param_comm_ms: list[float] = []
    for device, groups in sorted(by_device.items()):
        compute = merge_intervals(groups["compute"], window_start, window_end)
        grad_comm = merge_intervals(groups["grad_comm"] or groups["allreduce"], window_start, window_end)
        param_comm = merge_intervals(groups["param_comm"] or groups["allgather"], window_start, window_end)
        all_comm = merge_intervals(groups["nccl"], window_start, window_end)
        busy = merge_intervals(groups["all"], window_start, window_end)
        grad_overlap_ns = intersection_total(grad_comm, compute)
        param_overlap_ns = intersection_total(param_comm, compute)
        all_overlap_ns = intersection_total(all_comm, compute)
        grad_comm_ns = interval_total(grad_comm)
        param_comm_ns = interval_total(param_comm)
        all_comm_ns = interval_total(all_comm)
        grad_exposed_ns = grad_comm_ns - grad_overlap_ns
        param_exposed_ns = param_comm_ns - param_overlap_ns
        overlap_percent = (100.0 * all_overlap_ns / all_comm_ns) if all_comm_ns else 0.0
        overlap_percents.append(overlap_percent)
        exposed_grad_ms.append(grad_exposed_ns / NS_PER_MS / iterations)
        exposed_param_ms.append(param_exposed_ns / NS_PER_MS / iterations)
        grad_comm_ms.append(grad_comm_ns / NS_PER_MS / iterations)
        param_comm_ms.append(param_comm_ns / NS_PER_MS / iterations)
        per_device[str(device)] = {
            "grad_comm_ms_per_step": grad_comm_ns / NS_PER_MS / iterations,
            "param_comm_ms_per_step": param_comm_ns / NS_PER_MS / iterations,
            "all_comm_ms_per_step": all_comm_ns / NS_PER_MS / iterations,
            "grad_overlap_percent": (100.0 * grad_overlap_ns / grad_comm_ns) if grad_comm_ns else 0.0,
            "param_overlap_percent": (
                100.0 * param_overlap_ns / param_comm_ns if param_comm_ns else 0.0
            ),
            "overlap_percent": overlap_percent,
            "exposed_grad_comm_ms_per_step": grad_exposed_ns / NS_PER_MS / iterations,
            "exposed_param_comm_ms_per_step": param_exposed_ns / NS_PER_MS / iterations,
            "backward_grad_comm_ms_per_step": intersection_total(grad_comm, backward)
            / NS_PER_MS
            / iterations,
            "forward_param_comm_ms_per_step": intersection_total(param_comm, forward)
            / NS_PER_MS
            / iterations,
            "optimizer_param_comm_ms_per_step": intersection_total(param_comm, optimizer)
            / NS_PER_MS
            / iterations,
        }

    top_kernels = sorted(
        (
            {"name": name, "launches": count, "total_ms": kernel_ms[name]}
            for name, count in kernel_counts.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )[:10]

    connection.close()
    return {
        "sqlite": str(sqlite_path),
        "window_ms": window_ns / NS_PER_MS,
        "measured_iterations": iterations,
        "per_device": per_device,
        "mean_grad_comm_ms_per_step": statistics.fmean(grad_comm_ms) if grad_comm_ms else 0.0,
        "mean_param_comm_ms_per_step": statistics.fmean(param_comm_ms) if param_comm_ms else 0.0,
        "mean_overlap_percent": statistics.fmean(overlap_percents) if overlap_percents else 0.0,
        "mean_exposed_grad_comm_ms_per_step": (
            statistics.fmean(exposed_grad_ms) if exposed_grad_ms else 0.0
        ),
        "mean_exposed_param_comm_ms_per_step": (
            statistics.fmean(exposed_param_ms) if exposed_param_ms else 0.0
        ),
        "comm_class_launch_count": dict(comm_class_counts),
        "comm_class_total_ms": {key: value / iterations for key, value in comm_class_ms.items()},
        "allreduce_launches_per_step": comm_class_counts.get("allreduce", 0) / iterations,
        "reducescatter_launches_per_step": comm_class_counts.get("reducescatter", 0) / iterations,
        "allgather_launches_per_step": comm_class_counts.get("allgather", 0) / iterations,
        "sendrecv_launches_per_step": comm_class_counts.get("sendrecv", 0) / iterations,
        "nvtx_ms_per_step": {
            "dp_start_grad_sync": interval_total(start_grad) / NS_PER_MS / iterations,
            "dp_finish_grad_sync": interval_total(finish_grad) / NS_PER_MS / iterations,
            "dp_start_param_sync": interval_total(start_param) / NS_PER_MS / iterations,
        },
        "top_comm_kernels": top_kernels,
    }


def pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def loss_trajectory_delta(a_losses: list[float], b_losses: list[float]) -> dict[str, float]:
    n = min(len(a_losses), len(b_losses))
    if n == 0:
        return {"max_abs_delta": 0.0, "mean_abs_delta": 0.0}
    deltas = [abs(a_losses[i] - b_losses[i]) for i in range(n)]
    return {
        "max_abs_delta": max(deltas),
        "mean_abs_delta": statistics.fmean(deltas),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a = payload["variant_a"]
    b = payload["variant_b"]
    c = payload["variant_c"]
    comp = payload["comparison"]
    traces = payload["traces"]
    infra = payload["infrastructure"]
    topo = payload["topology"]
    formal = payload.get("formal")
    lines = [
        "# Phase 9.2: DP=2 Megatron Distributed Optimizer",
        "",
        "FAST ITERATION MODE (5 warmup + 20 measured) unless a formal 20+100 cell is noted.",
        "CUDA Graph off. TP=1, PP=1, DP=2. `overlap_grad_reduce=True` on all variants.",
        "Default MCore bucket size; buckets not tuned.",
        "",
        "Pinned API (Megatron-LM `09fde85`):",
        "- `DistributedDataParallelConfig.use_distributed_optimizer` / `--use-distributed-optimizer`",
        "- `overlap_grad_reduce` / `--overlap-grad-reduce`",
        "- `overlap_param_gather` / `--overlap-param-gather` (requires distributed optimizer)",
        "",
        "A: standard FP32Optimizer + gradient All-Reduce.",
        "B/C: `DistributedOptimizer` + gradient Reduce-Scatter + parameter All-Gather.",
        "C adds `overlap_param_gather=True`.",
        "",
        "## Outcome",
        "",
        f"- Status: **{payload['status']}**",
        f"- Formal B/C 20+100: **{'yes' if payload.get('formal_validation_ran') else 'no'}**",
        f"- A→B throughput: **{comp['a_to_b_throughput_percent']:+.2f}%**",
        f"- B→C throughput: **{comp['b_to_c_throughput_percent']:+.2f}%**",
        f"- Gradient comm transform A→B: {comp['comm_transform_a_to_b']}",
        f"- Param-gather overlap B→C: {comp['param_overlap_b_to_c']}",
        "",
        "## Infrastructure",
        "",
        f"- RunPod Pod: `{infra['pod_id']}` ({infra.get('pod_status', 'unknown')})",
        f"- Data center / public IP: {infra.get('data_center')}, `{infra.get('public_ip')}`",
        f"- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, ${infra['price_per_hour_usd']:.2f}/h",
        f"- Topology path: **{topo.get('gpu0_gpu1_path')}**",
        f"- Lab commit: `{infra.get('project_commit')}`",
        "",
        "## Correctness",
        "",
        f"- Variant A optimizer state sharded: {a.get('optimizer_state_sharding', {}).get('sharded')}",
        f"- Variant B optimizer state sharded: {b.get('optimizer_state_sharding', {}).get('sharded')}",
        f"- Variant C optimizer state sharded: {c.get('optimizer_state_sharding', {}).get('sharded')}",
        f"- B optimizer bytes/rank: {b.get('optimizer_state_bytes_per_rank')}",
        f"- A optimizer bytes/rank: {a.get('optimizer_state_bytes_per_rank')}",
        f"- A↔B loss max abs delta (measured): {comp['loss_trajectory_a_vs_b']['max_abs_delta']:.4f}",
        f"- B↔C loss max abs delta (measured): {comp['loss_trajectory_b_vs_c']['max_abs_delta']:.4f}",
        "",
        "## A/B/C throughput",
        "",
        "| Variant | dist opt | param gather overlap | tok/s | step ms | per-GPU MFU | VRAM smi | opt state B/rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| A baseline | off | off | {fmt(a['tokens_per_second'])} | {fmt(a['average_step_time_ms'])} | "
        f"{a['per_gpu_mfu_percent']:.2f}% | {fmt((a.get('smi_peak_memory_mib') or [None])[0], 0)} | "
        f"{fmt((a.get('optimizer_state_bytes_per_rank') or [0])[0], 0)} |",
        f"| B dist opt | on | off | {fmt(b['tokens_per_second'])} | {fmt(b['average_step_time_ms'])} | "
        f"{b['per_gpu_mfu_percent']:.2f}% | {fmt((b.get('smi_peak_memory_mib') or [None])[0], 0)} | "
        f"{fmt((b.get('optimizer_state_bytes_per_rank') or [0])[0], 0)} |",
        f"| C dist opt + overlap | on | on | {fmt(c['tokens_per_second'])} | {fmt(c['average_step_time_ms'])} | "
        f"{c['per_gpu_mfu_percent']:.2f}% | {fmt((c.get('smi_peak_memory_mib') or [None])[0], 0)} | "
        f"{fmt((c.get('optimizer_state_bytes_per_rank') or [0])[0], 0)} |",
        "",
        "## Communication transformation",
        "",
        f"- A All-Reduce launches/step: {traces['A']['allreduce_launches_per_step']:.2f} "
        f"(RS={traces['A']['reducescatter_launches_per_step']:.2f}, "
        f"AG={traces['A']['allgather_launches_per_step']:.2f})",
        f"- B Reduce-Scatter launches/step: {traces['B']['reducescatter_launches_per_step']:.2f} "
        f"(AR={traces['B']['allreduce_launches_per_step']:.2f}, "
        f"AG={traces['B']['allgather_launches_per_step']:.2f})",
        f"- C Reduce-Scatter launches/step: {traces['C']['reducescatter_launches_per_step']:.2f} "
        f"(AG={traces['C']['allgather_launches_per_step']:.2f})",
        "",
        "### Exposed communication (ms/step/GPU)",
        "",
        f"- Gradient exposed: A {fmt(traces['A']['mean_exposed_grad_comm_ms_per_step'])} → "
        f"B {fmt(traces['B']['mean_exposed_grad_comm_ms_per_step'])}",
        f"- Param-gather exposed: B {fmt(traces['B']['mean_exposed_param_comm_ms_per_step'])} → "
        f"C {fmt(traces['C']['mean_exposed_param_comm_ms_per_step'])}",
        f"- Overall overlap %: A {traces['A']['mean_overlap_percent']:.1f}% → "
        f"B {traces['B']['mean_overlap_percent']:.1f}% → C {traces['C']['mean_overlap_percent']:.1f}%",
        "",
        "## Memory",
        "",
        f"- A peak allocated MiB/rank: {a.get('peak_allocated_memory_mib')}",
        f"- B peak allocated MiB/rank: {b.get('peak_allocated_memory_mib')}",
        f"- C peak allocated MiB/rank: {c.get('peak_allocated_memory_mib')}",
        f"- Optimizer state saving B vs A (per rank): "
        f"{comp['optimizer_state_saving_percent_b_vs_a']:+.1f}%",
        "",
    ]
    if formal:
        lines.extend(
            [
                "## Formal B/C 20+100",
                "",
                f"- B: {fmt(formal['B']['tokens_per_second'])} tok/s",
                f"- C: {fmt(formal['C']['tokens_per_second'])} tok/s",
                f"- B→C: {formal['throughput_change_percent']:+.2f}%",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            payload["decision_text"],
            "",
            "## Commands",
            "",
            "```bash",
            f"bash scripts/phase92_distopt_pod.sh {infra['pod_id']} {infra['price_per_hour_usd']}",
            "```",
            "",
            "Raw outputs: `results/phase92_work/`. Summary: `results/phase9_distributed_optimizer.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    variant_a = load(args.variant_a)
    variant_b = load(args.variant_b)
    variant_c = load(args.variant_c)
    traces = {
        "A": analyze_sqlite(args.sqlite_a, variant_a),
        "B": analyze_sqlite(args.sqlite_b, variant_b),
        "C": analyze_sqlite(args.sqlite_c, variant_c),
    }
    for label, trace_path, sqlite_path in (
        ("A", args.trace_a, args.sqlite_a),
        ("B", args.trace_b, args.sqlite_b),
        ("C", args.trace_c, args.sqlite_c),
    ):
        traces[label]["nsys"] = str(trace_path)
        traces[label]["sha256"] = sha256(sqlite_path) if sqlite_path.exists() else None

    a_tps = float(variant_a["tokens_per_second"])
    b_tps = float(variant_b["tokens_per_second"])
    c_tps = float(variant_c["tokens_per_second"])
    a_to_b = (b_tps - a_tps) / a_tps * 100.0 if a_tps else 0.0
    b_to_c = (c_tps - b_tps) / b_tps * 100.0 if b_tps else 0.0

    a_opt = sum(variant_a.get("optimizer_state_bytes_per_rank") or [0])
    b_opt = sum(variant_b.get("optimizer_state_bytes_per_rank") or [0])
    a_per_rank = (variant_a.get("optimizer_state_bytes_per_rank") or [0])[0] or 1
    b_per_rank = (variant_b.get("optimizer_state_bytes_per_rank") or [0])[0] or 1
    opt_saving = 100.0 * (1.0 - b_per_rank / a_per_rank) if a_per_rank else 0.0

    ar_a = traces["A"]["allreduce_launches_per_step"]
    rs_b = traces["B"]["reducescatter_launches_per_step"]
    ag_b = traces["B"]["allgather_launches_per_step"]
    comm_transform = (
        f"All-Reduce {ar_a:.1f}/step → RS {rs_b:.1f}/step + AG {ag_b:.1f}/step"
        if rs_b > 0 or ag_b > 0
        else "expected RS+AG not observed in B trace"
    )
    param_overlap = (
        f"{traces['B']['mean_exposed_param_comm_ms_per_step']:.2f} → "
        f"{traces['C']['mean_exposed_param_comm_ms_per_step']:.2f} ms exposed"
    )

    comparison = {
        "a_to_b_throughput_percent": a_to_b,
        "b_to_c_throughput_percent": b_to_c,
        "step_time_a_to_b_ms": float(variant_b["average_step_time_ms"])
        - float(variant_a["average_step_time_ms"]),
        "step_time_b_to_c_ms": float(variant_c["average_step_time_ms"])
        - float(variant_b["average_step_time_ms"]),
        "comm_transform_a_to_b": comm_transform,
        "param_overlap_b_to_c": param_overlap,
        "loss_trajectory_a_vs_b": loss_trajectory_delta(
            variant_a.get("measured_losses") or [], variant_b.get("measured_losses") or []
        ),
        "loss_trajectory_b_vs_c": loss_trajectory_delta(
            variant_b.get("measured_losses") or [], variant_c.get("measured_losses") or []
        ),
        "optimizer_state_saving_percent_b_vs_a": opt_saving,
        "optimizer_state_bytes_total_a": a_opt,
        "optimizer_state_bytes_total_b": b_opt,
    }

    b_corr = variant_b.get("correctness") or {}
    correctness_ok = all(
        bool(b_corr.get(key))
        for key in (
            "both_ranks_initialized",
            "finite_loss",
            "no_deadlock",
            "parameters_sharded_after_optimizer",
        )
    )
    formal_ran = args.formal_b is not None and args.formal_c is not None
    if correctness_ok and b_to_c >= 2.0 and not formal_ran:
        decision = (
            f"Correctness passed. FAST B→C gain {b_to_c:+.2f}% met 2% gate but formal files missing."
        )
        status = "success"
    elif correctness_ok and b_to_c >= 2.0:
        decision = (
            f"Correctness passed. FAST B→C {b_to_c:+.2f}% met 2% gate; formal B/C 20+100 ran. "
            f"A→B dist-opt effect {a_to_b:+.2f}%."
        )
        status = "success"
    elif correctness_ok:
        decision = (
            f"Correctness passed. FAST B→C {b_to_c:+.2f}% below 2%; formal B/C skipped. "
            f"A→B dist-opt effect {a_to_b:+.2f}%."
        )
        status = "success"
    else:
        decision = "Correctness failed; do not interpret throughput."
        status = "failed"

    env = variant_b.get("environment") or {}
    formal_payload = None
    if formal_ran:
        formal_b = load(args.formal_b)
        formal_c = load(args.formal_c)
        formal_gain = (
            (formal_c["tokens_per_second"] - formal_b["tokens_per_second"])
            / formal_b["tokens_per_second"]
            * 100.0
        )
        formal_payload = {
            "B": {
                "tokens_per_second": formal_b["tokens_per_second"],
                "average_step_time_ms": formal_b["average_step_time_ms"],
            },
            "C": {
                "tokens_per_second": formal_c["tokens_per_second"],
                "average_step_time_ms": formal_c["average_step_time_ms"],
            },
            "throughput_change_percent": formal_gain,
        }

    payload = {
        "status": status,
        "experiment": "Phase 9.2 DP=2 Megatron Distributed Optimizer",
        "iteration_mode": "FAST ITERATION MODE",
        "mechanism": (
            "DistributedDataParallelConfig.use_distributed_optimizer + overlap_param_gather"
        ),
        "formal_validation_ran": formal_ran,
        "accepted_dp2_baseline": {
            "tokens_per_second": ACCEPTED_DP2_BASELINE_TOKENS_PER_SECOND,
            "source": ACCEPTED_DP2_BASELINE_SOURCE,
        },
        "infrastructure": {
            "pod_id": args.pod_id,
            "price_per_hour_usd": args.price_per_hour_usd,
            "pod_status": "deleted",
            "public_ip": topology.get("public_ip") or topology.get("host_public_ip"),
            "data_center": topology.get("data_center") or topology.get("data_center_id"),
            "project_commit": env.get("project_commit"),
            "megatron_lm_commit": env.get("megatron_lm_commit"),
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
        "variant_c": variant_c,
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
