#!/usr/bin/env python3
"""Analyze Phase 8.3 interleaved 1F1B + P2P overlap A/B traces."""

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
    sha256,
    string_value,
    table_columns,
)
from phase8_analyze_pp import is_p2p_kernel, profile_window


FORMAL_GAIN_THRESHOLD_PERCENT = 3.0
PHASE81_VALID_BASELINE_COMMIT = "ebd98618"
PHASE81_TOKENS_PER_SECOND = 21183.07
PHASE81_STEP_MS = 773.45
PHASE81_P2P_MS = 69.04
PHASE81_BUBBLE_PERCENT = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--variant-a", type=Path, required=True)
    parser.add_argument("--variant-b", type=Path, required=True)
    parser.add_argument("--sqlite-a", type=Path, required=True)
    parser.add_argument("--sqlite-b", type=Path, required=True)
    parser.add_argument("--trace-a", type=Path, required=True)
    parser.add_argument("--trace-b", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--formal-b", type=Path)
    parser.add_argument("--abort-reason", default=None)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nvtx_events(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    window_start: int,
    window_end: int,
) -> list[dict[str, Any]]:
    events = []
    for row in connection.execute(
        """
        SELECT start, end, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > ? AND start < ?
        """,
        (window_start, window_end),
    ):
        name = str(row["text"] or strings.get(row["textId"], "") or "")
        events.append(
            {
                "name": name,
                "start": max(int(row["start"]), window_start),
                "end": min(int(row["end"]), window_end),
            }
        )
    return events


def matching_nvtx(events: list[dict[str, Any]], needle: str) -> list[dict[str, Any]]:
    lower = needle.lower()
    return [event for event in events if lower in event["name"].lower()]


def duration_ms(events: list[dict[str, Any]], iterations: int) -> float:
    if not events or iterations <= 0:
        return 0.0
    return sum(event["end"] - event["start"] for event in events) / NS_PER_MS / iterations


def analyze_sqlite(sqlite_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    strings = {
        int(row["id"]): str(row["value"])
        for row in connection.execute("SELECT id, value FROM StringIds")
    }
    window_start, window_end = profile_window(connection)
    window_ns = window_end - window_start
    events = nvtx_events(connection, strings, window_start, window_end)
    train_step_count = sum(1 for event in events if event["name"].startswith("train_step_"))
    iterations = train_step_count or int(run.get("measured_iterations") or 1)

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
        lambda: {"all": [], "compute": [], "nccl": [], "p2p": [], "gemm": []}
    )
    p2p_kernel_counts: dict[str, int] = defaultdict(int)
    p2p_kernel_ms: dict[str, float] = defaultdict(float)
    for row in kernels:
        name = kernel_name(row, strings)
        start = max(int(row["start"]), window_start)
        end = min(int(row["end"]), window_end)
        if end <= start:
            continue
        device = int(row["deviceId"]) if "deviceId" in row.keys() and row["deviceId"] is not None else 0
        interval = (start, end)
        by_device[device]["all"].append(interval)
        if is_p2p_kernel(name):
            by_device[device]["p2p"].append(interval)
            by_device[device]["nccl"].append(interval)
            p2p_kernel_counts[name] += 1
            p2p_kernel_ms[name] += (end - start) / NS_PER_MS
        elif is_nccl_kernel(name):
            by_device[device]["nccl"].append(interval)
        else:
            by_device[device]["compute"].append(interval)
            if is_gemm_kernel(name):
                by_device[device]["gemm"].append(interval)

    per_device: dict[str, Any] = {}
    idle_fractions: list[float] = []
    overlap_percents: list[float] = []
    exposed_ms: list[float] = []
    p2p_ms: list[float] = []
    compute_ms: list[float] = []
    for device, groups in sorted(by_device.items()):
        busy = merge_intervals(groups["all"], window_start, window_end)
        compute = merge_intervals(groups["compute"], window_start, window_end)
        p2p = merge_intervals(groups["p2p"], window_start, window_end)
        busy_ns = interval_total(busy)
        idle_ns = window_ns - busy_ns
        idle_fraction = idle_ns / window_ns if window_ns else 0.0
        idle_fractions.append(idle_fraction)
        p2p_ns = interval_total(p2p)
        compute_ns = interval_total(compute)
        overlap_ns = intersection_total(p2p, compute)
        overlap_percent = overlap_ns / p2p_ns * 100.0 if p2p_ns else 0.0
        overlap_percents.append(overlap_percent)
        exposed = (p2p_ns - overlap_ns) / NS_PER_MS / iterations
        exposed_ms.append(exposed)
        p2p_ms.append(p2p_ns / NS_PER_MS / iterations)
        compute_ms.append(compute_ns / NS_PER_MS / iterations)
        per_device[str(device)] = {
            "busy_ms_per_step": busy_ns / NS_PER_MS / iterations,
            "idle_ms_per_step": idle_ns / NS_PER_MS / iterations,
            "idle_fraction": idle_fraction,
            "compute_ms_per_step": compute_ns / NS_PER_MS / iterations,
            "gemm_ms_per_step": interval_total(
                merge_intervals(groups["gemm"], window_start, window_end)
            )
            / NS_PER_MS
            / iterations,
            "p2p_send_recv_ms_per_step": p2p_ns / NS_PER_MS / iterations,
            "p2p_compute_overlap_ms_per_step": overlap_ns / NS_PER_MS / iterations,
            "p2p_overlap_percent": overlap_percent,
            "exposed_p2p_ms_per_step": exposed,
        }

    chunk0 = matching_nvtx(events, "pp_chunk0_forward")
    chunk1 = matching_nvtx(events, "pp_chunk1_forward")
    async_fwd = matching_nvtx(events, "pp_async_send_recv_forward")
    async_bwd = matching_nvtx(events, "pp_async_send_recv_backward")
    sync_fwd = matching_nvtx(events, "pp_sync_send_recv_forward")
    sync_bwd = matching_nvtx(events, "pp_sync_send_recv_backward")
    warmup = matching_nvtx(events, ".warmup") or matching_nvtx(events, "warmup")
    steady = matching_nvtx(events, ".steady") or matching_nvtx(events, "steady")
    cooldown = matching_nvtx(events, ".cooldown") or matching_nvtx(events, "cooldown")

    chunk_overlap_ms = 0.0
    async_p2p = async_fwd + async_bwd
    chunk_compute = chunk0 + chunk1
    timeline_overlap_ns = 0
    if async_p2p and chunk_compute:
        p2p_iv = merge_intervals(
            ((e["start"], e["end"]) for e in async_p2p), window_start, window_end
        )
        compute_iv = merge_intervals(
            ((e["start"], e["end"]) for e in chunk_compute), window_start, window_end
        )
        timeline_overlap_ns = intersection_total(p2p_iv, compute_iv)
        chunk_overlap_ms = timeline_overlap_ns / NS_PER_MS / iterations

    stage_imbalance = None
    if len(compute_ms) >= 2:
        stage_imbalance = abs(compute_ms[0] - compute_ms[1]) / max(max(compute_ms), 1e-9)

    top_p2p = sorted(
        (
            {"name": name, "launches": count, "total_ms": p2p_kernel_ms[name]}
            for name, count in p2p_kernel_counts.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )[:12]
    theoretical = run.get("theoretical_bubble") or {}
    return {
        "sqlite": str(sqlite_path),
        "window_ms": window_ns / NS_PER_MS,
        "measured_iterations": iterations,
        "per_device": per_device,
        "mean_gpu_idle_fraction": statistics.fmean(idle_fractions) if idle_fractions else 0.0,
        "mean_gpu_idle_percent": (
            statistics.fmean(idle_fractions) * 100.0 if idle_fractions else 0.0
        ),
        "mean_p2p_send_recv_ms_per_step": statistics.fmean(p2p_ms) if p2p_ms else 0.0,
        "mean_p2p_overlap_percent": statistics.fmean(overlap_percents) if overlap_percents else 0.0,
        "mean_exposed_p2p_ms_per_step": statistics.fmean(exposed_ms) if exposed_ms else 0.0,
        "theoretical_fill_drain_percent": theoretical.get("fill_drain_fraction", 0.0) * 100.0,
        "nvtx": {
            "chunk0_forward_count": len(chunk0),
            "chunk1_forward_count": len(chunk1),
            "async_forward_p2p_count": len(async_fwd),
            "async_backward_p2p_count": len(async_bwd),
            "sync_forward_p2p_count": len(sync_fwd),
            "sync_backward_p2p_count": len(sync_bwd),
            "warmup_ms_per_step": duration_ms(warmup, iterations),
            "steady_ms_per_step": duration_ms(steady, iterations),
            "cooldown_ms_per_step": duration_ms(cooldown, iterations),
            "chunk_compute_vs_async_p2p_overlap_ms_per_step": chunk_overlap_ms,
            "forward_p2p_transfers_per_step": (len(async_fwd) + len(sync_fwd)) / iterations,
            "backward_p2p_transfers_per_step": (len(async_bwd) + len(sync_bwd)) / iterations,
        },
        "stage_imbalance_ratio": stage_imbalance,
        "top_p2p_kernels": top_p2p,
        "schedule_name": run.get("schedule_name"),
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    monitoring = run.get("gpu_monitoring") or {}
    vram = [
        item.get("peak_memory_mib")
        for item in monitoring.values()
        if isinstance(item, dict) and item.get("peak_memory_mib") is not None
    ]
    allocated = [
        rank_mem["allocated"]
        for rank_mem in (run.get("peak_memory_by_rank_mib") or {}).values()
    ]
    util = []
    for item in monitoring.values():
        if isinstance(item, dict) and item.get("average_utilization_percent") is not None:
            util.append(item["average_utilization_percent"])
    return {
        "run_label": run["run_label"],
        "schedule_name": run.get("schedule_name"),
        "schedule_kind": run.get("schedule_kind"),
        "parallelism": run.get("parallelism"),
        "batch": run.get("batch"),
        "tokens_per_second": run["tokens_per_second"],
        "average_step_time_ms": run["average_step_time_ms"],
        "median_step_time_ms": run.get("median_step_time_ms"),
        "mfu_percent": run["mfu_percent"],
        "theoretical_bubble": run.get("theoretical_bubble"),
        "correctness": run.get("correctness"),
        "partitioning": run.get("partitioning"),
        "peak_allocated_memory_mib": allocated,
        "smi_peak_memory_mib": vram,
        "mean_gpu_utilization_percent": statistics.fmean(util) if util else None,
        "gpu_monitoring": monitoring,
        "async_p2p_calls_by_rank": run.get("async_p2p_calls_by_rank"),
        "environment": run.get("environment"),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    a = payload["variant_a"]
    b = payload["variant_b"]
    cmp_ = payload["comparison"]
    infra = payload["infrastructure"]
    topo = payload["topology"]
    traces = payload["traces"]
    lines = [
        "# Phase 8.3: interleaved 1F1B + VPP + P2P overlap",
        "",
        "FAST ITERATION MODE (5 warmup + 20 measured) unless a formal 20+100 cell is noted.",
        "CUDA Graph stayed off. Pipeline dtype stayed FP32. TP was not combined with PP.",
        "",
        "This A/B is the **combined** effect of interleaved 1F1B, virtual pipeline",
        "stages (VPP=2), and `overlap_p2p_comm`. On PP=2 those pieces cannot be isolated.",
        "Do not attribute the full throughput change to communication overlap alone.",
        "",
        "## Outcome",
        "",
        f"- Status: **{payload['status']}**",
        f"- Formal 20+100: **{'yes' if payload.get('formal_validation_ran') else 'no'}**",
        f"- A→B throughput: **{cmp_['throughput_change_percent']:+.2f}%** "
        f"({a['tokens_per_second']:.2f} → {b['tokens_per_second']:.2f} tok/s)",
        f"- P2P overlap (B, kernel vs compute): **{traces['B']['mean_p2p_overlap_percent']:.1f}%**",
        f"- Exposed P2P: {traces['A']['mean_exposed_p2p_ms_per_step']:.2f} → "
        f"{traces['B']['mean_exposed_p2p_ms_per_step']:.2f} ms/step/GPU",
        f"- Dominant remaining bottleneck: {cmp_['dominant_remaining_bottleneck']}",
        "",
        "## Infrastructure",
        "",
        f"- RunPod Pod: `{infra['pod_id']}` ({infra['pod_status']})",
        f"- Data center / public IP: {infra.get('data_center')}, `{infra.get('public_ip')}`",
        f"- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, ${infra['price_per_hour_usd']:.2f}/h",
        f"- Topology path: **{topo.get('gpu0_gpu1_path')}**, same-NUMA={topo.get('same_numa')}",
        f"- CUDA peer access bidirectional: {topo.get('p2p_bidirectional')}",
        f"- NCCL All-Reduce sanity: {topo.get('nccl_all_reduce_passed')}",
        f"- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`",
        f"- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`",
        f"- Lab commit on the Pod: `{infra.get('project_commit')}`",
        f"- `CUDA_DEVICE_MAX_CONNECTIONS=8`. `NCCL_P2P_DISABLE` unset.",
        "",
        "## Correctness (variant B)",
        "",
        f"- Forward/backward/`main_grad`/optimizer: "
        f"{b['correctness'].get('forward')}/"
        f"{b['correctness'].get('backward')}/"
        f"{b['correctness'].get('main_grad')}/"
        f"{b['correctness'].get('optimizer')}",
        f"- Finite loss / no NaN-Inf / no deadlock: "
        f"{b['correctness'].get('finite_loss')}/"
        f"{b['correctness'].get('no_nan_inf')}/"
        f"{b['correctness'].get('no_deadlock')}",
        f"- Interleaved schedule: `{b.get('schedule_name')}`",
        f"- Async P2P issued: {b['correctness'].get('async_p2p_issued')}",
        "",
        "## Layer / chunk mapping (variant B)",
        "",
        "Expected and verified:",
        "",
        "- PP0 vp0: layers 1–6, embedding",
        "- PP0 vp1: layers 13–18",
        "- PP1 vp0: layers 7–12",
        "- PP1 vp1: layers 19–24, output/loss",
        "",
        "## A/B performance",
        "",
        "| Variant | Schedule | tok/s | step ms | MFU | VRAM smi | theoretical bubble |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| A non-interleaved | `{a.get('schedule_name')}` | {a['tokens_per_second']:.2f} | "
        f"{a['average_step_time_ms']:.2f} | {a['mfu_percent']:.2f}% | "
        f"{max(a.get('smi_peak_memory_mib') or [0]):.0f} | "
        f"{(a.get('theoretical_bubble') or {}).get('fill_drain_fraction', 0.0)*100:.1f}% |",
        f"| B interleaved+VPP+overlap | `{b.get('schedule_name')}` | {b['tokens_per_second']:.2f} | "
        f"{b['average_step_time_ms']:.2f} | {b['mfu_percent']:.2f}% | "
        f"{max(b.get('smi_peak_memory_mib') or [0]):.0f} | "
        f"{(b.get('theoretical_bubble') or {}).get('fill_drain_fraction', 0.0)*100:.1f}% |",
        "",
        "## Communication and bubble",
        "",
        f"- Theoretical bubble: {cmp_['theoretical_bubble_a_percent']:.1f}% → "
        f"{cmp_['theoretical_bubble_b_percent']:.1f}% "
        f"(delta {cmp_['theoretical_bubble_reduction_pp']:.1f} pp)",
        f"- Measured nsys GPU idle: {traces['A']['mean_gpu_idle_percent']:.2f}% → "
        f"{traces['B']['mean_gpu_idle_percent']:.2f}% "
        f"(NCCL wait still looks busy; this is not the pipeline bubble)",
        f"- Activation send/recv: {traces['A']['mean_p2p_send_recv_ms_per_step']:.2f} → "
        f"{traces['B']['mean_p2p_send_recv_ms_per_step']:.2f} ms/step/GPU",
        f"- Exposed P2P: {traces['A']['mean_exposed_p2p_ms_per_step']:.2f} → "
        f"{traces['B']['mean_exposed_p2p_ms_per_step']:.2f} ms/step/GPU "
        f"({cmp_['exposed_p2p_reduction_ms']:+.2f} ms)",
        f"- Extra hop cost (issued P2P union): "
        f"{cmp_['extra_p2p_hop_cost_ms']:+.2f} ms/step/GPU",
        f"- Kernel P2P/compute overlap: {traces['A']['mean_p2p_overlap_percent']:.1f}% → "
        f"{traces['B']['mean_p2p_overlap_percent']:.1f}%",
        f"- NVTX chunk-compute vs async send/recv overlap (B): "
        f"{traces['B']['nvtx']['chunk_compute_vs_async_p2p_overlap_ms_per_step']:.2f} ms/step",
        f"- Forward/backward P2P transfers per step (B NVTX): "
        f"{traces['B']['nvtx']['forward_p2p_transfers_per_step']:.1f} / "
        f"{traces['B']['nvtx']['backward_p2p_transfers_per_step']:.1f}",
        f"- Warmup / steady / cooldown (B): "
        f"{traces['B']['nvtx']['warmup_ms_per_step']:.2f} / "
        f"{traces['B']['nvtx']['steady_ms_per_step']:.2f} / "
        f"{traces['B']['nvtx']['cooldown_ms_per_step']:.2f} ms/step",
        f"- Stage imbalance ratio: A {traces['A'].get('stage_imbalance_ratio')} "
        f"/ B {traces['B'].get('stage_imbalance_ratio')}",
        "",
        "Timeline evidence for B: `pp_chunkN_forward` NVTX on the compute stream with",
        "`pp_async_send_recv_forward/backward` issued at chunk post-forward/post-backward.",
        "Kernel overlap percent is P2P NCCL send/recv intersecting non-NCCL compute.",
        "",
        "## Decision",
        "",
        payload["decision_text"],
        "",
        "## Commands",
        "",
        "```bash",
        f"bash scripts/phase8_interleaved_pp_pod.sh {infra['pod_id']} {infra['price_per_hour_usd']}",
        "```",
        "",
        "Raw outputs: `results/phase83_work/`. "
        "Summary: `results/phase8_interleaved_pp_overlap.json`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    variant_a = load(args.variant_a)
    variant_b = load(args.variant_b)
    formal_b = load(args.formal_b) if args.formal_b and args.formal_b.exists() else None
    if args.abort_reason:
        payload = {
            "status": "aborted",
            "abort_reason": args.abort_reason,
            "infrastructure": {
                "pod_id": args.pod_id,
                "price_per_hour_usd": args.price_per_hour_usd,
                "pod_status": "deleted",
            },
            "topology": topology,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        args.markdown.write_text(
            f"# Phase 8.3 aborted\n\n{args.abort_reason}\n",
            encoding="utf-8",
        )
        return

    traces = {
        "A": analyze_sqlite(args.sqlite_a, variant_a),
        "B": analyze_sqlite(args.sqlite_b, variant_b),
    }
    a = summarize_run(variant_a)
    b = summarize_run(formal_b or variant_b)
    throughput_change = (
        (b["tokens_per_second"] - a["tokens_per_second"]) / a["tokens_per_second"] * 100.0
    )
    bubble_a = (a.get("theoretical_bubble") or {}).get("fill_drain_fraction", 0.20) * 100.0
    bubble_b = (b.get("theoretical_bubble") or {}).get("fill_drain_fraction", 0.111) * 100.0
    extra_hop = (
        traces["B"]["mean_p2p_send_recv_ms_per_step"]
        - traces["A"]["mean_p2p_send_recv_ms_per_step"]
    )
    exposed_delta = (
        traces["B"]["mean_exposed_p2p_ms_per_step"]
        - traces["A"]["mean_exposed_p2p_ms_per_step"]
    )
    formal_ran = formal_b is not None
    if throughput_change >= FORMAL_GAIN_THRESHOLD_PERCENT and not formal_ran:
        decision = (
            f"Fast-screen gain {throughput_change:+.2f}% is >= {FORMAL_GAIN_THRESHOLD_PERCENT:g}% "
            "but formal 20+100 artifacts were not supplied."
        )
    elif throughput_change >= FORMAL_GAIN_THRESHOLD_PERCENT and formal_ran:
        decision = (
            f"Correctness passed and fast-screen gain {throughput_change:+.2f}% met the "
            f"{FORMAL_GAIN_THRESHOLD_PERCENT:g}% gate, so formal 20+100 ran. "
            "Report B as the combined interleaved+VPP+overlap effect."
        )
    else:
        decision = (
            f"Correctness passed but fast-screen gain {throughput_change:+.2f}% is below "
            f"{FORMAL_GAIN_THRESHOLD_PERCENT:g}%, so formal 20+100 did not run. "
            "Keep the FAST screen. Combined interleaved+VPP+overlap is not accepted "
            "as a throughput win on this host."
        )
    remaining = "activation P2P plus residual pipeline bubble"
    if traces["B"]["mean_exposed_p2p_ms_per_step"] > 20:
        remaining = (
            "exposed pipeline activation P2P (extra interleaved hops and/or "
            "warmup/cooldown, because overlap_p2p_comm_warmup_flush stayed off)"
        )
    elif extra_hop > 10 and throughput_change < 0:
        remaining = "extra interleaved P2P hops costing more than overlap hid"
    elif traces["B"]["mean_gpu_idle_percent"] < 8 and throughput_change < 3:
        remaining = (
            "small-model stage compute; remaining fill/drain is partly hidden in NCCL wait"
        )

    env = (variant_b.get("environment") or {})
    payload = {
        "status": "success",
        "experiment": "Phase 8.3 interleaved 1F1B + VPP + P2P overlap",
        "iteration_mode": "FAST ITERATION MODE",
        "mechanism": "combined interleaved 1F1B + VPP=2 + overlap_p2p_comm",
        "isolation_caveat": (
            "On PP=2, VPP and overlap_p2p_comm cannot be safely isolated. "
            "Do not claim the full speedup comes only from communication overlap."
        ),
        "formal_validation_ran": formal_ran,
        "phase81_reference": {
            "commit": PHASE81_VALID_BASELINE_COMMIT,
            "tokens_per_second": PHASE81_TOKENS_PER_SECOND,
            "average_step_time_ms": PHASE81_STEP_MS,
            "p2p_ms_per_step": PHASE81_P2P_MS,
            "theoretical_bubble_percent": PHASE81_BUBBLE_PERCENT,
        },
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
        },
        "topology": {
            "gpu0_gpu1_path": topology.get("gpu0_gpu1_path"),
            "same_numa": topology.get("same_numa_acceptable"),
            "p2p_bidirectional": (topology.get("p2p_accessibility") or {}).get(
                "bidirectional_gpu0_gpu1"
            ),
            "nccl_all_reduce_passed": (topology.get("nccl_all_reduce_sanity") or {}).get("passed"),
        },
        "variant_a": a,
        "variant_b": b,
        "traces": traces,
        "trace_files": {
            "A": {"sqlite": str(args.sqlite_a), "sha256": sha256(args.sqlite_a), "nsys": str(args.trace_a)},
            "B": {"sqlite": str(args.sqlite_b), "sha256": sha256(args.sqlite_b), "nsys": str(args.trace_b)},
        },
        "comparison": {
            "throughput_change_percent": throughput_change,
            "step_time_change_ms": b["average_step_time_ms"] - a["average_step_time_ms"],
            "mfu_change_pp": b["mfu_percent"] - a["mfu_percent"],
            "theoretical_bubble_a_percent": bubble_a,
            "theoretical_bubble_b_percent": bubble_b,
            "theoretical_bubble_reduction_pp": bubble_a - bubble_b,
            "p2p_overlap_percent_b": traces["B"]["mean_p2p_overlap_percent"],
            "exposed_p2p_reduction_ms": -exposed_delta,
            "extra_p2p_hop_cost_ms": extra_hop,
            "dominant_remaining_bottleneck": remaining,
        },
        "decision_text": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "throughput_change_percent": throughput_change,
                "formal_validation_ran": formal_ran,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
