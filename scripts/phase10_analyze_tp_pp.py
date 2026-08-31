#!/usr/bin/env python3
"""Analyze Phase 10.1 hybrid TP+PP baseline traces and scaling references."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
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
    table_exists,
)
from phase8_analyze_pp import is_p2p_kernel, profile_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--best-run", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--microbatch-sweep", action="append", default=[], dest="microbatch_sweep")
    parser.add_argument("--reference", action="append", default=[], dest="references")
    parser.add_argument("--reference-label", action="append", default=[], dest="reference_labels")
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_nccl(name: str) -> str:
    lower = name.lower()
    compact = lower.replace("-", "").replace("_", "")
    if "allreduce" in compact or "all_reduce" in lower:
        return "allreduce"
    if "allgather" in compact or "all_gather" in lower:
        return "allgather"
    if "reducescatter" in compact or "reduce_scatter" in lower:
        return "reducescatter"
    if is_p2p_kernel(name):
        return "p2p"
    return "other"


def analyze_hybrid_trace(sqlite_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    strings = {
        int(row["id"]): str(row["value"])
        for row in connection.execute("SELECT id, value FROM StringIds")
    }
    window_start, window_end = profile_window(connection)
    window_ns = window_end - window_start
    train_steps = list(
        connection.execute(
            """
            SELECT start, end, text, textId
            FROM NVTX_EVENTS
            WHERE end IS NOT NULL AND end > ? AND start < ?
            """,
            (window_start, window_end),
        )
    )
    train_step_count = 0
    for row in train_steps:
        name = str(row["text"] or strings.get(row["textId"] if "textId" in row.keys() else None, "") or "")
        if name.startswith("train_step_"):
            train_step_count += 1
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
        lambda: {
            "all": [],
            "compute": [],
            "tp_nccl": [],
            "pp_p2p": [],
            "gemm": [],
        }
    )
    tp_counts: Counter[str] = Counter()
    tp_ms: dict[str, float] = defaultdict(float)
    pp_counts: Counter[str] = Counter()
    pp_ms: dict[str, float] = defaultdict(float)

    for row in kernels:
        name = kernel_name(row, strings)
        start = max(int(row["start"]), window_start)
        end = min(int(row["end"]), window_end)
        if end <= start:
            continue
        device = int(row["deviceId"]) if "deviceId" in row.keys() and row["deviceId"] is not None else 0
        interval = (start, end)
        by_device[device]["all"].append(interval)
        if is_nccl_kernel(name):
            kind = classify_nccl(name)
            if kind == "p2p":
                by_device[device]["pp_p2p"].append(interval)
                pp_counts[name] += 1
                pp_ms[name] += (end - start) / NS_PER_MS
            else:
                by_device[device]["tp_nccl"].append(interval)
                tp_counts[kind] += 1
                tp_ms[kind] += (end - start) / NS_PER_MS
        elif is_p2p_kernel(name):
            by_device[device]["pp_p2p"].append(interval)
            pp_counts[name] += 1
            pp_ms[name] += (end - start) / NS_PER_MS
        else:
            by_device[device]["compute"].append(interval)
            if is_gemm_kernel(name):
                by_device[device]["gemm"].append(interval)

    per_device: dict[str, Any] = {}
    tp_exposed: list[float] = []
    pp_exposed: list[float] = []
    tp_overlap_pct: list[float] = []
    pp_overlap_pct: list[float] = []
    idle_fractions: list[float] = []
    compute_ms: list[float] = []

    for device, groups in sorted(by_device.items()):
        busy = merge_intervals(groups["all"], window_start, window_end)
        compute = merge_intervals(groups["compute"], window_start, window_end)
        tp_nccl = merge_intervals(groups["tp_nccl"], window_start, window_end)
        pp_p2p = merge_intervals(groups["pp_p2p"], window_start, window_end)
        gemm = merge_intervals(groups["gemm"], window_start, window_end)
        busy_ns = interval_total(busy)
        idle_ns = window_ns - busy_ns
        idle_fraction = idle_ns / window_ns if window_ns else 0.0
        idle_fractions.append(idle_fraction)
        tp_ns = interval_total(tp_nccl)
        pp_ns = interval_total(pp_p2p)
        compute_ns = interval_total(compute)
        tp_overlap_ns = intersection_total(tp_nccl, compute)
        pp_overlap_ns = intersection_total(pp_p2p, compute)
        tp_exposed.append((tp_ns - tp_overlap_ns) / NS_PER_MS / iterations)
        pp_exposed.append((pp_ns - pp_overlap_ns) / NS_PER_MS / iterations)
        tp_overlap_pct.append(tp_overlap_ns / tp_ns * 100.0 if tp_ns else 0.0)
        pp_overlap_pct.append(pp_overlap_ns / pp_ns * 100.0 if pp_ns else 0.0)
        compute_ms.append(compute_ns / NS_PER_MS / iterations)
        per_device[str(device)] = {
            "tp_comm_ms_per_step": tp_ns / NS_PER_MS / iterations,
            "pp_comm_ms_per_step": pp_ns / NS_PER_MS / iterations,
            "tp_exposed_comm_ms_per_step": (tp_ns - tp_overlap_ns) / NS_PER_MS / iterations,
            "pp_exposed_comm_ms_per_step": (pp_ns - pp_overlap_ns) / NS_PER_MS / iterations,
            "tp_overlap_percent": tp_overlap_ns / tp_ns * 100.0 if tp_ns else 0.0,
            "pp_overlap_percent": pp_overlap_ns / pp_ns * 100.0 if pp_ns else 0.0,
            "compute_ms_per_step": compute_ns / NS_PER_MS / iterations,
            "gemm_ms_per_step": interval_total(gemm) / NS_PER_MS / iterations,
            "idle_fraction": idle_fraction,
            "idle_ms_per_step": idle_ns / NS_PER_MS / iterations,
        }

    stage_imbalance = None
    if len(compute_ms) >= 2:
        stage_imbalance = abs(compute_ms[0] - compute_ms[1]) / max(max(compute_ms), 1e-9)

    theoretical = run.get("theoretical_bubble") or {}
    measured_idle = sum(idle_fractions) / max(len(idle_fractions), 1)

    top_tp = sorted(
        (
            {"name": name, "launches": count, "total_ms": tp_ms.get(name, 0.0)}
            for name, count in tp_counts.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )[:12]
    top_pp = sorted(
        (
            {"name": name, "launches": count, "total_ms": pp_ms[name]}
            for name, count in pp_counts.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )[:12]

    return {
        "sqlite": str(sqlite_path),
        "window_ms": window_ns / NS_PER_MS,
        "measured_iterations": iterations,
        "per_device": per_device,
        "mean_tp_comm_ms_per_step": sum(per_device[d]["tp_comm_ms_per_step"] for d in per_device)
        / max(len(per_device), 1),
        "mean_pp_comm_ms_per_step": sum(per_device[d]["pp_comm_ms_per_step"] for d in per_device)
        / max(len(per_device), 1),
        "mean_tp_exposed_comm_ms_per_step": sum(tp_exposed) / max(len(tp_exposed), 1),
        "mean_pp_exposed_comm_ms_per_step": sum(pp_exposed) / max(len(pp_exposed), 1),
        "mean_tp_overlap_percent": sum(tp_overlap_pct) / max(len(tp_overlap_pct), 1),
        "mean_pp_overlap_percent": sum(pp_overlap_pct) / max(len(pp_overlap_pct), 1),
        "mean_gpu_idle_fraction": measured_idle,
        "mean_gpu_idle_percent": measured_idle * 100.0,
        "theoretical_fill_drain_percent": theoretical.get("fill_drain_fraction", 0.0) * 100.0,
        "theoretical_1f1b_warmup_percent": theoretical.get("one_f_one_b_warmup_over_steady_state", 0.0)
        * 100.0,
        "stage_imbalance_ratio": stage_imbalance,
        "tp_comm_class_launch_count": dict(tp_counts),
        "tp_comm_class_total_ms": {k: tp_ms[k] for k in tp_counts},
        "pp_comm_launch_count": dict(pp_counts),
        "pp_comm_total_ms": {k: pp_ms[k] for k in pp_counts},
        "allreduce_launches_per_step": tp_counts.get("allreduce", 0) / iterations,
        "allgather_launches_per_step": tp_counts.get("allgather", 0) / iterations,
        "reducescatter_launches_per_step": tp_counts.get("reducescatter", 0) / iterations,
        "pp_sendrecv_launches_per_step": sum(pp_counts.values()) / iterations,
        "top_tp_kernels": top_tp,
        "top_pp_kernels": top_pp,
        "both_tp_and_pp_present": bool(tp_counts) and bool(pp_counts),
        "schedule_name": run.get("schedule_name"),
        "sha256": sha256(sqlite_path),
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
    return {
        "run_label": run["run_label"],
        "tokens_per_second": run["tokens_per_second"],
        "average_step_time_ms": run["average_step_time_ms"],
        "mfu_percent": run["mfu_percent"],
        "parallelism": run.get("parallelism"),
        "batch": run.get("batch"),
        "theoretical_bubble": run.get("theoretical_bubble"),
        "peak_allocated_memory_mib": allocated,
        "smi_peak_memory_mib": vram,
        "gpu_monitoring": monitoring,
    }


def scaling_analysis(references: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {item["label"]: item["summary"] for item in references}
    r1 = by_label.get("R1")
    r2 = by_label.get("R2")
    r3 = by_label.get("R3")
    r4 = by_label.get("R4")
    result: dict[str, Any] = {"references": references, "limitations": []}
    if not r1:
        result["limitations"].append("R1 missing; cannot compute scaling efficiencies")
        return result
    r1_tps = r1["tokens_per_second"]
    if r2:
        result["tp_only_speedup_vs_r1"] = r2["tokens_per_second"] / r1_tps
    if r3:
        result["pp_only_speedup_vs_r1"] = r3["tokens_per_second"] / r1_tps
    if r4:
        result["tp2_pp2_speedup_vs_r1"] = r4["tokens_per_second"] / r1_tps
        result["strong_scaling_efficiency_vs_r1"] = r4["tokens_per_second"] / (4 * r1_tps)
    if r2 and r3 and r4:
        result["hybrid_vs_tp2_only"] = r4["tokens_per_second"] / r2["tokens_per_second"]
        result["hybrid_vs_pp2_only"] = r4["tokens_per_second"] / r3["tokens_per_second"]
    for label, summary in by_label.items():
        batch = summary.get("batch") or {}
        result.setdefault("batch_semantics", {})[label] = batch
    result["limitations"].append(
        "Scaling uses same-host R1-R4 only; global batch held at 8 where PP>1, "
        "but R1/R2 use micro_batch=8 without pipeline microbatches."
    )
    return result


def markdown_report(payload: dict[str, Any]) -> str:
    best = payload["best_configuration"]
    trace = payload["profile_trace"]
    scaling = payload["scaling"]
    lines = [
        "# Phase 10.1: TP=2 + PP=2 hybrid baseline (DP=1)",
        "",
        "FAST ITERATION MODE (5 warmup + 20 measured). CUDA Graph off.",
        "Not yet 3D parallelism because DP=1.",
        "",
        "## Outcome",
        "",
        f"- Status: **{payload['status']}**",
        f"- Best microbatches M: **{best['batch']['num_microbatches']}**",
        f"- Best throughput: **{best['tokens_per_second']:,.2f} tok/s**",
        f"- Step time: **{best['average_step_time_ms']:.2f} ms**",
        f"- MFU: **{best['mfu_percent']:.2f}%**",
        "",
        "## Infrastructure",
        "",
        f"- Pod: `{payload['infrastructure']['pod_id']}` ({payload['infrastructure']['pod_status']})",
        f"- Price: ${payload['infrastructure']['price_per_hour_usd']:.2f}/h",
        f"- Topology recorded: full 4-GPU matrix in JSON",
        "",
        "## Rank mapping (runtime)",
        "",
    ]
    for report in payload.get("rank_reports", []):
        groups = report["process_groups"]
        part = report["partitioning"]
        lines.append(
            f"- rank {groups['global_rank']}: TP {groups['tensor_parallel_rank']}"
            f"/{groups['tensor_parallel_world_size']} group {groups['tensor_parallel_group_ranks']}; "
            f"PP {groups['pipeline_parallel_rank']}/{groups['pipeline_parallel_world_size']} "
            f"group {groups['pipeline_parallel_group_ranks']}; "
            f"DP {groups['data_parallel_rank']}/{groups['data_parallel_world_size']}; "
            f"layers {part['global_layer_numbers']}; "
            f"embedding={part['owns_embedding']} output={part['owns_output_layer']}"
        )
    lines.extend(
        [
            "",
            "## Microbatch sweep",
            "",
            "| M | micro_batch | tok/s | step ms | MFU | bubble % |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["microbatch_sweep"]:
        bubble = (item.get("theoretical_bubble") or {}).get("fill_drain_fraction", 0.0) * 100.0
        lines.append(
            f"| {item['batch']['num_microbatches']} | {item['batch']['micro_batch_size']} | "
            f"{item['tokens_per_second']:,.2f} | {item['average_step_time_ms']:.2f} | "
            f"{item['mfu_percent']:.2f}% | {bubble:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Communication (best config profile)",
            "",
            f"- TP comm: {trace['mean_tp_comm_ms_per_step']:.2f} ms/step/GPU",
            f"- PP comm: {trace['mean_pp_comm_ms_per_step']:.2f} ms/step/GPU",
            f"- TP exposed: {trace['mean_tp_exposed_comm_ms_per_step']:.2f} ms/step/GPU",
            f"- PP exposed: {trace['mean_pp_exposed_comm_ms_per_step']:.2f} ms/step/GPU",
            f"- TP overlap: {trace['mean_tp_overlap_percent']:.1f}%",
            f"- PP overlap: {trace['mean_pp_overlap_percent']:.1f}%",
            f"- Measured idle/bubble: {trace['mean_gpu_idle_percent']:.1f}%",
            f"- Theoretical fill-drain bubble: {trace['theoretical_fill_drain_percent']:.1f}%",
            f"- Both TP and PP kernels present: {trace['both_tp_and_pp_present']}",
            "",
            "## Same-host scaling references",
            "",
        ]
    )
    for ref in scaling.get("references", []):
        summary = ref["summary"]
        lines.append(
            f"- {ref['label']}: {summary['tokens_per_second']:,.2f} tok/s "
            f"({summary['parallelism']})"
        )
    if "tp2_pp2_speedup_vs_r1" in scaling:
        lines.append(
            f"- TP2+PP2 vs R1 speedup: {scaling['tp2_pp2_speedup_vs_r1']:.3f}x"
        )
        lines.append(
            f"- 4-GPU strong-scaling efficiency vs R1: {scaling['strong_scaling_efficiency_vs_r1']*100:.1f}%"
        )
    lines.append("")
    lines.append(f"Dominant bottleneck: {payload.get('dominant_bottleneck', 'see JSON')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    best_run = load(args.best_run)
    trace = analyze_hybrid_trace(args.sqlite, best_run)
    sweep = [summarize_run(load(Path(path))) for path in args.microbatch_sweep]
    references = []
    labels = args.reference_labels or [f"R{index+1}" for index in range(len(args.references))]
    for label, path in zip(labels, args.references, strict=False):
        run = load(Path(path))
        references.append({"label": label, "summary": summarize_run(run), "path": str(path)})
    scaling = scaling_analysis(references)

    idle = trace["mean_gpu_idle_percent"]
    tp_exp = trace["mean_tp_exposed_comm_ms_per_step"]
    pp_exp = trace["mean_pp_exposed_comm_ms_per_step"]
    if idle > max(tp_exp, pp_exp) * 2:
        bottleneck = "pipeline bubble / idle time"
    elif pp_exp > tp_exp * 1.2:
        bottleneck = "pipeline P2P communication"
    elif tp_exp > pp_exp * 1.2:
        bottleneck = "tensor-parallel collectives"
    else:
        bottleneck = "mixed TP+PP communication and compute"

    payload = {
        "status": "success",
        "experiment": "Phase 10.1 TP=2 PP=2 hybrid baseline",
        "iteration_mode": "FAST ITERATION MODE",
        "infrastructure": {
            "pod_id": args.pod_id,
            "price_per_hour_usd": args.price_per_hour_usd,
            "pod_status": "deleted",
        },
        "topology": topology,
        "rank_reports": best_run.get("rank_reports", []),
        "microbatch_sweep": sweep,
        "best_configuration": summarize_run(best_run),
        "profile_trace": trace,
        "scaling": scaling,
        "dominant_bottleneck": bottleneck,
        "correctness": best_run.get("correctness"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"status": "success", "best_tokens_per_second": best_run["tokens_per_second"]}, indent=2))


if __name__ == "__main__":
    main()
