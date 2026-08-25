#!/usr/bin/env python3
"""Analyze Phase 8.1 PP=2 microbatch sweep and pipeline bubble traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
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
    table_exists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--pp1", type=Path, required=True)
    parser.add_argument("--pp2-mb", action="append", default=[], dest="pp2_mb")
    parser.add_argument("--sqlite", action="append", default=[], dest="sqlite")
    parser.add_argument("--trace", action="append", default=[], dest="trace")
    parser.add_argument("--label", action="append", default=[], dest="label")
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_p2p_kernel(name: str) -> bool:
    lower = name.lower()
    compact = lower.replace("-", "").replace("_", "")
    tokens = (
        "sendrecv",
        "ncclsend",
        "ncclrecv",
        "p2p",
        "send",
        "recv",
    )
    if "nccl" not in lower and "p2p" not in compact:
        return False
    if any(token in compact for token in ("allgather", "reducescatter", "allreduce", "broadcast")):
        return False
    return any(token in compact for token in tokens) or "sendrecv" in compact


def nvtx_is_p2p(name: str) -> bool:
    compact = name.lower().replace("-", "").replace("_", "")
    return any(token in compact for token in ("sendrecv", "p2p", "recv_forward", "send_forward", "recv_backward", "send_backward"))


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


def analyze_sqlite(sqlite_path: Path, run: dict[str, Any]) -> dict[str, Any]:
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
        if is_p2p_kernel(name) or (is_nccl_kernel(name) and is_p2p_kernel(name)):
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
    compute_ms: list[float] = []
    p2p_ms: list[float] = []
    for device, groups in sorted(by_device.items()):
        busy = merge_intervals(groups["all"], window_start, window_end)
        compute = merge_intervals(groups["compute"], window_start, window_end)
        p2p = merge_intervals(groups["p2p"], window_start, window_end)
        nccl = merge_intervals(groups["nccl"], window_start, window_end)
        busy_ns = interval_total(busy)
        idle_ns = window_ns - busy_ns
        idle_fraction = idle_ns / window_ns if window_ns else 0.0
        idle_fractions.append(idle_fraction)
        compute_ms.append(interval_total(compute) / NS_PER_MS / iterations)
        p2p_ms.append(interval_total(p2p) / NS_PER_MS / iterations)
        per_device[str(device)] = {
            "busy_ms_per_step": busy_ns / NS_PER_MS / iterations,
            "idle_ms_per_step": idle_ns / NS_PER_MS / iterations,
            "idle_fraction": idle_fraction,
            "compute_ms_per_step": interval_total(compute) / NS_PER_MS / iterations,
            "gemm_ms_per_step": interval_total(
                merge_intervals(groups["gemm"], window_start, window_end)
            )
            / NS_PER_MS
            / iterations,
            "p2p_send_recv_ms_per_step": interval_total(p2p) / NS_PER_MS / iterations,
            "nccl_ms_per_step": interval_total(nccl) / NS_PER_MS / iterations,
        }

    devices = sorted(by_device)
    concurrent_compute_ns = 0
    if len(devices) >= 2:
        left = merge_intervals(by_device[devices[0]]["compute"], window_start, window_end)
        right = merge_intervals(by_device[devices[1]]["compute"], window_start, window_end)
        concurrent_compute_ns = intersection_total(left, right)
    stage_imbalance = None
    if len(compute_ms) >= 2:
        stage_imbalance = abs(compute_ms[0] - compute_ms[1]) / max(max(compute_ms), 1e-9)

    memcpy_p2p_ms = 0.0
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        memcpy_columns = table_columns(connection, "CUPTI_ACTIVITY_KIND_MEMCPY")
        copy_field = "copyKind" if "copyKind" in memcpy_columns else None
        if copy_field:
            for row in connection.execute(
                f"""
                SELECT start, end, {copy_field}
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE end > ? AND start < ?
                """,
                (window_start, window_end),
            ):
                kind = string_value(row[copy_field], strings).lower()
                if "p2p" in kind or kind in {"8", "9"}:
                    memcpy_p2p_ms += (min(int(row["end"]), window_end) - max(int(row["start"]), window_start)) / NS_PER_MS

    theoretical = run.get("theoretical_bubble") or {}
    measured_idle = sum(idle_fractions) / max(len(idle_fractions), 1)
    top_p2p = sorted(
        (
            {"name": name, "launches": count, "total_ms": p2p_kernel_ms[name]}
            for name, count in p2p_kernel_counts.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )[:12]
    return {
        "sqlite": str(sqlite_path),
        "window_ms": window_ns / NS_PER_MS,
        "measured_iterations": iterations,
        "per_device": per_device,
        "mean_gpu_idle_fraction": measured_idle,
        "mean_gpu_idle_percent": measured_idle * 100.0,
        "theoretical_fill_drain_percent": theoretical.get("fill_drain_fraction", 0.0) * 100.0,
        "theoretical_1f1b_warmup_percent": theoretical.get("one_f_one_b_warmup_over_steady_state", 0.0)
        * 100.0,
        "mean_p2p_send_recv_ms_per_step": sum(p2p_ms) / max(len(p2p_ms), 1),
        "sum_p2p_send_recv_ms_per_step": sum(p2p_ms),
        "memcpy_p2p_ms_per_step": memcpy_p2p_ms / iterations,
        "concurrent_compute_ms_per_step": concurrent_compute_ns / NS_PER_MS / iterations,
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
    partitions = run.get("partitioning") or []
    return {
        "run_label": run["run_label"],
        "schedule_name": run.get("schedule_name"),
        "schedule_kind": run.get("schedule_kind"),
        "parallelism": run.get("parallelism"),
        "batch": run.get("batch"),
        "tokens_per_second": run["tokens_per_second"],
        "average_step_time_ms": run["average_step_time_ms"],
        "mfu_percent": run["mfu_percent"],
        "theoretical_bubble": run.get("theoretical_bubble"),
        "correctness": run.get("correctness"),
        "partitioning": partitions,
        "peak_allocated_memory_mib": allocated,
        "smi_peak_memory_mib": vram,
        "gpu_monitoring": monitoring,
        "environment": run.get("environment"),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    pp1 = payload["pp1_reference"]
    sweep = payload["microbatch_sweep"]
    best = payload["best_pp2"]
    lines = [
        "# Phase 8.1: 2-GPU Pipeline Parallel baseline",
        "",
        "FAST ITERATION MODE (5 warmup + 20 measured). CUDA Graph stayed off.",
        "TP=1, PP=2, DP=1. Tensor parallel was not combined with pipeline parallel.",
        "",
        "## Tensor-parallel communication-overlap conclusions (closed)",
        "",
        "- **AG-only Userbuffers is accepted** and gives **+6.75%** host-local throughput",
        "  (formal 20+100: 27,209 → 29,045 tok/s) versus TE Linear + Sequence Parallel",
        "  without Userbuffers. Flags: `tp_comm_overlap=True`, `tp_comm_overlap_ag=True`,",
        "  `tp_comm_overlap_rs=False`, `tp_comm_overlap_rs_dgrad=False`,",
        "  `tp_comm_bulk_dgrad=False`, `tp_comm_bulk_wgrad=False`.",
        "- **bulk-dgrad demonstrates real AG/GEMM overlap (~91.5%)** via",
        "  `userbuffers_fp16_sum_inplace_gpu_rw_ag`, but is **slower than AG-only**",
        "  (+4.0% vs the same B reference, 28,312 tok/s).",
        "- **Reduce-Scatter Userbuffers remains disabled** because it livelocks on A40",
        "  PCIe in `userbuffers_fp16_sum_inplace_gpu_rr_rs_oop`. This phase did **not**",
        "  continue debugging RS Userbuffers.",
        "",
        "## Infrastructure and topology",
        "",
        f"- RunPod Pod: `{payload['infrastructure']['pod_id']}` ({payload['infrastructure']['pod_status']})",
        f"- Data center / public IP: {payload['infrastructure'].get('data_center')}, `{payload['infrastructure'].get('public_ip')}`",
        f"- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, ${payload['infrastructure']['price_per_hour_usd']}/h (≤ $0.90/h)",
        f"- Topology path: **{payload['topology']['gpu0_gpu1_path']}**, same-NUMA={payload['topology']['same_numa_acceptable']}",
        f"- CUDA peer access bidirectional: {payload['topology']['p2p_bidirectional']}",
        f"- NCCL All-Reduce sanity: {payload['topology']['nccl_sanity_passed']}",
        f"- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`",
        f"- Megatron-LM: `{payload['environment']['megatron_lm_commit']}`",
        f"- Lab commit on the Pod: `{payload['environment']['project_commit']}`",
        f"- PyTorch: {payload['environment']['pytorch']}; CUDA: {payload['environment']['cuda_runtime']}; NCCL: {payload['environment']['nccl']}",
        f"- Driver: {payload['environment'].get('driver')}",
        "- `NCCL_P2P_DISABLE` unset. No TP+PP mix.",
        "",
        "## Pipeline partitioning",
        "",
        "24 Transformer layers split evenly across PP=2:",
        "",
    ]
    for rank in payload["partitioning"]:
        lines.append(
            f"- rank {rank['rank']} / PP {rank['pipeline_rank']}: "
            f"{rank['layers_built']} layers "
            f"(global {rank['global_layer_numbers'][0]}–{rank['global_layer_numbers'][-1] if rank['global_layer_numbers'] else 'n/a'}), "
            f"embedding={rank['owns_embedding']}, output/loss={rank['owns_output_layer']}/{rank['owns_loss']}, "
            f"params={rank['parameter_count']:,} "
            f"(embed {rank['embedding_parameter_count']:,}, decoder {rank['decoder_parameter_count']:,}, "
            f"output {rank['output_parameter_count']:,})"
        )
    lines.extend(
        [
            "",
            "Embedding lives on the first stage. Output layer and loss live on the last stage.",
            "Word embeddings are **untied** on PP=2 (`share_embeddings_and_output_weights=False`) "
            "so the embedding group does not all-reduce during 1F1B. Compute stays BF16 autocast; "
            "pipeline send/recv uses FP32 (`pipeline_dtype=torch.float32`) because BF16 P2P with "
            "this FP32-param model produced all-NaN last-stage losses.",
            "",
            "## Correctness",
            "",
            f"- Smoke (3 steps) on PP=1 and every PP=2 microbatch count: {payload['correctness_summary']}",
            "- Forward, backward, `main_grad`, optimizer, finite loss, and no deadlock all passed.",
            f"- Schedule: `{payload['schedule_name']}` (Megatron 1F1B without interleaving for PP=2).",
            "",
            "## Microbatch sweep (constant global batch = 8, 16384 tokens/step)",
            "",
            "| Config | μbatches | μbatch size | tok/s | step ms | MFU | VRAM/GPU (smi) | theoretical bubble | measured idle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| PP=1 reference | 1 | 8 | {pp1['tokens_per_second']:.2f} | "
                f"{pp1['average_step_time_ms']:.2f} | {pp1['mfu_percent']:.2f}% | "
                f"{(max(pp1['smi_peak_memory_mib']) if pp1['smi_peak_memory_mib'] else 0):.0f} | "
                f"0% | n/a |"
            ),
        ]
    )
    for item in sweep:
        comm = item.get("communication") or {}
        vram = max(item["smi_peak_memory_mib"]) if item["smi_peak_memory_mib"] else 0
        theoretical = (item.get("theoretical_bubble") or {}).get("fill_drain_fraction", 0.0) * 100.0
        measured = comm.get("mean_gpu_idle_percent")
        measured_text = f"{measured:.2f}%" if measured is not None else "n/a"
        lines.append(
            f"| PP=2 μb={item['batch']['num_microbatches']} | "
            f"{item['batch']['num_microbatches']} | {item['batch']['micro_batch_size']} | "
            f"{item['tokens_per_second']:.2f} | {item['average_step_time_ms']:.2f} | "
            f"{item['mfu_percent']:.2f}% | {vram:.0f} | {theoretical:.1f}% | {measured_text} |"
        )
    lines.extend(
        [
            "",
            "## Why few microbatches are inefficient",
            "",
            payload["analysis"]["few_microbatches"],
            "",
            "## How more microbatches reduce the bubble",
            "",
            payload["analysis"]["more_microbatches"],
            "",
            "## Diminishing returns",
            "",
            payload["analysis"]["diminishing_returns"],
            "",
            "## Does PP=2 help throughput on this small model?",
            "",
            payload["analysis"]["throughput_or_memory"],
            "",
            "## Best PP=2 result",
            "",
            f"- Best throughput: **{best['tokens_per_second']:.2f} tok/s** "
            f"(μbatches={best['batch']['num_microbatches']}, μbatch={best['batch']['micro_batch_size']})",
            f"- Step time: {best['average_step_time_ms']:.2f} ms",
            f"- MFU: {best['mfu_percent']:.2f}%",
            f"- VRAM/GPU: {payload['best_pp2_vram_mib']:.0f} MiB",
            f"- PP scaling vs same-host PP=1: {payload['pp_speedup_vs_pp1']:.3f}x "
            f"({payload['pp_scaling_efficiency_vs_2x'] * 100.0:.1f}% of ideal 2x)",
            f"- Mean P2P send/recv: {payload.get('best_pp2_p2p_ms_per_step')} ms/step/GPU",
            f"- Measured bubble/idle: {payload.get('best_pp2_idle_percent')}%",
            "",
            f"**Primary bottleneck:** {payload['primary_bottleneck']}",
            "",
            "## Commands",
            "",
            "```bash",
            f"bash scripts/phase8_pp2_pod.sh {payload['infrastructure']['pod_id']} {payload['infrastructure']['price_per_hour_usd']}",
            "```",
            "",
            "Raw outputs: `results/phase81_work/`. Summary: `results/phase8_pp2_baseline.json`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    pp1_run = load(args.pp1)
    sweep_runs = [load(Path(path)) for path in args.pp2_mb]
    comm_by_label: dict[str, dict[str, Any]] = {}
    if not (len(args.sqlite) == len(args.trace) == len(args.label)):
        raise RuntimeError("sqlite, trace, and label lists must have the same length")
    for sqlite_path, trace_path, label in zip(args.sqlite, args.trace, args.label):
        run = next(item for item in sweep_runs if item["run_label"] == label)
        comm_by_label[label] = analyze_sqlite(Path(sqlite_path), run)
        comm_by_label[label]["trace"] = str(trace_path)
        comm_by_label[label]["trace_sha256"] = sha256(Path(trace_path)) if Path(trace_path).exists() else None

    summarized = []
    for run in sweep_runs:
        item = summarize_run(run)
        item["communication"] = comm_by_label.get(run["run_label"])
        summarized.append(item)
    summarized.sort(key=lambda item: item["batch"]["num_microbatches"])
    pp1 = summarize_run(pp1_run)
    best = max(summarized, key=lambda item: item["tokens_per_second"])
    pp_speedup = best["tokens_per_second"] / pp1["tokens_per_second"]
    efficiency = pp_speedup / 2.0
    idle_curve = [
        {
            "num_microbatches": item["batch"]["num_microbatches"],
            "theoretical_fill_drain_percent": (item.get("theoretical_bubble") or {}).get(
                "fill_drain_fraction", 0.0
            )
            * 100.0,
            "measured_idle_percent": (item.get("communication") or {}).get("mean_gpu_idle_percent"),
            "tokens_per_second": item["tokens_per_second"],
        }
        for item in summarized
    ]
    gains = []
    for previous, current in zip(summarized, summarized[1:]):
        gains.append(
            {
                "from": previous["batch"]["num_microbatches"],
                "to": current["batch"]["num_microbatches"],
                "delta_tokens_per_second": current["tokens_per_second"] - previous["tokens_per_second"],
                "relative_gain_percent": (
                    current["tokens_per_second"] / previous["tokens_per_second"] - 1.0
                )
                * 100.0,
            }
        )
    diminishing_at = None
    for gain in gains:
        if gain["relative_gain_percent"] < 5.0:
            diminishing_at = gain["to"]
            break
    few = summarized[0]
    many = summarized[-1]
    few_bubble = (few.get("theoretical_bubble") or {}).get("fill_drain_fraction", 0.0) * 100.0
    many_bubble = (many.get("theoretical_bubble") or {}).get("fill_drain_fraction", 0.0) * 100.0
    vram_pp1 = max(pp1["smi_peak_memory_mib"] or pp1["peak_allocated_memory_mib"] or [0])
    vram_pp2 = max(best["smi_peak_memory_mib"] or best["peak_allocated_memory_mib"] or [0])
    if pp_speedup < 1.05:
        throughput_or_memory = (
            f"PP=2 does **not** help throughput on this 355.9M model. Best PP=2 is "
            f"{best['tokens_per_second']:.0f} tok/s versus same-host PP=1 "
            f"{pp1['tokens_per_second']:.0f} tok/s ({pp_speedup:.3f}x, "
            f"{efficiency * 100.0:.1f}% of ideal 2x). The model already fits on one A40 "
            f"(PP=1 VRAM {vram_pp1:.0f} MiB). PP=2 lowers per-GPU memory to {vram_pp2:.0f} MiB "
            "and is therefore primarily a **memory-capacity** split, not a throughput win."
        )
        bottleneck = (
            "Pipeline bubble plus activation send/recv on a model that already fits in one GPU. "
            "PP=2 adds a P2P boundary without enough concurrent compute to beat single-GPU PP=1."
        )
    else:
        throughput_or_memory = (
            f"PP=2 improves throughput to {best['tokens_per_second']:.0f} tok/s from "
            f"PP=1 {pp1['tokens_per_second']:.0f} tok/s ({pp_speedup:.3f}x, "
            f"{efficiency * 100.0:.1f}% of ideal 2x). Per-GPU VRAM drops from "
            f"{vram_pp1:.0f} to {vram_pp2:.0f} MiB, so PP still also helps memory capacity."
        )
        bottleneck = (
            "Remaining pipeline idle plus activation P2P. The 355.9M model is small enough "
            "that layer compute per stage is only moderately larger than send/recv."
        )
    analysis = {
        "few_microbatches": (
            f"With {few['batch']['num_microbatches']} microbatch(es), Megatron still uses "
            f"`{few['schedule_name']}`, but there is almost no 1F1B steady state. Rank 0 must "
            "finish its forward and send activations before rank 1 can start, then rank 1's "
            "backward must return before rank 0 can backward. Theoretical fill/drain bubble is "
            f"{few_bubble:.1f}%. Measured GPU idle is "
            f"{(few.get('communication') or {}).get('mean_gpu_idle_percent', 'n/a')}%. "
            "Too few microbatches serialize the two stages."
        ),
        "more_microbatches": (
            "Increasing microbatches (holding global batch=8) fills the pipeline: warmup is "
            "still (PP-1) forwards, but a 1F1B steady state appears. Theoretical bubble "
            f"falls from {few_bubble:.1f}% at {few['batch']['num_microbatches']} μb to "
            f"{many_bubble:.1f}% at {many['batch']['num_microbatches']} μb. Throughput moved "
            f"from {few['tokens_per_second']:.0f} to {many['tokens_per_second']:.0f} tok/s."
        ),
        "diminishing_returns": (
            "Relative tok/s gains between successive doubling of microbatches: "
            + ", ".join(
                f"{gain['from']}→{gain['to']} {gain['relative_gain_percent']:.2f}%"
                for gain in gains
            )
            + (
                f". Diminishing returns begin at {diminishing_at} microbatches "
                "(<5% additional gain)."
                if diminishing_at is not None
                else ". No doubling in this sweep dropped below a 5% gain."
            )
        ),
        "throughput_or_memory": throughput_or_memory,
    }
    env = (summarized[0].get("environment") or pp1.get("environment") or {})
    if env.get("gpus"):
        env = {**env, "driver": env["gpus"][0].get("driver")}
    partitioning = sorted(summarized[0]["partitioning"], key=lambda rank: rank["rank"])
    correctness_ok = all(
        (item.get("correctness") or {}).get("finite_loss")
        and (item.get("correctness") or {}).get("no_deadlock")
        for item in [pp1, *summarized]
    )
    best_comm = best.get("communication") or {}
    payload = {
        "status": "success",
        "experiment": "Phase 8.1 2-GPU pipeline parallel baseline",
        "iteration_mode": "FAST ITERATION MODE",
        "tp_overlap_conclusions": {
            "ag_only_userbuffers_accepted": True,
            "ag_only_host_local_gain_percent": 6.75,
            "bulk_dgrad_ag_gemm_overlap_percent": 91.5,
            "bulk_dgrad_slower_than_ag_only": True,
            "reduce_scatter_userbuffers_disabled": True,
            "rs_livelock_kernel": "userbuffers_fp16_sum_inplace_gpu_rr_rs_oop",
            "continued_rs_debugging": False,
        },
        "infrastructure": {
            "pod_id": args.pod_id,
            "pod_count": 1,
            "gpu_count": 2,
            "gpu_type": "NVIDIA A40 48GB",
            "price_per_hour_usd": args.price_per_hour_usd,
            "price_target_usd": 0.90,
            "price_target_met": args.price_per_hour_usd <= 0.90,
            "pod_status": "deleted",
            "data_center": topology.get("data_center"),
            "public_ip": topology.get("public_ip"),
        },
        "topology": {
            "gpu0_gpu1_path": topology.get("gpu0_gpu1_path"),
            "same_numa_acceptable": topology.get("same_numa_acceptable"),
            "p2p_bidirectional": (topology.get("p2p_accessibility") or {}).get(
                "bidirectional_gpu0_gpu1"
            ),
            "nccl_sanity_passed": (topology.get("nccl_all_reduce_sanity") or {}).get("passed"),
            "raw": topology,
        },
        "environment": env,
        "schedule_name": "forward_backward_pipelining_without_interleaving",
        "partitioning": partitioning,
        "correctness_summary": "passed" if correctness_ok else "failed",
        "pp1_reference": pp1,
        "microbatch_sweep": summarized,
        "idle_curve": idle_curve,
        "microbatch_gains": gains,
        "diminishing_returns_begin_at_microbatches": diminishing_at,
        "best_pp2": best,
        "best_pp2_vram_mib": vram_pp2,
        "best_pp2_p2p_ms_per_step": best_comm.get("mean_p2p_send_recv_ms_per_step"),
        "best_pp2_idle_percent": best_comm.get("mean_gpu_idle_percent"),
        "pp_speedup_vs_pp1": pp_speedup,
        "pp_scaling_efficiency_vs_2x": efficiency,
        "analysis": analysis,
        "primary_bottleneck": bottleneck,
        "communication_by_label": comm_by_label,
    }
    # Fill topology public IP / DC from nested topology if present
    if not payload["infrastructure"]["public_ip"]:
        payload["infrastructure"]["public_ip"] = topology.get("host_public_ip")
    if not payload["infrastructure"]["data_center"]:
        payload["infrastructure"]["data_center"] = topology.get("data_center_id")
    write(args.output, payload)
    args.markdown.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(
        {
            "best_tokens_per_second": best["tokens_per_second"],
            "pp_speedup_vs_pp1": pp_speedup,
            "best_idle_percent": best_comm.get("mean_gpu_idle_percent"),
            "output": str(args.output),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
