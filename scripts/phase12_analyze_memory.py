#!/usr/bin/env python3
"""Analyze Phase 12 A/B/C/D memory and capacity results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--variant-a", type=Path, required=True)
    parser.add_argument("--variant-b", type=Path, required=True)
    parser.add_argument("--variant-c", type=Path, required=True)
    parser.add_argument("--variant-d", type=Path, required=True)
    parser.add_argument("--capacity-a", type=Path, required=True)
    parser.add_argument("--capacity-b", type=Path, required=True)
    parser.add_argument("--capacity-c", type=Path, required=True)
    parser.add_argument("--capacity-d", type=Path, required=True)
    parser.add_argument("--capacity-bench-a", type=Path, default=None)
    parser.add_argument("--capacity-bench-b", type=Path, default=None)
    parser.add_argument("--capacity-bench-c", type=Path, default=None)
    parser.add_argument("--capacity-bench-d", type=Path, default=None)
    parser.add_argument("--sqlite-a", type=Path, default=None)
    parser.add_argument("--sqlite-c", type=Path, default=None)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def peak_mib(payload: dict[str, Any], key: str = "peak_allocated_memory_mib") -> float:
    values = payload.get(key) or []
    if not values:
        return float("nan")
    return float(max(values))


def gib(mib: float) -> float:
    return mib / 1024.0


def summarize_fixed(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    opt_bytes = payload.get("optimizer_state_bytes_per_rank") or [0, 0]
    return {
        "variant": label,
        "status": payload.get("status"),
        "use_distributed_optimizer": payload.get("use_distributed_optimizer"),
        "activation_checkpointing": payload.get("activation_checkpointing", {}).get("enabled"),
        "tokens_per_second": payload.get("tokens_per_second"),
        "average_step_time_ms": payload.get("average_step_time_ms"),
        "median_step_time_ms": payload.get("median_step_time_ms"),
        "mfu_percent": payload.get("mfu_percent"),
        "peak_allocated_mib": peak_mib(payload, "peak_allocated_memory_mib"),
        "peak_reserved_mib": peak_mib(payload, "peak_reserved_memory_mib"),
        "smi_peak_mib": max(
            [v for v in (payload.get("smi_peak_memory_mib") or []) if v is not None],
            default=None,
        ),
        "optimizer_state_bytes_per_rank": opt_bytes,
        "optimizer_state_gib_per_rank": [b / 1024**3 for b in opt_bytes],
        "tokens_per_step": payload.get("batch", {}).get("tokens_per_step"),
        "micro_batch_size_per_gpu": payload.get("batch", {}).get("micro_batch_size_per_gpu"),
        "correctness_ok": bool(
            payload.get("correctness", {}).get("finite_loss")
            and payload.get("correctness", {}).get("no_deadlock", True)
        ),
        "recompute": payload.get("activation_checkpointing"),
    }


def summarize_capacity(
    label: str,
    capacity_payload: dict[str, Any],
    bench_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    max_mb = capacity_payload.get("max_micro_batch_size")
    seq = capacity_payload.get("sequence_length", 2048)
    out: dict[str, Any] = {
        "variant": label,
        "max_micro_batch_size": max_mb,
        "sequence_length": seq,
        "global_batch_size": (max_mb or 0) * 2,
        "max_tokens_per_step": (max_mb or 0) * 2 * seq,
        "search_log": capacity_payload.get("search_log", []),
        "peak_allocated_mib_at_max": capacity_payload.get("peak_allocated_mib_at_max"),
    }
    if bench_payload and bench_payload.get("status") == "success":
        out.update(
            {
                "tokens_per_second_at_max": bench_payload.get("tokens_per_second"),
                "average_step_time_ms_at_max": bench_payload.get("average_step_time_ms"),
                "mfu_percent_at_max": bench_payload.get("mfu_percent"),
                "peak_allocated_mib_bench": peak_mib(bench_payload),
                "smi_peak_mib_bench": max(
                    [v for v in (bench_payload.get("smi_peak_memory_mib") or []) if v is not None],
                    default=None,
                ),
            }
        )
    return out


def delta_pct(new: float | None, base: float | None) -> float | None:
    if new is None or base is None or base == 0:
        return None
    return (new - base) / base * 100.0


def analyze_nsys_recompute(sqlite_a: Path | None, sqlite_c: Path | None) -> dict[str, Any]:
    """Lightweight evidence: compare total CUDA kernel time if sqlite available."""
    result: dict[str, Any] = {
        "available": False,
        "note": "optional nsys evidence for recompute cost",
    }
    if sqlite_a is None or sqlite_c is None:
        return result
    if not sqlite_a.exists() or not sqlite_c.exists():
        result["note"] = "sqlite traces missing"
        return result
    try:
        import sqlite3

        def cuda_ms(path: Path) -> float:
            conn = sqlite3.connect(path)
            try:
                row = conn.execute(
                    "SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
                ).fetchone()
            except sqlite3.Error:
                # Older/newer schemas may differ; try generic.
                tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
                    return float("nan")
                row = (None,)
            finally:
                conn.close()
            total_ns = row[0] or 0
            return total_ns / 1e6

        a_ms = cuda_ms(sqlite_a)
        c_ms = cuda_ms(sqlite_c)
        result.update(
            {
                "available": True,
                "variant_a_cuda_kernel_ms": a_ms,
                "variant_c_cuda_kernel_ms": c_ms,
                "c_over_a_kernel_time_ratio": (c_ms / a_ms) if a_ms else None,
                "interpretation": (
                    "Activation checkpointing should increase total CUDA work during "
                    "backward due to recomputed forwards; ratio > 1 supports that."
                ),
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    fixed = payload["fixed_workload"]
    cap = payload["capacity"]
    decisions = payload["decisions"]
    lines = [
        "# Phase 12: Training Memory and Capacity Engineering",
        "",
        "FAST ITERATION MODE (5 warmup + 20 measured for fixed workload).",
        "",
        "## Scope",
        "",
        "- DP=2, TP=1, PP=1 on one 2×A40 Secure Pod",
        "- No TP/PP/SP/VPP/Userbuffers/CUDA Graph",
        "- After Phase 12, remaining work is Phase 15 packaging only",
        "",
        "## Pinned activation recompute (Megatron 09fde85)",
        "",
        "```",
        "recompute_granularity = 'full'",
        "recompute_method      = 'uniform'",
        "recompute_num_layers  = 1",
        "```",
        "",
        "- Effect: each of 24 Transformer layers is an independent recompute unit.",
        "- Implementation: `TransformerBlock` → `checkpointed_forward` →",
        "  `tensor_parallel.checkpoint` (`megatron/core/recompute.py`).",
        "- Expected memory: discard layer activations after forward; recompute during backward.",
        "",
        "## Infrastructure",
        "",
        f"- Pod: `{payload['infrastructure']['pod_id']}`",
        f"- Price: ${payload['infrastructure']['price_per_hour_usd']:.2f}/h",
        f"- Host / DC: `{payload['infrastructure'].get('host_suffix')}` / "
        f"`{payload['infrastructure'].get('data_center')}`",
        f"- Topology path: `{payload['infrastructure'].get('gpu0_gpu1_path')}`",
        "",
        "## Part 1 — Fixed workload (~355.9M, MB=8, seq=2048)",
        "",
        "| Variant | DistOpt | AC | tok/s | step ms | MFU % | Peak alloc GiB | Opt state GiB/rank |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "B", "C", "D"):
        row = fixed[key]
        lines.append(
            f"| {key} | {row['use_distributed_optimizer']} | {row['activation_checkpointing']} | "
            f"{row['tokens_per_second']:.1f} | {row['average_step_time_ms']:.2f} | "
            f"{row['mfu_percent']:.2f} | {gib(row['peak_allocated_mib']):.2f} | "
            f"{statistics.fmean(row['optimizer_state_gib_per_rank']):.3f} |"
        )
    lines.extend(
        [
            "",
            "### Memory deltas vs A",
            "",
            f"- A→B (DistOpt): {decisions['memory_saving_distopt_gib']:.2f} GiB "
            f"({decisions['memory_saving_distopt_pct']:.1f}%)",
            f"- A→C (AC): {decisions['memory_saving_ac_gib']:.2f} GiB "
            f"({decisions['memory_saving_ac_pct']:.1f}%)",
            f"- A→D (Combined): {decisions['memory_saving_combined_gib']:.2f} GiB "
            f"({decisions['memory_saving_combined_pct']:.1f}%)",
            f"- C→D (DistOpt on top of AC): {decisions['memory_saving_c_to_d_gib']:.2f} GiB",
            "",
            "## Part 2 — Capacity search (max microbatch @ seq=2048)",
            "",
            "| Variant | Max MB/GPU | Global batch | Max tokens/step | Peak alloc GiB @ max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("A", "B", "C", "D"):
        row = cap[key]
        peak = row.get("peak_allocated_mib_at_max") or row.get("peak_allocated_mib_bench")
        peak_s = f"{gib(peak):.2f}" if peak else "n/a"
        lines.append(
            f"| {key} | {row['max_micro_batch_size']} | {row['global_batch_size']} | "
            f"{row['max_tokens_per_step']} | {peak_s} |"
        )
    lines.extend(
        [
            "",
            "## Part 3 — Throughput vs capacity",
            "",
            "| Variant | Max tokens/step | tok/s @ max | MFU @ max | Capacity gain | Perf cost vs A@MB8 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("A", "B", "C", "D"):
        row = cap[key]
        tps = row.get("tokens_per_second_at_max")
        mfu = row.get("mfu_percent_at_max")
        gain = decisions["capacity_gain_tokens"][key]
        cost = decisions["throughput_cost_vs_a_fixed"][key]
        tps_s = f"{tps:.1f}" if tps is not None else "n/a"
        mfu_s = f"{mfu:.2f}" if mfu is not None else "n/a"
        lines.append(
            f"| {key} | {row['max_tokens_per_step']} | "
            f"{tps_s} | {mfu_s} | "
            f"{gain:+.1f}% | {cost:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"1. DistOpt memory save: **{decisions['memory_saving_distopt_gib']:.2f} GiB** "
            f"({decisions['memory_saving_distopt_pct']:.1f}%)",
            f"2. Activation checkpointing memory save: **{decisions['memory_saving_ac_gib']:.2f} GiB** "
            f"({decisions['memory_saving_ac_pct']:.1f}%)",
            f"3. Combined capacity gain vs A: **{decisions['capacity_gain_tokens']['D']:+.1f}%** "
            f"tokens/step",
            f"4. Checkpointing throughput penalty (C vs A @ fixed MB=8): "
            f"**{decisions['throughput_cost_vs_a_fixed']['C']:+.1f}%**",
            f"5. Throughput-optimal: **{decisions['throughput_optimal']}**",
            f"6. Capacity-optimal: **{decisions['capacity_optimal']}**",
            f"7. When memory saving justifies recompute: {decisions['justify_recompute']}",
            "",
            "## Correctness",
            "",
            f"- All fixed variants finite/no-deadlock: `{decisions['correctness_all_ok']}`",
            "",
            "## Profiling evidence",
            "",
            "```json",
            json.dumps(payload.get("profiling_evidence", {}), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    fixed_raw = {
        "A": load(args.variant_a),
        "B": load(args.variant_b),
        "C": load(args.variant_c),
        "D": load(args.variant_d),
    }
    capacity_raw = {
        "A": load(args.capacity_a),
        "B": load(args.capacity_b),
        "C": load(args.capacity_c),
        "D": load(args.capacity_d),
    }
    bench_paths = {
        "A": args.capacity_bench_a,
        "B": args.capacity_bench_b,
        "C": args.capacity_bench_c,
        "D": args.capacity_bench_d,
    }
    benches: dict[str, dict[str, Any] | None] = {}
    for key, path in bench_paths.items():
        benches[key] = load(path) if path and path.exists() else None

    fixed = {k: summarize_fixed(k, v) for k, v in fixed_raw.items()}
    capacity = {
        k: summarize_capacity(k, capacity_raw[k], benches[k]) for k in ("A", "B", "C", "D")
    }

    a_peak = fixed["A"]["peak_allocated_mib"]
    decisions = {
        "memory_saving_distopt_gib": gib(a_peak - fixed["B"]["peak_allocated_mib"]),
        "memory_saving_distopt_pct": -delta_pct(fixed["B"]["peak_allocated_mib"], a_peak) or 0.0,
        "memory_saving_ac_gib": gib(a_peak - fixed["C"]["peak_allocated_mib"]),
        "memory_saving_ac_pct": -delta_pct(fixed["C"]["peak_allocated_mib"], a_peak) or 0.0,
        "memory_saving_combined_gib": gib(a_peak - fixed["D"]["peak_allocated_mib"]),
        "memory_saving_combined_pct": -delta_pct(fixed["D"]["peak_allocated_mib"], a_peak) or 0.0,
        "memory_saving_c_to_d_gib": gib(
            fixed["C"]["peak_allocated_mib"] - fixed["D"]["peak_allocated_mib"]
        ),
        "capacity_gain_tokens": {
            k: delta_pct(
                float(capacity[k]["max_tokens_per_step"]),
                float(capacity["A"]["max_tokens_per_step"]),
            )
            or 0.0
            for k in ("A", "B", "C", "D")
        },
        "throughput_cost_vs_a_fixed": {
            k: delta_pct(fixed[k]["tokens_per_second"], fixed["A"]["tokens_per_second"]) or 0.0
            for k in ("A", "B", "C", "D")
        },
        "correctness_all_ok": all(fixed[k]["correctness_ok"] for k in ("A", "B", "C", "D")),
    }
    # Throughput-optimal: highest tok/s at fixed MB=8.
    decisions["throughput_optimal"] = max(
        ("A", "B", "C", "D"), key=lambda k: fixed[k]["tokens_per_second"] or 0.0
    )
    # Capacity-optimal: highest max tokens/step.
    decisions["capacity_optimal"] = max(
        ("A", "B", "C", "D"), key=lambda k: capacity[k]["max_tokens_per_step"] or 0
    )
    ac_penalty = decisions["throughput_cost_vs_a_fixed"]["C"]
    ac_capacity = decisions["capacity_gain_tokens"]["C"]
    decisions["justify_recompute"] = (
        f"Activation checkpointing costs {ac_penalty:+.1f}% throughput at fixed MB=8 "
        f"but unlocks {ac_capacity:+.1f}% tokens/step capacity. Prefer AC (or Combined) when "
        f"the target microbatch/seq would OOM under baseline/DistOpt-only; otherwise keep "
        f"{decisions['throughput_optimal']} for throughput."
    )

    host = None
    ssh = topology.get("ssh_proxy_username") or ""
    if "-" in str(topology.get("pod_id", args.pod_id)):
        pass
    # Prefer hostname-like suffix from topology if present.
    host = topology.get("host_suffix") or topology.get("hostname")
    if not host:
        # RunPod proxy usernames often look like <podid>-<hostsuf>
        for key in ("public_ip", "host_public_ip"):
            _ = topology.get(key)
        host = (topology.get("environment") or {}).get("hostname")

    payload = {
        "status": "success",
        "experiment": "Phase 12 training memory and capacity",
        "iteration_mode": "FAST ITERATION MODE",
        "recompute_configuration": fixed["C"]["recompute"],
        "infrastructure": {
            "pod_id": args.pod_id,
            "price_per_hour_usd": args.price_per_hour_usd,
            "gpu_count": 2,
            "gpu_type": "NVIDIA A40 48GB",
            "data_center": topology.get("data_center") or topology.get("data_center_id"),
            "gpu0_gpu1_path": topology.get("gpu0_gpu1_path"),
            "host_suffix": host,
            "pod_status": "to_be_deleted",
        },
        "fixed_workload": fixed,
        "capacity": capacity,
        "decisions": decisions,
        "profiling_evidence": analyze_nsys_recompute(args.sqlite_a, args.sqlite_c),
        "memory_breakdown_notes": {
            "parameters": "replicated on each DP rank (TP=1); DistOpt does not shard params in BF16 copy permanently the same way ZeRO-3 would — Megatron distopt shards optimizer states / FP32 shards",
            "gradients_main_grad": "reduce-scatter under DistOpt vs all-reduce without",
            "optimizer_states": "Adam moments; primary DistOpt saving observed in optimizer_state_bytes_per_rank",
            "activations": "primary AC saving observed in peak_allocated delta A→C at fixed batch",
            "communication_workspace": "NCCL buckets / overlap workspaces included in reserved/peak",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(f"PHASE12_ANALYZE_OK throughput_opt={decisions['throughput_optimal']} "
          f"capacity_opt={decisions['capacity_optimal']}")


if __name__ == "__main__":
    main()
