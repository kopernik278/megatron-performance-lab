#!/usr/bin/env python3
"""Analyze Phase 7.4 TP Userbuffers A/B/C timing and communication."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from phase7_analyze_tp import analyze_trace, load, timing_summary


FORMAL_GAIN_THRESHOLD_PERCENT = 3.0
PHASE71_VALID_BASELINE_COMMIT = "709437d"
PHASE71_VALID_POD_ID = "7rpwv95a5j6axg"
PHASE72_P2P_DISABLED_POD_ID = "wtd9cxr3q8obuh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--variant-a", type=Path, required=True)
    parser.add_argument("--variant-b", type=Path, required=True)
    parser.add_argument("--variant-c", type=Path, required=True)
    parser.add_argument("--variant-a-profile", type=Path, required=True)
    parser.add_argument("--variant-b-profile", type=Path, required=True)
    parser.add_argument("--variant-c-profile", type=Path, required=True)
    parser.add_argument("--sqlite-a", type=Path, required=True)
    parser.add_argument("--sqlite-b", type=Path, required=True)
    parser.add_argument("--sqlite-c", type=Path, required=True)
    parser.add_argument("--trace-a", type=Path, required=True)
    parser.add_argument("--trace-b", type=Path, required=True)
    parser.add_argument("--trace-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-b", type=Path)
    parser.add_argument("--formal-c", type=Path)
    parser.add_argument("--abort-reason", default=None)
    return parser.parse_args()


def overlap_summary(communication: dict[str, Any]) -> dict[str, Any]:
    per_device = list(communication["per_device_overlap"].values())
    return {
        "average_communication_compute_overlap_percent": statistics.fmean(
            item["communication_overlap_percent"] for item in per_device
        ),
        "average_exposed_communication_ms_per_step": statistics.fmean(
            item["exposed_communication_ms_per_step"] for item in per_device
        ),
        "average_idle_or_gap_ms_per_step": statistics.fmean(
            item.get("idle_or_gap_ms_per_step", 0.0) for item in per_device
        ),
        "per_device_overlap": communication["per_device_overlap"],
    }


def peak_vram_mib(run: dict[str, Any]) -> dict[str, Any]:
    rank_memory = [
        {
            "rank": item["rank"],
            "peak_allocated_memory_mib": item["peak_allocated_memory_mib"],
            "peak_reserved_memory_mib": item["peak_reserved_memory_mib"],
        }
        for item in run["rank_timing_and_memory"]
    ]
    smi = {
        device: values["peak_memory_mib"]
        for device, values in (run.get("gpu_monitoring") or {}).items()
    }
    return {
        "per_rank_torch": rank_memory,
        "per_gpu_nvidia_smi_mib": smi,
        "max_allocated_mib": max(item["peak_allocated_memory_mib"] for item in rank_memory),
        "max_nvidia_smi_mib": max(smi.values()) if smi else None,
    }


def userbuffers_active(run: dict[str, Any]) -> bool:
    runtimes = run.get("userbuffers_runtime") or []
    return bool(runtimes) and all(item.get("active") for item in runtimes)


def assert_variant_roles(
    variant_a: dict[str, Any],
    variant_b: dict[str, Any],
    variant_c: dict[str, Any],
) -> None:
    for name, run, want_sp, want_te, want_ub in (
        ("A", variant_a, False, False, False),
        ("B", variant_b, True, True, False),
        ("C", variant_c, True, True, True),
    ):
        parallelism = run["parallelism"]
        if parallelism["tensor_parallel"] != 2:
            raise RuntimeError(f"Variant {name} is not TP=2")
        if bool(parallelism.get("sequence_parallel")) != want_sp:
            raise RuntimeError(f"Variant {name} sequence_parallel mismatch")
        if bool(parallelism.get("te_linear")) != want_te:
            raise RuntimeError(f"Variant {name} te_linear mismatch")
        if bool(parallelism.get("tp_comm_overlap")) != want_ub:
            raise RuntimeError(f"Variant {name} tp_comm_overlap mismatch")
        if run["model_config"]["cuda_graph_impl"] != "none":
            raise RuntimeError("CUDA Graph must remain disabled")
        if run["model_config"]["bias_activation_fusion"]:
            raise RuntimeError("bias_gelu_fusion must remain False")
        if not run["model_config"]["bias_dropout_fusion"]:
            raise RuntimeError("bias_dropout_fusion must remain True")
    if userbuffers_active(variant_a) or userbuffers_active(variant_b):
        raise RuntimeError("Userbuffers leaked into A or B")
    if not userbuffers_active(variant_c):
        raise RuntimeError("Variant C did not activate Userbuffers")


def collective_snapshot(communication: dict[str, Any]) -> dict[str, Any]:
    types = communication["collective_types"]
    overlap = overlap_summary(communication)
    return {
        "all_reduce_count_per_step": types["All-Reduce"][
            "estimated_logical_count_per_step"
        ],
        "all_gather_count_per_step": types["All-Gather"][
            "estimated_logical_count_per_step"
        ],
        "reduce_scatter_count_per_step": types["Reduce-Scatter"][
            "estimated_logical_count_per_step"
        ],
        "all_reduce_ms_per_step": types["All-Reduce"][
            "average_kernel_time_ms_per_step_per_gpu"
        ],
        "all_gather_ms_per_step": types["All-Gather"][
            "average_kernel_time_ms_per_step_per_gpu"
        ],
        "reduce_scatter_ms_per_step": types["Reduce-Scatter"][
            "average_kernel_time_ms_per_step_per_gpu"
        ],
        "nccl_ms_per_step": communication[
            "average_nccl_kernel_time_ms_per_step_per_gpu"
        ],
        "userbuffer_ms_per_step": communication.get(
            "average_userbuffer_kernel_time_ms_per_step_per_gpu", 0.0
        ),
        "p2p_memcpy_ms_per_step": communication.get(
            "average_p2p_memcpy_ms_per_step_per_gpu", 0.0
        ),
        "communication_ms_per_step": communication.get(
            "average_communication_kernel_time_ms_per_step_per_gpu",
            communication["average_nccl_kernel_time_ms_per_step_per_gpu"],
        ),
        **overlap,
    }


def speedup(before: float, after: float) -> dict[str, float]:
    return {
        "speedup": after / before if before else 0.0,
        "throughput_gain_percent": (
            (after / before - 1.0) * 100.0 if before else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    if args.abort_reason or topology.get("status") == "abort":
        payload = {
            "status": "aborted",
            "experiment": "Phase 7.4 TE Userbuffers TP communication overlap",
            "iteration_mode": "FAST ITERATION MODE",
            "abort_reason": args.abort_reason or topology.get("abort_reason"),
            "topology": topology,
            "valid_tp2_baseline_reference": {
                "commit": PHASE71_VALID_BASELINE_COMMIT,
                "pod_id": PHASE71_VALID_POD_ID,
                "phase72_p2p_disabled_pod_id": PHASE72_P2P_DISABLED_POD_ID,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    variant_a = load(args.variant_a)
    variant_b = load(args.variant_b)
    variant_c = load(args.variant_c)
    assert_variant_roles(variant_a, variant_b, variant_c)

    communication_a = analyze_trace(args.sqlite_a, args.trace_a, load(args.variant_a_profile))
    communication_b = analyze_trace(args.sqlite_b, args.trace_b, load(args.variant_b_profile))
    communication_c = analyze_trace(args.sqlite_c, args.trace_c, load(args.variant_c_profile))
    collectives = {
        "A": collective_snapshot(communication_a),
        "B": collective_snapshot(communication_b),
        "C": collective_snapshot(communication_c),
    }

    summary_a = timing_summary(variant_a)
    summary_b = timing_summary(variant_b)
    summary_c = timing_summary(variant_c)
    b_to_c = speedup(summary_b["tokens_per_second"], summary_c["tokens_per_second"])
    a_to_c = speedup(summary_a["tokens_per_second"], summary_c["tokens_per_second"])
    a_to_b = speedup(summary_a["tokens_per_second"], summary_b["tokens_per_second"])

    exposed_b = collectives["B"]["average_exposed_communication_ms_per_step"]
    exposed_c = collectives["C"]["average_exposed_communication_ms_per_step"]
    formal_required = b_to_c["throughput_gain_percent"] >= FORMAL_GAIN_THRESHOLD_PERCENT
    formal_b = load(args.formal_b) if args.formal_b else None
    formal_c = load(args.formal_c) if args.formal_c else None
    if formal_required and formal_b and formal_c:
        reported_b = timing_summary(formal_b)
        reported_c = timing_summary(formal_c)
        reported_protocol = "formal_20_plus_100"
        reported_b_to_c = speedup(
            reported_b["tokens_per_second"], reported_c["tokens_per_second"]
        )
    else:
        reported_b = summary_b
        reported_c = summary_c
        reported_protocol = "fast_screen_5_plus_20"
        reported_b_to_c = b_to_c

    overlap_c = collectives["C"]["average_communication_compute_overlap_percent"]
    remaining = "exposed TP communication"
    if overlap_c >= 40.0 and b_to_c["throughput_gain_percent"] >= 3.0:
        remaining = "compute / GEMM after hiding a substantial fraction of TP communication"
    elif collectives["C"]["nccl_ms_per_step"] > collectives["C"]["userbuffer_ms_per_step"]:
        remaining = "remaining NCCL collectives plus unhidden Userbuffer/PCIe traffic"

    payload = {
        "status": "success",
        "experiment": "Phase 7.4 TE Userbuffers TP communication overlap",
        "iteration_mode": "FAST ITERATION MODE",
        "valid_tp2_baseline_reference": {
            "commit": PHASE71_VALID_BASELINE_COMMIT,
            "pod_id": PHASE71_VALID_POD_ID,
            "phase72_p2p_disabled_pod_id": PHASE72_P2P_DISABLED_POD_ID,
            "note": (
                "Use the corrected same-NUMA Phase 7.1 TP=2 result as the normal TP "
                "baseline. Phase 7.2 is a memory experiment on a P2P-disabled host."
            ),
        },
        "infrastructure": topology.get("infrastructure"),
        "topology": {
            "gpu0_gpu1_path": topology.get("gpu0_gpu1_path"),
            "same_numa_acceptable": topology.get("same_numa_acceptable"),
            "p2p_bidirectional": topology["p2p_accessibility"]["bidirectional_gpu0_gpu1"],
            "nccl_sanity_passed": topology["nccl_all_reduce_sanity"]["passed"],
            "commands": topology.get("commands"),
            "p2p_accessibility": topology.get("p2p_accessibility"),
        },
        "correctness": {
            "A": variant_a["correctness_smoke"],
            "B": variant_b["correctness_smoke"],
            "C": variant_c["correctness_smoke"],
        },
        "userbuffers_runtime": {
            "A": variant_a.get("userbuffers_runtime"),
            "B": variant_b.get("userbuffers_runtime"),
            "C": variant_c.get("userbuffers_runtime"),
        },
        "fast_screen": {
            "A": summary_a,
            "B": summary_b,
            "C": summary_c,
            "B_over_A": a_to_b,
            "C_over_B": b_to_c,
            "C_over_A": a_to_c,
        },
        "memory": {
            "A": peak_vram_mib(variant_a),
            "B": peak_vram_mib(variant_b),
            "C": peak_vram_mib(variant_c),
        },
        "communication": collectives,
        "exposed_nccl_reduction_B_to_C": {
            "B_exposed_ms": exposed_b,
            "C_exposed_ms": exposed_c,
            "delta_ms": exposed_c - exposed_b,
            "percent": (
                (exposed_b - exposed_c) / exposed_b * 100.0 if exposed_b else 0.0
            ),
        },
        "formal_benchmark": (
            {
                "B": timing_summary(formal_b),
                "C": timing_summary(formal_c),
                "C_over_B": reported_b_to_c,
            }
            if formal_b and formal_c
            else None
        ),
        "reported_protocol": reported_protocol,
        "reported_throughput": {
            "A": summary_a,
            "B": reported_b,
            "C": reported_c,
            "B_to_C_userbuffers_speedup": reported_b_to_c,
            "A_to_C_net_speedup": speedup(
                summary_a["tokens_per_second"],
                reported_c["tokens_per_second"],
            ),
        },
        "decision": {
            "cuda_graph_enabled": False,
            "other_optimization_added": False,
            "formal_20_plus_100_required": formal_required,
            "formal_20_plus_100_ran": bool(formal_b and formal_c),
            "keep_fast_screen": not bool(formal_b and formal_c),
            "dominant_remaining_bottleneck": remaining,
        },
        "environment": variant_c.get("environment"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PHASE74_ANALYSIS_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
