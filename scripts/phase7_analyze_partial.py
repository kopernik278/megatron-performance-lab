#!/usr/bin/env python3
"""Analyze Phase 7.4b B vs C1 (AG-only Userbuffers) timing and overlap."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from phase7_analyze_tp import analyze_trace, load, timing_summary
from phase7_analyze_ub import (
    collective_snapshot,
    peak_vram_mib,
    speedup,
    userbuffers_active,
)


FORMAL_GAIN_THRESHOLD_PERCENT = 2.0
PHASE71_VALID_BASELINE_COMMIT = "709437d"
PHASE71_VALID_POD_ID = "7rpwv95a5j6axg"
PHASE72_P2P_DISABLED_POD_ID = "wtd9cxr3q8obuh"
PHASE74_HANG_KERNEL = "userbuffers_fp16_sum_inplace_gpu_rr_rs_oop"

C1_REQUIRED_FLAGS = {
    "tp_comm_overlap": True,
    "tp_comm_overlap_ag": True,
    "tp_comm_overlap_rs": False,
    "tp_comm_bulk_dgrad": False,
    "tp_comm_bulk_wgrad": False,
    "tp_comm_overlap_rs_dgrad": False,
}
C2_REQUIRED_FLAGS = {
    **C1_REQUIRED_FLAGS,
    "tp_comm_bulk_dgrad": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--variant-b", type=Path, required=True)
    parser.add_argument("--variant-c1", type=Path)
    parser.add_argument("--variant-b-profile", type=Path)
    parser.add_argument("--variant-c1-profile", type=Path)
    parser.add_argument("--sqlite-b", type=Path)
    parser.add_argument("--sqlite-c1", type=Path)
    parser.add_argument("--trace-b", type=Path)
    parser.add_argument("--trace-c1", type=Path)
    parser.add_argument("--variant-c2", type=Path)
    parser.add_argument("--variant-c2-profile", type=Path)
    parser.add_argument("--sqlite-c2", type=Path)
    parser.add_argument("--trace-c2", type=Path)
    parser.add_argument("--formal-b", type=Path)
    parser.add_argument("--formal-c1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abort-reason", default=None)
    return parser.parse_args()


def overlap_flags(run: dict[str, Any]) -> dict[str, bool]:
    parallelism = run.get("parallelism") or {}
    model = run.get("model_config") or {}
    return {
        name: bool(parallelism.get(name, model.get(name)))
        for name in (
            "tp_comm_overlap",
            "tp_comm_overlap_ag",
            "tp_comm_overlap_rs",
            "tp_comm_bulk_dgrad",
            "tp_comm_bulk_wgrad",
            "tp_comm_overlap_rs_dgrad",
        )
    }


def assert_common_controls(run: dict[str, Any], name: str) -> None:
    if run["parallelism"]["tensor_parallel"] != 2:
        raise RuntimeError(f"Variant {name} is not TP=2")
    if not bool(run["parallelism"].get("sequence_parallel")):
        raise RuntimeError(f"Variant {name} must use sequence_parallel")
    if not bool(run["parallelism"].get("te_linear")):
        raise RuntimeError(f"Variant {name} must use TE Linear")
    if run["model_config"]["cuda_graph_impl"] != "none":
        raise RuntimeError("CUDA Graph must remain disabled")
    if run["model_config"]["bias_activation_fusion"]:
        raise RuntimeError("bias_gelu_fusion must remain False")
    if not run["model_config"]["bias_dropout_fusion"]:
        raise RuntimeError("bias_dropout_fusion must remain True")


def assert_flags(run: dict[str, Any], name: str, expected: dict[str, bool]) -> None:
    observed = overlap_flags(run)
    for flag, want in expected.items():
        if observed[flag] != want:
            raise RuntimeError(
                f"Variant {name} {flag}={observed[flag]} expected {want}"
            )


def ag_overlap_snapshot(communication: dict[str, Any]) -> dict[str, Any]:
    per_device = list(communication["per_device_overlap"].values())
    return {
        "average_ag_gemm_overlap_percent": statistics.fmean(
            item.get("ag_gemm_overlap_percent", 0.0) for item in per_device
        ),
        "average_ag_communication_ms_per_step": statistics.fmean(
            item.get("ag_communication_union_ms_per_step", 0.0) for item in per_device
        ),
        "average_ag_gemm_overlap_ms_per_step": statistics.fmean(
            item.get("ag_gemm_overlap_ms_per_step", 0.0) for item in per_device
        ),
        "average_exposed_ag_ms_per_step": statistics.fmean(
            item.get("exposed_ag_communication_ms_per_step", 0.0) for item in per_device
        ),
        "ag_communication_kernel_launches": communication.get(
            "ag_communication_kernel_launches", 0
        ),
        "rs_communication_kernel_launches": communication.get(
            "rs_communication_kernel_launches", 0
        ),
        "hang_rs_kernel_launches": communication.get("hang_rs_kernel_launches", 0),
        "gemm_kernel_launches": communication.get("gemm_kernel_launches", 0),
        "top_ag_communication_kernels": communication.get(
            "top_ag_communication_kernels", []
        ),
        "top_userbuffer_kernels": communication.get("top_userbuffer_kernels", []),
        "top_gemm_kernels": communication.get("top_gemm_kernels", []),
        "per_device": {
            device: {
                "ag_gemm_overlap_percent": values.get("ag_gemm_overlap_percent", 0.0),
                "ag_communication_union_ms_per_step": values.get(
                    "ag_communication_union_ms_per_step", 0.0
                ),
                "ag_gemm_overlap_ms_per_step": values.get(
                    "ag_gemm_overlap_ms_per_step", 0.0
                ),
                "exposed_ag_communication_ms_per_step": values.get(
                    "exposed_ag_communication_ms_per_step", 0.0
                ),
            }
            for device, values in communication["per_device_overlap"].items()
        },
    }


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def abort_payload(
    topology: dict[str, Any],
    reason: str,
    variant_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "aborted",
        "experiment": "Phase 7.4b targeted TP communication overlap",
        "iteration_mode": "FAST ITERATION MODE",
        "abort_reason": reason,
        "phase74_hang_kernel": PHASE74_HANG_KERNEL,
        "overlap_flags_c1": C1_REQUIRED_FLAGS,
        "overlap_flags_c2": C2_REQUIRED_FLAGS,
        "collectives_targeted": ["All-Gather"],
        "collectives_disabled": ["Reduce-Scatter", "RS dgrad", "bulk wgrad RS"],
        "topology": topology,
        "variant_b": timing_summary(variant_b) if variant_b else None,
        "valid_tp2_baseline_reference": {
            "commit": PHASE71_VALID_BASELINE_COMMIT,
            "pod_id": PHASE71_VALID_POD_ID,
            "phase72_p2p_disabled_pod_id": PHASE72_P2P_DISABLED_POD_ID,
        },
    }


def main() -> None:
    args = parse_args()
    topology = load(args.topology) if args.topology.exists() else {}
    variant_b = load(args.variant_b) if args.variant_b and args.variant_b.exists() else None
    if args.abort_reason or topology.get("status") == "abort":
        write(
            args.output,
            abort_payload(
                topology,
                args.abort_reason or topology.get("abort_reason") or "aborted",
                variant_b,
            ),
        )
        return

    if variant_b is None:
        raise RuntimeError("Variant B result is required")
    assert_common_controls(variant_b, "B")
    if bool(variant_b["parallelism"].get("tp_comm_overlap")):
        raise RuntimeError("Variant B must keep Userbuffers off")
    if userbuffers_active(variant_b):
        raise RuntimeError("Userbuffers leaked into B")

    if args.variant_c1 is None or not args.variant_c1.exists():
        write(args.output, abort_payload(topology, "Variant C1 result missing", variant_b))
        return

    variant_c1 = load(args.variant_c1)
    assert_common_controls(variant_c1, "C1")
    assert_flags(variant_c1, "C1", C1_REQUIRED_FLAGS)
    if not userbuffers_active(variant_c1):
        raise RuntimeError("Variant C1 did not activate Userbuffers")
    c1_runtime = (variant_c1.get("userbuffers_runtime") or [{}])[0]
    if c1_runtime.get("mode") != "ag_only":
        raise RuntimeError(f"C1 Userbuffers mode is {c1_runtime.get('mode')}, expected ag_only")
    if c1_runtime.get("observed_paths", {}).get("row_rs_fprop"):
        raise RuntimeError("C1 activated the Reduce-Scatter fprop Userbuffers path")
    if c1_runtime.get("observed_paths", {}).get("column_bulk_wgrad"):
        raise RuntimeError("C1 activated bulk wgrad RS overlap")

    communication_b = None
    communication_c1 = None
    if args.sqlite_b and args.trace_b and args.variant_b_profile:
        communication_b = analyze_trace(
            args.sqlite_b, args.trace_b, load(args.variant_b_profile)
        )
    if args.sqlite_c1 and args.trace_c1 and args.variant_c1_profile:
        communication_c1 = analyze_trace(
            args.sqlite_c1, args.trace_c1, load(args.variant_c1_profile)
        )
    if communication_c1 and communication_c1.get("hang_rs_kernel_launches", 0):
        write(
            args.output,
            abort_payload(
                topology,
                f"C1 launched {PHASE74_HANG_KERNEL}",
                variant_b,
            ),
        )
        return

    collectives = {
        "B": collective_snapshot(communication_b) if communication_b else None,
        "C1": collective_snapshot(communication_c1) if communication_c1 else None,
    }
    ag_overlap = {
        "B": ag_overlap_snapshot(communication_b) if communication_b else None,
        "C1": ag_overlap_snapshot(communication_c1) if communication_c1 else None,
    }
    summary_b = timing_summary(variant_b)
    summary_c1 = timing_summary(variant_c1)
    b_to_c1 = speedup(summary_b["tokens_per_second"], summary_c1["tokens_per_second"])
    c1_overlap_percent = (
        ag_overlap["C1"]["average_ag_gemm_overlap_percent"]
        if ag_overlap["C1"]
        else collectives["C1"]["average_communication_compute_overlap_percent"]
        if collectives["C1"]
        else 0.0
    )
    comm_overlap_percent = (
        collectives["C1"]["average_communication_compute_overlap_percent"]
        if collectives["C1"]
        else 0.0
    )
    measured_overlap = max(c1_overlap_percent, comm_overlap_percent)
    formal_required = (
        measured_overlap > 0.0
        and b_to_c1["throughput_gain_percent"] >= FORMAL_GAIN_THRESHOLD_PERCENT
    )
    formal_b = load(args.formal_b) if args.formal_b and args.formal_b.exists() else None
    formal_c1 = (
        load(args.formal_c1) if args.formal_c1 and args.formal_c1.exists() else None
    )
    if formal_required and formal_b and formal_c1:
        reported_b = timing_summary(formal_b)
        reported_c1 = timing_summary(formal_c1)
        reported_protocol = "formal_20_plus_100"
        reported_b_to_c1 = speedup(
            reported_b["tokens_per_second"], reported_c1["tokens_per_second"]
        )
    else:
        reported_b = summary_b
        reported_c1 = summary_c1
        reported_protocol = "fast_screen_5_plus_20"
        reported_b_to_c1 = b_to_c1

    variant_c2 = load(args.variant_c2) if args.variant_c2 and args.variant_c2.exists() else None
    communication_c2 = None
    if (
        variant_c2
        and args.sqlite_c2
        and args.trace_c2
        and args.variant_c2_profile
        and args.variant_c2_profile.exists()
    ):
        assert_common_controls(variant_c2, "C2")
        assert_flags(variant_c2, "C2", C2_REQUIRED_FLAGS)
        communication_c2 = analyze_trace(
            args.sqlite_c2, args.trace_c2, load(args.variant_c2_profile)
        )
        collectives["C2"] = collective_snapshot(communication_c2)
        ag_overlap["C2"] = ag_overlap_snapshot(communication_c2)
    elif variant_c2:
        assert_common_controls(variant_c2, "C2")
        assert_flags(variant_c2, "C2", C2_REQUIRED_FLAGS)
        collectives["C2"] = None
        ag_overlap["C2"] = None

    remaining = "exposed TP All-Gather on A40 PCIe"
    if measured_overlap > 0 and b_to_c1["throughput_gain_percent"] >= 2.0:
        remaining = "remaining Reduce-Scatter plus unhidden AG after partial overlap"
    elif measured_overlap > 0:
        remaining = "AG overlap present but step time still dominated by compute or remaining RS"

    payload = {
        "status": "success",
        "experiment": "Phase 7.4b targeted TP communication overlap",
        "iteration_mode": "FAST ITERATION MODE",
        "phase74_hang_kernel": PHASE74_HANG_KERNEL,
        "overlap_flags_c1": C1_REQUIRED_FLAGS,
        "overlap_flags_c2": C2_REQUIRED_FLAGS if variant_c2 else None,
        "collectives_overlapped": ["All-Gather"],
        "collectives_not_overlapped": [
            "Reduce-Scatter fprop",
            "RS dgrad",
            "bulk wgrad RS",
        ],
        "valid_tp2_baseline_reference": {
            "commit": PHASE71_VALID_BASELINE_COMMIT,
            "pod_id": PHASE71_VALID_POD_ID,
            "phase72_p2p_disabled_pod_id": PHASE72_P2P_DISABLED_POD_ID,
            "note": (
                "Reference B is TP=2 + Sequence Parallel + TE Linear + Userbuffers "
                "OFF on this host. Do not mix with the Phase 7.2 P2P-disabled host."
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
            "B": variant_b["correctness_smoke"],
            "C1": variant_c1["correctness_smoke"],
            "C2": variant_c2["correctness_smoke"] if variant_c2 else None,
        },
        "userbuffers_runtime": {
            "B": variant_b.get("userbuffers_runtime"),
            "C1": variant_c1.get("userbuffers_runtime"),
            "C2": variant_c2.get("userbuffers_runtime") if variant_c2 else None,
        },
        "fast_screen": {
            "B": summary_b,
            "C1": summary_c1,
            "C2": timing_summary(variant_c2) if variant_c2 else None,
            "C1_over_B": b_to_c1,
            "C2_over_B": (
                speedup(
                    summary_b["tokens_per_second"],
                    variant_c2["tokens_per_second"],
                )
                if variant_c2
                else None
            ),
        },
        "memory": {
            "B": peak_vram_mib(variant_b),
            "C1": peak_vram_mib(variant_c1),
            "C2": peak_vram_mib(variant_c2) if variant_c2 else None,
        },
        "communication": collectives,
        "ag_gemm_overlap": ag_overlap,
        "exposed_communication_B_to_C1": {
            "B_exposed_ms": (
                collectives["B"]["average_exposed_communication_ms_per_step"]
                if collectives["B"]
                else None
            ),
            "C1_exposed_ms": (
                collectives["C1"]["average_exposed_communication_ms_per_step"]
                if collectives["C1"]
                else None
            ),
            "delta_ms": (
                collectives["C1"]["average_exposed_communication_ms_per_step"]
                - collectives["B"]["average_exposed_communication_ms_per_step"]
                if collectives["B"] and collectives["C1"]
                else None
            ),
        },
        "formal_benchmark": (
            {
                "B": timing_summary(formal_b),
                "C1": timing_summary(formal_c1),
                "C1_over_B": reported_b_to_c1,
            }
            if formal_b and formal_c1
            else None
        ),
        "reported_protocol": reported_protocol,
        "reported_throughput": {
            "B": reported_b,
            "C1": reported_c1,
            "B_to_C1_speedup": reported_b_to_c1,
        },
        "decision": {
            "cuda_graph_enabled": False,
            "rs_overlap_enabled": False,
            "ag_only_overlap_works": True,
            "ag_gemm_overlap_percent": c1_overlap_percent,
            "communication_compute_overlap_percent": comm_overlap_percent,
            "measured_overlap_gt_zero": measured_overlap > 0.0,
            "throughput_gain_percent": b_to_c1["throughput_gain_percent"],
            "formal_20_plus_100_required": formal_required,
            "formal_20_plus_100_ran": bool(formal_b and formal_c1),
            "c2_bulk_dgrad_tested": bool(variant_c2),
            "keep_fast_screen": not bool(formal_b and formal_c1),
            "dominant_remaining_bottleneck": remaining,
        },
        "environment": variant_c1.get("environment"),
    }
    write(args.output, payload)
    print("PHASE74B_ANALYSIS_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
