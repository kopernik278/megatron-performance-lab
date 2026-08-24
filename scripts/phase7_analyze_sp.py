#!/usr/bin/env python3
"""Analyze Phase 7.2 TP=2 sequence-parallel A/B timing and communication."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from phase7_analyze_tp import analyze_trace, load, timing_summary


FORMAL_GAIN_THRESHOLD_PERCENT = 2.0
PHASE71_VALID_BASELINE_COMMIT = "709437d"
PHASE71_VALID_POD_ID = "7rpwv95a5j6axg"
DISCARDED_CROSS_NUMA_POD_ID = "x72b8bn80zqdeg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--variant-a", type=Path, required=True)
    parser.add_argument("--variant-b", type=Path, required=True)
    parser.add_argument("--variant-a-profile", type=Path, required=True)
    parser.add_argument("--variant-b-profile", type=Path, required=True)
    parser.add_argument("--sqlite-a", type=Path, required=True)
    parser.add_argument("--sqlite-b", type=Path, required=True)
    parser.add_argument("--trace-a", type=Path, required=True)
    parser.add_argument("--trace-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-a", type=Path)
    parser.add_argument("--formal-b", type=Path)
    return parser.parse_args()


def model_controls_without_sp(run: dict[str, Any]) -> dict[str, Any]:
    config = dict(run["model_config"])
    config.pop("sequence_parallel", None)
    return config


def compare_controls(variant_a: dict[str, Any], variant_b: dict[str, Any]) -> None:
    if model_controls_without_sp(variant_a) != model_controls_without_sp(variant_b):
        raise RuntimeError("A/B model controls differ beyond sequence_parallel")
    for field in (
        "precision",
        "optimizer",
        "data",
        "tokens_per_step",
        "smoke_iterations",
        "warmup_iterations",
        "measured_iterations",
    ):
        if variant_a[field] != variant_b[field]:
            raise RuntimeError(f"A/B control differs: {field}")
    if variant_a["parallelism"]["tensor_parallel"] != 2:
        raise RuntimeError("Variant A is not TP=2")
    if variant_b["parallelism"]["tensor_parallel"] != 2:
        raise RuntimeError("Variant B is not TP=2")
    if variant_a["parallelism"].get("sequence_parallel"):
        raise RuntimeError("Variant A must have sequence_parallel=False")
    if not variant_b["parallelism"].get("sequence_parallel"):
        raise RuntimeError("Variant B must have sequence_parallel=True")
    if variant_a["model_config"]["cuda_graph_impl"] != "none":
        raise RuntimeError("CUDA Graph must remain disabled")
    if variant_b["model_config"]["bias_activation_fusion"]:
        raise RuntimeError("bias_gelu_fusion must remain False")
    if not variant_a["model_config"]["bias_dropout_fusion"]:
        raise RuntimeError("bias_dropout_fusion must remain True")


def peak_vram_mib(run: dict[str, Any]) -> dict[str, Any]:
    rank_memory = [
        {
            "rank": item["rank"],
            "peak_allocated_memory_mib": item["peak_allocated_memory_mib"],
            "peak_reserved_memory_mib": item["peak_reserved_memory_mib"],
        }
        for item in run["rank_timing_and_memory"]
    ]
    smi = {}
    monitoring = run.get("gpu_monitoring") or {}
    for device, stats in monitoring.items():
        smi[str(device)] = stats.get("peak_memory_mib")
    allocated = [item["peak_allocated_memory_mib"] for item in rank_memory]
    smi_values = [value for value in smi.values() if value is not None]
    return {
        "per_rank_torch": rank_memory,
        "per_gpu_nvidia_smi_mib": smi,
        "max_allocated_mib": max(allocated) if allocated else None,
        "max_nvidia_smi_mib": max(smi_values) if smi_values else None,
    }


def overlap_summary(communication: dict[str, Any]) -> dict[str, Any]:
    per_device = list(communication["per_device_overlap"].values())
    return {
        "average_communication_compute_overlap_percent": statistics.fmean(
            item["communication_overlap_percent"] for item in per_device
        ),
        "average_exposed_communication_ms_per_step": statistics.fmean(
            item["exposed_communication_ms_per_step"] for item in per_device
        ),
        "per_device_overlap": communication["per_device_overlap"],
    }


def collective_summary(communication: dict[str, Any]) -> dict[str, Any]:
    types = communication["collective_types"]
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
        "total_nccl_ms_per_step": communication[
            "average_nccl_kernel_time_ms_per_step_per_gpu"
        ],
        "nvtx_collective_shapes": communication.get("nvtx_collective_shapes"),
    }


def ring_volume_bytes(elements: float, dtype_bytes: int, kind: str, tp: int) -> float:
    payload = elements * dtype_bytes
    if kind == "All-Reduce":
        return 2.0 * (tp - 1) / tp * payload
    if kind in {"All-Gather", "Reduce-Scatter"}:
        return (tp - 1) / tp * payload
    return payload


def estimate_volume(communication: dict[str, Any], tp: int) -> dict[str, Any]:
    totals: dict[str, float] = {
        "All-Reduce": 0.0,
        "All-Gather": 0.0,
        "Reduce-Scatter": 0.0,
    }
    details = []
    shapes = communication.get("nvtx_collective_shapes") or {}
    for kind, entries in shapes.items():
        for entry in entries:
            elements = float(entry["elements"])
            calls = float(entry["calls_per_step_per_rank"])
            dtype_bytes = 2 if elements >= 1_000_000 else 4
            bytes_per_call = ring_volume_bytes(elements, dtype_bytes, kind, tp)
            bytes_per_step = bytes_per_call * calls
            totals[kind] += bytes_per_step
            details.append(
                {
                    "kind": kind,
                    "shape": entry["shape"],
                    "elements": elements,
                    "calls_per_step_per_rank": calls,
                    "assumed_dtype_bytes": dtype_bytes,
                    "ring_volume_bytes_per_step_per_rank": bytes_per_step,
                }
            )
    total = sum(totals.values())
    return {
        "method": (
            "NVTX collective input shapes with ring-algorithm volume. "
            "Tensors with >=1e6 elements are treated as BF16; smaller tensors "
            "as FP32. All-Gather/Reduce-Scatter volume uses (TP-1)/TP of the "
            "NVTX-reported tensor; All-Reduce uses 2*(TP-1)/TP."
        ),
        "bytes_per_step_per_rank": totals,
        "total_bytes_per_step_per_rank": total,
        "mib_per_step_per_rank": total / 1024.0 / 1024.0,
        "details": details,
    }


def gpu_utilization(run: dict[str, Any]) -> dict[str, Any]:
    monitoring = run.get("gpu_monitoring") or {}
    return {
        str(device): {
            "average_utilization_percent": stats.get("average_utilization_percent"),
            "median_utilization_percent": stats.get("median_utilization_percent"),
        }
        for device, stats in monitoring.items()
    }


def topology_path(topology: dict[str, Any]) -> str:
    matrix = topology["commands"]["nvidia_smi_topology"]["output"]
    if "\nGPU0\t X \tSYS" in matrix:
        return "cross-NUMA SYS PCIe path"
    if "\nGPU0\t X \tNODE" in matrix:
        return "same-NUMA NODE PCIe path"
    return "observed PCIe path"


def source_communication_map(num_layers: int) -> dict[str, Any]:
    per_layer_all_reduces = num_layers * 4
    return {
        "baseline_tp2_sp_false": {
            "large_activation_all_reduces_per_step": per_layer_all_reduces + 1,
            "per_layer_all_reduces": {
                "attention_projection_forward": num_layers,
                "mlp_fc2_forward": num_layers,
                "qkv_dgrad_backward": num_layers,
                "fc1_dgrad_backward": num_layers,
            },
            "output_layer_dgrad_all_reduce": 1,
            "remaining_all_reduces": {
                "embedding_forward": 1,
                "vocab_parallel_cross_entropy": 3,
            },
            "all_gather": 0,
            "reduce_scatter": 0,
            "expected_total_all_reduces": per_layer_all_reduces + 1 + 1 + 3,
        },
        "sequence_parallel_tp2": {
            "replaced_per_layer_all_reduces": per_layer_all_reduces,
            "replacement": {
                "attention_projection_forward_all_reduce": (
                    "row-parallel reduce_scatter_to_sequence_parallel_region"
                ),
                "mlp_fc2_forward_all_reduce": (
                    "row-parallel reduce_scatter_to_sequence_parallel_region"
                ),
                "qkv_dgrad_backward_all_reduce": (
                    "column-parallel reduce-scatter of sequence-parallel dgrad"
                ),
                "fc1_dgrad_backward_all_reduce": (
                    "column-parallel reduce-scatter of sequence-parallel dgrad"
                ),
                "added_column_parallel_forward_all_gather": (
                    "QKV, FC1, and output gather_from_sequence_parallel_region"
                ),
                "added_row_parallel_backward_all_gather": (
                    "projection and FC2 Reduce-Scatter backward All-Gather"
                ),
            },
            "output_layer_dgrad_all_reduce": (
                "replaced by output-layer Reduce-Scatter plus a forward All-Gather"
            ),
            "not_replaced": {
                "embedding_forward_all_reduce": (
                    "learned_absolute position embeddings keep "
                    "reduce_scatter_embeddings=False, so the vocab embedding still "
                    "All-Reduces and then splits into the sequence-parallel region"
                ),
                "vocab_parallel_cross_entropy": 3,
                "layernorm_weight_grad_all_reduce": (
                    "finalize_model_grads coalesces sequence-parallel LayerNorm "
                    "parameter grads into one TP All-Reduce"
                ),
            },
            "expected_all_gather": per_layer_all_reduces + 1 + 1,
            "expected_reduce_scatter": per_layer_all_reduces + 1,
            "expected_remaining_all_reduces": 5,
        },
    }


def sp_active(run: dict[str, Any]) -> bool:
    runtimes = run.get("sequence_parallel_runtime") or []
    if not runtimes:
        return False
    return all(item.get("active") for item in runtimes)


def main() -> None:
    args = parse_args()
    topology = load(args.topology)
    variant_a = load(args.variant_a)
    variant_b = load(args.variant_b)
    profile_a = load(args.variant_a_profile)
    profile_b = load(args.variant_b_profile)
    compare_controls(variant_a, variant_b)
    if model_controls_without_sp(profile_a) != model_controls_without_sp(variant_a):
        raise RuntimeError("Profile A changed model controls")
    if model_controls_without_sp(profile_b) != model_controls_without_sp(variant_b):
        raise RuntimeError("Profile B changed model controls")
    if not profile_b["parallelism"].get("sequence_parallel"):
        raise RuntimeError("Profile B did not enable sequence parallel")
    if not sp_active(variant_b) or not sp_active(profile_b):
        raise RuntimeError("Sequence parallel was not active at runtime on variant B")
    if sp_active(variant_a) or sp_active(profile_a):
        raise RuntimeError("Sequence parallel leaked into variant A")

    communication_a = analyze_trace(args.sqlite_a, args.trace_a, profile_a)
    communication_b = analyze_trace(args.sqlite_b, args.trace_b, profile_b)
    summary_a = collective_summary(communication_a)
    summary_b = collective_summary(communication_b)
    overlap_a = overlap_summary(communication_a)
    overlap_b = overlap_summary(communication_b)
    volume_a = estimate_volume(communication_a, 2)
    volume_b = estimate_volume(communication_b, 2)
    vram_a = peak_vram_mib(variant_a)
    vram_b = peak_vram_mib(variant_b)

    speedup = variant_b["tokens_per_second"] / variant_a["tokens_per_second"]
    throughput_gain_percent = (speedup - 1.0) * 100.0
    nccl_ratio = (
        summary_b["total_nccl_ms_per_step"] / summary_a["total_nccl_ms_per_step"]
        if summary_a["total_nccl_ms_per_step"]
        else None
    )
    volume_ratio = (
        volume_b["total_bytes_per_step_per_rank"]
        / volume_a["total_bytes_per_step_per_rank"]
        if volume_a["total_bytes_per_step_per_rank"]
        else None
    )
    vram_delta_mib = None
    vram_reduction_percent = None
    if vram_a["max_nvidia_smi_mib"] and vram_b["max_nvidia_smi_mib"]:
        vram_delta_mib = vram_b["max_nvidia_smi_mib"] - vram_a["max_nvidia_smi_mib"]
        vram_reduction_percent = (
            -vram_delta_mib / vram_a["max_nvidia_smi_mib"] * 100.0
        )

    smoke_loss_errors = [
        abs(left - right)
        for left, right in zip(
            variant_a["correctness_smoke"]["rank0_losses"],
            variant_b["correctness_smoke"]["rank0_losses"],
        )
    ]
    replaced_all_reduces = (
        summary_a["all_reduce_count_per_step"] - summary_b["all_reduce_count_per_step"]
    )
    added_all_gather = summary_b["all_gather_count_per_step"]
    added_reduce_scatter = summary_b["reduce_scatter_count_per_step"]

    used_formal = args.formal_a is not None and args.formal_b is not None
    formal = None
    reported_a = variant_a
    reported_b = variant_b
    protocol = "fast_screen_5_plus_20"
    if used_formal:
        formal_a = load(args.formal_a)
        formal_b = load(args.formal_b)
        compare_controls(formal_a, formal_b)
        formal_speedup = formal_b["tokens_per_second"] / formal_a["tokens_per_second"]
        formal = {
            "protocol": {
                "warmup_iterations": formal_a["warmup_iterations"],
                "measured_iterations": formal_a["measured_iterations"],
            },
            "A": timing_summary(formal_a),
            "B": timing_summary(formal_b),
            "speedup": formal_speedup,
            "throughput_gain_percent": (formal_speedup - 1.0) * 100.0,
        }
        if throughput_gain_percent >= FORMAL_GAIN_THRESHOLD_PERCENT:
            reported_a = formal_a
            reported_b = formal_b
            speedup = formal_speedup
            throughput_gain_percent = (speedup - 1.0) * 100.0
            vram_a = peak_vram_mib(formal_a)
            vram_b = peak_vram_mib(formal_b)
            protocol = "formal_20_plus_100"
    elif throughput_gain_percent >= FORMAL_GAIN_THRESHOLD_PERCENT:
        raise RuntimeError(
            "Fast-screen throughput gain is >=2% but formal 20+100 results "
            "were not provided"
        )

    path = topology_path(topology)
    result = {
        "status": "success",
        "experiment": "Phase 7.2 sequence-parallel A/B",
        "iteration_mode": "FAST ITERATION MODE",
        "valid_tp2_baseline_reference": {
            "commit": PHASE71_VALID_BASELINE_COMMIT,
            "pod_id": PHASE71_VALID_POD_ID,
            "discarded_cross_numa_pod_id": DISCARDED_CROSS_NUMA_POD_ID,
            "note": (
                "Use the corrected same-NUMA Phase 7.1 TP=2 result. The earlier "
                "cross-NUMA SYS result is invalid and must not be compared."
            ),
        },
        "infrastructure": topology["infrastructure"],
        "topology": topology,
        "configuration": {
            "model": model_controls_without_sp(variant_a),
            "precision": variant_a["precision"],
            "optimizer": variant_a["optimizer"],
            "A": {
                "tensor_model_parallel_size": 2,
                "sequence_parallel": False,
            },
            "B": {
                "tensor_model_parallel_size": 2,
                "sequence_parallel": True,
            },
            "unchanged": {
                "fused_attention": True,
                "bias_dropout_fusion": True,
                "bias_gelu_fusion": False,
                "cuda_graph": False,
                "micro_batch_size": variant_a["model_config"]["micro_batch_size"],
                "sequence_length": variant_a["model_config"]["sequence_length"],
            },
        },
        "sequence_parallel_runtime": {
            "A": variant_a.get("sequence_parallel_runtime"),
            "B": variant_b.get("sequence_parallel_runtime"),
            "B_active_on_both_ranks": sp_active(variant_b),
            "inspected_megatron_commit": variant_b["environment"]["megatron_lm_commit"],
        },
        "correctness": {
            "A": variant_a["correctness_smoke"],
            "B": variant_b["correctness_smoke"],
            "A_vs_B_smoke_loss_absolute_errors": smoke_loss_errors,
            "A_vs_B_smoke_loss_max_absolute_error": max(smoke_loss_errors),
            "passed": bool(
                topology["nccl_all_reduce_sanity"]["passed"]
                and variant_a["correctness_smoke"]["parameters_updated"]
                and variant_b["correctness_smoke"]["parameters_updated"]
                and all(
                    item["loss_finite"] and item["main_grads_finite"]
                    for item in variant_a["correctness_smoke"]["per_step_checks"]
                    + variant_b["correctness_smoke"]["per_step_checks"]
                )
                and sp_active(variant_b)
                and not sp_active(variant_a)
                and not variant_a["correctness_smoke"]["deadlock"]
                and not variant_b["correctness_smoke"]["deadlock"]
            ),
        },
        "fast_benchmark": {
            "protocol": {
                "warmup_iterations": variant_a["warmup_iterations"],
                "measured_iterations": variant_a["measured_iterations"],
                "micro_batch_size": variant_a["model_config"]["micro_batch_size"],
                "tokens_per_step": variant_a["tokens_per_step"],
            },
            "A_tp2_sp_false": timing_summary(variant_a),
            "B_tp2_sp_true": timing_summary(variant_b),
            "speedup": variant_b["tokens_per_second"] / variant_a["tokens_per_second"],
            "throughput_gain_percent": (
                variant_b["tokens_per_second"] / variant_a["tokens_per_second"] - 1.0
            )
            * 100.0,
        },
        "formal_benchmark": formal,
        "reported_protocol": protocol,
        "reported_throughput": {
            "A": timing_summary(reported_a),
            "B": timing_summary(reported_b),
            "speedup": speedup,
            "throughput_gain_percent": throughput_gain_percent,
        },
        "memory": {
            "A": vram_a,
            "B": vram_b,
            "B_minus_A_nvidia_smi_mib": vram_delta_mib,
            "activation_memory_reduction_percent_smi": vram_reduction_percent,
        },
        "gpu_utilization": {
            "A": gpu_utilization(reported_a),
            "B": gpu_utilization(reported_b),
        },
        "communication_profile": {
            "A": communication_a,
            "B": communication_b,
        },
        "communication_summary": {
            "A": {**summary_a, **overlap_a},
            "B": {**summary_b, **overlap_b},
            "replaced_all_reduces_per_step": replaced_all_reduces,
            "added_all_gather_per_step": added_all_gather,
            "added_reduce_scatter_per_step": added_reduce_scatter,
            "nccl_time_ratio_B_over_A": nccl_ratio,
            "volume_estimate": {
                "A": volume_a,
                "B": volume_b,
                "B_over_A": volume_ratio,
            },
        },
        "communication_transformation": {
            "topology_path": path,
            "source_map": source_communication_map(
                variant_a["model_config"]["num_layers"]
            ),
            "measured": {
                "A_all_reduce": summary_a["all_reduce_count_per_step"],
                "A_all_gather": summary_a["all_gather_count_per_step"],
                "A_reduce_scatter": summary_a["reduce_scatter_count_per_step"],
                "B_all_reduce": summary_b["all_reduce_count_per_step"],
                "B_all_gather": summary_b["all_gather_count_per_step"],
                "B_reduce_scatter": summary_b["reduce_scatter_count_per_step"],
            },
            "answers": {
                "which_all_reduces_replaced": (
                    "The 24 attention-projection forward All-Reduces, 24 MLP FC2 "
                    "forward All-Reduces, 24 QKV dgrad All-Reduces, and 24 FC1 "
                    "dgrad All-Reduces, plus the tied output-layer dgrad All-Reduce. "
                    f"Measured All-Reduce count fell by {replaced_all_reduces:.1f} "
                    "per step."
                ),
                "replaced_by": (
                    "Row-parallel forward Reduce-Scatter, column-parallel forward "
                    "All-Gather, column-parallel backward Reduce-Scatter, and "
                    "row-parallel backward All-Gather. Measured "
                    f"{added_all_gather:.1f} All-Gathers and "
                    f"{added_reduce_scatter:.1f} Reduce-Scatters per step."
                ),
                "communication_volume_or_time_improved": {
                    "nccl_ms_per_step_A": summary_a["total_nccl_ms_per_step"],
                    "nccl_ms_per_step_B": summary_b["total_nccl_ms_per_step"],
                    "nccl_time_ratio_B_over_A": nccl_ratio,
                    "volume_mib_per_step_A": volume_a["mib_per_step_per_rank"],
                    "volume_mib_per_step_B": volume_b["mib_per_step_per_rank"],
                    "volume_ratio_B_over_A": volume_ratio,
                },
                "activation_memory_reduction": {
                    "nvidia_smi_peak_A_mib": vram_a["max_nvidia_smi_mib"],
                    "nvidia_smi_peak_B_mib": vram_b["max_nvidia_smi_mib"],
                    "delta_mib": vram_delta_mib,
                    "percent": vram_reduction_percent,
                },
                "throughput_impact": {
                    "speedup": speedup,
                    "throughput_gain_percent": throughput_gain_percent,
                },
            },
        },
        "decision": {
            "cuda_graph_enabled": False,
            "other_optimization_added": False,
            "formal_20_plus_100_required": (
                (
                    variant_b["tokens_per_second"] / variant_a["tokens_per_second"]
                    - 1.0
                )
                * 100.0
                >= FORMAL_GAIN_THRESHOLD_PERCENT
            ),
            "formal_20_plus_100_ran": used_formal,
            "keep_fast_screen": protocol == "fast_screen_5_plus_20",
        },
        "environment": variant_b["environment"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PHASE7_SP_ANALYSIS_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
