#!/usr/bin/env python3
"""Four-rank NCCL all-reduce sanity; host topo comes from phase10_preflight.py."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from phase7_tp_run import collect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = json.loads(args.preflight_json.read_text(encoding="utf-8"))
    if preflight.get("abort_reason"):
        raise RuntimeError(preflight["abort_reason"])

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    try:
        world = dist.get_world_size()
        if world != 4:
            raise RuntimeError(f"Topology test requires exactly four ranks, got {world}")
        if torch.cuda.device_count() != 4:
            raise RuntimeError(
                f"Topology test requires exactly four visible GPUs, got {torch.cuda.device_count()}"
            )

        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        scalar = torch.tensor([float(rank + 1)], device=device)
        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        all_reduce_ms = (time.perf_counter() - start) * 1000.0
        expected_sum = sum(range(1, world + 1))
        sanity = {
            "rank": rank,
            "result": float(scalar.item()),
            "expected": float(expected_sum),
            "passed": float(scalar.item()) == float(expected_sum),
            "latency_ms": all_reduce_ms,
        }
        sanity_by_rank: list[Any] = [None] * world
        dist.all_gather_object(sanity_by_rank, sanity)
        abort_reason = None
        if rank == 0:
            p2p_matrix = preflight["p2p_matrix"]
            result = {
                "status": "success",
                "experiment": "Phase 10.1 four-A40 topology and NCCL P2P sanity",
                "infrastructure": {
                    "pod_id": args.pod_id,
                    "pod_count": 1,
                    "gpu_count": 4,
                    "gpu_type": "NVIDIA A40 48GB",
                    "same_physical_host": True,
                    "price_per_hour_usd": args.price_per_hour_usd,
                    "price_target_usd": 1.80,
                    "price_target_met": args.price_per_hour_usd <= 1.80,
                },
                "preflight": preflight,
                "commands": preflight.get("commands", {}),
                "topology_matrix": preflight.get("topology_matrix", {}),
                "gpu_pair_paths": preflight.get("gpu_pair_paths", []),
                "numa_affinity_by_gpu": preflight.get("numa_affinity_by_gpu", {}),
                "same_numa": preflight.get("same_numa", True),
                "abort_reason": None,
                "p2p_accessibility": {
                    "matrix": p2p_matrix,
                    "fully_bidirectional": all(
                        p2p_matrix[i][j] and p2p_matrix[j][i]
                        for i in range(4)
                        for j in range(4)
                        if i != j
                    ),
                },
                "nccl_all_reduce_sanity": {
                    "backend": dist.get_backend(),
                    "per_rank": sanity_by_rank,
                    "passed": all(item["passed"] for item in sanity_by_rank),
                },
                "pairwise_nccl_tests": preflight.get("pairwise_nccl_tests", []),
                "environment": collect_environment(tensor_parallel_size=4),
            }
            if not result["nccl_all_reduce_sanity"]["passed"]:
                result["status"] = "abort"
                result["abort_reason"] = "NCCL All-Reduce sanity failed"
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print("PHASE10_TOPOLOGY_JSON=" + json.dumps({"status": result["status"]}))
            abort_reason = result["abort_reason"]

        dist.barrier()
        if abort_reason is not None:
            raise RuntimeError(f"PHASE101_ABORT:{abort_reason}")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
