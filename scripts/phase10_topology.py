#!/usr/bin/env python3
"""Record four-GPU topology and run NCCL/P2P sanity for Phase 10.1."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
    parser.add_argument(
        "--allow-sys-topology",
        action="store_true",
        help="Continue when some GPU pairs report SYS if NCCL/P2P pass",
    )
    return parser.parse_args()


def run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output": completed.stdout.strip(),
    }


def parse_topology_matrix(topology_output: str, gpu_count: int) -> dict[str, Any]:
    """Parse nvidia-smi topo -m into a GPUxGPU link matrix."""

    matrix: dict[str, dict[str, str | None]] = {}
    numa: dict[str, str] = {}
    gpu_rows = [
        line.strip()
        for line in topology_output.splitlines()
        if re.match(r"^GPU\d+\s", line.strip())
    ]
    for row in gpu_rows:
        fields = row.split()
        if len(fields) < 2 or not fields[0].startswith("GPU"):
            continue
        src = fields[0]
        numa[src] = fields[4] if len(fields) >= 5 else "N/A"
        matrix[src] = {}
        for dst_index in range(gpu_count):
            dst = f"GPU{dst_index}"
            col = dst_index + 1
            matrix[src][dst] = fields[col] if col < len(fields) else None
    return {"matrix": matrix, "numa_affinity_by_gpu": numa}


def is_acceptable_link(path: str | None) -> bool:
    if path is None:
        return False
    if path in {"NODE", "PIX", "PHB", "PXB"}:
        return True
    return path.startswith("NV") and path[2:].isdigit()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    try:
        world = dist.get_world_size()
        gpu_count = torch.cuda.device_count()
        if world != 4:
            raise RuntimeError(f"Topology test requires exactly four ranks, got {world}")
        if gpu_count != 4:
            raise RuntimeError(f"Topology test requires exactly four visible GPUs, got {gpu_count}")

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

        p2p_matrix = [
            [
                i == j or torch.cuda.can_device_access_peer(i, j)
                for j in range(gpu_count)
            ]
            for i in range(gpu_count)
        ]
        p2p_by_rank: list[Any] = [None] * world
        dist.all_gather_object(p2p_by_rank, {"rank": rank, "matrix": p2p_matrix})

        if rank != 0:
            return

        commands = {
            "nvidia_smi": run(["nvidia-smi"]),
            "nvidia_smi_query": run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "nvidia_smi_topology": run(["nvidia-smi", "topo", "-m"]),
            "nvidia_smi_nvlink": run(["nvidia-smi", "nvlink", "--status"]),
            "lspci_nvidia": run(
                [
                    "bash",
                    "-lc",
                    "lspci -D | awk 'tolower($0) ~ /nvidia|vga|3d controller/'",
                ]
            ),
        }
        topology_output = commands["nvidia_smi_topology"]["output"]
        parsed = parse_topology_matrix(topology_output, gpu_count)
        matrix = parsed["matrix"]
        numa = parsed["numa_affinity_by_gpu"]

        pair_paths: list[dict[str, Any]] = []
        unacceptable_pairs: list[str] = []
        for i in range(gpu_count):
            for j in range(i + 1, gpu_count):
                path = matrix.get(f"GPU{i}", {}).get(f"GPU{j}")
                acceptable = is_acceptable_link(path)
                pair_paths.append(
                    {
                        "gpu_i": i,
                        "gpu_j": j,
                        "path": path,
                        "acceptable": acceptable,
                        "p2p_bidirectional": p2p_matrix[i][j] and p2p_matrix[j][i],
                    }
                )
                if path == "SYS" and not args.allow_sys_topology:
                    unacceptable_pairs.append(f"GPU{i}-GPU{j}:SYS")
                elif not acceptable:
                    unacceptable_pairs.append(f"GPU{i}-GPU{j}:{path}")

        numa_values = {value for value in numa.values() if value not in {"N/A", "NA"}}
        abort_reason = None
        if unacceptable_pairs:
            abort_reason = f"unsupported GPU links: {', '.join(unacceptable_pairs)}"
        elif not all(
            p2p_matrix[i][j] and p2p_matrix[j][i]
            for i in range(gpu_count)
            for j in range(gpu_count)
            if i != j
        ):
            abort_reason = "CUDA peer access is not fully bidirectional across all GPU pairs"
        elif not all(item["passed"] for item in sanity_by_rank):
            abort_reason = "NCCL All-Reduce sanity failed"
        elif len(numa_values) > 1:
            abort_reason = f"cross-NUMA GPU affinities {numa}"

        result = {
            "status": "success" if abort_reason is None else "abort",
            "experiment": "Phase 10.1 four-A40 topology and NCCL P2P sanity",
            "infrastructure": {
                "pod_id": args.pod_id,
                "pod_count": 1,
                "gpu_count": gpu_count,
                "gpu_type": "NVIDIA A40 48GB",
                "same_physical_host": True,
                "price_per_hour_usd": args.price_per_hour_usd,
                "price_target_usd": 1.80,
                "price_target_met": args.price_per_hour_usd <= 1.80,
            },
            "commands": commands,
            "topology_matrix": matrix,
            "gpu_pair_paths": pair_paths,
            "numa_affinity_by_gpu": numa,
            "same_numa": len(numa_values) <= 1,
            "abort_reason": abort_reason,
            "p2p_accessibility": {
                "matrix": p2p_matrix,
                "consistent_across_ranks": all(
                    item["matrix"] == p2p_matrix for item in p2p_by_rank
                ),
                "fully_bidirectional": all(
                    p2p_matrix[i][j] and p2p_matrix[j][i]
                    for i in range(gpu_count)
                    for j in range(gpu_count)
                    if i != j
                ),
            },
            "nccl_all_reduce_sanity": {
                "backend": dist.get_backend(),
                "per_rank": sanity_by_rank,
                "passed": all(item["passed"] for item in sanity_by_rank),
            },
            "environment": collect_environment(tensor_parallel_size=4),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE10_TOPOLOGY_JSON=" + json.dumps(result, sort_keys=True))
        if abort_reason is not None:
            raise RuntimeError(f"PHASE101_ABORT:{abort_reason}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
