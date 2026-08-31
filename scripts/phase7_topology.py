#!/usr/bin/env python3
"""Record two-GPU topology and run a minimal NCCL sanity check."""

from __future__ import annotations

import argparse
import json
import os
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
        help="Phase 9.2 host scarcity: continue when nvidia-smi reports SYS if NCCL/P2P pass",
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


def gpu0_gpu1_path(topology_output: str) -> str | None:
    for line in topology_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU0"):
            fields = stripped.split()
            if len(fields) >= 3:
                return fields[2]
    return None


def is_acceptable_gpu0_gpu1_path(path: str | None) -> bool:
    """Accept PCIe same-NUMA paths and NVLink. Reject SYS and unknown labels."""

    if path is None:
        return False
    if path in {"NODE", "PIX", "PHB", "PXB"}:
        return True
    return path.startswith("NV") and path[2:].isdigit()


def numa_affinity_by_gpu(topology_output: str) -> dict[str, str]:
    """Parse NUMA Affinity from `nvidia-smi topo -m` GPU rows."""

    affinities: dict[str, str] = {}
    for line in topology_output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("GPU") or stripped.startswith("GPU "):
            continue
        fields = stripped.split()
        if len(fields) < 5 or not fields[0][:3] == "GPU" or not fields[0][3:].isdigit():
            continue
        # Two-GPU rows look like: GPU0 X PIX 0-31 0 N/A
        affinities[fields[0]] = fields[4]
    return affinities


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
        if dist.get_world_size() != 2:
            raise RuntimeError("Topology test requires exactly two ranks")
        if torch.cuda.device_count() != 2:
            raise RuntimeError("Topology test requires exactly two visible GPUs")

        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        scalar = torch.tensor([float(rank + 1)], device=device)
        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        all_reduce_ms = (time.perf_counter() - start) * 1000.0
        sanity = {
            "rank": rank,
            "result": float(scalar.item()),
            "expected": 3.0,
            "passed": float(scalar.item()) == 3.0,
            "latency_ms": all_reduce_ms,
        }
        sanity_by_rank: list[Any] = [None, None]
        dist.all_gather_object(sanity_by_rank, sanity)

        p2p_matrix = [
            [
                i == j or torch.cuda.can_device_access_peer(i, j)
                for j in range(torch.cuda.device_count())
            ]
            for i in range(torch.cuda.device_count())
        ]
        p2p_by_rank: list[Any] = [None, None]
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
        interconnect = gpu0_gpu1_path(topology_output)
        numa = numa_affinity_by_gpu(topology_output)
        same_numa_ok = is_acceptable_gpu0_gpu1_path(interconnect)
        numa_values = {value for value in numa.values() if value not in {"N/A", "NA"}}
        abort_reason = None
        if interconnect == "SYS" and not args.allow_sys_topology:
            abort_reason = "cross-NUMA SYS topology"
        elif not same_numa_ok:
            abort_reason = f"unsupported GPU0-GPU1 path {interconnect}"
        elif len(numa_values) > 1:
            abort_reason = f"cross-NUMA GPU affinities {numa}"
        elif not (p2p_matrix[0][1] and p2p_matrix[1][0]):
            abort_reason = "CUDA peer access is not bidirectional"
        elif not all(item["passed"] for item in sanity_by_rank):
            abort_reason = "NCCL All-Reduce sanity failed"
        result = {
            "status": "success" if abort_reason is None else "abort",
            "experiment": "Phase 7.4 two-A40 topology and NCCL P2P sanity",
            "infrastructure": {
                "pod_id": args.pod_id,
                "pod_count": 1,
                "gpu_count": 2,
                "gpu_type": "NVIDIA A40 48GB",
                "same_physical_host": True,
                "price_per_hour_usd": args.price_per_hour_usd,
                "price_target_usd": 0.90,
                "price_target_met": args.price_per_hour_usd <= 0.90,
            },
            "commands": commands,
            "gpu0_gpu1_path": interconnect,
            "numa_affinity_by_gpu": numa,
            "same_numa_acceptable": same_numa_ok and len(numa_values) <= 1,
            "abort_reason": abort_reason,
            "p2p_accessibility": {
                "matrix": p2p_matrix,
                "consistent_across_ranks": all(
                    item["matrix"] == p2p_matrix for item in p2p_by_rank
                ),
                "bidirectional_gpu0_gpu1": (
                    p2p_matrix[0][1] and p2p_matrix[1][0]
                ),
            },
            "nccl_all_reduce_sanity": {
                "backend": dist.get_backend(),
                "per_rank": sanity_by_rank,
                "passed": all(item["passed"] for item in sanity_by_rank),
            },
            "environment": collect_environment(tensor_parallel_size=2),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print("PHASE7_TOPOLOGY_JSON=" + json.dumps(result, sort_keys=True))
        if abort_reason is not None:
            raise RuntimeError(f"PHASE74_ABORT:{abort_reason}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
