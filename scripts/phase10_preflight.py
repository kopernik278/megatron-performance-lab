#!/usr/bin/env python3
"""Single-process host preflight before 4-rank NCCL topology (Phase 10.1)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--price-per-hour-usd", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-sys-topology", action="store_true")
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
    matrix: dict[str, dict[str, str | None]] = {}
    numa: dict[str, str] = {}
    for line in topology_output.splitlines():
        stripped = line.strip()
        if not re.match(r"^GPU\d+\s", stripped):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            continue
        src = fields[0]
        numa[src] = fields[4] if len(fields) >= 5 else "N/A"
        matrix[src] = {}
        for dst_index in range(gpu_count):
            dst = f"GPU{dst_index}"
            col = dst_index + 1
            matrix[src][dst] = fields[col] if col < len(fields) else None
    return {"matrix": matrix, "numa_affinity_by_gpu": numa}


def is_acceptable_link(path: str | None, allow_sys: bool) -> bool:
    if path is None:
        return False
    if path == "SYS":
        return allow_sys
    if path in {"NODE", "PIX", "PHB", "PXB"}:
        return True
    return path.startswith("NV") and path[2:].isdigit()


def nccl_pair_sanity(gpu_a: int, gpu_b: int, timeout_sec: int = 90) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{gpu_a},{gpu_b}"
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_NVLS_ENABLE", "0")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).with_name("phase10_nccl_pair.py")),
        "--expected-sum",
        "3",
    ]
    started = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        check=False,
    )
    return {
        "gpu_a": gpu_a,
        "gpu_b": gpu_b,
        "exit_code": started.returncode,
        "passed": started.returncode == 0,
        "output_tail": started.stdout.strip()[-2000:],
    }


def main() -> None:
    args = parse_args()
    gpu_count = torch.cuda.device_count()
    if gpu_count != 4:
        raise SystemExit(f"Need exactly 4 GPUs, got {gpu_count}")

    commands = {
        "nvidia_smi": run(["nvidia-smi"]),
        "nvidia_smi_topology": run(["nvidia-smi", "topo", "-m"]),
        "nvidia_smi_nvlink": run(["nvidia-smi", "nvlink", "--status"]),
    }
    topology_output = commands["nvidia_smi_topology"]["output"]
    parsed = parse_topology_matrix(topology_output, gpu_count)
    matrix = parsed["matrix"]
    numa = parsed["numa_affinity_by_gpu"]

    p2p_matrix = [
        [
            i == j or torch.cuda.can_device_access_peer(i, j)
            for j in range(gpu_count)
        ]
        for i in range(gpu_count)
    ]
    pair_paths: list[dict[str, Any]] = []
    unacceptable_pairs: list[str] = []
    for i in range(gpu_count):
        for j in range(i + 1, gpu_count):
            path = matrix.get(f"GPU{i}", {}).get(f"GPU{j}")
            acceptable = is_acceptable_link(path, args.allow_sys_topology)
            bidirectional = p2p_matrix[i][j] and p2p_matrix[j][i]
            pair_paths.append(
                {
                    "gpu_i": i,
                    "gpu_j": j,
                    "path": path,
                    "acceptable": acceptable,
                    "p2p_bidirectional": bidirectional,
                }
            )
            if not acceptable:
                unacceptable_pairs.append(f"GPU{i}-GPU{j}:{path}")
            elif not bidirectional:
                unacceptable_pairs.append(f"GPU{i}-GPU{j}:p2p_not_bidirectional")

    pair_tests = [
        nccl_pair_sanity(0, 1),
        nccl_pair_sanity(2, 3),
        nccl_pair_sanity(0, 2),
        nccl_pair_sanity(1, 3),
    ]
    numa_values = {value for value in numa.values() if value not in {"N/A", "NA"}}
    abort_reason = None
    if unacceptable_pairs:
        abort_reason = f"preflight rejected pairs: {', '.join(unacceptable_pairs)}"
    elif len(numa_values) > 1:
        abort_reason = f"cross-NUMA GPU affinities {numa}"
    elif not all(item["passed"] for item in pair_tests):
        failed = [f"{t['gpu_a']}-{t['gpu_b']}" for t in pair_tests if not t["passed"]]
        abort_reason = f"pairwise NCCL sanity failed for GPU pairs {failed}"

    payload = {
        "status": "success" if abort_reason is None else "abort",
        "stage": "preflight",
        "pod_id": args.pod_id,
        "price_per_hour_usd": args.price_per_hour_usd,
        "topology_matrix": matrix,
        "gpu_pair_paths": pair_paths,
        "numa_affinity_by_gpu": numa,
        "same_numa": len(numa_values) <= 1,
        "p2p_matrix": p2p_matrix,
        "pairwise_nccl_tests": pair_tests,
        "commands": commands,
        "abort_reason": abort_reason,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PHASE10_PREFLIGHT_JSON=" + json.dumps({"status": payload["status"]}))
    if abort_reason is not None:
        raise SystemExit(abort_reason)


if __name__ == "__main__":
    main()
