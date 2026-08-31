#!/usr/bin/env python3
"""Bounded microbatch capacity search for one Phase 12 variant."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--variant", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--run-script", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--start-mb", type=int, default=8)
    parser.add_argument("--max-mb-cap", type=int, default=64)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def variant_flags(variant: str) -> list[str]:
    # A: baseline; B: DistOpt+overlaps; C: AC; D: Combined
    common = ["--overlap-grad-reduce"]
    if variant == "A":
        return common + [
            "--no-use-distributed-optimizer",
            "--no-overlap-param-gather",
            "--no-activation-checkpointing",
        ]
    if variant == "B":
        return common + [
            "--use-distributed-optimizer",
            "--overlap-param-gather",
            "--no-activation-checkpointing",
        ]
    if variant == "C":
        return common + [
            "--no-use-distributed-optimizer",
            "--no-overlap-param-gather",
            "--activation-checkpointing",
        ]
    return common + [
        "--use-distributed-optimizer",
        "--overlap-param-gather",
        "--activation-checkpointing",
    ]


def probe(
    args: argparse.Namespace,
    micro_batch: int,
) -> dict[str, Any]:
    out = args.work_dir / f"capacity_{args.variant}_mb{micro_batch}.json"
    cmd = [
        "timeout",
        str(args.timeout_sec),
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(args.run_script),
        "--variant",
        args.variant,
        "--run-label",
        f"phase12_{args.variant}_cap_mb{micro_batch}",
        "--mode",
        "capacity-probe",
        "--micro-batch-size",
        str(micro_batch),
        "--sequence-length",
        str(args.sequence_length),
        "--output-json",
        str(out),
        *variant_flags(args.variant),
    ]
    completed = subprocess.run(cmd, check=False)
    status = "timeout" if completed.returncode == 124 else "unknown"
    payload: dict[str, Any] = {}
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        status = payload.get("status", status)
    elif completed.returncode != 0:
        status = "failed"
    peak = None
    if payload.get("peak_allocated_memory_mib"):
        peak = max(payload["peak_allocated_memory_mib"])
    return {
        "micro_batch_size": micro_batch,
        "status": status,
        "returncode": completed.returncode,
        "peak_allocated_mib": peak,
        "output_json": str(out) if out.exists() else None,
    }


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    search_log: list[dict[str, Any]] = []

    # Exponential growth from start_mb until fail or cap.
    mb = args.start_mb
    last_success = None
    first_fail = None
    while mb <= args.max_mb_cap:
        result = probe(args, mb)
        search_log.append(result)
        print(
            f"PHASE12_CAPACITY_PROBE variant={args.variant} mb={mb} status={result['status']}",
            flush=True,
        )
        if result["status"] == "success":
            last_success = result
            next_mb = mb * 2
            if next_mb > args.max_mb_cap and mb < args.max_mb_cap:
                mb = args.max_mb_cap
                continue
            mb = next_mb
            continue
        first_fail = result
        break

    # Binary search between last_success and first_fail (exclusive fail).
    if last_success is None:
        # Try stepping down from start_mb.
        mb = max(1, args.start_mb // 2)
        while mb >= 1 and last_success is None:
            result = probe(args, mb)
            search_log.append(result)
            print(
                f"PHASE12_CAPACITY_PROBE variant={args.variant} mb={mb} status={result['status']}",
                flush=True,
            )
            if result["status"] == "success":
                last_success = result
                break
            mb //= 2
        if last_success is None:
            payload = {
                "status": "failed",
                "variant": args.variant,
                "sequence_length": args.sequence_length,
                "max_micro_batch_size": 0,
                "search_log": search_log,
                "error": "no successful microbatch found",
            }
            args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(2)

    low = last_success["micro_batch_size"]
    high = first_fail["micro_batch_size"] - 1 if first_fail else low
    # If exponential never failed, high==low already at max.
    while low < high:
        mid = (low + high + 1) // 2
        if any(item["micro_batch_size"] == mid for item in search_log):
            # Already probed.
            prior = next(item for item in search_log if item["micro_batch_size"] == mid)
            ok = prior["status"] == "success"
        else:
            prior = probe(args, mid)
            search_log.append(prior)
            print(
                f"PHASE12_CAPACITY_PROBE variant={args.variant} mb={mid} status={prior['status']}",
                flush=True,
            )
            ok = prior["status"] == "success"
        if ok:
            low = mid
            last_success = prior
        else:
            high = mid - 1

    payload = {
        "status": "success",
        "variant": args.variant,
        "sequence_length": args.sequence_length,
        "max_micro_batch_size": last_success["micro_batch_size"],
        "global_batch_size": last_success["micro_batch_size"] * 2,
        "max_tokens_per_step": last_success["micro_batch_size"] * 2 * args.sequence_length,
        "peak_allocated_mib_at_max": last_success.get("peak_allocated_mib"),
        "search_log": search_log,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"PHASE12_CAPACITY_OK variant={args.variant} max_mb={payload['max_micro_batch_size']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
