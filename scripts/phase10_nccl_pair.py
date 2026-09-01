#!/usr/bin/env python3
"""Minimal 2-rank NCCL all-reduce sanity for Phase 10.1 preflight."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sum", type=float, default=3.0)
    return parser.parse_args()


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
            raise RuntimeError("pair sanity requires world_size=2")
        device = torch.device(f"cuda:{local_rank}")
        value = torch.tensor([float(dist.get_rank() + 1)], device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        if float(value.item()) != args.expected_sum:
            raise RuntimeError(f"expected {args.expected_sum}, got {float(value.item())}")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
