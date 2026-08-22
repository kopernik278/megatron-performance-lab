#!/usr/bin/env python3
"""Minimal single-GPU Megatron Core GPT smoke test."""

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

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ffn-hidden-size", type=int, default=256)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/phase1_smoke_test/checkpoint"))
    parser.add_argument("--output-json", type=Path, default=Path("results/phase1_smoke_test/metrics.json"))
    return parser.parse_args()


def run_command(command: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {exc}"


def initialize_single_gpu_distributed(seed: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    return local_rank


def build_model(args: argparse.Namespace, dtype: torch.dtype) -> GPTModel:
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=False,
        attention_backend=AttnBackend.unfused,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=args.vocab_size,
        max_sequence_length=args.sequence_length,
        parallel_output=True,
    )
    return model.cuda()


def synthetic_batch(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    tokens = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.micro_batch_size, args.sequence_length),
        dtype=torch.long,
        device=device,
    )
    labels = torch.roll(tokens, shifts=-1, dims=1)
    position_ids = torch.arange(args.sequence_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(args.micro_batch_size, -1)
    loss_mask = torch.ones((args.micro_batch_size, args.sequence_length), dtype=torch.float32, device=device)
    attention_mask = torch.tril(
        torch.ones((args.sequence_length, args.sequence_length), dtype=torch.bool, device=device)
    )
    attention_mask = ~attention_mask.view(1, 1, args.sequence_length, args.sequence_length)
    attention_mask = attention_mask.expand(args.micro_batch_size, -1, -1, -1)
    return {
        "tokens": tokens,
        "labels": labels,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
    }


def masked_language_model_loss(output_tensor: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    losses = output_tensor.float()
    mask = loss_mask.view(-1).float()
    return torch.sum(losses.view(-1) * mask) / mask.sum()


def save_and_load_checkpoint(
    args: argparse.Namespace,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    final_loss: float,
) -> dict[str, Any]:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "single_gpu_smoke.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "final_loss": final_loss,
            "config": {
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "ffn_hidden_size": args.ffn_hidden_size,
                "num_attention_heads": args.num_attention_heads,
                "vocab_size": args.vocab_size,
                "sequence_length": args.sequence_length,
            },
        },
        checkpoint_path,
    )

    reloaded = build_model(args, torch.float32)
    payload = torch.load(checkpoint_path, map_location="cuda")
    missing, unexpected = reloaded.load_state_dict(payload["model"], strict=False)
    checks = []
    reloaded_state = reloaded.state_dict()
    for name, value in model.state_dict().items():
        loaded_value = reloaded_state[name]
        if torch.is_tensor(value):
            checks.append(torch.allclose(value.detach(), loaded_value.detach()))
        else:
            checks.append(value == loaded_value)
    del reloaded
    return {
        "path": str(checkpoint_path),
        "loaded": not missing and not unexpected and all(checks),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def collect_versions() -> dict[str, Any]:
    nccl_version = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
    megatron_version = run_command(["python", "-c", "import importlib.metadata as m; print(m.version('megatron-core'))"])
    megatron_commit = run_command(["git", "rev-parse", "HEAD"], cwd="/workspace/Megatron-LM")
    gpu_name = torch.cuda.get_device_name(0)
    driver = run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    )
    return {
        "python": run_command(["python", "--version"]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": ".".join(str(part) for part in nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
        "megatron": megatron_version,
        "megatron_commit": megatron_commit,
        "gpu": gpu_name,
        "driver": driver,
        "nvcc": run_command(["nvcc", "--version"]).splitlines()[-1],
    }


def main() -> None:
    args = parse_args()
    local_rank = initialize_single_gpu_distributed(args.seed)
    try:
        device = torch.device(f"cuda:{local_rank}")
        use_bf16 = torch.cuda.is_bf16_supported()

        # Keep parameters in FP32 because the no-Apex local LayerNorm fallback emits FP32.
        # BF16 is still exercised through CUDA autocast during the forward pass.
        model = build_model(args, torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        batch = synthetic_batch(args, device)
        losses: list[float] = []
        step_times_ms: list[float] = []

        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.iterations):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output_tensor = model(
                    batch["tokens"],
                    batch["position_ids"],
                    batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = masked_language_model_loss(output_tensor, batch["loss_mask"])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            step_times_ms.append((time.perf_counter() - start) * 1000.0)
            losses.append(float(loss.detach().cpu()))

        final_loss = losses[-1]
        checkpoint = save_and_load_checkpoint(args, model, optimizer, final_loss)
        peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2

        result = {
            "status": "success",
            "iterations": args.iterations,
            "losses": losses,
            "final_loss": final_loss,
            "average_step_time_ms": sum(step_times_ms) / len(step_times_ms),
            "step_times_ms": step_times_ms,
            "peak_memory_mib": peak_memory_mib,
            "dtype": "bf16_autocast_fp32_params" if use_bf16 else "fp32",
            "model_config": {
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "ffn_hidden_size": args.ffn_hidden_size,
                "num_attention_heads": args.num_attention_heads,
                "vocab_size": args.vocab_size,
                "sequence_length": args.sequence_length,
            },
            "batch_size": args.micro_batch_size,
            "sequence_length": args.sequence_length,
            "checkpoint": checkpoint,
            "versions": collect_versions(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE1_SMOKE_TEST_METRICS_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
