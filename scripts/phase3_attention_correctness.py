#!/usr/bin/env python3
"""Compare local and TE fused attention outputs and gradients on a small GPT."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state

from phase1_baseline import (
    LOCAL_UNFUSED_ATTENTION,
    TE_FUSED_ATTENTION,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
)


ATOL = 0.05
RTOL = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase3_attention_correctness.json"),
    )
    return parser.parse_args()


def small_model_args() -> argparse.Namespace:
    return argparse.Namespace(
        sequence_length=128,
        vocab_size=1024,
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=512,
        num_attention_heads=4,
        learning_rate=1.0e-4,
        seed=1234,
    )


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    absolute = (candidate_float - reference_float).abs()
    relative = absolute / reference_float.abs().clamp_min(1.0e-6)
    return {
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "max_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
    }


def run_variant(
    model_args: argparse.Namespace,
    batch: dict[str, torch.Tensor],
    attention_implementation: str,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(model_args.seed)
    model = build_model(
        model_args,
        attention_implementation=attention_implementation,
        attention_dropout=0.0,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = masked_language_model_loss(output, batch["loss_mask"])
    loss.backward()
    torch.cuda.synchronize()

    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    saved_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    return output.detach().cpu(), gradients, saved_state


def fused_backend_status() -> dict[str, Any]:
    module = importlib.import_module(
        "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention"
    )
    backend_state = module._attention_backends
    fused_backend = backend_state["fused_attention_backend"]
    fused_backend_number = int(fused_backend) if fused_backend is not None else None
    status = {
        "flash_attention": bool(backend_state["use_flash_attention"]),
        "fused_attention": bool(backend_state["use_fused_attention"]),
        "unfused_attention": bool(backend_state["use_unfused_attention"]),
        "fused_sub_backend": fused_backend_number,
    }
    if status != {
        "flash_attention": False,
        "fused_attention": True,
        "unfused_attention": False,
        "fused_sub_backend": 1,
    }:
        raise RuntimeError(f"Unexpected Transformer Engine backend selection: {status}")
    return status


def main() -> None:
    args = parse_args()
    if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
        raise RuntimeError("TRANSFORMER_ENGINE_DISABLE must be unset")

    model_args = small_model_args()
    local_rank = initialize_single_gpu_distributed(model_args.seed)
    try:
        device = torch.device(f"cuda:{local_rank}")
        batch = synthetic_batch(model_args, micro_batch_size=1, device=device)

        local_output, local_gradients, state_dict = run_variant(
            model_args,
            batch,
            LOCAL_UNFUSED_ATTENTION,
        )
        torch.cuda.empty_cache()
        fused_output, fused_gradients, _ = run_variant(
            model_args,
            batch,
            TE_FUSED_ATTENTION,
            state_dict=state_dict,
        )
        backend = fused_backend_status()

        if local_gradients.keys() != fused_gradients.keys():
            raise RuntimeError("Local and fused variants produced different gradient sets")
        torch.testing.assert_close(fused_output, local_output, atol=ATOL, rtol=RTOL)

        gradient_errors: dict[str, dict[str, float]] = {}
        for name in local_gradients:
            torch.testing.assert_close(
                fused_gradients[name],
                local_gradients[name],
                atol=ATOL,
                rtol=RTOL,
            )
            gradient_errors[name] = tensor_error(
                local_gradients[name], fused_gradients[name]
            )

        worst_gradient = max(
            gradient_errors,
            key=lambda name: gradient_errors[name]["max_absolute_error"],
        )
        result = {
            "status": "success",
            "comparison": "Megatron local DotProductAttention vs TE cuDNN FusedAttention",
            "precision": "BF16 autocast with FP32 parameters and gradients",
            "attention_dropout": 0.0,
            "atol": ATOL,
            "rtol": RTOL,
            "model_config": {
                "num_layers": model_args.num_layers,
                "hidden_size": model_args.hidden_size,
                "ffn_hidden_size": model_args.ffn_hidden_size,
                "num_attention_heads": model_args.num_attention_heads,
                "sequence_length": model_args.sequence_length,
                "vocab_size": model_args.vocab_size,
                "micro_batch_size": 1,
            },
            "forward": tensor_error(local_output, fused_output),
            "gradients": {
                "compared_parameter_count": len(gradient_errors),
                "worst_parameter": worst_gradient,
                "worst_error": gradient_errors[worst_gradient],
            },
            "transformer_engine_backend": backend,
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("FusedAttention backend (sub-backend 1)")
        print("PHASE3_CORRECTNESS_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
