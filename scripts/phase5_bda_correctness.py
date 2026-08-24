#!/usr/bin/env python3
"""Check Phase 5.2 BDA fusion outputs, losses, and gradients."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state

from phase1_baseline import (
    TE_FUSED_ATTENTION,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
)
from phase3_attention_correctness import fused_backend_status


ATOL = 1.0e-5
RTOL = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase5_bda_correctness.json"),
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


def finite_summary(tensor: torch.Tensor) -> dict[str, int | bool]:
    nan_count = int(torch.isnan(tensor).sum()) if tensor.is_floating_point() else 0
    inf_count = int(torch.isinf(tensor).sum()) if tensor.is_floating_point() else 0
    return {
        "all_finite": nan_count == 0 and inf_count == 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "element_count": tensor.numel(),
    }


def run_variant(
    model_args: argparse.Namespace,
    batch: dict[str, torch.Tensor],
    bias_dropout_fusion: bool,
    state_dict: dict[str, Any] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, Any],
    dict[str, Any],
]:
    torch.manual_seed(model_args.seed)
    model = build_model(
        model_args,
        attention_implementation=TE_FUSED_ATTENTION,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bias_dropout_fusion=bias_dropout_fusion,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.train()
    if bool(model.config.bias_dropout_fusion) != bias_dropout_fusion:
        raise RuntimeError("Model BDA fusion configuration did not match the requested variant")

    torch.manual_seed(model_args.seed + 2)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            forward_output = model(
                batch["tokens"],
                batch["position_ids"],
                batch["attention_mask"],
            )

    torch.manual_seed(model_args.seed + 3)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        per_token_loss = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = masked_language_model_loss(per_token_loss, batch["loss_mask"])
    loss.backward()
    torch.cuda.synchronize()

    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    saved_state = {
        name: (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
        )
        for name, value in model.state_dict().items()
    }
    finite = {
        "forward_output": finite_summary(forward_output),
        "per_token_loss": finite_summary(per_token_loss),
        "loss": finite_summary(loss),
        "gradients": {
            "all_finite": all(
                bool(finite_summary(gradient)["all_finite"])
                for gradient in gradients.values()
            ),
            "nan_count": sum(
                int(finite_summary(gradient)["nan_count"])
                for gradient in gradients.values()
            ),
            "inf_count": sum(
                int(finite_summary(gradient)["inf_count"])
                for gradient in gradients.values()
            ),
            "element_count": sum(gradient.numel() for gradient in gradients.values()),
        },
    }
    controls = {
        "bias_dropout_fusion": bool(model.config.bias_dropout_fusion),
        "bias_activation_fusion": bool(model.config.bias_activation_fusion),
        "masked_softmax_fusion": bool(model.config.masked_softmax_fusion),
        "cross_entropy_loss_fusion": bool(model.config.cross_entropy_loss_fusion),
        "parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        ),
    }
    output_cpu = forward_output.detach().cpu()
    per_token_loss_cpu = per_token_loss.detach().cpu()
    loss_cpu = loss.detach().cpu()
    del loss, per_token_loss, forward_output, model
    torch.cuda.empty_cache()
    return (
        output_cpu,
        per_token_loss_cpu,
        loss_cpu,
        gradients,
        saved_state,
        {"finite": finite, "controls": controls},
    )


def gradient_comparison(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        raise RuntimeError("A/B variants produced different gradient sets")

    errors = {
        name: tensor_error(reference[name], candidate[name]) for name in reference
    }
    worst_absolute = max(
        errors,
        key=lambda name: errors[name]["max_absolute_error"],
    )
    worst_relative = max(
        errors,
        key=lambda name: errors[name]["max_relative_error"],
    )
    all_close = all(
        torch.allclose(candidate[name], reference[name], atol=ATOL, rtol=RTOL)
        for name in reference
    )
    dot = 0.0
    reference_norm_squared = 0.0
    candidate_norm_squared = 0.0
    for name in reference:
        reference_flat = reference[name].double().view(-1)
        candidate_flat = candidate[name].double().view(-1)
        dot += float(torch.dot(reference_flat, candidate_flat))
        reference_norm_squared += float(torch.dot(reference_flat, reference_flat))
        candidate_norm_squared += float(torch.dot(candidate_flat, candidate_flat))
    denominator = math.sqrt(reference_norm_squared * candidate_norm_squared)
    cosine_similarity = dot / denominator if denominator else 1.0
    return {
        "compared_parameter_count": len(errors),
        "all_close": all_close,
        "global_cosine_similarity": cosine_similarity,
        "worst_absolute_parameter": worst_absolute,
        "worst_absolute_error": errors[worst_absolute],
        "worst_relative_parameter": worst_relative,
        "worst_relative_error": errors[worst_relative],
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
        raise RuntimeError("Transformer Engine must remain enabled")

    model_args = small_model_args()
    local_rank = initialize_single_gpu_distributed(model_args.seed)
    try:
        device = torch.device(f"cuda:{local_rank}")
        batch = synthetic_batch(model_args, micro_batch_size=1, device=device)
        (
            baseline_output,
            baseline_per_token_loss,
            baseline_loss,
            baseline_gradients,
            state_dict,
            baseline_diagnostics,
        ) = run_variant(
            model_args,
            batch,
            bias_dropout_fusion=False,
        )
        (
            fused_output,
            fused_per_token_loss,
            fused_loss,
            fused_gradients,
            _,
            fused_diagnostics,
        ) = run_variant(
            model_args,
            batch,
            bias_dropout_fusion=True,
            state_dict=state_dict,
        )
        backend = fused_backend_status()

        forward_close = torch.allclose(
            fused_output,
            baseline_output,
            atol=ATOL,
            rtol=RTOL,
        )
        per_token_loss_close = torch.allclose(
            fused_per_token_loss,
            baseline_per_token_loss,
            atol=ATOL,
            rtol=RTOL,
        )
        scalar_loss_close = torch.allclose(
            fused_loss,
            baseline_loss,
            atol=ATOL,
            rtol=RTOL,
        )
        gradients = gradient_comparison(baseline_gradients, fused_gradients)
        finite = all(
            bool(variant["finite"][field]["all_finite"])
            for variant in (baseline_diagnostics, fused_diagnostics)
            for field in ("forward_output", "per_token_loss", "loss", "gradients")
        )
        controls_ok = (
            baseline_diagnostics["controls"]
            == {
                "bias_dropout_fusion": False,
                "bias_activation_fusion": False,
                "masked_softmax_fusion": False,
                "cross_entropy_loss_fusion": False,
                "parameter_dtypes": ["torch.float32"],
            }
            and fused_diagnostics["controls"]
            == {
                "bias_dropout_fusion": True,
                "bias_activation_fusion": False,
                "masked_softmax_fusion": False,
                "cross_entropy_loss_fusion": False,
                "parameter_dtypes": ["torch.float32"],
            }
        )
        passed = (
            forward_close
            and per_token_loss_close
            and scalar_loss_close
            and bool(gradients["all_close"])
            and finite
            and controls_ok
        )
        result = {
            "status": "success" if passed else "failed",
            "comparison": "bias_dropout_fusion=False versus True",
            "passed": passed,
            "same_initial_weights": True,
            "same_seed": True,
            "model_seed": model_args.seed,
            "dropout": {
                "hidden_dropout": 0.0,
                "attention_dropout": 0.0,
                "correctness_only": True,
            },
            "precision": "BF16 autocast with FP32 parameters and residual stream",
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
                "attention_implementation": TE_FUSED_ATTENTION,
            },
            "loss": {
                "baseline": float(baseline_loss),
                "fused": float(fused_loss),
                "absolute_error": abs(float(fused_loss) - float(baseline_loss)),
                "all_close": scalar_loss_close,
            },
            "per_token_loss": {
                **tensor_error(baseline_per_token_loss, fused_per_token_loss),
                "all_close": per_token_loss_close,
            },
            "forward_output": {
                **tensor_error(baseline_output, fused_output),
                "all_close": forward_close,
            },
            "gradients": gradients,
            "finite": {
                "all_finite": finite,
                "baseline": baseline_diagnostics["finite"],
                "fused": fused_diagnostics["finite"],
            },
            "controls": {
                "only_intended_difference": "bias_dropout_fusion",
                "verified": controls_ok,
                "baseline": baseline_diagnostics["controls"],
                "fused": fused_diagnostics["controls"],
                "state_dict_tensor_count": len(state_dict),
            },
            "transformer_engine_backend": backend,
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE5_BDA_CORRECTNESS_JSON=" + json.dumps(result, sort_keys=True))
        if not passed:
            raise RuntimeError("BDA fusion correctness comparison failed")
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
