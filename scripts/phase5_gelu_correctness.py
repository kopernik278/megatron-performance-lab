#!/usr/bin/env python3
"""Compare Phase 5.3 unfused and fused bias-plus-GELU numerics."""

from __future__ import annotations

import argparse
import copy
import json
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
from phase5_bda_correctness import (
    ATOL,
    RTOL,
    finite_summary,
    gradient_comparison,
    tensor_error,
)
from phase5_gelu_run import probe_runtime_gelu_path, verify_static_controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase5_gelu_correctness.json"),
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


def run_variant(
    model_args: argparse.Namespace,
    batch: dict[str, torch.Tensor],
    bias_gelu_fusion: bool,
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
        bias_activation_fusion=bias_gelu_fusion,
        bias_dropout_fusion=True,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.train()
    static_controls = verify_static_controls(model, bias_gelu_fusion)
    runtime_path = probe_runtime_gelu_path(model, batch, bias_gelu_fusion)

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
    diagnostics = {
        "finite": finite,
        "static_controls": static_controls,
        "runtime_path": runtime_path,
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
        diagnostics,
    )


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
            bias_gelu_fusion=False,
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
            bias_gelu_fusion=True,
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
        runtime_verified = (
            baseline_diagnostics["runtime_path"]["verified"]
            and fused_diagnostics["runtime_path"]["verified"]
            and baseline_diagnostics["runtime_path"][
                "observed_bias_gelu_impl_calls"
            ]
            == 0
            and fused_diagnostics["runtime_path"]["observed_bias_gelu_impl_calls"]
            == model_args.num_layers
        )
        passed = (
            forward_close
            and per_token_loss_close
            and scalar_loss_close
            and bool(gradients["all_close"])
            and finite
            and runtime_verified
        )
        result = {
            "status": "success",
            "comparison": "bias_gelu_fusion=False versus True",
            "passed": passed,
            "strict_allclose_passed": passed,
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
            "known_formula_difference": {
                "baseline": "torch.nn.functional.gelu exact/erf path",
                "fused": "Megatron bias_gelu tanh approximation",
                "source": "megatron/core/fusions/fused_bias_gelu.py",
            },
            "model_config": {
                "num_layers": model_args.num_layers,
                "hidden_size": model_args.hidden_size,
                "ffn_hidden_size": model_args.ffn_hidden_size,
                "num_attention_heads": model_args.num_attention_heads,
                "sequence_length": model_args.sequence_length,
                "vocab_size": model_args.vocab_size,
                "micro_batch_size": 1,
                "attention_implementation": TE_FUSED_ATTENTION,
                "bias_dropout_fusion": True,
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
                "only_intended_difference": "bias_activation_fusion",
                "verified": runtime_verified,
                "baseline": baseline_diagnostics["static_controls"],
                "fused": fused_diagnostics["static_controls"],
                "state_dict_tensor_count": len(state_dict),
            },
            "runtime_path_verification": {
                "verified": runtime_verified,
                "baseline": baseline_diagnostics["runtime_path"],
                "fused": fused_diagnostics["runtime_path"],
            },
            "transformer_engine_backend": backend,
            "environment": collect_environment(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE5_GELU_CORRECTNESS_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
