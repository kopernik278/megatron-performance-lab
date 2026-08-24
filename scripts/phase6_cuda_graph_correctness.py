#!/usr/bin/env python3
"""Verify MCore local CUDA Graph loss and gradients against eager execution."""

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
from megatron.core.transformer.cuda_graphs import create_cudagraphs

from phase1_baseline import (
    TE_FUSED_ATTENTION,
    build_model,
    collect_environment,
    initialize_single_gpu_distributed,
    masked_language_model_loss,
    synthetic_batch,
)
from phase3_attention_correctness import fused_backend_status
from phase6_cuda_graph_run import (
    CUDA_GRAPH_IMPL,
    graph_parameters,
    prepare_main_grad_buffers,
    verify_graph_state,
)


ATOL = 1.0e-5
RTOL = 1.0e-5
GRAPH_WARMUP_ITERATIONS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase6_cuda_graph_correctness.json"),
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


def forward_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        per_token_loss = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = masked_language_model_loss(per_token_loss, batch["loss_mask"])
    return per_token_loss, loss


def finite_summary(tensor: torch.Tensor) -> dict[str, int | bool]:
    float_tensor = tensor.float()
    nan_count = int(torch.isnan(float_tensor).sum())
    inf_count = int(torch.isinf(float_tensor).sum())
    return {
        "all_finite": nan_count == 0 and inf_count == 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "element_count": tensor.numel(),
    }


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


def collect_eager_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def collect_graphed_gradients(
    model: torch.nn.Module,
    captured_parameters: list[torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    captured_ids = {id(parameter) for parameter in captured_parameters}
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        gradient = (
            parameter.main_grad
            if id(parameter) in captured_ids
            else parameter.grad
        )
        if gradient is None:
            raise RuntimeError(f"Missing gradient for {name}")
        gradients[name] = gradient.detach().cpu().clone()
    return gradients


def gradient_comparison(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        raise RuntimeError("Eager and graphed variants produced different gradient sets")
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
    return {
        "compared_parameter_count": len(errors),
        "all_close": all_close,
        "global_cosine_similarity": dot / denominator if denominator else 1.0,
        "worst_absolute_parameter": worst_absolute,
        "worst_absolute_error": errors[worst_absolute],
        "worst_relative_parameter": worst_relative,
        "worst_relative_error": errors[worst_relative],
    }


def saved_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    return {
        name: (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
        )
        for name, value in model.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
        raise RuntimeError("Transformer Engine must remain enabled")

    model_args = small_model_args()
    initialize_single_gpu_distributed(
        model_args.seed,
        use_te_rng_tracker=True,
    )
    try:
        device = torch.device("cuda:0")
        batch = synthetic_batch(model_args, micro_batch_size=1, device=device)

        torch.manual_seed(model_args.seed)
        eager_model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bias_dropout_fusion=True,
            cuda_graph_impl="none",
            cuda_graph_warmup_steps=GRAPH_WARMUP_ITERATIONS,
        )
        eager_model.train()
        state_dict = saved_state_dict(eager_model)
        eager_model.zero_grad(set_to_none=True)
        eager_per_token_loss, eager_loss = forward_loss(eager_model, batch)
        eager_loss.backward()
        torch.cuda.synchronize()
        eager_gradients = collect_eager_gradients(eager_model)
        eager_finite = {
            "per_token_loss": finite_summary(eager_per_token_loss),
            "loss": finite_summary(eager_loss),
            "gradients": finite_summary(torch.cat([
                gradient.float().view(-1) for gradient in eager_gradients.values()
            ])),
        }
        eager_per_token_loss_cpu = eager_per_token_loss.detach().cpu()
        eager_loss_cpu = eager_loss.detach().cpu()
        del eager_loss, eager_per_token_loss, eager_model
        torch.cuda.empty_cache()

        torch.manual_seed(model_args.seed)
        graphed_model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bias_dropout_fusion=True,
            cuda_graph_impl=CUDA_GRAPH_IMPL,
            cuda_graph_warmup_steps=GRAPH_WARMUP_ITERATIONS,
        )
        graphed_model.load_state_dict(state_dict, strict=True)
        graphed_model.train()
        captured_parameters = graph_parameters(graphed_model)
        prepare_main_grad_buffers(captured_parameters)

        # First eager forward/backward records MCore's graph ordering.
        graphed_model.zero_grad(set_to_none=True)
        _, record_loss = forward_loss(graphed_model, batch)
        record_loss.backward()
        torch.cuda.synchronize()
        graphed_model.zero_grad(set_to_none=True)
        for parameter in captured_parameters:
            parameter.main_grad.zero_()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
            cache_enabled=False,
        ):
            capture_stats_raw = create_cudagraphs()
        torch.cuda.synchronize()
        if capture_stats_raw is None:
            raise RuntimeError("No CUDA graphs were created during correctness check")
        state_after_capture = verify_graph_state(graphed_model, True)

        graphed_model.zero_grad(set_to_none=True)
        for parameter in captured_parameters:
            parameter.main_grad.zero_()
        graphed_per_token_loss, graphed_loss = forward_loss(graphed_model, batch)
        graphed_loss.backward()
        torch.cuda.synchronize()
        graphed_gradients = collect_graphed_gradients(
            graphed_model,
            captured_parameters,
        )
        state_after_replay = verify_graph_state(graphed_model, True)
        graphed_finite = {
            "per_token_loss": finite_summary(graphed_per_token_loss),
            "loss": finite_summary(graphed_loss),
            "gradients": finite_summary(torch.cat([
                gradient.float().view(-1) for gradient in graphed_gradients.values()
            ])),
        }
        graphed_per_token_loss_cpu = graphed_per_token_loss.detach().cpu()
        graphed_loss_cpu = graphed_loss.detach().cpu()

        per_token_close = torch.allclose(
            graphed_per_token_loss_cpu,
            eager_per_token_loss_cpu,
            atol=ATOL,
            rtol=RTOL,
        )
        scalar_loss_close = torch.allclose(
            graphed_loss_cpu,
            eager_loss_cpu,
            atol=ATOL,
            rtol=RTOL,
        )
        gradients = gradient_comparison(eager_gradients, graphed_gradients)
        finite = all(
            bool(summary[field]["all_finite"])
            for summary in (eager_finite, graphed_finite)
            for field in ("per_token_loss", "loss", "gradients")
        )
        replay_verified = bool(
            state_after_capture["replay_ready"]
            and state_after_replay["replay_ready"]
            and state_after_replay["forward_graph_count"] == model_args.num_layers
            and state_after_replay["backward_graph_count"] == model_args.num_layers
        )
        controls_ok = (
            eager_finite
            and graphed_model.config.cuda_graph_impl == CUDA_GRAPH_IMPL
            and graphed_model.config.bias_dropout_fusion is True
            and graphed_model.config.bias_activation_fusion is False
            and sorted(
                {str(parameter.dtype) for parameter in graphed_model.parameters()}
            )
            == ["torch.float32"]
        )
        passed = (
            per_token_close
            and scalar_loss_close
            and bool(gradients["all_close"])
            and finite
            and replay_verified
            and bool(controls_ok)
        )
        capture_stats = {
            "time_seconds": float(capture_stats_raw["time"]),
            "allocated_bytes": int(capture_stats_raw["allocated_bytes"]),
            "reserved_bytes": int(capture_stats_raw["reserved_bytes"]),
        }
        result = {
            "status": "success" if passed else "failed",
            "comparison": "eager versus MCore local CUDA Graph replay",
            "passed": passed,
            "same_initial_weights": True,
            "same_seed": True,
            "model_seed": model_args.seed,
            "dropout": {
                "hidden_dropout": 0.0,
                "attention_dropout": 0.0,
                "correctness_only": True,
                "production_dropout_unchanged": 0.1,
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
                "bias_dropout_fusion": True,
                "bias_activation_fusion": False,
            },
            "loss": {
                "eager": float(eager_loss_cpu),
                "graphed": float(graphed_loss_cpu),
                "absolute_error": abs(
                    float(graphed_loss_cpu) - float(eager_loss_cpu)
                ),
                "all_close": scalar_loss_close,
            },
            "per_token_loss": {
                **tensor_error(eager_per_token_loss_cpu, graphed_per_token_loss_cpu),
                "all_close": per_token_close,
            },
            "gradients": gradients,
            "finite": {
                "all_finite": finite,
                "eager": eager_finite,
                "graphed": graphed_finite,
            },
            "capture_replay": {
                "mechanism": (
                    "Megatron Core cuda_graph_impl=local, empty "
                    "cuda_graph_modules (whole TransformerLayer)"
                ),
                "capture_warmup_iterations": GRAPH_WARMUP_ITERATIONS,
                "capture_stats": capture_stats,
                "state_after_capture": state_after_capture,
                "state_after_replay": state_after_replay,
                "replay_verified": replay_verified,
                "raw_adamw_main_grad_bridge_required": True,
            },
            "controls": {
                "only_intended_runtime_difference": "CUDA Graph capture/replay",
                "verified": bool(controls_ok),
                "state_dict_tensor_count": len(state_dict),
                "graph_safe_rng_tracker_for_both_variants": "Transformer Engine",
            },
            "transformer_engine_backend": fused_backend_status(),
            "environment": {
                **collect_environment(),
                "cuda_graph_enabled": True,
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE6_CUDA_GRAPH_CORRECTNESS_JSON=" + json.dumps(result, sort_keys=True))
        if not passed:
            raise RuntimeError("CUDA Graph correctness comparison failed")
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
