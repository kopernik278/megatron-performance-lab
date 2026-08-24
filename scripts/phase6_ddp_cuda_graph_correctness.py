#!/usr/bin/env python3
"""Validate MCore DDP and local CUDA Graph gradient lifecycles."""

from __future__ import annotations

import argparse
import copy
import gc
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
from phase6_cuda_graph_correctness import finite_summary, tensor_error
from phase6_cuda_graph_run import CUDA_GRAPH_IMPL, verify_graph_state
from phase6_megatron_ddp_lifecycle import (
    DDPOptimizerBundle,
    assert_main_grad_pointers,
    assert_optimizer_consumed_main_grad,
    assert_zeroed_lifecycle,
    build_ddp_optimizer_bundle,
    collect_main_gradients,
    create_local_cudagraphs_preserving_gradients,
    finalize_gradients,
    main_grad_pointers,
    named_trainable_parameters,
    unwrap_model,
    zero_gradients,
)


ATOL = 1.0e-5
RTOL = 1.0e-5
GRAPH_WARMUP_ITERATIONS = 5
CORRECTNESS_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/phase6_ddp_cuda_graph_correctness.json"),
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


def saved_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    return {
        name: (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
        )
        for name, value in unwrap_model(model).state_dict().items()
    }


def parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_trainable_parameters(model)
    }


def raw_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Raw harness is missing a gradient for {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    return gradients


def optimizer_state_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, dict[str, torch.Tensor | int | float]]:
    parameter_names = {
        parameter: name for name, parameter in named_trainable_parameters(model)
    }
    result: dict[str, dict[str, torch.Tensor | int | float]] = {}
    for parameter, state in optimizer.state.items():
        name = parameter_names[parameter]
        result[name] = {}
        for key, value in state.items():
            result[name][key] = (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
            )
    return result


def forward_output(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = masked_language_model_loss(output, batch["loss_mask"])
    return output, loss


def graph_identity(model: torch.nn.Module) -> list[tuple[int, int, int]]:
    raw_model = unwrap_model(model)
    return [
        (id(runner), id(runner.fwd_graph), id(runner.bwd_graph))
        for layer in raw_model.decoder.layers
        for runner in layer.cudagraph_manager.cudagraph_runners
    ]


def run_raw_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    parameters_before = parameter_snapshot(model)
    output, loss = forward_output(model, batch)
    loss.backward()
    torch.cuda.synchronize()
    gradients = raw_gradients(model)
    optimizer.step()
    torch.cuda.synchronize()
    return {
        "output": output.detach().cpu().clone(),
        "loss": loss.detach().cpu().clone(),
        "gradients": gradients,
        "parameters_before": parameters_before,
        "parameters_after": parameter_snapshot(model),
        "optimizer_state": optimizer_state_snapshot(model, optimizer),
    }


def run_ddp_step(
    bundle: DDPOptimizerBundle,
    batch: dict[str, torch.Tensor],
    expected_pointers: dict[str, int],
    create_graphs: bool = False,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    zero_gradients(bundle)
    torch.cuda.synchronize()
    assert_zeroed_lifecycle(bundle)
    assert_main_grad_pointers(bundle.model, expected_pointers)

    parameters_before = parameter_snapshot(bundle.model)
    output, loss = forward_output(bundle.model, batch)
    loss.backward()
    finalize_gradients(bundle.model)
    torch.cuda.synchronize()

    for name, parameter in named_trainable_parameters(bundle.model):
        if parameter.grad is not None:
            raise RuntimeError(f"DDP left a stale param.grad after backward for {name}")

    capture_stats = None
    if create_graphs:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
            cache_enabled=False,
        ):
            capture_stats = create_local_cudagraphs_preserving_gradients(
                bundle.model
            )

    assert_main_grad_pointers(bundle.model, expected_pointers)
    gradients = collect_main_gradients(bundle.model)
    update_successful, grad_norm, num_zeros = bundle.optimizer.step()
    torch.cuda.synchronize()
    if not update_successful:
        raise RuntimeError("FP32Optimizer unexpectedly rejected an update")
    assert_optimizer_consumed_main_grad(bundle)
    assert_main_grad_pointers(bundle.model, expected_pointers)
    return (
        {
            "output": output.detach().cpu().clone(),
            "loss": loss.detach().cpu().clone(),
            "gradients": gradients,
            "parameters_before": parameters_before,
            "parameters_after": parameter_snapshot(bundle.model),
            "optimizer_state": optimizer_state_snapshot(
                bundle.model,
                bundle.base_optimizer,
            ),
            "grad_norm": grad_norm,
            "num_zeros": num_zeros,
        },
        capture_stats,
    )


def mapping_finite_summary(
    mapping: dict[str, torch.Tensor],
) -> dict[str, int | bool]:
    return finite_summary(
        torch.cat([tensor.float().reshape(-1) for tensor in mapping.values()])
    )


def mapping_comparison(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        raise RuntimeError("Compared mappings have different keys")
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
        reference_flat = reference[name].double().reshape(-1)
        candidate_flat = candidate[name].double().reshape(-1)
        dot += float(torch.dot(reference_flat, candidate_flat))
        reference_norm_squared += float(torch.dot(reference_flat, reference_flat))
        candidate_norm_squared += float(torch.dot(candidate_flat, candidate_flat))
    denominator = math.sqrt(reference_norm_squared * candidate_norm_squared)
    return {
        "tensor_count": len(reference),
        "all_close": all_close,
        "global_cosine_similarity": dot / denominator if denominator else 1.0,
        "worst_absolute_tensor": worst_absolute,
        "worst_absolute_error": errors[worst_absolute],
        "worst_relative_tensor": worst_relative,
        "worst_relative_error": errors[worst_relative],
    }


def update_mapping(step: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: step["parameters_after"][name] - step["parameters_before"][name]
        for name in step["parameters_before"]
    }


def optimizer_state_comparison(
    reference: dict[str, dict[str, torch.Tensor | int | float]],
    candidate: dict[str, dict[str, torch.Tensor | int | float]],
) -> dict[str, Any]:
    reference_tensors: dict[str, torch.Tensor] = {}
    candidate_tensors: dict[str, torch.Tensor] = {}
    scalar_equal = True
    if reference.keys() != candidate.keys():
        raise RuntimeError("Optimizer states have different parameter keys")
    for name in reference:
        if reference[name].keys() != candidate[name].keys():
            raise RuntimeError(f"Optimizer state fields differ for {name}")
        for field, reference_value in reference[name].items():
            candidate_value = candidate[name][field]
            key = f"{name}:{field}"
            if isinstance(reference_value, torch.Tensor):
                if not isinstance(candidate_value, torch.Tensor):
                    raise RuntimeError(f"Optimizer state type differs for {key}")
                reference_tensors[key] = reference_value
                candidate_tensors[key] = candidate_value
            else:
                scalar_equal = scalar_equal and reference_value == candidate_value
    tensor_comparison = mapping_comparison(reference_tensors, candidate_tensors)
    return {
        **tensor_comparison,
        "scalar_fields_equal": scalar_equal,
        "all_close": bool(tensor_comparison["all_close"] and scalar_equal),
    }


def compare_steps(
    reference_steps: list[dict[str, Any]],
    candidate_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference_steps) != len(candidate_steps):
        raise RuntimeError("Compared runs have different step counts")
    results = []
    for index, (reference, candidate) in enumerate(
        zip(reference_steps, candidate_steps)
    ):
        output = {
            **tensor_error(reference["output"], candidate["output"]),
            "all_close": torch.allclose(
                candidate["output"],
                reference["output"],
                atol=ATOL,
                rtol=RTOL,
            ),
        }
        loss = {
            "reference": float(reference["loss"]),
            "candidate": float(candidate["loss"]),
            "absolute_error": abs(
                float(candidate["loss"]) - float(reference["loss"])
            ),
            "all_close": torch.allclose(
                candidate["loss"],
                reference["loss"],
                atol=ATOL,
                rtol=RTOL,
            ),
        }
        gradients = mapping_comparison(
            reference["gradients"],
            candidate["gradients"],
        )
        parameter_updates = mapping_comparison(
            update_mapping(reference),
            update_mapping(candidate),
        )
        parameters_after = mapping_comparison(
            reference["parameters_after"],
            candidate["parameters_after"],
        )
        optimizer_state = optimizer_state_comparison(
            reference["optimizer_state"],
            candidate["optimizer_state"],
        )
        finite = all(
            bool(summary["all_finite"])
            for summary in (
                finite_summary(reference["output"]),
                finite_summary(candidate["output"]),
                finite_summary(reference["loss"]),
                finite_summary(candidate["loss"]),
                mapping_finite_summary(reference["gradients"]),
                mapping_finite_summary(candidate["gradients"]),
                mapping_finite_summary(reference["parameters_after"]),
                mapping_finite_summary(candidate["parameters_after"]),
            )
        )
        parameters_updated = any(
            torch.count_nonzero(update).item() > 0
            for update in update_mapping(candidate).values()
        )
        passed = bool(
            output["all_close"]
            and loss["all_close"]
            and gradients["all_close"]
            and parameter_updates["all_close"]
            and parameters_after["all_close"]
            and optimizer_state["all_close"]
            and finite
            and parameters_updated
        )
        results.append(
            {
                "step": index,
                "passed": passed,
                "output": output,
                "loss": loss,
                "gradients": gradients,
                "parameter_updates": parameter_updates,
                "parameters_after": parameters_after,
                "optimizer_state": optimizer_state,
                "all_finite": finite,
                "parameters_updated": parameters_updated,
            }
        )
    return {
        "passed": all(step["passed"] for step in results),
        "steps": results,
    }


def build_candidate(
    args: argparse.Namespace,
    state_dict: dict[str, Any],
    cuda_graph: bool,
) -> DDPOptimizerBundle:
    torch.manual_seed(args.seed)
    model = build_model(
        args,
        attention_implementation=TE_FUSED_ATTENTION,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bias_dropout_fusion=True,
        cuda_graph_impl=CUDA_GRAPH_IMPL if cuda_graph else "none",
        cuda_graph_warmup_steps=GRAPH_WARMUP_ITERATIONS,
    )
    model.load_state_dict(state_dict, strict=True)
    model.train()
    return build_ddp_optimizer_bundle(model, args.learning_rate)


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
        initial_model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bias_dropout_fusion=True,
            cuda_graph_impl="none",
            cuda_graph_warmup_steps=GRAPH_WARMUP_ITERATIONS,
        )
        initial_state = saved_state_dict(initial_model)
        del initial_model
        torch.cuda.empty_cache()

        # Phase A: compare the old raw lifecycle with eager MCore DDP.
        torch.manual_seed(model_args.seed)
        raw_model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bias_dropout_fusion=True,
            cuda_graph_impl="none",
            cuda_graph_warmup_steps=GRAPH_WARMUP_ITERATIONS,
        )
        raw_model.load_state_dict(initial_state, strict=True)
        raw_model.train()
        raw_optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=model_args.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            foreach=False,
            fused=False,
        )
        ddp_bundle = build_candidate(model_args, initial_state, cuda_graph=False)
        ddp_pointers = main_grad_pointers(ddp_bundle.model)

        raw_steps = []
        ddp_steps = []
        for _ in range(CORRECTNESS_STEPS):
            raw_steps.append(run_raw_step(raw_model, raw_optimizer, batch))
            ddp_step, _ = run_ddp_step(
                ddp_bundle,
                batch,
                ddp_pointers,
            )
            ddp_steps.append(ddp_step)
        lifecycle_comparison = compare_steps(raw_steps, ddp_steps)
        lifecycle_passed = bool(lifecycle_comparison["passed"])
        if not lifecycle_passed:
            raise RuntimeError("MCore DDP lifecycle does not match the raw harness")

        del raw_steps, ddp_steps, raw_optimizer, raw_model, ddp_bundle
        gc.collect()
        torch.cuda.empty_cache()

        # Phase B: compare eager DDP with local CUDA Graph DDP.
        eager_bundle = build_candidate(model_args, initial_state, cuda_graph=False)
        graph_bundle = build_candidate(model_args, initial_state, cuda_graph=True)
        eager_pointers = main_grad_pointers(eager_bundle.model)
        graph_pointers = main_grad_pointers(graph_bundle.model)
        eager_steps = []
        graph_steps = []
        capture_stats = None
        graph_state_after_capture = None
        graph_identities: list[list[tuple[int, int, int]]] = []

        for step_index in range(CORRECTNESS_STEPS):
            eager_step, _ = run_ddp_step(
                eager_bundle,
                batch,
                eager_pointers,
            )
            graph_step, step_capture_stats = run_ddp_step(
                graph_bundle,
                batch,
                graph_pointers,
                create_graphs=step_index == 0,
            )
            if step_capture_stats is not None:
                capture_stats = step_capture_stats
                graph_state_after_capture = verify_graph_state(
                    unwrap_model(graph_bundle.model),
                    True,
                )
            eager_steps.append(eager_step)
            graph_steps.append(graph_step)
            graph_identities.append(graph_identity(graph_bundle.model))

        graph_state_after_replay = verify_graph_state(
            unwrap_model(graph_bundle.model),
            True,
        )
        graph_comparison = compare_steps(eager_steps, graph_steps)
        graph_reused = bool(
            len(graph_identities) == CORRECTNESS_STEPS
            and graph_identities[0]
            and all(identity == graph_identities[0] for identity in graph_identities[1:])
        )
        graph_passed = bool(
            graph_comparison["passed"]
            and capture_stats is not None
            and graph_state_after_capture is not None
            and graph_state_after_replay["replay_ready"]
            and graph_state_after_replay["forward_graph_count"]
            == model_args.num_layers
            and graph_state_after_replay["backward_graph_count"]
            == model_args.num_layers
            and graph_reused
        )

        result = {
            "status": "success" if graph_passed else "failed",
            "experiment": "Phase 6.3 MCore DDP and local CUDA Graph correctness",
            "passed": bool(lifecycle_passed and graph_passed),
            "atol": ATOL,
            "rtol": RTOL,
            "correctness_steps": CORRECTNESS_STEPS,
            "same_initial_weights": True,
            "same_input": True,
            "dropout": {
                "hidden_dropout": 0.0,
                "attention_dropout": 0.0,
                "correctness_only": True,
                "production_dropout_unchanged": 0.1,
            },
            "phase_a_ddp_lifecycle": {
                "passed": lifecycle_passed,
                "comparison": "raw PyTorch lifecycle versus eager MCore DDP lifecycle",
                "main_grad_parameter_count": len(ddp_pointers),
                "main_grad_addresses_stable": True,
                "optimizer_consumed_main_grad": True,
                "zero_grad_reset_verified_each_step": True,
                "comparison_details": lifecycle_comparison,
            },
            "phase_b_cuda_graph": {
                "passed": graph_passed,
                "comparison": "eager MCore DDP versus local CUDA Graph MCore DDP",
                "capture_stats": capture_stats,
                "state_after_capture": graph_state_after_capture,
                "state_after_replay": graph_state_after_replay,
                "graph_identity_stable_across_steps": graph_reused,
                "main_grad_addresses_stable": True,
                "optimizer_consumed_main_grad": True,
                "comparison_details": graph_comparison,
            },
            "original_phase61_gradient_mismatch_fixed": bool(
                graph_comparison["passed"]
                and all(
                    step["gradients"]["all_close"]
                    for step in graph_comparison["steps"]
                )
            ),
            "controls": {
                "precision": "BF16 autocast with FP32 parameters, gradients, and AdamW",
                "attention_implementation": TE_FUSED_ATTENTION,
                "bias_dropout_fusion": True,
                "bias_activation_fusion": False,
                "distributed_optimizer": False,
                "optimizer": "MCore FP32Optimizer wrapping torch.optim.AdamW",
                "graph_safe_rng_tracker_for_all_candidates": "Transformer Engine",
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": 1,
            },
            "model_config": {
                "num_layers": model_args.num_layers,
                "hidden_size": model_args.hidden_size,
                "ffn_hidden_size": model_args.ffn_hidden_size,
                "num_attention_heads": model_args.num_attention_heads,
                "sequence_length": model_args.sequence_length,
                "vocab_size": model_args.vocab_size,
                "micro_batch_size": 1,
            },
            "transformer_engine_backend": fused_backend_status(),
            "environment": {
                **collect_environment(),
                "cuda_graph_enabled": True,
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print("PHASE6_DDP_CUDA_GRAPH_CORRECTNESS_JSON=" + json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise RuntimeError("Phase 6.3 correctness gate failed")
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
