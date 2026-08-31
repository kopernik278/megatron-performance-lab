#!/usr/bin/env python3
"""Run one Phase 12 DP=2 memory/capacity variant (A/B/C/D).

Pinned Megatron-LM 09fde85:
- Distributed optimizer: DistributedDataParallelConfig.use_distributed_optimizer
  (+ overlap_grad_reduce / overlap_param_gather)
- Activation checkpointing: TransformerConfig.recompute_granularity='full',
  recompute_method='uniform', recompute_num_layers=1
  (full Transformer-layer recomputation; see megatron/core/recompute.py)

TP=1, PP=1, DP=2. No SP/VPP/Userbuffers/CUDA Graph.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from phase1_baseline import (
    A40_DENSE_BF16_PEAK_TFLOPS,
    TE_FUSED_ATTENTION,
    build_model,
    masked_language_model_loss,
    training_flops_per_iteration,
)
from phase3_attention_correctness import fused_backend_status
from phase3_attention_profile import baseline_model_args
from phase6_megatron_ddp_lifecycle import (
    DDPOptimizerBundle,
    assert_main_grad_pointers,
    assert_optimizer_consumed_main_grad,
    assert_zeroed_lifecycle,
    build_megatron_optimizer_bundle,
    collect_bucket_metadata,
    collect_optimizer_state_metadata,
    collect_param_partition_metadata,
    finalize_gradients,
    instrument_grad_sync_nvtx,
    instrument_param_sync_nvtx,
    lifecycle_metadata,
    main_grad_pointers,
    named_trainable_parameters,
    zero_gradients,
)
from phase7_tp_run import MultiGpuNvidiaSmiSampler


# Full Transformer-layer recompute (pinned Megatron 09fde85).
RECOMPUTE_GRANULARITY = "full"
RECOMPUTE_METHOD = "uniform"
RECOMPUTE_NUM_LAYERS = 1
RECOMPUTE_SOURCE = (
    "megatron/core/transformer/transformer_config.py "
    "(recompute_granularity/method/num_layers); "
    "megatron/core/transformer/transformer_block.py "
    "(recompute_granularity=='full' -> checkpointed_forward); "
    "megatron/core/recompute.py (uniform chunks of recompute_num_layers)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-distributed-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--overlap-grad-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overlap-param-gather",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable full/uniform/1 Transformer-layer recompute",
    )
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--smoke-iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=100)
    parser.add_argument("--profile-mode", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("benchmark", "smoke", "capacity-probe"),
        default="benchmark",
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--variant", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.micro_batch_size < 1:
        parser.error("micro_batch_size must be positive")
    if args.sequence_length < 1:
        parser.error("sequence_length must be positive")
    if args.overlap_param_gather and not args.use_distributed_optimizer:
        parser.error("overlap_param_gather requires use_distributed_optimizer")
    if args.mode == "smoke":
        # Keep smoke path short; caller may still pass measured args unused.
        pass
    return args


def run_command(command: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def initialize_distributed(seed: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"Phase 12 requires DP=2, got world size {world}")
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    actual_tp = parallel_state.get_tensor_model_parallel_world_size()
    actual_pp = parallel_state.get_pipeline_model_parallel_world_size()
    actual_dp = parallel_state.get_data_parallel_world_size()
    if actual_tp != 1 or actual_pp != 1 or actual_dp != 2:
        raise RuntimeError(
            f"Phase 12 requires TP=1 PP=1 DP=2, got TP={actual_tp} PP={actual_pp} DP={actual_dp}"
        )
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    return local_rank


def dp_synthetic_batch(
    model_args: argparse.Namespace,
    micro_batch_size: int,
    sequence_length: int,
    device: torch.device,
    dp_rank: int,
    step_index: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(model_args.seed + 1 + 1_000_003 * step_index + 17 * dp_rank)
    tokens = torch.randint(
        low=0,
        high=model_args.vocab_size,
        size=(micro_batch_size, sequence_length),
        dtype=torch.long,
        generator=generator,
    )
    labels = torch.roll(tokens, shifts=-1, dims=1)
    position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0).expand(
        micro_batch_size, -1
    )
    loss_mask = torch.ones((micro_batch_size, sequence_length), dtype=torch.float32)
    attention_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool))
    attention_mask = ~attention_mask.view(1, 1, sequence_length, sequence_length)
    attention_mask = attention_mask.expand(micro_batch_size, -1, -1, -1)
    return {
        "tokens": tokens.to(device),
        "labels": labels.to(device),
        "position_ids": position_ids.to(device),
        "loss_mask": loss_mask.to(device),
        "attention_mask": attention_mask.to(device),
    }


def gather_objects(value: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def forward_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        return masked_language_model_loss(output, batch["loss_mask"])


def train_step(
    bundle: DDPOptimizerBundle,
    batch: dict[str, torch.Tensor],
    step_index: int,
) -> float:
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
        with torch.cuda.nvtx.range("optimizer_zero_grad"):
            zero_gradients(bundle)
        with torch.cuda.nvtx.range("forward"):
            loss = forward_loss(bundle.model, batch)
        with torch.cuda.nvtx.range("backward"):
            # Under full recompute, Megatron re-runs layer forwards inside backward.
            loss.backward()
        with torch.cuda.nvtx.range("finalize_model_grads"):
            finalize_gradients(bundle.model)
        with torch.cuda.nvtx.range("optimizer_step"):
            update_successful, _, _ = bundle.optimizer.step()
            if not update_successful:
                raise RuntimeError("Optimizer unexpectedly rejected an update")
    return float(loss.detach().cpu())


def main_grads_finite(bundle: DDPOptimizerBundle) -> bool:
    return all(
        bool(torch.isfinite(parameter.main_grad).all().item())
        for _, parameter in named_trainable_parameters(bundle.model)
    )


def tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().sum().cpu())


def parameter_checksum(bundle: DDPOptimizerBundle) -> float:
    total = 0.0
    for _, parameter in named_trainable_parameters(bundle.model):
        total += tensor_checksum(parameter)
    return total


def main_grad_checksum(bundle: DDPOptimizerBundle) -> float:
    total = 0.0
    for _, parameter in named_trainable_parameters(bundle.model):
        total += tensor_checksum(parameter.main_grad)
    return total


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message


def collect_environment(gpu_count: int) -> dict[str, Any]:
    try:
        te_version = importlib.metadata.version("transformer_engine")
    except importlib.metadata.PackageNotFoundError:
        te_version = None
    nccl_version = torch.cuda.nccl.version()
    gpus = []
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "driver": run_command(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                        f"--id={index}",
                    ]
                ),
                "memory_total_mib": props.total_memory / 1024**2,
                "pci_bus_id": getattr(props, "pci_bus_id", None),
            }
        )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": ".".join(str(part) for part in nccl_version),
        "cudnn": torch.backends.cudnn.version(),
        "megatron_lm_commit": run_command(
            ["git", "rev-parse", "HEAD"],
            cwd="/workspace/Megatron-LM",
        ),
        "project_commit": run_command(
            ["git", "rev-parse", "HEAD"],
            cwd="/workspace/megatron-performance-lab",
        ),
        "transformer_engine_installed": (
            importlib.util.find_spec("transformer_engine") is not None
        ),
        "transformer_engine_version": te_version,
        "gpus": gpus,
        "cuda_graph_enabled": False,
        "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"),
    }


def write_rank0_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_optimizer_state_sharding(
    rank_metadata: list[dict[str, Any]],
    use_distributed_optimizer: bool,
) -> dict[str, Any]:
    bytes_per_rank = [item["optimizer_state_bytes"] for item in rank_metadata]
    result: dict[str, Any] = {
        "optimizer_state_bytes_per_rank": bytes_per_rank,
        "optimizer_state_bytes_total": sum(bytes_per_rank),
    }
    if not use_distributed_optimizer:
        if len(set(bytes_per_rank)) > 1:
            max_b = max(bytes_per_rank)
            min_b = min(bytes_per_rank)
            if max_b > 0 and (max_b - min_b) / max_b > 0.05:
                raise RuntimeError(
                    f"Replicated optimizer state diverged across ranks: {bytes_per_rank}"
                )
        result["sharded"] = False
        result["sharding_ratio_max_over_min"] = (
            max(bytes_per_rank) / min(bytes_per_rank) if min(bytes_per_rank) else None
        )
        return result
    if any(value <= 0 for value in bytes_per_rank):
        raise RuntimeError(f"Empty optimizer state on a rank: {bytes_per_rank}")
    ratio = max(bytes_per_rank) / min(bytes_per_rank)
    if ratio > 1.10:
        raise RuntimeError(
            f"Optimizer state not evenly sharded across DP ranks: {bytes_per_rank}"
        )
    result["sharded"] = True
    result["sharding_ratio_max_over_min"] = ratio
    return result


def recompute_config(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "recompute_granularity": None,
            "recompute_method": None,
            "recompute_num_layers": None,
            "layers_recomputation_unit": None,
            "expected_behavior": "all activations saved; no recompute during backward",
            "source": RECOMPUTE_SOURCE,
        }
    return {
        "enabled": True,
        "recompute_granularity": RECOMPUTE_GRANULARITY,
        "recompute_method": RECOMPUTE_METHOD,
        "recompute_num_layers": RECOMPUTE_NUM_LAYERS,
        "layers_recomputation_unit": (
            "each of the 24 Transformer layers is its own uniform recompute chunk "
            "(recompute_num_layers=1)"
        ),
        "expected_behavior": (
            "layer inputs checkpointed; activations discarded after forward; "
            "backward recomputes each layer's forward before local grads"
        ),
        "source": RECOMPUTE_SOURCE,
    }


def run_smoke_correctness(
    bundle: DDPOptimizerBundle,
    model_args: argparse.Namespace,
    micro_batch_size: int,
    sequence_length: int,
    device: torch.device,
    dp_rank: int,
    rank: int,
    args: argparse.Namespace,
    pointers: dict[str, int],
) -> dict[str, Any]:
    use_dist = args.use_distributed_optimizer
    smoke_batch = dp_synthetic_batch(
        model_args, micro_batch_size, sequence_length, device, dp_rank, step_index=0
    )
    token_fingerprint = (
        int(smoke_batch["tokens"][0, 0].item()),
        int(smoke_batch["tokens"][0, 1].item()),
        int(smoke_batch["tokens"].sum().item()),
    )
    rank_tokens = gather_objects(
        {"rank": rank, "dp_rank": dp_rank, "token_fingerprint": token_fingerprint}
    )
    if len({item["token_fingerprint"] for item in rank_tokens}) != 2:
        raise RuntimeError(f"DP ranks did not receive different data: {rank_tokens}")

    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in named_trainable_parameters(bundle.model)
        if name.endswith("decoder.layers.0.self_attention.linear_qkv.weight")
    )
    parameters_before = tracked_parameter.detach().cpu().clone()
    smoke_losses: list[float] = []
    lifecycle_checks: list[dict[str, Any]] = []
    for smoke_index in range(args.smoke_iterations):
        batch = dp_synthetic_batch(
            model_args,
            micro_batch_size,
            sequence_length,
            device,
            dp_rank,
            step_index=smoke_index,
        )
        zero_gradients(bundle)
        torch.cuda.synchronize(device)
        assert_zeroed_lifecycle(bundle)
        assert_main_grad_pointers(bundle.model, pointers)
        loss = forward_loss(bundle.model, batch)
        loss.backward()
        finalize_gradients(bundle.model)
        torch.cuda.synchronize(device)
        gradients_finite = main_grads_finite(bundle)
        grad_checksum = main_grad_checksum(bundle)
        rank_grads = gather_objects({"rank": rank, "checksum": grad_checksum})
        gradients_synchronized = True
        if not use_dist:
            checksums = [item["checksum"] for item in rank_grads]
            if max(checksums) - min(checksums) > 1.0e-3:
                raise RuntimeError(f"main_grad not synchronized after finalize: {rank_grads}")
        else:
            checksums = [item["checksum"] for item in rank_grads]
            gradients_synchronized = len(set(round(c, 1) for c in checksums)) == 2

        update_successful, _, _ = bundle.optimizer.step()
        torch.cuda.synchronize(device)
        if not use_dist:
            assert_optimizer_consumed_main_grad(bundle)
        assert_main_grad_pointers(bundle.model, pointers)
        param_checksum = parameter_checksum(bundle)
        rank_params = gather_objects({"rank": rank, "checksum": param_checksum})
        parameters_identical = True
        if not use_dist:
            param_checksums = [item["checksum"] for item in rank_params]
            if max(param_checksums) - min(param_checksums) > 1.0e-3:
                raise RuntimeError(f"parameters diverged after optimizer step: {rank_params}")
        else:
            param_checksums = [item["checksum"] for item in rank_params]
            parameters_identical = max(param_checksums) - min(param_checksums) > 1.0e-3

        loss_value = float(loss.detach().cpu())
        rank_losses = gather_objects({"rank": rank, "loss": loss_value})
        loss_values = [item["loss"] for item in rank_losses]
        if not all(math.isfinite(value) for value in loss_values):
            raise RuntimeError(f"Non-finite loss on a rank: {rank_losses}")
        loss_matched_across_ranks = max(loss_values) - min(loss_values) <= 1.0e-3
        if not update_successful or not gradients_finite or not math.isfinite(loss_value):
            raise RuntimeError("Smoke-step lifecycle failed")
        smoke_losses.append(loss_value)
        lifecycle_checks.append(
            {
                "step": smoke_index,
                "loss_finite": math.isfinite(loss_value),
                "main_grads_finite": gradients_finite,
                "gradients_synchronized": gradients_synchronized,
                "parameters_identical_across_ranks": parameters_identical,
                "optimizer_consumed_main_grad": True,
                "main_grad_addresses_stable": True,
                "deadlock": False,
            }
        )
    if torch.equal(parameters_before, tracked_parameter.detach().cpu()):
        raise RuntimeError(f"Parameter {tracked_name} did not change during smoke steps")

    opt_meta = collect_optimizer_state_metadata(bundle)
    rank_opt = gather_objects({"rank": rank, **opt_meta})
    sharding = verify_optimizer_state_sharding(rank_opt, use_dist)

    return {
        "both_ranks_initialized": sorted(item["dp_rank"] for item in rank_tokens) == [0, 1],
        "different_data_per_dp_rank": True,
        "forward": True,
        "backward": True,
        "main_grad": True,
        "optimizer": True,
        "gradients_synchronized": not use_dist or gradients_synchronized,
        "parameters_identical_after_optimizer": not use_dist,
        "parameters_sharded_after_optimizer": use_dist,
        "finite_loss": all(math.isfinite(loss) for loss in smoke_losses),
        "loss_matched_across_ranks": loss_matched_across_ranks,
        "no_nan_inf": True,
        "no_deadlock": True,
        "parameters_updated": True,
        "smoke_losses": smoke_losses,
        "lifecycle_checks": lifecycle_checks,
        "rank_token_fingerprints": rank_tokens,
        "optimizer_state_sharding": sharding,
        "param_partition": collect_param_partition_metadata(bundle.model),
    }


def build_payload_common(
    args: argparse.Namespace,
    model_args: argparse.Namespace,
    *,
    status: str,
    correctness: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "experiment": "Phase 12 training memory and capacity",
        "run_label": args.run_label,
        "variant": args.variant,
        "iteration_mode": "FAST ITERATION MODE",
        "use_distributed_optimizer": args.use_distributed_optimizer,
        "overlap_grad_reduce": args.overlap_grad_reduce,
        "overlap_param_gather": args.overlap_param_gather,
        "activation_checkpointing": recompute_config(args.activation_checkpointing),
        "parallelism": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "world_size": 2,
            "sequence_parallel": False,
            "virtual_pipeline": False,
        },
        "batch": {
            "micro_batch_size_per_gpu": args.micro_batch_size,
            "gradient_accumulation": 1,
            "global_batch_size": args.micro_batch_size * 2,
            "sequence_length": args.sequence_length,
            "tokens_per_step": args.micro_batch_size * 2 * args.sequence_length,
        },
        "model": {
            "num_layers": model_args.num_layers,
            "hidden_size": model_args.hidden_size,
            "ffn_hidden_size": model_args.ffn_hidden_size,
            "num_attention_heads": model_args.num_attention_heads,
            "vocab_size": model_args.vocab_size,
            "precision": "bf16-autocast",
            "fused_attention": True,
            "bias_dropout_fusion": True,
            "bias_gelu_fusion": False,
            "cuda_graph": False,
        },
    }
    if correctness is not None:
        payload["correctness"] = correctness
    if extra:
        payload.update(extra)
    return payload


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    # Capacity / fixed workload may override sequence length independently of model max.
    # GPTModel is built with max_sequence_length from model_args; raise if needed.
    if args.sequence_length > model_args.sequence_length:
        model_args.sequence_length = args.sequence_length
    local_rank = initialize_distributed(model_args.seed)
    rank = dist.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    device = torch.device(f"cuda:{local_rank}")
    oom_status = False
    error_message: str | None = None
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("Transformer Engine must remain enabled")
        if os.environ.get("NCCL_P2P_DISABLE") == "1":
            raise RuntimeError("NCCL_P2P_DISABLE must not be set")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required")
        max_connections = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")
        if max_connections == "1":
            raise RuntimeError(
                "DP overlap needs CUDA_DEVICE_MAX_CONNECTIONS>1 so NCCL can run on a side stream"
            )

        build_kwargs: dict[str, Any] = {
            "attention_implementation": TE_FUSED_ATTENTION,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "bias_dropout_fusion": True,
            "cuda_graph_impl": "none",
        }
        if args.activation_checkpointing:
            build_kwargs.update(
                {
                    "recompute_granularity": RECOMPUTE_GRANULARITY,
                    "recompute_method": RECOMPUTE_METHOD,
                    "recompute_num_layers": RECOMPUTE_NUM_LAYERS,
                    "distribute_saved_activations": False,
                }
            )

        model = build_model(model_args, **build_kwargs)
        if model.config.cuda_graph_impl != "none":
            raise RuntimeError("CUDA Graph must remain disabled")
        if bool(model.config.bias_dropout_fusion) is not True:
            raise RuntimeError("bias_dropout_fusion must stay True")
        if bool(model.config.bias_activation_fusion):
            raise RuntimeError("bias_gelu / bias_activation fusion must stay False")
        if args.activation_checkpointing:
            if model.config.recompute_granularity != RECOMPUTE_GRANULARITY:
                raise RuntimeError("recompute_granularity not honored")
            if model.config.recompute_method != RECOMPUTE_METHOD:
                raise RuntimeError("recompute_method not honored")
            if model.config.recompute_num_layers != RECOMPUTE_NUM_LAYERS:
                raise RuntimeError("recompute_num_layers not honored")
        elif model.config.recompute_granularity is not None:
            raise RuntimeError("activation checkpointing unexpectedly enabled")

        bundle = build_megatron_optimizer_bundle(
            model,
            model_args.learning_rate,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            overlap_param_gather=args.overlap_param_gather,
            disable_bucketing=False,
        )
        if bundle.ddp_config.overlap_grad_reduce != args.overlap_grad_reduce:
            raise RuntimeError("DDP did not honor overlap_grad_reduce")
        if bundle.ddp_config.use_distributed_optimizer != args.use_distributed_optimizer:
            raise RuntimeError("DDP did not honor use_distributed_optimizer")
        if bundle.ddp_config.overlap_param_gather != args.overlap_param_gather:
            raise RuntimeError("DDP did not honor overlap_param_gather")

        instrument_grad_sync_nvtx(bundle.model)
        instrument_param_sync_nvtx(bundle.model)
        pointers = main_grad_pointers(bundle.model)
        buckets = collect_bucket_metadata(bundle.model)
        lifecycle = lifecycle_metadata(bundle)
        # TE populates _attention_backends on first DotProductAttention forward.
        # Check fused status only after smoke steps (matches Phase 9.2 / 8.1).
        fused: dict[str, Any] | None = None

        if args.mode == "smoke":
            correctness = run_smoke_correctness(
                bundle,
                model_args,
                args.micro_batch_size,
                args.sequence_length,
                device,
                dp_rank,
                rank,
                args,
                pointers,
            )
            fused = fused_backend_status()
            if rank == 0:
                write_rank0_json(
                    args.output_json,
                    build_payload_common(
                        args,
                        model_args,
                        status="success",
                        correctness=correctness,
                        extra={
                            "run_mode": "smoke",
                            "buckets": buckets,
                            "ddp_lifecycle": lifecycle,
                            "fused_backend_status": fused,
                        },
                    ),
                )
            return

        # Capacity probe: short multi-step train; report OOM without raising across ranks.
        probe_steps = 3 if args.mode == "capacity-probe" else 0
        smoke_iters = 1 if args.mode == "capacity-probe" else args.smoke_iterations
        warmup_iters = 0 if args.mode == "capacity-probe" else args.warmup_iterations
        measured_iters = (
            probe_steps if args.mode == "capacity-probe" else args.measured_iterations
        )
        args.smoke_iterations = smoke_iters

        try:
            correctness = run_smoke_correctness(
                bundle,
                model_args,
                args.micro_batch_size,
                args.sequence_length,
                device,
                dp_rank,
                rank,
                args,
                pointers,
            )
            fused = fused_backend_status()
        except Exception as exc:
            if is_oom_error(exc):
                oom_status = True
                error_message = f"OOM during smoke: {exc}"
                torch.cuda.empty_cache()
            else:
                raise

        if not oom_status:
            for warmup_index in range(warmup_iters):
                batch = dp_synthetic_batch(
                    model_args,
                    args.micro_batch_size,
                    args.sequence_length,
                    device,
                    dp_rank,
                    step_index=100 + warmup_index,
                )
                try:
                    train_step(bundle, batch, warmup_index)
                    torch.cuda.synchronize(device)
                except Exception as exc:
                    if is_oom_error(exc):
                        oom_status = True
                        error_message = f"OOM during warmup: {exc}"
                        torch.cuda.empty_cache()
                        break
                    raise

        torch.cuda.reset_peak_memory_stats(device)
        sampler = None
        measured_losses: list[float] = []
        local_step_times_ms: list[float] = []
        if not oom_status:
            if rank == 0 and args.mode == "benchmark":
                sampler = MultiGpuNvidiaSmiSampler(
                    list(range(2)),
                    args.gpu_sample_interval_ms,
                )
                sampler.start()
            dist.barrier()
            if args.profile_mode and rank == 0:
                torch.cuda.cudart().cudaProfilerStart()
            dist.barrier()
            try:
                emit_context = (
                    torch.autograd.profiler.emit_nvtx(record_shapes=True)
                    if args.profile_mode
                    else torch.autograd.profiler.emit_nvtx(enabled=False)
                )
                with torch.cuda.nvtx.range("profile_window"):
                    with emit_context:
                        for step_index in range(measured_iters):
                            batch = dp_synthetic_batch(
                                model_args,
                                args.micro_batch_size,
                                args.sequence_length,
                                device,
                                dp_rank,
                                step_index=1_000 + step_index,
                            )
                            torch.cuda.synchronize(device)
                            start = time.perf_counter()
                            try:
                                measured_losses.append(
                                    train_step(bundle, batch, step_index)
                                )
                            except Exception as exc:
                                if is_oom_error(exc):
                                    oom_status = True
                                    error_message = f"OOM during measured: {exc}"
                                    torch.cuda.empty_cache()
                                    break
                                raise
                            torch.cuda.synchronize(device)
                            local_step_times_ms.append(
                                (time.perf_counter() - start) * 1000.0
                            )
            finally:
                dist.barrier()
                if args.profile_mode and rank == 0:
                    torch.cuda.cudart().cudaProfilerStop()
                dist.barrier()
                gpu_monitoring = sampler.stop() if sampler is not None else None
        else:
            gpu_monitoring = None

        # Sync OOM across ranks so all exit cleanly with a consistent JSON.
        oom_flags = gather_objects({"rank": rank, "oom": oom_status, "error": error_message})
        if any(item["oom"] for item in oom_flags):
            oom_status = True
            error_message = next(
                (item["error"] for item in oom_flags if item["error"]),
                "OOM on a peer rank",
            )

        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        opt_meta = collect_optimizer_state_metadata(bundle) if not oom_status else {
            "optimizer_state_bytes": 0
        }
        rank_times = gather_objects(
            {
                "rank": rank,
                "dp_rank": dp_rank,
                "step_times_ms": local_step_times_ms,
                "peak_allocated_memory_mib": peak_allocated,
                "peak_reserved_memory_mib": peak_reserved,
                **opt_meta,
            }
        )
        rank_losses = gather_objects({"rank": rank, "losses": measured_losses})
        if rank != 0:
            return

        if oom_status:
            write_rank0_json(
                args.output_json,
                build_payload_common(
                    args,
                    model_args,
                    status="oom",
                    extra={
                        "run_mode": args.mode,
                        "oom": True,
                        "error": error_message,
                        "peak_allocated_memory_mib": [
                            item["peak_allocated_memory_mib"] for item in rank_times
                        ],
                        "peak_reserved_memory_mib": [
                            item["peak_reserved_memory_mib"] for item in rank_times
                        ],
                        "environment": collect_environment(2),
                    },
                ),
            )
            return

        if not local_step_times_ms:
            raise RuntimeError("No measured steps completed")

        global_step_times_ms = [
            max(rank_time["step_times_ms"][index] for rank_time in rank_times)
            for index in range(len(local_step_times_ms))
        ]
        average_step_time_ms = statistics.fmean(global_step_times_ms)
        median_step_time_ms = statistics.median(global_step_times_ms)
        global_batch_size = args.micro_batch_size * 2
        tokens_per_step = global_batch_size * args.sequence_length
        tokens_per_second = tokens_per_step / (average_step_time_ms / 1000.0)
        flops_per_iteration = training_flops_per_iteration(
            global_batch_size,
            args.sequence_length,
            model_args.num_layers,
            model_args.hidden_size,
            model_args.vocab_size,
        )
        # Activation checkpointing adds ~1/3 extra FLOPs (recompute forward).
        # Report measured MFU against theoretical non-recompute FLOPs (standard MFU).
        aggregate_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1e12
        mfu_percent = aggregate_tflops / (2 * A40_DENSE_BF16_PEAK_TFLOPS) * 100.0
        peak_smi = [
            (gpu_monitoring or {}).get(str(index), {}).get("peak_memory_mib")
            for index in range(2)
        ]
        mean_util = None
        if gpu_monitoring:
            utils = [
                item.get("average_utilization_percent")
                for item in gpu_monitoring.values()
                if item.get("average_utilization_percent") is not None
            ]
            if utils:
                mean_util = statistics.fmean(utils)

        sharding = verify_optimizer_state_sharding(
            [{"optimizer_state_bytes": item["optimizer_state_bytes"]} for item in rank_times],
            args.use_distributed_optimizer,
        )

        write_rank0_json(
            args.output_json,
            build_payload_common(
                args,
                model_args,
                status="success",
                correctness={
                    **correctness,
                    "finite_loss": all(
                        math.isfinite(loss)
                        for row in rank_losses
                        for loss in row["losses"]
                    )
                    and correctness["finite_loss"],
                },
                extra={
                    "run_mode": "nsight_profile" if args.profile_mode else args.mode,
                    "oom": False,
                    "buckets": buckets,
                    "ddp_lifecycle": lifecycle,
                    "fused_backend_status": fused,
                    "optimizer_state_sharding": sharding,
                    "optimizer_state_bytes_per_rank": [
                        item["optimizer_state_bytes"] for item in rank_times
                    ],
                    "param_partition": collect_param_partition_metadata(bundle.model),
                    "tokens_per_second": tokens_per_second,
                    "average_step_time_ms": average_step_time_ms,
                    "median_step_time_ms": median_step_time_ms,
                    "mfu_percent": mfu_percent,
                    "peak_allocated_memory_mib": [
                        item["peak_allocated_memory_mib"] for item in rank_times
                    ],
                    "peak_reserved_memory_mib": [
                        item["peak_reserved_memory_mib"] for item in rank_times
                    ],
                    "smi_peak_memory_mib": peak_smi,
                    "mean_gpu_utilization_percent": mean_util,
                    "gpu_monitoring": gpu_monitoring,
                    "step_times_ms": global_step_times_ms,
                    "measured_losses": measured_losses,
                    "environment": collect_environment(2),
                    "warmup_iterations": warmup_iters,
                    "measured_iterations": measured_iters,
                },
            ),
        )
    except Exception as exc:
        if is_oom_error(exc):
            if rank == 0:
                write_rank0_json(
                    args.output_json,
                    build_payload_common(
                        args,
                        model_args,
                        status="oom",
                        extra={
                            "run_mode": args.mode,
                            "oom": True,
                            "error": str(exc),
                            "traceback": traceback.format_exc()[-2000:],
                            "environment": collect_environment(2),
                        },
                    ),
                )
            return
        raise
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
