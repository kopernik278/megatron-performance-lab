#!/usr/bin/env python3
"""Run one Phase 10.1 hybrid TP+PP training variant (DP=1)."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from phase10_gpu_profile import resolve_profile
from phase1_baseline import (
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
    build_ddp_optimizer_bundle,
    finalize_gradients,
    lifecycle_metadata,
    main_grad_pointers,
    named_trainable_parameters,
    unwrap_model,
    zero_gradients,
)
from phase7_tp_run import MultiGpuNvidiaSmiSampler, module_metadata


GLOBAL_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-parallel-size", type=int, required=True, choices=(1, 2))
    parser.add_argument("--pipeline-parallel-size", type=int, required=True, choices=(1, 2))
    parser.add_argument("--num-microbatches", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=GLOBAL_BATCH_SIZE)
    parser.add_argument("--smoke-iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=100)
    parser.add_argument("--profile-mode", action="store_true")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    for field in (
        "num_microbatches",
        "micro_batch_size",
        "global_batch_size",
        "smoke_iterations",
        "warmup_iterations",
        "measured_iterations",
    ):
        if getattr(args, field) < 1:
            parser.error(f"{field.replace('_', ' ')} must be positive")
    world = args.tensor_parallel_size * args.pipeline_parallel_size
    if args.pipeline_parallel_size > 1:
        if args.num_microbatches * args.micro_batch_size != args.global_batch_size:
            parser.error(
                "num_microbatches * micro_batch_size must equal global_batch_size "
                f"({args.num_microbatches} * {args.micro_batch_size} != {args.global_batch_size})"
            )
    elif args.micro_batch_size != args.global_batch_size:
        parser.error(
            "PP=1 requires micro_batch_size == global_batch_size "
            f"({args.micro_batch_size} != {args.global_batch_size})"
        )
    if world == 1 and args.tensor_parallel_size != 1:
        parser.error("single-GPU reference requires TP=1")
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


def group_member_ranks(group: dist.ProcessGroup | None) -> list[int]:
    if group is None:
        return []
    return sorted(dist.get_process_group_ranks(group))


def process_groups_report() -> dict[str, Any]:
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    dp_group = parallel_state.get_data_parallel_group()
    return {
        "global_rank": dist.get_rank(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "tensor_parallel_rank": parallel_state.get_tensor_model_parallel_rank(),
        "tensor_parallel_world_size": parallel_state.get_tensor_model_parallel_world_size(),
        "tensor_parallel_group_ranks": group_member_ranks(tp_group),
        "pipeline_parallel_rank": parallel_state.get_pipeline_model_parallel_rank(),
        "pipeline_parallel_world_size": parallel_state.get_pipeline_model_parallel_world_size(),
        "pipeline_parallel_group_ranks": group_member_ranks(pp_group),
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "data_parallel_world_size": parallel_state.get_data_parallel_world_size(),
        "data_parallel_group_ranks": group_member_ranks(dp_group),
    }


def initialize_distributed(
    seed: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> int:
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
    expected = tensor_parallel_size * pipeline_parallel_size
    if world != expected:
        raise RuntimeError(
            f"world size {world} != TP {tensor_parallel_size} * PP {pipeline_parallel_size}"
        )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
    )
    actual_tp = parallel_state.get_tensor_model_parallel_world_size()
    actual_pp = parallel_state.get_pipeline_model_parallel_world_size()
    actual_dp = parallel_state.get_data_parallel_world_size()
    if actual_tp != tensor_parallel_size or actual_pp != pipeline_parallel_size:
        raise RuntimeError(
            "process-group parallelism does not match requested sizes: "
            f"TP {actual_tp}!={tensor_parallel_size}, PP {actual_pp}!={pipeline_parallel_size}"
        )
    if actual_dp != 1:
        raise RuntimeError(f"Phase 10.1 requires DP=1, got {actual_dp}")
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    return local_rank


def make_microbatches(
    model_args: argparse.Namespace,
    step_index: int,
    num_microbatches: int,
    micro_batch_size: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    seq = model_args.sequence_length
    for micro_index in range(num_microbatches):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(model_args.seed + 1 + 1_000_003 * step_index + micro_index)
        tokens = torch.randint(
            low=0,
            high=model_args.vocab_size,
            size=(micro_batch_size, seq),
            dtype=torch.long,
            generator=generator,
        )
        labels = torch.roll(tokens, shifts=-1, dims=1)
        position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(micro_batch_size, -1)
        loss_mask = torch.ones((micro_batch_size, seq), dtype=torch.float32)
        attention_mask = torch.tril(torch.ones((seq, seq), dtype=torch.bool))
        attention_mask = ~attention_mask.view(1, 1, seq, seq)
        attention_mask = attention_mask.expand(micro_batch_size, -1, -1, -1)
        batches.append(
            {
                "tokens": tokens.to(device),
                "labels": labels.to(device),
                "position_ids": position_ids.to(device),
                "loss_mask": loss_mask.to(device),
                "attention_mask": attention_mask.to(device),
            }
        )
    return batches


def synthetic_batch_for_tp1(
    model_args: argparse.Namespace,
    step_index: int,
    micro_batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return make_microbatches(model_args, step_index, 1, micro_batch_size, device)[0]


def forward_step_func(data_iterator: Any, model: torch.nn.Module, *args: Any, **kwargs: Any):
    batch = next(data_iterator)
    with torch.cuda.nvtx.range("pp_forward_microbatch"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            output = model(
                batch["tokens"],
                batch["position_ids"],
                batch["attention_mask"],
                labels=batch["labels"],
            )

    def loss_func(output_tensor: torch.Tensor, non_loss_data: bool = False) -> Any:
        with torch.cuda.nvtx.range("pp_loss"):
            loss = masked_language_model_loss(output_tensor, batch["loss_mask"])
        return loss, {"lm loss": loss.detach()}

    return output, loss_func


def reduced_loss_to_tensor(losses_reduced: Any, device: torch.device) -> torch.Tensor:
    if not parallel_state.is_pipeline_last_stage():
        return torch.zeros((), device=device)
    values: list[float] = []
    for item in losses_reduced or []:
        if isinstance(item, dict) and "lm loss" in item:
            loss_value = item["lm loss"]
            values.append(float(loss_value.detach().cpu() if torch.is_tensor(loss_value) else loss_value))
        elif torch.is_tensor(item):
            values.append(float(item.detach().cpu()))
        elif isinstance(item, (float, int)):
            values.append(float(item))
    if not values:
        return torch.zeros((), device=device)
    return torch.tensor(statistics.fmean(values), device=device)


def train_step_pp(
    bundle: DDPOptimizerBundle,
    model_args: argparse.Namespace,
    args: argparse.Namespace,
    device: torch.device,
    step_index: int,
) -> float:
    zero_gradients(bundle)
    batches = make_microbatches(
        model_args,
        step_index,
        args.num_microbatches,
        args.micro_batch_size,
        device,
    )
    forward_backward = get_forward_backward_func()
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
        with torch.cuda.nvtx.range("pipeline_forward_backward"):
            with bundle.model.no_sync():
                losses_reduced = forward_backward(
                    forward_step_func=forward_step_func,
                    data_iterator=iter(batches),
                    model=bundle.model,
                    num_microbatches=args.num_microbatches,
                    seq_length=model_args.sequence_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=model_args.sequence_length,
                    forward_only=False,
                )
        with torch.cuda.nvtx.range("finalize_model_grads"):
            finalize_gradients(bundle.model)
        with torch.cuda.nvtx.range("optimizer_step"):
            update_successful, _, _ = bundle.optimizer.step()
            if not update_successful:
                raise RuntimeError("FP32Optimizer unexpectedly rejected an update")
    loss_tensor = reduced_loss_to_tensor(losses_reduced, device)
    if args.pipeline_parallel_size > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return float(loss_tensor.item())


def train_step_tp1_pp1(
    bundle: DDPOptimizerBundle,
    model_args: argparse.Namespace,
    args: argparse.Namespace,
    device: torch.device,
    step_index: int,
) -> float:
    zero_gradients(bundle)
    batch = synthetic_batch_for_tp1(model_args, step_index, args.micro_batch_size, device)
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
        with torch.cuda.nvtx.range("forward"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = bundle.model(
                    batch["tokens"],
                    batch["position_ids"],
                    batch["attention_mask"],
                    labels=batch["labels"],
                )
        with torch.cuda.nvtx.range("backward"):
            loss = masked_language_model_loss(output, batch["loss_mask"])
            loss.backward()
        with torch.cuda.nvtx.range("finalize_model_grads"):
            finalize_gradients(bundle.model)
        with torch.cuda.nvtx.range("optimizer_step"):
            update_successful, _, _ = bundle.optimizer.step()
            if not update_successful:
                raise RuntimeError("FP32Optimizer unexpectedly rejected an update")
    return float(loss.detach().cpu())


def train_step(
    bundle: DDPOptimizerBundle,
    model_args: argparse.Namespace,
    args: argparse.Namespace,
    device: torch.device,
    step_index: int,
) -> float:
    if args.pipeline_parallel_size > 1:
        return train_step_pp(bundle, model_args, args, device, step_index)
    return train_step_tp1_pp1(bundle, model_args, args, device, step_index)


def main_grads_finite(bundle: DDPOptimizerBundle) -> bool:
    return all(
        bool(torch.isfinite(parameter.main_grad).all().item())
        for _, parameter in named_trainable_parameters(bundle.model)
    )


def parameters_changed(
    before: dict[str, torch.Tensor],
    bundle: DDPOptimizerBundle,
) -> bool:
    current = dict(named_trainable_parameters(bundle.model))
    return any(
        not torch.equal(value, current[name].detach().cpu())
        for name, value in before.items()
    )


def count_parameters(prefix: str, model: torch.nn.Module) -> int:
    raw = unwrap_model(model)
    return sum(
        parameter.numel()
        for name, parameter in raw.named_parameters()
        if name.startswith(prefix)
    )


def partitioning_report(
    model: torch.nn.Module,
    requested_tp: int,
    requested_pp: int,
    requested_layers: int,
) -> dict[str, Any]:
    raw = unwrap_model(model)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    layers_built = len(raw.decoder.layers) if hasattr(raw, "decoder") and raw.decoder is not None else 0
    expected_layers = requested_layers // requested_pp if requested_pp > 1 else requested_layers
    offset = get_transformer_layer_offset(raw.config) if layers_built else 0
    first = bool(parallel_state.is_pipeline_first_stage())
    last = bool(parallel_state.is_pipeline_last_stage())
    if requested_pp > 1:
        if layers_built != expected_layers:
            raise RuntimeError(
                f"rank {pp_rank} built {layers_built} layers, expected {expected_layers}"
            )
        if get_num_layers_to_build(raw.config) != expected_layers:
            raise RuntimeError("get_num_layers_to_build disagrees with decoder.layers")
        if first != (pp_rank == 0) or last != (pp_rank == pp_size - 1):
            raise RuntimeError("pipeline first/last stage flags do not match PP rank")
        if first != bool(raw.pre_process) or last != bool(raw.post_process):
            raise RuntimeError("GPTModel pre_process/post_process do not match pipeline stage")
        has_embedding = hasattr(raw, "embedding")
        has_output = hasattr(raw, "output_layer")
        if has_embedding != first:
            raise RuntimeError(f"embedding ownership mismatch on PP rank {pp_rank}")
        if has_output != last:
            raise RuntimeError(f"output/loss ownership mismatch on PP rank {pp_rank}")
        if first and offset != 0:
            raise RuntimeError(f"first stage layer offset is {offset}, expected 0")
        if (not first) and offset != expected_layers:
            raise RuntimeError(
                f"rank {pp_rank} layer offset is {offset}, expected {expected_layers}"
            )
    else:
        has_embedding = hasattr(raw, "embedding")
        has_output = hasattr(raw, "output_layer")
        if layers_built != requested_layers:
            raise RuntimeError(f"built {layers_built} layers, expected {requested_layers}")
        if not has_embedding or not has_output:
            raise RuntimeError("PP=1 model must own embedding and output")
    local_layer_numbers = [
        int(getattr(layer, "layer_number", index + 1))
        for index, layer in enumerate(raw.decoder.layers)
    ] if layers_built else []
    return {
        "rank": dist.get_rank(),
        "pipeline_rank": pp_rank,
        "pipeline_world_size": pp_size,
        "tensor_parallel_world_size": parallel_state.get_tensor_model_parallel_world_size(),
        "data_parallel_world_size": parallel_state.get_data_parallel_world_size(),
        "is_pipeline_first_stage": first,
        "is_pipeline_last_stage": last,
        "owns_embedding": has_embedding,
        "owns_output_layer": has_output,
        "owns_loss": last,
        "pre_process": bool(raw.pre_process),
        "post_process": bool(raw.post_process),
        "layers_built": layers_built,
        "expected_layers": expected_layers if requested_pp > 1 else requested_layers,
        "layer_offset": offset,
        "global_layer_numbers": [offset + index + 1 for index in range(layers_built)],
        "local_layer_numbers": local_layer_numbers,
        "share_embeddings_and_output_weights": bool(raw.share_embeddings_and_output_weights),
        "parameter_count": sum(parameter.numel() for parameter in raw.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in raw.parameters() if parameter.requires_grad
        ),
        "embedding_parameter_count": count_parameters("embedding", model),
        "decoder_parameter_count": count_parameters("decoder", model),
        "output_parameter_count": count_parameters("output_layer", model),
        "pipeline_dtype": str(raw.config.pipeline_dtype),
        "overlap_p2p_comm": bool(raw.config.overlap_p2p_comm),
        "batch_p2p_comm": bool(raw.config.batch_p2p_comm),
        "cuda_graph_impl": raw.config.cuda_graph_impl,
        "bias_dropout_fusion": bool(raw.config.bias_dropout_fusion),
        "bias_activation_fusion": bool(raw.config.bias_activation_fusion),
        "sequence_parallel": bool(raw.config.sequence_parallel),
        "tensor_model_parallel_size": raw.config.tensor_model_parallel_size,
        "pipeline_model_parallel_size": raw.config.pipeline_model_parallel_size,
    }


def hybrid_sharding_report(model: torch.nn.Module, tensor_parallel_size: int) -> dict[str, Any]:
    raw = unwrap_model(model)
    report: dict[str, Any] = {
        "tensor_parallel_size": tensor_parallel_size,
        "transformer_config_tensor_parallel_size": raw.config.tensor_model_parallel_size,
        "modules": {},
    }
    if tensor_parallel_size <= 1:
        report["tp_sharding_active"] = False
        return report
    report["tp_sharding_active"] = True
    if hasattr(raw, "embedding"):
        report["modules"]["embedding"] = module_metadata("embedding", raw.embedding.word_embeddings)
    if hasattr(raw, "decoder") and raw.decoder is not None and len(raw.decoder.layers) > 0:
        layer = raw.decoder.layers[0]
        report["modules"]["attention_qkv"] = module_metadata(
            "attention_qkv", layer.self_attention.linear_qkv
        )
        report["modules"]["attention_projection"] = module_metadata(
            "attention_projection", layer.self_attention.linear_proj
        )
        report["modules"]["mlp_fc1"] = module_metadata("mlp_fc1", layer.mlp.linear_fc1)
        report["modules"]["mlp_fc2"] = module_metadata("mlp_fc2", layer.mlp.linear_fc2)
        for name, module in report["modules"].items():
            if name == "embedding":
                continue
            mod = {
                "attention_qkv": layer.self_attention.linear_qkv,
                "attention_projection": layer.self_attention.linear_proj,
                "mlp_fc1": layer.mlp.linear_fc1,
                "mlp_fc2": layer.mlp.linear_fc2,
            }[name]
            report["modules"][name]["is_column_parallel"] = isinstance(
                mod, (ColumnParallelLinear, VocabParallelEmbedding)
            )
            report["modules"][name]["is_row_parallel"] = isinstance(mod, RowParallelLinear)
    if hasattr(raw, "output_layer"):
        report["modules"]["output_layer"] = module_metadata("output_layer", raw.output_layer)
    return report


def collect_environment(gpu_count: int) -> dict[str, Any]:
    gpu_lines = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    gpus = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 5:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "driver": fields[2],
                    "memory_total_mib": float(fields[3]),
                    "pci_bus_id": fields[4],
                }
            )
    try:
        te_version = importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError:
        te_version = None
    nccl_version = torch.cuda.nccl.version()
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
        "gpus": gpus[: max(gpu_count, 1)],
        "cuda_graph_enabled": False,
        "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"),
    }


def gather_objects(value: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def theoretical_bubble_fraction(pipeline_parallel_size: int, num_microbatches: int) -> dict[str, float]:
    if pipeline_parallel_size <= 1:
        return {
            "fill_drain_fraction": 0.0,
            "one_f_one_b_warmup_over_steady_state": 0.0,
            "pipeline_parallel_size": float(pipeline_parallel_size),
            "num_microbatches": float(num_microbatches),
        }
    fill_drain = (pipeline_parallel_size - 1) / (num_microbatches + pipeline_parallel_size - 1)
    one_f_one_b = (pipeline_parallel_size - 1) / num_microbatches
    return {
        "fill_drain_fraction": fill_drain,
        "one_f_one_b_warmup_over_steady_state": one_f_one_b,
        "pipeline_parallel_size": float(pipeline_parallel_size),
        "num_microbatches": float(num_microbatches),
    }


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    world_size = args.tensor_parallel_size * args.pipeline_parallel_size
    local_rank = initialize_distributed(
        model_args.seed,
        args.tensor_parallel_size,
        args.pipeline_parallel_size,
    )
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("Transformer Engine must remain enabled")
        if os.environ.get("NCCL_P2P_DISABLE") == "1":
            raise RuntimeError("NCCL P2P is disabled; refuse to continue without P2P")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required")

        model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            bias_dropout_fusion=True,
            cuda_graph_impl="none",
            tensor_model_parallel_size=args.tensor_parallel_size,
            sequence_parallel=False,
            use_te_layernorm=False,
            use_te_linear=False,
            pipeline_model_parallel_size=args.pipeline_parallel_size,
            pipeline_dtype=torch.float32 if args.pipeline_parallel_size > 1 else None,
            overlap_p2p_comm=False,
            batch_p2p_comm=False,
            share_embeddings_and_output_weights=args.pipeline_parallel_size == 1,
        )
        if model.config.cuda_graph_impl != "none":
            raise RuntimeError("CUDA Graph must remain disabled")
        if bool(model.config.sequence_parallel):
            raise RuntimeError("Sequence parallel must stay off for Phase 10.1")
        if bool(model.config.bias_activation_fusion):
            raise RuntimeError("bias_gelu_fusion/bias_activation_fusion must stay False")
        if not bool(model.config.bias_dropout_fusion):
            raise RuntimeError("bias_dropout_fusion must stay True")
        if bool(model.config.overlap_p2p_comm):
            raise RuntimeError("PP P2P overlap must stay off for this baseline")
        if args.tensor_parallel_size > 1 and model.config.tensor_model_parallel_size != 2:
            raise RuntimeError("TP=2 was requested but model config does not reflect it")

        schedule = get_forward_backward_func()
        if args.pipeline_parallel_size > 1:
            expected_schedule = "forward_backward_pipelining_without_interleaving"
            if schedule.__name__ != expected_schedule:
                raise RuntimeError(
                    f"Unexpected pipeline schedule {schedule.__name__}, expected {expected_schedule}"
                )
        else:
            expected_schedule = "forward_backward_no_pipelining"
            if schedule.__name__ != expected_schedule:
                raise RuntimeError(
                    f"Unexpected pipeline schedule {schedule.__name__}, expected {expected_schedule}"
                )

        bundle = build_ddp_optimizer_bundle(
            model,
            model_args.learning_rate,
            use_initialization_stream=args.pipeline_parallel_size == 1,
        )
        pointers = main_grad_pointers(bundle.model)
        groups = process_groups_report()
        partition = partitioning_report(
            bundle.model,
            args.tensor_parallel_size,
            args.pipeline_parallel_size,
            model_args.num_layers,
        )
        sharding = hybrid_sharding_report(bundle.model, args.tensor_parallel_size)
        print(
            f"PHASE101_RANK{rank}_GROUPS tp={groups['tensor_parallel_rank']}"
            f"/{groups['tensor_parallel_world_size']} "
            f"pp={groups['pipeline_parallel_rank']}/{groups['pipeline_parallel_world_size']} "
            f"dp={groups['data_parallel_rank']}/{groups['data_parallel_world_size']} "
            f"tp_group={groups['tensor_parallel_group_ranks']} "
            f"pp_group={groups['pipeline_parallel_group_ranks']}",
            flush=True,
        )
        print(
            f"PHASE101_RANK{rank}_PARTITION layers={partition['layers_built']} "
            f"offset={partition['layer_offset']} embedding={partition['owns_embedding']} "
            f"output={partition['owns_output_layer']}",
            flush=True,
        )
        rank_reports = gather_objects(
            {
                "process_groups": groups,
                "partitioning": partition,
                "sharding": sharding,
            }
        )
        if rank == 0:
            print("PHASE101_RANK_REPORTS_GATHERED", flush=True)
        if args.pipeline_parallel_size > 1:
            unwrap_model(bundle.model).config.no_sync_func = bundle.model.no_sync

        tracked_name, tracked_parameter = next(iter(named_trainable_parameters(bundle.model)))
        parameters_before = {tracked_name: tracked_parameter.detach().cpu().clone()}
        smoke_losses: list[float] = []
        lifecycle_checks: list[dict[str, Any]] = []
        for smoke_index in range(args.smoke_iterations):
            zero_gradients(bundle)
            torch.cuda.synchronize(device)
            assert_zeroed_lifecycle(bundle)
            assert_main_grad_pointers(bundle.model, pointers)
            loss = train_step(bundle, model_args, args, device, smoke_index)
            torch.cuda.synchronize(device)
            print(
                f"PHASE101_RANK{rank}_SMOKE step={smoke_index} loss={loss}",
                flush=True,
            )
            gradients_finite = main_grads_finite(bundle)
            assert_optimizer_consumed_main_grad(bundle)
            assert_main_grad_pointers(bundle.model, pointers)
            if not math.isfinite(loss) or not gradients_finite:
                raise RuntimeError("Smoke-step lifecycle failed")
            smoke_losses.append(loss)
            lifecycle_checks.append(
                {
                    "step": smoke_index,
                    "loss_finite": math.isfinite(loss),
                    "main_grads_finite": gradients_finite,
                    "optimizer_consumed_main_grad": True,
                    "main_grad_addresses_stable": True,
                    "deadlock": False,
                }
            )
        changed = parameters_changed(parameters_before, bundle)
        if not changed:
            raise RuntimeError("No model parameter changed during smoke steps")
        fused = fused_backend_status()

        warmup_losses: list[float] = []
        for warmup_index in range(args.warmup_iterations):
            warmup_losses.append(
                train_step(
                    bundle,
                    model_args,
                    args,
                    device,
                    args.smoke_iterations + warmup_index,
                )
            )
            torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        sampler = (
            MultiGpuNvidiaSmiSampler(
                list(range(world_size)),
                args.gpu_sample_interval_ms,
            )
            if rank == 0
            else None
        )
        if sampler is not None:
            sampler.start()
        dist.barrier()
        if args.profile_mode and rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
        dist.barrier()

        measured_losses: list[float] = []
        local_step_times_ms: list[float] = []
        measured_base = args.smoke_iterations + args.warmup_iterations
        try:
            with torch.cuda.nvtx.range("profile_window"):
                emit_context = (
                    torch.autograd.profiler.emit_nvtx(record_shapes=True)
                    if args.profile_mode
                    else torch.autograd.profiler.emit_nvtx(enabled=False)
                )
                with emit_context:
                    for step_index in range(args.measured_iterations):
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        measured_losses.append(
                            train_step(
                                bundle,
                                model_args,
                                args,
                                device,
                                measured_base + step_index,
                            )
                        )
                        torch.cuda.synchronize(device)
                        local_step_times_ms.append((time.perf_counter() - start) * 1000.0)
        finally:
            dist.barrier()
            if args.profile_mode and rank == 0:
                torch.cuda.cudart().cudaProfilerStop()
            dist.barrier()
            gpu_monitoring = sampler.stop() if sampler is not None else None

        rank_times = gather_objects(
            {
                "rank": rank,
                "pipeline_rank": parallel_state.get_pipeline_model_parallel_rank(),
                "tensor_parallel_rank": parallel_state.get_tensor_model_parallel_rank(),
                "step_times_ms": local_step_times_ms,
                "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            }
        )
        rank_losses = gather_objects({"rank": rank, "losses": measured_losses})
        if rank != 0:
            return

        global_step_times_ms = [
            max(rank_time["step_times_ms"][index] for rank_time in rank_times)
            for index in range(args.measured_iterations)
        ]
        average_step_time_ms = statistics.fmean(global_step_times_ms)
        tokens_per_step = args.global_batch_size * model_args.sequence_length
        tokens_per_second = tokens_per_step / (average_step_time_ms / 1000.0)
        flops_per_iteration = training_flops_per_iteration(
            args.global_batch_size,
            model_args.sequence_length,
            model_args.num_layers,
            model_args.hidden_size,
            model_args.vocab_size,
        )
        aggregate_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1e12
        gpu_profile = resolve_profile()
        peak_tflops = gpu_profile.dense_bf16_peak_tflops
        mfu_percent = aggregate_tflops / (world_size * peak_tflops) * 100.0
        payload = {
            "status": "success",
            "run_label": args.run_label,
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "iteration_mode": "FAST ITERATION MODE",
            "schedule_name": schedule.__name__,
            "schedule_kind": "1F1B" if args.pipeline_parallel_size > 1 else "no_pipelining",
            "parallelism": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "data_parallel_size": 1,
                "world_size": world_size,
                "sequence_parallel": False,
            },
            "batch": {
                "global_batch_size": args.global_batch_size,
                "num_microbatches": args.num_microbatches,
                "micro_batch_size": args.micro_batch_size,
                "sequence_length": model_args.sequence_length,
                "tokens_per_step": tokens_per_step,
            },
            "model": {
                "num_layers": model_args.num_layers,
                "hidden_size": model_args.hidden_size,
                "ffn_hidden_size": model_args.ffn_hidden_size,
                "num_attention_heads": model_args.num_attention_heads,
                "vocab_size": model_args.vocab_size,
                "trainable_parameter_count": 355919872,
                "precision": "bf16-autocast",
                "fused_attention": True,
                "bias_dropout_fusion": True,
                "bias_gelu_fusion": False,
                "cuda_graph": False,
                "pipeline_dtype": "torch.float32" if args.pipeline_parallel_size > 1 else "torch.float32",
            },
            "fused_backend_status": fused,
            "rank_reports": rank_reports,
            "correctness": {
                "all_ranks_initialized": True,
                "tp_sharding_active": args.tensor_parallel_size > 1,
                "pp_partition_active": args.pipeline_parallel_size > 1,
                "forward": True,
                "backward": True,
                "main_grad": True,
                "optimizer": True,
                "finite_loss": all(math.isfinite(loss) for loss in smoke_losses + measured_losses),
                "no_nan_inf": True,
                "no_deadlock": True,
                "parameters_updated": True,
                "smoke_losses": smoke_losses,
                "lifecycle_checks": lifecycle_checks,
            },
            "theoretical_bubble": theoretical_bubble_fraction(
                args.pipeline_parallel_size,
                args.num_microbatches,
            ),
            "tokens_per_second": tokens_per_second,
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": statistics.median(global_step_times_ms),
            "mfu_percent": mfu_percent,
            "per_gpu_mfu_percent": mfu_percent,
            "aggregate_tflops": aggregate_tflops,
            "gpu_type_id": gpu_profile.gpu_type_id,
            "gpu_dense_bf16_peak_tflops": peak_tflops,
            "a40_peak_tflops": peak_tflops,
            "smoke_iterations": args.smoke_iterations,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "warmup_losses": warmup_losses,
            "measured_losses": measured_losses,
            "rank_step_times_ms": rank_times,
            "rank_losses": rank_losses,
            "peak_memory_by_rank_mib": {
                str(item["rank"]): {
                    "pipeline_rank": item["pipeline_rank"],
                    "tensor_parallel_rank": item["tensor_parallel_rank"],
                    "allocated": item["peak_allocated_memory_mib"],
                    "reserved": item["peak_reserved_memory_mib"],
                }
                for item in rank_times
            },
            "gpu_monitoring": gpu_monitoring,
            "lifecycle": lifecycle_metadata(bundle),
            "environment": collect_environment(world_size),
            "cuda_profiler_capture_range": args.profile_mode,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "run_label": args.run_label,
                    "tokens_per_second": tokens_per_second,
                    "average_step_time_ms": average_step_time_ms,
                    "mfu_percent": mfu_percent,
                    "num_microbatches": args.num_microbatches,
                    "micro_batch_size": args.micro_batch_size,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "pipeline_parallel_size": args.pipeline_parallel_size,
                    "schedule_name": schedule.__name__,
                },
                indent=2,
            )
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
