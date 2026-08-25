#!/usr/bin/env python3
"""Run Phase 8.3 interleaved 1F1B + overlap_p2p_comm (variant B)."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import configure_nvtx_profiling

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
    OptimizerConfig,
    FP32Optimizer,
    assert_main_grad_pointers,
    lifecycle_metadata,
    unwrap_model,
    wrap_with_megatron_ddp,
)
from phase7_tp_run import MultiGpuNvidiaSmiSampler
from phase8_pp_run import (
    GLOBAL_BATCH_SIZE,
    collect_environment,
    count_parameters,
    gather_objects,
    make_microbatches,
    reduced_loss_to_tensor,
)


VIRTUAL_PIPELINE_SIZE = 2
EXPECTED_LAYERS_PER_CHUNK = 6
EXPECTED_LAYER_MAP = {
    (0, 0): [1, 2, 3, 4, 5, 6],
    (0, 1): [13, 14, 15, 16, 17, 18],
    (1, 0): [7, 8, 9, 10, 11, 12],
    (1, 1): [19, 20, 21, 22, 23, 24],
}

P2P_OVERLAP_CALLS = {
    "send_forward_recv_forward_overlap": 0,
    "send_backward_recv_backward_overlap": 0,
    "send_forward_recv_forward_sync": 0,
    "send_backward_recv_backward_sync": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, choices=(1,))
    parser.add_argument("--pipeline-parallel-size", type=int, default=2, choices=(2,))
    parser.add_argument("--virtual-pipeline-size", type=int, default=VIRTUAL_PIPELINE_SIZE)
    parser.add_argument("--num-microbatches", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=GLOBAL_BATCH_SIZE)
    parser.add_argument("--smoke-iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=100)
    parser.add_argument("--profile-mode", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
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
        "virtual_pipeline_size",
    ):
        if getattr(args, field) < 1:
            parser.error(f"{field.replace('_', ' ')} must be positive")
    if args.num_microbatches * args.micro_batch_size != args.global_batch_size:
        parser.error("num_microbatches * micro_batch_size must equal global_batch_size")
    if args.virtual_pipeline_size != VIRTUAL_PIPELINE_SIZE:
        parser.error("Phase 8.3 uses virtual_pipeline_model_parallel_size=2")
    return args


def initialize_distributed(
    seed: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    virtual_pipeline_size: int,
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
        virtual_pipeline_model_parallel_size=virtual_pipeline_size,
    )
    actual_tp = parallel_state.get_tensor_model_parallel_world_size()
    actual_pp = parallel_state.get_pipeline_model_parallel_world_size()
    actual_dp = parallel_state.get_data_parallel_world_size()
    actual_vpp = parallel_state.get_virtual_pipeline_model_parallel_world_size()
    if actual_tp != tensor_parallel_size or actual_pp != pipeline_parallel_size:
        raise RuntimeError(
            "process-group parallelism does not match requested sizes: "
            f"TP {actual_tp}!={tensor_parallel_size}, PP {actual_pp}!={pipeline_parallel_size}"
        )
    if actual_dp != 1:
        raise RuntimeError(f"Phase 8.3 requires DP=1, got {actual_dp}")
    if actual_vpp != virtual_pipeline_size:
        raise RuntimeError(
            f"virtual pipeline size {actual_vpp} != requested {virtual_pipeline_size}"
        )
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    return local_rank


def install_p2p_overlap_probe() -> None:
    original_forward = P2PCommunicator.send_forward_recv_forward
    original_backward = P2PCommunicator.send_backward_recv_backward

    def wrapped_forward(self, *args, **kwargs):
        overlap = bool(kwargs.get("overlap_p2p_comm", False))
        if len(args) >= 4:
            overlap = bool(args[3])
        key = (
            "send_forward_recv_forward_overlap" if overlap else "send_forward_recv_forward_sync"
        )
        P2P_OVERLAP_CALLS[key] += 1
        with torch.cuda.nvtx.range(
            "pp_async_send_recv_forward" if overlap else "pp_sync_send_recv_forward"
        ):
            return original_forward(self, *args, **kwargs)

    def wrapped_backward(self, *args, **kwargs):
        overlap = bool(kwargs.get("overlap_p2p_comm", False))
        if len(args) >= 4:
            overlap = bool(args[3])
        key = (
            "send_backward_recv_backward_overlap"
            if overlap
            else "send_backward_recv_backward_sync"
        )
        P2P_OVERLAP_CALLS[key] += 1
        with torch.cuda.nvtx.range(
            "pp_async_send_recv_backward" if overlap else "pp_sync_send_recv_backward"
        ):
            return original_backward(self, *args, **kwargs)

    P2PCommunicator.send_forward_recv_forward = wrapped_forward
    P2PCommunicator.send_backward_recv_backward = wrapped_backward


def forward_step_func(data_iterator: Any, model: torch.nn.Module, *args: Any, **kwargs: Any):
    batch = next(data_iterator)
    raw = unwrap_model(model)
    vp_stage = int(getattr(raw, "vp_stage", 0) or 0)
    with torch.cuda.nvtx.range(f"pp_chunk{vp_stage}_forward"):
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


def named_trainable_parameters_chunks(
    chunks: list[DistributedDataParallel],
) -> list[tuple[str, torch.nn.Parameter]]:
    named: list[tuple[str, torch.nn.Parameter]] = []
    for index, chunk in enumerate(chunks):
        for name, parameter in unwrap_model(chunk).named_parameters():
            if parameter.requires_grad:
                named.append((f"chunk{index}.{name}", parameter))
    return named


def build_chunked_bundle(
    chunks: list[torch.nn.Module],
    learning_rate: float,
) -> tuple[DDPOptimizerBundle, list[DistributedDataParallel]]:
    wrapped: list[DistributedDataParallel] = []
    ddp_config = None
    for chunk in chunks:
        ddp_chunk, ddp_config = wrap_with_megatron_ddp(
            chunk,
            use_initialization_stream=False,
        )
        wrapped.append(ddp_chunk)
    parameters: list[torch.nn.Parameter] = []
    for chunk in wrapped:
        parameters.extend(list(chunk.parameters()))
    base_optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
        fused=False,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        decoupled_weight_decay=True,
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
        clip_grad=0.0,
        log_num_zeros_in_grad=False,
        use_distributed_optimizer=False,
        optimizer_cuda_graph=False,
    )
    optimizer = FP32Optimizer(
        base_optimizer,
        optimizer_config,
        init_state_fn=lambda *_args, **_kwargs: None,
    )
    optimizer.grad_stats_parallel_group = parallel_state.get_model_parallel_group()
    optimizer.tp_group = parallel_state.get_tensor_model_parallel_group()
    optimizer.expert_tp_group = optimizer.tp_group
    bundle = DDPOptimizerBundle(
        model=wrapped[0],
        optimizer=optimizer,
        base_optimizer=base_optimizer,
        ddp_config=ddp_config,
        optimizer_config=optimizer_config,
    )
    return bundle, wrapped


def zero_chunk_gradients(bundle: DDPOptimizerBundle, chunks: list[DistributedDataParallel]) -> None:
    for chunk in chunks:
        chunk.zero_grad_buffer()
    bundle.optimizer.zero_grad(set_to_none=True)


def main_grad_pointers_chunks(
    chunks: list[DistributedDataParallel],
) -> dict[str, int]:
    pointers: dict[str, int] = {}
    for name, parameter in named_trainable_parameters_chunks(chunks):
        if not hasattr(parameter, "main_grad"):
            raise RuntimeError(f"MCore DDP did not assign main_grad to {name}")
        pointers[name] = parameter.main_grad.data_ptr()
    return pointers


def assert_chunk_main_grad_pointers(
    chunks: list[DistributedDataParallel],
    expected: dict[str, int],
) -> None:
    actual = main_grad_pointers_chunks(chunks)
    if actual != expected:
        changed = sorted(
            name
            for name in expected.keys() | actual.keys()
            if expected.get(name) != actual.get(name)
        )
        raise RuntimeError(f"main_grad addresses changed: {changed[:10]}")


def assert_chunk_zeroed(bundle: DDPOptimizerBundle, chunks: list[DistributedDataParallel]) -> None:
    for name, parameter in named_trainable_parameters_chunks(chunks):
        if parameter.grad is not None:
            raise RuntimeError(f"param.grad was not cleared for {name}")
        if torch.count_nonzero(parameter.main_grad).item() != 0:
            raise RuntimeError(f"main_grad was not zeroed for {name}")
        if getattr(parameter, "grad_added_to_main_grad", None) is not False:
            raise RuntimeError(f"DDP accumulation flag was not reset for {name}")


def assert_chunk_optimizer_consumed(chunks: list[DistributedDataParallel]) -> None:
    for name, parameter in named_trainable_parameters_chunks(chunks):
        if parameter.grad is None:
            raise RuntimeError(f"Optimizer did not receive a gradient for {name}")
        if parameter.grad.data_ptr() != parameter.main_grad.data_ptr():
            raise RuntimeError(f"Optimizer gradient is not main_grad for {name}")


def main_grads_finite_chunks(chunks: list[DistributedDataParallel]) -> bool:
    return all(
        bool(torch.isfinite(parameter.main_grad).all().item())
        for _, parameter in named_trainable_parameters_chunks(chunks)
    )


def parameters_changed_chunks(
    before: dict[str, torch.Tensor],
    chunks: list[DistributedDataParallel],
) -> bool:
    current = dict(named_trainable_parameters_chunks(chunks))
    return any(
        not torch.equal(value, current[name].detach().cpu())
        for name, value in before.items()
    )


def pipeline_train_step(
    bundle: DDPOptimizerBundle,
    chunks: list[DistributedDataParallel],
    model_args: argparse.Namespace,
    args: argparse.Namespace,
    device: torch.device,
    step_index: int,
) -> float:
    zero_chunk_gradients(bundle, chunks)
    iterators = []
    for _chunk_index in range(args.virtual_pipeline_size):
        iterators.append(
            iter(
                make_microbatches(
                    model_args,
                    step_index,
                    args.num_microbatches,
                    args.micro_batch_size,
                    device,
                )
            )
        )
    forward_backward = get_forward_backward_func()
    with torch.cuda.nvtx.range(f"train_step_{step_index:03d}"):
        with torch.cuda.nvtx.range("pipeline_forward_backward"):
            with contextlib.ExitStack() as stack:
                for chunk in chunks:
                    stack.enter_context(chunk.no_sync())
                losses_reduced = forward_backward(
                    forward_step_func=forward_step_func,
                    data_iterator=iterators,
                    model=chunks,
                    num_microbatches=args.num_microbatches,
                    seq_length=model_args.sequence_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=model_args.sequence_length,
                    forward_only=False,
                )
        with torch.cuda.nvtx.range("finalize_model_grads"):
            finalize_model_grads(chunks)
        with torch.cuda.nvtx.range("optimizer_step"):
            update_successful, _, _ = bundle.optimizer.step()
            if not update_successful:
                raise RuntimeError("FP32Optimizer unexpectedly rejected an update")
    loss_tensor = reduced_loss_to_tensor(losses_reduced, device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return float(loss_tensor.item())


def chunk_partition_report(
    chunk: torch.nn.Module,
    requested_pp: int,
    requested_layers: int,
    virtual_pipeline_size: int,
) -> dict[str, Any]:
    raw = unwrap_model(chunk)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    vp_stage = int(raw.vp_stage)
    layers_built = len(raw.decoder.layers)
    expected_layers = requested_layers // requested_pp // virtual_pipeline_size
    offset = get_transformer_layer_offset(raw.config, vp_stage=vp_stage)
    first = bool(
        parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
    )
    last = bool(
        parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    )
    if layers_built != expected_layers:
        raise RuntimeError(
            f"PP{pp_rank} vp{vp_stage} built {layers_built} layers, expected {expected_layers}"
        )
    if get_num_layers_to_build(raw.config, vp_stage=vp_stage) != expected_layers:
        raise RuntimeError("get_num_layers_to_build disagrees with decoder.layers")
    if first != bool(raw.pre_process) or last != bool(raw.post_process):
        raise RuntimeError("GPTModel pre_process/post_process do not match virtual stage")
    if is_vp_first_stage(vp_stage, virtual_pipeline_size) != (vp_stage == 0):
        raise RuntimeError("vp first-stage helper mismatch")
    if is_vp_last_stage(vp_stage, virtual_pipeline_size) != (
        vp_stage == virtual_pipeline_size - 1
    ):
        raise RuntimeError("vp last-stage helper mismatch")
    has_embedding = hasattr(raw, "embedding")
    has_output = hasattr(raw, "output_layer")
    if has_embedding != first:
        raise RuntimeError(f"embedding ownership mismatch on PP{pp_rank} vp{vp_stage}")
    if has_output != last:
        raise RuntimeError(f"output/loss ownership mismatch on PP{pp_rank} vp{vp_stage}")
    global_layers = [offset + index + 1 for index in range(layers_built)]
    expected_global = EXPECTED_LAYER_MAP[(pp_rank, vp_stage)]
    if global_layers != expected_global:
        raise RuntimeError(
            f"PP{pp_rank} vp{vp_stage} layers {global_layers} != expected {expected_global}"
        )
    local_layer_numbers = [
        int(getattr(layer, "layer_number", index + 1))
        for index, layer in enumerate(raw.decoder.layers)
    ]
    return {
        "rank": dist.get_rank(),
        "pipeline_rank": pp_rank,
        "pipeline_world_size": pp_size,
        "virtual_pipeline_stage": vp_stage,
        "virtual_pipeline_size": virtual_pipeline_size,
        "is_pipeline_first_stage": first,
        "is_pipeline_last_stage": last,
        "owns_embedding": has_embedding,
        "owns_output_layer": has_output,
        "owns_loss": last,
        "pre_process": bool(raw.pre_process),
        "post_process": bool(raw.post_process),
        "layers_built": layers_built,
        "expected_layers": expected_layers,
        "layer_offset": offset,
        "global_layer_numbers": global_layers,
        "local_layer_numbers": local_layer_numbers,
        "share_embeddings_and_output_weights": bool(raw.share_embeddings_and_output_weights),
        "parameter_count": sum(parameter.numel() for parameter in raw.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in raw.parameters() if parameter.requires_grad
        ),
        "embedding_parameter_count": count_parameters("embedding", chunk),
        "decoder_parameter_count": count_parameters("decoder", chunk),
        "output_parameter_count": count_parameters("output_layer", chunk),
        "pipeline_dtype": str(raw.config.pipeline_dtype),
        "overlap_p2p_comm": bool(raw.config.overlap_p2p_comm),
        "overlap_p2p_comm_warmup_flush": bool(raw.config.overlap_p2p_comm_warmup_flush),
        "batch_p2p_comm": bool(raw.config.batch_p2p_comm),
        "cuda_graph_impl": raw.config.cuda_graph_impl,
        "bias_dropout_fusion": bool(raw.config.bias_dropout_fusion),
        "bias_activation_fusion": bool(raw.config.bias_activation_fusion),
        "sequence_parallel": bool(raw.config.sequence_parallel),
        "tensor_model_parallel_size": raw.config.tensor_model_parallel_size,
        "pipeline_model_parallel_size": raw.config.pipeline_model_parallel_size,
        "virtual_pipeline_model_parallel_size": raw.config.virtual_pipeline_model_parallel_size,
        "microbatch_group_size_per_vp_stage": raw.config.microbatch_group_size_per_vp_stage,
    }


def theoretical_bubble_fraction(
    pipeline_parallel_size: int,
    num_microbatches: int,
    virtual_pipeline_size: int,
) -> dict[str, float]:
    fill_drain = (pipeline_parallel_size - 1) / (
        num_microbatches * virtual_pipeline_size + pipeline_parallel_size - 1
    )
    one_f_one_b = (pipeline_parallel_size - 1) / (num_microbatches * virtual_pipeline_size)
    return {
        "fill_drain_fraction": fill_drain,
        "one_f_one_b_warmup_over_steady_state": one_f_one_b,
        "pipeline_parallel_size": float(pipeline_parallel_size),
        "num_microbatches": float(num_microbatches),
        "virtual_pipeline_size": float(virtual_pipeline_size),
        "formula": "(PP-1)/(M*VPP+PP-1)",
    }


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_distributed(
        model_args.seed,
        args.tensor_parallel_size,
        args.pipeline_parallel_size,
        args.virtual_pipeline_size,
    )
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    install_p2p_overlap_probe()
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("Transformer Engine must remain enabled")
        if os.environ.get("NCCL_P2P_DISABLE") == "1":
            raise RuntimeError("NCCL P2P is disabled; refuse to continue without P2P")
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "8":
            raise RuntimeError("CUDA_DEVICE_MAX_CONNECTIONS must be 8")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required")
        if args.profile_mode:
            configure_nvtx_profiling(True)

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        raw_chunks: list[torch.nn.Module] = []
        for vp_stage in range(args.virtual_pipeline_size):
            chunk = build_model(
                model_args,
                attention_implementation=TE_FUSED_ATTENTION,
                attention_dropout=0.1,
                hidden_dropout=0.1,
                bias_dropout_fusion=True,
                cuda_graph_impl="none",
                tensor_model_parallel_size=1,
                sequence_parallel=False,
                use_te_layernorm=False,
                use_te_linear=False,
                pipeline_model_parallel_size=2,
                pipeline_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
                overlap_p2p_comm_warmup_flush=False,
                virtual_pipeline_model_parallel_size=args.virtual_pipeline_size,
                microbatch_group_size_per_vp_stage=2,
                vp_stage=vp_stage,
                share_embeddings_and_output_weights=False,
            )
            config = chunk.config
            if config.cuda_graph_impl != "none":
                raise RuntimeError("CUDA Graph must remain disabled")
            if bool(config.sequence_parallel):
                raise RuntimeError("Sequence parallel must stay off")
            if config.tensor_model_parallel_size != 1:
                raise RuntimeError("Phase 8.3 is TP=1")
            if bool(config.bias_activation_fusion):
                raise RuntimeError("bias_gelu_fusion must stay False")
            if not bool(config.overlap_p2p_comm):
                raise RuntimeError("overlap_p2p_comm must be True")
            if bool(config.batch_p2p_comm):
                raise RuntimeError("batch_p2p_comm must stay False")
            if bool(config.overlap_p2p_comm_warmup_flush):
                raise RuntimeError("overlap_p2p_comm_warmup_flush must stay False")
            if config.pipeline_dtype != torch.float32:
                raise RuntimeError("pipeline_dtype must stay float32")
            if config.virtual_pipeline_model_parallel_size != args.virtual_pipeline_size:
                raise RuntimeError("virtual_pipeline_model_parallel_size mismatch")
            if int(chunk.vp_stage) != vp_stage:
                raise RuntimeError("GPTModel.vp_stage mismatch")
            raw_chunks.append(chunk)
            print(
                f"PHASE83_RANK{rank}_CHUNK_BUILT pp={pp_rank} vp={vp_stage} "
                f"layers={len(chunk.decoder.layers)}",
                flush=True,
            )

        schedule = get_forward_backward_func()
        if schedule.__name__ != "forward_backward_pipelining_with_interleaving":
            raise RuntimeError(
                f"Unexpected pipeline schedule {schedule.__name__}, "
                "expected forward_backward_pipelining_with_interleaving"
            )

        bundle, chunks = build_chunked_bundle(raw_chunks, model_args.learning_rate)
        no_sync_funcs = [chunk.no_sync for chunk in chunks]
        for chunk in chunks:
            unwrap_model(chunk).config.no_sync_func = no_sync_funcs
        pointers = main_grad_pointers_chunks(chunks)
        partitions = [
            chunk_partition_report(
                chunk,
                args.pipeline_parallel_size,
                model_args.num_layers,
                args.virtual_pipeline_size,
            )
            for chunk in chunks
        ]
        used_chunks = [part["layers_built"] == EXPECTED_LAYERS_PER_CHUNK for part in partitions]
        if not all(used_chunks) or len(partitions) != args.virtual_pipeline_size:
            raise RuntimeError("both virtual pipeline chunks were not built/used")
        for part in partitions:
            print(
                f"PHASE83_RANK{rank}_PARTITION vp={part['virtual_pipeline_stage']} "
                f"layers={part['global_layer_numbers']} embedding={part['owns_embedding']} "
                f"output={part['owns_output_layer']}",
                flush=True,
            )
        rank_partitions = gather_objects(partitions)
        if rank == 0:
            print("PHASE83_PARTITIONS_GATHERED", flush=True)

        tracked_name, tracked_parameter = next(iter(named_trainable_parameters_chunks(chunks)))
        parameters_before = {tracked_name: tracked_parameter.detach().cpu().clone()}
        smoke_losses: list[float] = []
        lifecycle_checks: list[dict[str, Any]] = []
        for smoke_index in range(args.smoke_iterations):
            zero_chunk_gradients(bundle, chunks)
            torch.cuda.synchronize(device)
            assert_chunk_zeroed(bundle, chunks)
            assert_chunk_main_grad_pointers(chunks, pointers)
            loss = pipeline_train_step(bundle, chunks, model_args, args, device, smoke_index)
            torch.cuda.synchronize(device)
            print(
                f"PHASE83_RANK{rank}_SMOKE step={smoke_index} loss={loss} "
                f"async_fwd={P2P_OVERLAP_CALLS['send_forward_recv_forward_overlap']} "
                f"async_bwd={P2P_OVERLAP_CALLS['send_backward_recv_backward_overlap']}",
                flush=True,
            )
            gradients_finite = main_grads_finite_chunks(chunks)
            assert_chunk_optimizer_consumed(chunks)
            assert_chunk_main_grad_pointers(chunks, pointers)
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
                    "async_p2p_forward_calls": P2P_OVERLAP_CALLS[
                        "send_forward_recv_forward_overlap"
                    ],
                    "async_p2p_backward_calls": P2P_OVERLAP_CALLS[
                        "send_backward_recv_backward_overlap"
                    ],
                }
            )
        if P2P_OVERLAP_CALLS["send_forward_recv_forward_overlap"] < 1:
            raise RuntimeError("No async send_forward_recv_forward calls were issued")
        if P2P_OVERLAP_CALLS["send_backward_recv_backward_overlap"] < 1:
            raise RuntimeError("No async send_backward_recv_backward calls were issued")
        if not parameters_changed_chunks(parameters_before, chunks):
            raise RuntimeError("No model parameter changed during smoke steps")
        fused = fused_backend_status()
        if args.smoke_only:
            overlap_by_rank = gather_objects({"rank": rank, "calls": dict(P2P_OVERLAP_CALLS)})
            if rank == 0:
                payload = {
                    "status": "success",
                    "run_label": args.run_label,
                    "variant": "B",
                    "run_mode": "smoke_only",
                    "schedule_name": schedule.__name__,
                    "correctness": {
                        "forward": True,
                        "backward": True,
                        "main_grad": True,
                        "optimizer": True,
                        "finite_loss": all(math.isfinite(loss) for loss in smoke_losses),
                        "no_nan_inf": all(math.isfinite(loss) for loss in smoke_losses),
                        "no_deadlock": True,
                        "parameters_updated": True,
                        "interleaved_schedule": True,
                        "async_p2p_issued": True,
                        "smoke_losses": smoke_losses,
                        "lifecycle_checks": lifecycle_checks,
                    },
                    "partitioning": rank_partitions,
                    "async_p2p_calls_by_rank": overlap_by_rank,
                }
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print("PHASE83_SMOKE_OK", flush=True)
            return

        warmup_losses: list[float] = []
        for warmup_index in range(args.warmup_iterations):
            warmup_losses.append(
                pipeline_train_step(
                    bundle,
                    chunks,
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
                list(range(max(args.pipeline_parallel_size, 1))),
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
                            pipeline_train_step(
                                bundle,
                                chunks,
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

        overlap_by_rank = gather_objects({"rank": rank, "calls": dict(P2P_OVERLAP_CALLS)})
        rank_times = gather_objects(
            {
                "rank": rank,
                "pipeline_rank": parallel_state.get_pipeline_model_parallel_rank(),
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
        gpu_count = max(args.pipeline_parallel_size, 1)
        aggregate_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1e12
        mfu_percent = aggregate_tflops / (gpu_count * A40_DENSE_BF16_PEAK_TFLOPS) * 100.0
        payload = {
            "status": "success",
            "run_label": args.run_label,
            "variant": "B",
            "mechanism": "interleaved 1F1B + VPP + P2P overlap",
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "iteration_mode": "FAST ITERATION MODE"
            if args.measured_iterations <= 20
            else "formal 20+100",
            "schedule_name": schedule.__name__,
            "schedule_kind": "interleaved_1F1B",
            "parallelism": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "virtual_pipeline_size": args.virtual_pipeline_size,
                "data_parallel_size": 1,
                "world_size": gpu_count,
            },
            "batch": {
                "global_batch_size": args.global_batch_size,
                "num_microbatches": args.num_microbatches,
                "micro_batch_size": args.micro_batch_size,
                "sequence_length": model_args.sequence_length,
                "tokens_per_step": tokens_per_step,
                "microbatch_group_size_per_vp_stage": 2,
            },
            "model": {
                "num_layers": model_args.num_layers,
                "layers_per_chunk": EXPECTED_LAYERS_PER_CHUNK,
                "hidden_size": model_args.hidden_size,
                "ffn_hidden_size": model_args.ffn_hidden_size,
                "num_attention_heads": model_args.num_attention_heads,
                "vocab_size": model_args.vocab_size,
                "precision": "bf16-autocast",
                "pipeline_dtype": "torch.float32",
                "fused_attention": True,
                "bias_dropout_fusion": True,
                "bias_gelu_fusion": False,
                "cuda_graph": False,
                "overlap_p2p_comm": True,
                "overlap_p2p_comm_warmup_flush": False,
                "batch_p2p_comm": False,
            },
            "fused_backend_status": fused,
            "partitioning": rank_partitions,
            "both_chunks_used": True,
            "async_p2p_calls_by_rank": overlap_by_rank,
            "correctness": {
                "forward": True,
                "backward": True,
                "main_grad": True,
                "optimizer": True,
                "finite_loss": all(math.isfinite(loss) for loss in smoke_losses + measured_losses),
                "no_nan_inf": all(math.isfinite(loss) for loss in smoke_losses + measured_losses),
                "no_deadlock": True,
                "parameters_updated": True,
                "interleaved_schedule": True,
                "async_p2p_issued": True,
                "smoke_losses": smoke_losses,
                "lifecycle_checks": lifecycle_checks,
            },
            "theoretical_bubble": theoretical_bubble_fraction(
                args.pipeline_parallel_size,
                args.num_microbatches,
                args.virtual_pipeline_size,
            ),
            "tokens_per_second": tokens_per_second,
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": statistics.median(global_step_times_ms),
            "p95_step_time_ms": statistics.quantiles(global_step_times_ms, n=20)[-1]
            if len(global_step_times_ms) >= 20
            else max(global_step_times_ms),
            "mfu_percent": mfu_percent,
            "aggregate_tflops": aggregate_tflops,
            "a40_peak_tflops": A40_DENSE_BF16_PEAK_TFLOPS,
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
                    "allocated": item["peak_allocated_memory_mib"],
                    "reserved": item["peak_reserved_memory_mib"],
                }
                for item in rank_times
            },
            "gpu_monitoring": gpu_monitoring,
            "lifecycle": lifecycle_metadata(bundle),
            "environment": collect_environment(gpu_count),
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
                    "schedule_name": schedule.__name__,
                    "async_p2p_forward": P2P_OVERLAP_CALLS["send_forward_recv_forward_overlap"],
                    "async_p2p_backward": P2P_OVERLAP_CALLS["send_backward_recv_backward_overlap"],
                },
                indent=2,
            )
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
