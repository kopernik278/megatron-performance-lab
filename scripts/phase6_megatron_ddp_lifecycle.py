#!/usr/bin/env python3
"""Shared MCore DDP gradient lifecycle for Phase 6.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import FP32Optimizer


@dataclass
class DDPOptimizerBundle:
    """Model and optimizer objects that own the Phase 6.3 lifecycle."""

    model: DistributedDataParallel
    optimizer: FP32Optimizer
    base_optimizer: torch.optim.AdamW
    ddp_config: DistributedDataParallelConfig
    optimizer_config: OptimizerConfig


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the model under the MCore DDP wrapper."""

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def named_trainable_parameters(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return each unique trainable model parameter once."""

    return [
        (name, parameter)
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad
    ]


def wrap_with_megatron_ddp(
    model: torch.nn.Module,
    use_initialization_stream: bool = True,
    overlap_grad_reduce: bool = False,
    disable_bucketing: bool | None = None,
    bucket_size: int | None = None,
) -> tuple[DistributedDataParallel, DistributedDataParallelConfig]:
    """Wrap a model with MCore DDP.

    Flag names match pinned Megatron-LM 09fde85
    ``DistributedDataParallelConfig`` / ``--overlap-grad-reduce``.
    ``bucket_size=None`` keeps MCore's default
    ``max(40000000, 1000000 * dp_size)`` when overlap is on; MCore then
    forces ``bucket_size=None`` when overlap is off.
    """

    if disable_bucketing is None:
        disable_bucketing = not overlap_grad_reduce
    if overlap_grad_reduce and disable_bucketing:
        raise RuntimeError(
            "overlap_grad_reduce=True requires disable_bucketing=False so "
            "DDP can issue per-bucket async All-Reduce"
        )
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=False,
        use_distributed_optimizer=False,
        check_for_nan_in_grad=False,
        check_for_large_grads=False,
        average_in_collective=False,
        bucket_size=bucket_size,
    )

    def _wrap() -> DistributedDataParallel:
        return DistributedDataParallel(
            config=model.config,
            ddp_config=ddp_config,
            module=model,
            disable_bucketing=disable_bucketing,
        )

    if not use_initialization_stream:
        return _wrap(), ddp_config

    initialization_stream = torch.cuda.Stream()
    initialization_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(initialization_stream):
        ddp_model = _wrap()
    torch.cuda.current_stream().wait_stream(initialization_stream)
    return ddp_model, ddp_config


def build_fp32_optimizer(
    model: DistributedDataParallel,
    learning_rate: float,
) -> tuple[FP32Optimizer, torch.optim.AdamW, OptimizerConfig]:
    """Wrap the unchanged PyTorch AdamW with MCore's FP32 lifecycle."""

    base_optimizer = torch.optim.AdamW(
        model.parameters(),
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
    return optimizer, base_optimizer, optimizer_config


def build_ddp_optimizer_bundle(
    model: torch.nn.Module,
    learning_rate: float,
    use_initialization_stream: bool = True,
    overlap_grad_reduce: bool = False,
    disable_bucketing: bool | None = None,
    bucket_size: int | None = None,
) -> DDPOptimizerBundle:
    """Build DDP before creating the optimizer, as required by MCore."""

    ddp_model, ddp_config = wrap_with_megatron_ddp(
        model,
        use_initialization_stream=use_initialization_stream,
        overlap_grad_reduce=overlap_grad_reduce,
        disable_bucketing=disable_bucketing,
        bucket_size=bucket_size,
    )
    optimizer, base_optimizer, optimizer_config = build_fp32_optimizer(
        ddp_model,
        learning_rate,
    )
    return DDPOptimizerBundle(
        model=ddp_model,
        optimizer=optimizer,
        base_optimizer=base_optimizer,
        ddp_config=ddp_config,
        optimizer_config=optimizer_config,
    )


def main_grad_pointers(model: torch.nn.Module) -> dict[str, int]:
    """Return persistent main_grad pointers for lifecycle checks."""

    pointers: dict[str, int] = {}
    for name, parameter in named_trainable_parameters(model):
        if not hasattr(parameter, "main_grad"):
            raise RuntimeError(f"MCore DDP did not assign main_grad to {name}")
        main_grad = parameter.main_grad
        if main_grad.shape != parameter.shape:
            raise RuntimeError(
                f"main_grad shape mismatch for {name}: "
                f"{tuple(main_grad.shape)} != {tuple(parameter.shape)}"
            )
        if main_grad.dtype != torch.float32:
            raise RuntimeError(f"main_grad dtype changed for {name}: {main_grad.dtype}")
        pointers[name] = main_grad.data_ptr()
    return pointers


def assert_main_grad_pointers(
    model: torch.nn.Module,
    expected: dict[str, int],
) -> None:
    """Fail if DDP replaced any graph-visible main_grad tensor."""

    actual = main_grad_pointers(model)
    if actual != expected:
        changed = sorted(
            name
            for name in expected.keys() | actual.keys()
            if expected.get(name) != actual.get(name)
        )
        raise RuntimeError(f"main_grad addresses changed: {changed[:10]}")


def zero_gradients(bundle: DDPOptimizerBundle) -> None:
    """Apply Megatron's required DDP-buffer then optimizer zeroing order."""

    bundle.model.zero_grad_buffer()
    bundle.optimizer.zero_grad(set_to_none=True)


def finalize_gradients(model: DistributedDataParallel) -> None:
    """Run standard MCore gradient finalization before optimizer access."""

    finalize_model_grads([model])


def collect_main_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone all DDP-owned gradients to CPU."""

    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in named_trainable_parameters(model):
        if not hasattr(parameter, "main_grad"):
            raise RuntimeError(f"Missing main_grad for {name}")
        gradients[name] = parameter.main_grad.detach().cpu().clone()
    return gradients


def assert_zeroed_lifecycle(bundle: DDPOptimizerBundle) -> None:
    """Verify the beginning-of-step lifecycle cleared both gradient interfaces."""

    for name, parameter in named_trainable_parameters(bundle.model):
        if parameter.grad is not None:
            raise RuntimeError(f"param.grad was not cleared for {name}")
        if torch.count_nonzero(parameter.main_grad).item() != 0:
            raise RuntimeError(f"main_grad was not zeroed for {name}")
        if getattr(parameter, "grad_added_to_main_grad", None) is not False:
            raise RuntimeError(f"DDP accumulation flag was not reset for {name}")


def assert_optimizer_consumed_main_grad(bundle: DDPOptimizerBundle) -> None:
    """Verify FP32Optimizer exposed each DDP buffer to AdamW."""

    for name, parameter in named_trainable_parameters(bundle.model):
        if parameter.grad is None:
            raise RuntimeError(f"Optimizer did not receive a gradient for {name}")
        if parameter.grad.data_ptr() != parameter.main_grad.data_ptr():
            raise RuntimeError(f"Optimizer gradient is not main_grad for {name}")


def optimizer_state_dtypes(base_optimizer: torch.optim.Optimizer) -> list[str]:
    """Return floating optimizer-state dtypes."""

    return sorted(
        {
            str(value.dtype)
            for state in base_optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
    )


def collect_bucket_metadata(model: DistributedDataParallel) -> dict[str, Any]:
    """Describe DDP gradient buckets without touching private buffer layout APIs."""

    buckets: list[dict[str, Any]] = []
    for buffer_index, buffer in enumerate(getattr(model, "buffers", []) or []):
        param_dtype = str(getattr(buffer, "param_dtype", None))
        grad_dtype = str(getattr(buffer, "grad_dtype", None))
        for bucket_index, bucket in enumerate(getattr(buffer, "buckets", []) or []):
            grad_data = getattr(bucket, "grad_data", None)
            padded_numel = int(grad_data.numel()) if grad_data is not None else None
            element_size = int(grad_data.element_size()) if grad_data is not None else None
            params = getattr(bucket, "params", None) or getattr(bucket, "params_list", [])
            buckets.append(
                {
                    "buffer_index": buffer_index,
                    "bucket_index": bucket_index,
                    "param_dtype": param_dtype,
                    "grad_dtype": grad_dtype,
                    "numel_unpadded": int(getattr(bucket, "numel_unpadded", 0)),
                    "padded_numel": padded_numel,
                    "bytes": (
                        None
                        if padded_numel is None or element_size is None
                        else padded_numel * element_size
                    ),
                    "param_count": len(list(params)),
                }
            )
    return {
        "overlap_grad_reduce": bool(model.ddp_config.overlap_grad_reduce),
        "overlap_param_gather": bool(model.ddp_config.overlap_param_gather),
        "use_distributed_optimizer": bool(model.ddp_config.use_distributed_optimizer),
        "configured_bucket_size": model.ddp_config.bucket_size,
        "effective_bucket_size": getattr(model, "bucket_size", None),
        "bucket_count": len(buckets),
        "buckets": buckets,
    }


def instrument_grad_sync_nvtx(model: DistributedDataParallel) -> None:
    """Wrap DDP grad-sync entry points with NVTX for Nsight attribution."""

    original_start = model.start_grad_sync
    original_finish = model.finish_grad_sync

    def start_grad_sync(*args: Any, **kwargs: Any) -> Any:
        with torch.cuda.nvtx.range("dp_start_grad_sync"):
            return original_start(*args, **kwargs)

    def finish_grad_sync(*args: Any, **kwargs: Any) -> Any:
        with torch.cuda.nvtx.range("dp_finish_grad_sync"):
            return original_finish(*args, **kwargs)

    model.start_grad_sync = start_grad_sync  # type: ignore[method-assign]
    model.finish_grad_sync = finish_grad_sync  # type: ignore[method-assign]


def lifecycle_metadata(bundle: DDPOptimizerBundle) -> dict[str, Any]:
    """Describe controls that must remain fixed across the A/B."""

    return {
        "wrapper": type(bundle.model).__name__,
        "main_grad_parameter_count": len(main_grad_pointers(bundle.model)),
        "grad_reduce_in_fp32": bundle.ddp_config.grad_reduce_in_fp32,
        "overlap_grad_reduce": bundle.ddp_config.overlap_grad_reduce,
        "overlap_param_gather": bundle.ddp_config.overlap_param_gather,
        "use_distributed_optimizer": bundle.ddp_config.use_distributed_optimizer,
        "optimizer_wrapper": type(bundle.optimizer).__name__,
        "base_optimizer": type(bundle.base_optimizer).__name__,
        "clip_grad": bundle.optimizer_config.clip_grad,
        "optimizer_cuda_graph": bundle.optimizer_config.optimizer_cuda_graph,
        "optimizer_state_dtypes": optimizer_state_dtypes(bundle.base_optimizer),
    }
