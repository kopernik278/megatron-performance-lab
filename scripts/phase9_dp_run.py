#!/usr/bin/env python3
"""Run one Phase 9.1 DP=1 or DP=2 MCore DDP training variant.

Uses pinned Megatron-LM 09fde85 DistributedDataParallelConfig.overlap_grad_reduce
(CLI --overlap-grad-reduce). Distributed optimizer stays off. Bucket size is the
MCore default (not tuned). TP=1, PP=1.
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
    build_ddp_optimizer_bundle,
    collect_bucket_metadata,
    finalize_gradients,
    instrument_grad_sync_nvtx,
    lifecycle_metadata,
    main_grad_pointers,
    named_trainable_parameters,
    unwrap_model,
    zero_gradients,
)
from phase7_tp_run import MultiGpuNvidiaSmiSampler


MICRO_BATCH_SIZE = 8
SEQUENCE_LENGTH = 2048
ACCEPTED_DP1_TOKENS_PER_SECOND = 15801.942
ACCEPTED_DP1_SOURCE = "Phase 5.2 formal B (bias_dropout_fusion=True, MB=8, DP=1)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parallel-size", type=int, required=True, choices=(1, 2))
    parser.add_argument(
        "--overlap-grad-reduce",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Megatron DistributedDataParallelConfig.overlap_grad_reduce / --overlap-grad-reduce",
    )
    parser.add_argument("--smoke-iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=100)
    parser.add_argument("--profile-mode", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    for field in ("smoke_iterations", "warmup_iterations", "measured_iterations"):
        if getattr(args, field) < 1:
            parser.error(f"{field.replace('_', ' ')} must be positive")
    if args.data_parallel_size == 1 and args.overlap_grad_reduce:
        parser.error("DP=1 overlap is an identity; Phase 9.1 overlap is DP=2 only")
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


def initialize_distributed(seed: int, data_parallel_size: int) -> int:
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
    if world != data_parallel_size:
        raise RuntimeError(
            f"world size {world} != requested DP={data_parallel_size}"
        )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    actual_tp = parallel_state.get_tensor_model_parallel_world_size()
    actual_pp = parallel_state.get_pipeline_model_parallel_world_size()
    actual_dp = parallel_state.get_data_parallel_world_size()
    if actual_tp != 1 or actual_pp != 1:
        raise RuntimeError(f"Phase 9.1 requires TP=1 PP=1, got TP={actual_tp} PP={actual_pp}")
    if actual_dp != data_parallel_size:
        raise RuntimeError(f"Phase 9.1 requires DP={data_parallel_size}, got {actual_dp}")
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    return local_rank


def dp_synthetic_batch(
    model_args: argparse.Namespace,
    micro_batch_size: int,
    device: torch.device,
    dp_rank: int,
    step_index: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        model_args.seed + 1 + 1_000_003 * step_index + 17 * dp_rank
    )
    seq = model_args.sequence_length
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
            loss.backward()
        with torch.cuda.nvtx.range("finalize_model_grads"):
            finalize_gradients(bundle.model)
        with torch.cuda.nvtx.range("optimizer_step"):
            update_successful, _, _ = bundle.optimizer.step()
            if not update_successful:
                raise RuntimeError("FP32Optimizer unexpectedly rejected an update")
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


def collect_environment(gpu_count: int) -> dict[str, Any]:
    try:
        te_version = importlib.metadata.version("transformer-engine")
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
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if query and not query.startswith("unavailable"):
        parsed = []
        for line in query.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 5:
                continue
            parsed.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "pci_bus_id": fields[2],
                    "driver": fields[3],
                    "memory_total_mib": float(fields[4]),
                }
            )
        if parsed:
            gpus = parsed[:gpu_count]
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


def main() -> None:
    args = parse_args()
    model_args = baseline_model_args()
    local_rank = initialize_distributed(model_args.seed, args.data_parallel_size)
    rank = dist.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    device = torch.device(f"cuda:{local_rank}")
    try:
        if os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("Transformer Engine must remain enabled")
        if os.environ.get("NCCL_P2P_DISABLE") == "1":
            raise RuntimeError("NCCL_P2P_DISABLE must not be set")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required")
        if args.data_parallel_size > 1:
            max_connections = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")
            if max_connections == "1":
                raise RuntimeError(
                    "DP gradient overlap needs CUDA_DEVICE_MAX_CONNECTIONS>1 "
                    "so NCCL can run on a side stream"
                )

        model = build_model(
            model_args,
            attention_implementation=TE_FUSED_ATTENTION,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            bias_dropout_fusion=True,
            cuda_graph_impl="none",
        )
        if model.config.cuda_graph_impl != "none":
            raise RuntimeError("CUDA Graph must remain disabled")
        if bool(model.config.bias_dropout_fusion) is not True:
            raise RuntimeError("bias_dropout_fusion must stay True")
        if bool(model.config.bias_activation_fusion):
            raise RuntimeError("bias_gelu / bias_activation fusion must stay False")
        bundle = build_ddp_optimizer_bundle(
            model,
            model_args.learning_rate,
            overlap_grad_reduce=args.overlap_grad_reduce,
            disable_bucketing=not args.overlap_grad_reduce,
        )
        if bundle.ddp_config.overlap_grad_reduce != args.overlap_grad_reduce:
            raise RuntimeError("DDP did not honor overlap_grad_reduce")
        if bundle.ddp_config.use_distributed_optimizer:
            raise RuntimeError("Distributed optimizer must stay off")
        if bundle.ddp_config.overlap_param_gather:
            raise RuntimeError("overlap_param_gather must stay off")
        instrument_grad_sync_nvtx(bundle.model)
        pointers = main_grad_pointers(bundle.model)
        buckets = collect_bucket_metadata(bundle.model)
        lifecycle = lifecycle_metadata(bundle)

        smoke_batch = dp_synthetic_batch(
            model_args, MICRO_BATCH_SIZE, device, dp_rank, step_index=0
        )
        token_fingerprint = (
            int(smoke_batch["tokens"][0, 0].item()),
            int(smoke_batch["tokens"][0, 1].item()),
            int(smoke_batch["tokens"].sum().item()),
        )
        rank_tokens = gather_objects(
            {"rank": rank, "dp_rank": dp_rank, "token_fingerprint": token_fingerprint}
        )
        if args.data_parallel_size > 1:
            fingerprints = [item["token_fingerprint"] for item in rank_tokens]
            if len(set(fingerprints)) != args.data_parallel_size:
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
                model_args, MICRO_BATCH_SIZE, device, dp_rank, step_index=smoke_index
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
            if args.data_parallel_size > 1:
                checksums = [item["checksum"] for item in rank_grads]
                if max(checksums) - min(checksums) > 1.0e-3:
                    raise RuntimeError(
                        f"main_grad not synchronized after finalize: {rank_grads}"
                    )
            update_successful, _, _ = bundle.optimizer.step()
            torch.cuda.synchronize(device)
            assert_optimizer_consumed_main_grad(bundle)
            assert_main_grad_pointers(bundle.model, pointers)
            param_checksum = parameter_checksum(bundle)
            rank_params = gather_objects({"rank": rank, "checksum": param_checksum})
            if args.data_parallel_size > 1:
                param_checksums = [item["checksum"] for item in rank_params]
                if max(param_checksums) - min(param_checksums) > 1.0e-3:
                    raise RuntimeError(
                        f"parameters diverged after optimizer step: {rank_params}"
                    )
            loss_value = float(loss.detach().cpu())
            if not update_successful or not gradients_finite or not math.isfinite(loss_value):
                raise RuntimeError("Smoke-step lifecycle failed")
            smoke_losses.append(loss_value)
            lifecycle_checks.append(
                {
                    "step": smoke_index,
                    "loss_finite": math.isfinite(loss_value),
                    "main_grads_finite": gradients_finite,
                    "gradients_synchronized": True,
                    "parameters_identical_across_ranks": True,
                    "optimizer_consumed_main_grad": True,
                    "main_grad_addresses_stable": True,
                    "deadlock": False,
                }
            )
        if torch.equal(parameters_before, tracked_parameter.detach().cpu()):
            raise RuntimeError(f"Parameter {tracked_name} did not change during smoke steps")
        fused = fused_backend_status()

        correctness = {
            "both_ranks_initialized": sorted(item["dp_rank"] for item in rank_tokens)
            == list(range(args.data_parallel_size)),
            "different_data_per_dp_rank": (
                args.data_parallel_size == 1
                or len({item["token_fingerprint"] for item in rank_tokens})
                == args.data_parallel_size
            ),
            "forward": True,
            "backward": True,
            "main_grad": True,
            "optimizer": True,
            "gradients_synchronized": True,
            "parameters_identical_after_optimizer": True,
            "finite_loss": all(math.isfinite(loss) for loss in smoke_losses),
            "no_nan_inf": True,
            "no_deadlock": True,
            "parameters_updated": True,
            "smoke_losses": smoke_losses,
            "lifecycle_checks": lifecycle_checks,
            "rank_token_fingerprints": rank_tokens,
        }

        if args.smoke_only:
            if rank == 0:
                write_rank0_json(
                    args.output_json,
                    {
                        "status": "success",
                        "run_label": args.run_label,
                        "run_mode": "smoke",
                        "overlap_grad_reduce": args.overlap_grad_reduce,
                        "parallelism": {
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "data_parallel_size": args.data_parallel_size,
                            "world_size": args.data_parallel_size,
                        },
                        "correctness": correctness,
                        "buckets": buckets,
                        "ddp_lifecycle": lifecycle,
                        "fused_backend_status": fused,
                    },
                )
            return

        warmup_base = 100
        for warmup_index in range(args.warmup_iterations):
            batch = dp_synthetic_batch(
                model_args,
                MICRO_BATCH_SIZE,
                device,
                dp_rank,
                step_index=warmup_base + warmup_index,
            )
            train_step(bundle, batch, warmup_index)
            torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        sampler = None
        if rank == 0:
            sampler = MultiGpuNvidiaSmiSampler(
                list(range(args.data_parallel_size)),
                args.gpu_sample_interval_ms,
            )
            sampler.start()
        measured_losses: list[float] = []
        local_step_times_ms: list[float] = []
        measured_base = 1_000
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
                    for step_index in range(args.measured_iterations):
                        batch = dp_synthetic_batch(
                            model_args,
                            MICRO_BATCH_SIZE,
                            device,
                            dp_rank,
                            step_index=measured_base + step_index,
                        )
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        measured_losses.append(train_step(bundle, batch, step_index))
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
                "dp_rank": dp_rank,
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
        median_step_time_ms = statistics.median(global_step_times_ms)
        global_batch_size = MICRO_BATCH_SIZE * args.data_parallel_size
        tokens_per_step = global_batch_size * model_args.sequence_length
        tokens_per_second = tokens_per_step / (average_step_time_ms / 1000.0)
        flops_per_iteration = training_flops_per_iteration(
            global_batch_size,
            model_args.sequence_length,
            model_args.num_layers,
            model_args.hidden_size,
            model_args.vocab_size,
        )
        aggregate_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1e12
        mfu_percent = aggregate_tflops / (
            args.data_parallel_size * A40_DENSE_BF16_PEAK_TFLOPS
        ) * 100.0
        per_gpu_flops = training_flops_per_iteration(
            MICRO_BATCH_SIZE,
            model_args.sequence_length,
            model_args.num_layers,
            model_args.hidden_size,
            model_args.vocab_size,
        )
        per_gpu_mfu_percent = (
            (per_gpu_flops / (average_step_time_ms / 1000.0) / 1e12)
            / A40_DENSE_BF16_PEAK_TFLOPS
            * 100.0
        )
        peak_smi = [
            (gpu_monitoring or {}).get(str(index), {}).get("peak_memory_mib")
            for index in range(args.data_parallel_size)
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

        environment = collect_environment(args.data_parallel_size)
        payload = {
            "status": "success",
            "run_label": args.run_label,
            "run_mode": "nsight_profile" if args.profile_mode else "benchmark",
            "iteration_mode": "FAST ITERATION MODE",
            "overlap_grad_reduce": args.overlap_grad_reduce,
            "parallelism": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": args.data_parallel_size,
                "world_size": args.data_parallel_size,
            },
            "batch": {
                "micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
                "gradient_accumulation": 1,
                "global_batch_size": global_batch_size,
                "sequence_length": model_args.sequence_length,
                "tokens_per_step": tokens_per_step,
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
                "distributed_optimizer": False,
            },
            "fused_backend_status": fused,
            "ddp_lifecycle": lifecycle,
            "buckets": buckets,
            "correctness": {
                **correctness,
                "finite_loss": all(
                    math.isfinite(loss)
                    for row in rank_losses
                    for loss in row["losses"]
                )
                and correctness["finite_loss"],
            },
            "tokens_per_second": tokens_per_second,
            "average_step_time_ms": average_step_time_ms,
            "median_step_time_ms": median_step_time_ms,
            "mfu_percent": mfu_percent,
            "per_gpu_mfu_percent": per_gpu_mfu_percent,
            "peak_allocated_memory_mib": [
                item["peak_allocated_memory_mib"] for item in rank_times
            ],
            "smi_peak_memory_mib": peak_smi,
            "mean_gpu_utilization_percent": mean_util,
            "gpu_monitoring": gpu_monitoring,
            "step_times_ms": global_step_times_ms,
            "measured_losses": measured_losses,
            "accepted_dp1_tokens_per_second": ACCEPTED_DP1_TOKENS_PER_SECOND,
            "accepted_dp1_source": ACCEPTED_DP1_SOURCE,
            "environment": environment,
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
        }
        write_rank0_json(args.output_json, payload)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
