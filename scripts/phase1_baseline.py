#!/usr/bin/env python3
"""Run the Phase 1.2 single-GPU Megatron GPT baseline."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


A40_DENSE_BF16_PEAK_TFLOPS = 149.7
LOCAL_UNFUSED_ATTENTION = "local-unfused"
TE_FUSED_ATTENTION = "te-fused"
ATTENTION_IMPLEMENTATIONS = (LOCAL_UNFUSED_ATTENTION, TE_FUSED_ATTENTION)


class UnsafeMemoryMargin(RuntimeError):
    """Raised when a candidate batch size leaves too little VRAM headroom."""


class NvidiaSmiSampler:
    """Collect utilization and device-memory samples with a single nvidia-smi process."""

    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms
        self.utilization_samples: list[float] = []
        self.memory_samples_mib: list[float] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_samples, daemon=True)
        self._thread.start()

    def _read_samples(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                self.utilization_samples.append(float(fields[0]))
                self.memory_samples_mib.append(float(fields[1]))
            except ValueError:
                continue

    def stop(self) -> dict[str, Any]:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        return {
            "sample_interval_ms": self.interval_ms,
            "sample_count": len(self.utilization_samples),
            "average_gpu_utilization_percent": (
                statistics.fmean(self.utilization_samples) if self.utilization_samples else None
            ),
            "peak_nvidia_smi_memory_mib": (
                max(self.memory_samples_mib) if self.memory_samples_mib else None
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=100)
    parser.add_argument("--micro-batch-candidates", default="4,2,1")
    parser.add_argument("--memory-safety-fraction", type=float, default=0.90)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ffn-hidden-size", type=int, default=4096)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=200)
    parser.add_argument(
        "--attention-implementation",
        choices=ATTENTION_IMPLEMENTATIONS,
        default=LOCAL_UNFUSED_ATTENTION,
    )
    parser.add_argument("--output-json", type=Path, default=Path("results/phase1_baseline.json"))
    return parser.parse_args()


def run_command(command: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {exc}"


def initialize_single_gpu_distributed(seed: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    return local_rank


def bda_nvtx_factory(name: str, factory: Any) -> Any:
    """Wrap a BDA factory with a forward-only NVTX attribution range."""

    def instrumented_factory(training: bool, fused: bool) -> Any:
        bda = factory(training, fused)

        def instrumented_bda(
            x_with_bias: tuple[torch.Tensor, torch.Tensor | None],
            residual: torch.Tensor,
            probability: float,
        ) -> torch.Tensor:
            with torch.cuda.nvtx.range(f"bda::{name}"):
                return bda(x_with_bias, residual, probability)

        return instrumented_bda

    return instrumented_factory


def get_transformer_layer_spec(
    attention_implementation: str,
    instrument_bda: bool = False,
) -> Any:
    layer_spec = get_gpt_layer_local_spec()
    if instrument_bda:
        layer_spec.submodules.self_attn_bda = bda_nvtx_factory(
            "self_attention",
            layer_spec.submodules.self_attn_bda,
        )
        layer_spec.submodules.mlp_bda = bda_nvtx_factory(
            "mlp",
            layer_spec.submodules.mlp_bda,
        )
    if attention_implementation == LOCAL_UNFUSED_ATTENTION:
        return layer_spec
    if attention_implementation == TE_FUSED_ATTENTION:
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        class AutocastTEDotProductAttention(TEDotProductAttention):
            """Adapt FP32 local QKV projections to the active BF16 autocast dtype."""

            def forward(
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                *args: Any,
                **kwargs: Any,
            ) -> torch.Tensor:
                if torch.is_autocast_enabled("cuda"):
                    autocast_dtype = torch.get_autocast_dtype("cuda")
                    query = query.to(autocast_dtype)
                    key = key.to(autocast_dtype)
                    value = value.to(autocast_dtype)
                return super().forward(query, key, value, *args, **kwargs)

        layer_spec.submodules.self_attention.submodules.core_attention = (
            AutocastTEDotProductAttention
        )
        return layer_spec
    raise ValueError(f"Unsupported attention implementation: {attention_implementation}")


def build_model(
    args: argparse.Namespace,
    attention_implementation: str | None = None,
    attention_dropout: float = 0.1,
    hidden_dropout: float = 0.1,
    bias_dropout_fusion: bool = False,
    instrument_bda: bool = False,
) -> GPTModel:
    if attention_implementation is None:
        attention_implementation = getattr(
            args, "attention_implementation", LOCAL_UNFUSED_ATTENTION
        )
    attention_backend = (
        AttnBackend.fused
        if attention_implementation == TE_FUSED_ATTENTION
        else AttnBackend.unfused
    )
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        layernorm_epsilon=1.0e-5,
        add_bias_linear=True,
        gated_linear_unit=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=bias_dropout_fusion,
        masked_softmax_fusion=False,
        cross_entropy_loss_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        attention_backend=attention_backend,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_transformer_layer_spec(
            attention_implementation,
            instrument_bda=instrument_bda,
        ),
        vocab_size=args.vocab_size,
        max_sequence_length=args.sequence_length,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="learned_absolute",
    )
    return model.cuda()


def synthetic_batch(
    args: argparse.Namespace,
    micro_batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    tokens = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(micro_batch_size, args.sequence_length),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    labels = torch.roll(tokens, shifts=-1, dims=1)
    position_ids = torch.arange(args.sequence_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(micro_batch_size, -1)
    loss_mask = torch.ones(
        (micro_batch_size, args.sequence_length),
        dtype=torch.float32,
        device=device,
    )
    attention_mask = torch.tril(
        torch.ones(
            (args.sequence_length, args.sequence_length),
            dtype=torch.bool,
            device=device,
        )
    )
    attention_mask = ~attention_mask.view(1, 1, args.sequence_length, args.sequence_length)
    attention_mask = attention_mask.expand(micro_batch_size, -1, -1, -1)
    return {
        "tokens": tokens,
        "labels": labels,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
    }


def masked_language_model_loss(output_tensor: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    losses = output_tensor.float()
    mask = loss_mask.view(-1).float()
    return torch.sum(losses.view(-1) * mask) / mask.sum()


def train_step(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output_tensor = model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = masked_language_model_loss(output_tensor, batch["loss_mask"])
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def training_flops_per_iteration(
    micro_batch_size: int,
    sequence_length: int,
    num_layers: int,
    hidden_size: int,
    vocab_size: int,
) -> float:
    transformer_flops = 72.0 * micro_batch_size * sequence_length * num_layers * hidden_size**2
    # Megatron counts half of causal attention's S^2 matrix as active work.
    attention_flops = 6.0 * micro_batch_size * num_layers * sequence_length**2 * hidden_size
    logits_flops = 6.0 * micro_batch_size * sequence_length * hidden_size * vocab_size
    return transformer_flops + attention_flops + logits_flops


def collect_environment() -> dict[str, Any]:
    nccl_version = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
    gpu_fields = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).split(",")
    try:
        transformer_engine_version = importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError:
        transformer_engine_version = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": ".".join(str(part) for part in nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
        "cudnn": torch.backends.cudnn.version(),
        "megatron_core": run_command(
            ["python", "-c", "import importlib.metadata as m; print(m.version('megatron-core'))"]
        ),
        "megatron_lm_commit": run_command(["git", "rev-parse", "HEAD"], cwd="/workspace/Megatron-LM"),
        "project_commit": run_command(["git", "rev-parse", "HEAD"], cwd="/workspace/megatron-performance-lab"),
        "gpu": gpu_fields[0].strip() if gpu_fields else "unavailable",
        "driver": gpu_fields[1].strip() if len(gpu_fields) > 1 else "unavailable",
        "gpu_memory_total_mib": float(gpu_fields[2].strip()) if len(gpu_fields) > 2 else None,
        "compute_capability": torch.cuda.get_device_capability(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "transformer_engine_installed": importlib.util.find_spec("transformer_engine") is not None,
        "transformer_engine_version": transformer_engine_version,
        "transformer_engine_disabled": os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1",
        "cuda_graph_enabled": False,
    }


def run_candidate(
    args: argparse.Namespace,
    micro_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    attention_implementation = getattr(
        args, "attention_implementation", LOCAL_UNFUSED_ATTENTION
    )
    model = build_model(args, attention_implementation=attention_implementation)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        foreach=False,
        fused=False,
    )
    batch = synthetic_batch(args, micro_batch_size, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    torch.cuda.reset_peak_memory_stats(device)
    warmup_losses: list[float] = []
    for _ in range(args.warmup_iterations):
        warmup_losses.append(train_step(model, optimizer, batch))
        torch.cuda.synchronize(device)

    total_memory = torch.cuda.get_device_properties(device).total_memory
    warmup_peak_reserved = torch.cuda.max_memory_reserved(device)
    warmup_memory_fraction = warmup_peak_reserved / total_memory
    if warmup_memory_fraction > args.memory_safety_fraction:
        raise UnsafeMemoryMargin(
            f"micro-batch {micro_batch_size} reserved {warmup_memory_fraction:.1%} of VRAM "
            f"during warmup, above the {args.memory_safety_fraction:.1%} safety limit"
        )

    torch.cuda.reset_peak_memory_stats(device)
    sampler = NvidiaSmiSampler(args.gpu_sample_interval_ms)
    measured_losses: list[float] = []
    step_times_ms: list[float] = []
    sampler.start()
    try:
        for _ in range(args.measured_iterations):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            measured_losses.append(train_step(model, optimizer, batch))
            torch.cuda.synchronize(device)
            step_times_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        utilization = sampler.stop()

    average_step_time_ms = statistics.fmean(step_times_ms)
    median_step_time_ms = statistics.median(step_times_ms)
    tokens_per_iteration = micro_batch_size * args.sequence_length
    tokens_per_second = tokens_per_iteration / (average_step_time_ms / 1000.0)
    flops_per_iteration = training_flops_per_iteration(
        micro_batch_size,
        args.sequence_length,
        args.num_layers,
        args.hidden_size,
        args.vocab_size,
    )
    achieved_tflops = flops_per_iteration / (average_step_time_ms / 1000.0) / 1.0e12
    mfu_percent = achieved_tflops / A40_DENSE_BF16_PEAK_TFLOPS * 100.0

    return {
        "status": "success",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "model_config": {
            "architecture": "Megatron Core GPTModel",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "ffn_hidden_size": args.ffn_hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "head_dimension": args.hidden_size // args.num_attention_heads,
            "vocab_size": args.vocab_size,
            "max_position_embeddings": args.sequence_length,
            "position_embedding_type": "learned_absolute",
            "share_embeddings_and_output_weights": True,
            "hidden_dropout": 0.1,
            "attention_dropout": 0.1,
            "layernorm_epsilon": 1.0e-5,
            "attention_backend": (
                "fused"
                if attention_implementation == TE_FUSED_ATTENTION
                else "unfused"
            ),
            "attention_implementation": attention_implementation,
            "core_attention": (
                "TEDotProductAttention"
                if attention_implementation == TE_FUSED_ATTENTION
                else "DotProductAttention"
            ),
            "transformer_layer_spec": "get_gpt_layer_local_spec",
        },
        "parallelism": {
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "data_parallel": 1,
        },
        "precision": {
            "forward_backward": "BF16 autocast",
            "parameter_storage": "FP32",
            "optimizer_state": "FP32",
            "bf16_enabled": True,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "foreach": False,
            "fused": False,
        },
        "data": {
            "type": "fixed synthetic random token IDs",
            "seed": args.seed + 1,
        },
        "micro_batch_size": micro_batch_size,
        "global_batch_size": micro_batch_size,
        "sequence_length": args.sequence_length,
        "warmup_iterations": args.warmup_iterations,
        "measured_iterations": args.measured_iterations,
        "warmup_final_loss": warmup_losses[-1],
        "measured_losses": measured_losses,
        "final_loss": measured_losses[-1],
        "average_step_time_ms": average_step_time_ms,
        "median_step_time_ms": median_step_time_ms,
        "step_times_ms": step_times_ms,
        "tokens_per_second": tokens_per_second,
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "warmup_peak_reserved_memory_mib": warmup_peak_reserved / 1024**2,
        "warmup_memory_fraction": warmup_memory_fraction,
        "gpu_monitoring": utilization,
        "mfu": {
            "available": True,
            "training_flops_per_iteration": flops_per_iteration,
            "achieved_tflops": achieved_tflops,
            "gpu_dense_bf16_peak_tflops": A40_DENSE_BF16_PEAK_TFLOPS,
            "mfu_percent": mfu_percent,
            "formula": (
                "F_iter = 72*B*S*L*H^2 + 6*B*L*S^2*H + 6*B*S*H*V; "
                "MFU = (F_iter / step_seconds) / 149.7e12"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    local_rank = initialize_single_gpu_distributed(args.seed)
    try:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
        if args.attention_implementation == LOCAL_UNFUSED_ATTENTION:
            if os.environ.get("TRANSFORMER_ENGINE_DISABLE") != "1":
                raise RuntimeError("TRANSFORMER_ENGINE_DISABLE=1 is required for this baseline")
        elif os.environ.get("TRANSFORMER_ENGINE_DISABLE") == "1":
            raise RuntimeError("TRANSFORMER_ENGINE_DISABLE must be unset for TE fused attention")

        device = torch.device(f"cuda:{local_rank}")
        candidates = [int(value) for value in args.micro_batch_candidates.split(",")]
        rejected_candidates: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        for candidate in candidates:
            try:
                result = run_candidate(args, candidate, device)
                break
            except (torch.OutOfMemoryError, UnsafeMemoryMargin) as exc:
                rejected_candidates.append({"micro_batch_size": candidate, "reason": str(exc)})
                gc.collect()
                torch.cuda.empty_cache()

        if result is None:
            raise RuntimeError(f"No safe micro-batch size found: {rejected_candidates}")

        result["micro_batch_selection"] = {
            "candidates": candidates,
            "memory_safety_fraction": args.memory_safety_fraction,
            "rejected": rejected_candidates,
            "selected": result["micro_batch_size"],
        }
        result["environment"] = collect_environment()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PHASE1_BASELINE_METRICS_JSON=" + json.dumps(result, sort_keys=True))
    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
