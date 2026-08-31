#!/usr/bin/env python3
"""GPU profile metadata for Phase 10.1 multi-GPU pods."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuProfile:
    gpu_type_id: str
    display_name: str
    memory_gb: int
    dense_bf16_peak_tflops: float
    nvte_cuda_archs: str


GPU_PROFILES: dict[str, GpuProfile] = {
    "NVIDIA A40": GpuProfile(
        gpu_type_id="NVIDIA A40",
        display_name="NVIDIA A40 48GB",
        memory_gb=48,
        dense_bf16_peak_tflops=149.7,
        nvte_cuda_archs="86",
    ),
    "NVIDIA L40S": GpuProfile(
        gpu_type_id="NVIDIA L40S",
        display_name="NVIDIA L40S 48GB",
        memory_gb=48,
        dense_bf16_peak_tflops=181.0,
        nvte_cuda_archs="89",
    ),
    "NVIDIA GeForce RTX 4090": GpuProfile(
        gpu_type_id="NVIDIA GeForce RTX 4090",
        display_name="NVIDIA GeForce RTX 4090 24GB",
        memory_gb=24,
        dense_bf16_peak_tflops=165.2,
        nvte_cuda_archs="89",
    ),
    "NVIDIA GeForce RTX 5090": GpuProfile(
        gpu_type_id="NVIDIA GeForce RTX 5090",
        display_name="NVIDIA GeForce RTX 5090 32GB",
        memory_gb=32,
        dense_bf16_peak_tflops=209.0,
        nvte_cuda_archs="100",
    ),
    "NVIDIA GeForce RTX 3090": GpuProfile(
        gpu_type_id="NVIDIA GeForce RTX 3090",
        display_name="NVIDIA GeForce RTX 3090 24GB",
        memory_gb=24,
        dense_bf16_peak_tflops=142.6,
        nvte_cuda_archs="86",
    ),
    "NVIDIA A100-SXM4-80GB": GpuProfile(
        gpu_type_id="NVIDIA A100-SXM4-80GB",
        display_name="NVIDIA A100 SXM 80GB",
        memory_gb=80,
        dense_bf16_peak_tflops=312.0,
        nvte_cuda_archs="80",
    ),
}


def query_gpu_name() -> str:
    return (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .splitlines()[0]
    )


def match_profile_from_name(gpu_name: str) -> GpuProfile | None:
    upper = gpu_name.upper()
    if "L40S" in upper or "L40 S" in upper:
        return GPU_PROFILES["NVIDIA L40S"]
    if "A40" in upper:
        return GPU_PROFILES["NVIDIA A40"]
    if "4090" in upper:
        return GPU_PROFILES["NVIDIA GeForce RTX 4090"]
    if "5090" in upper:
        return GPU_PROFILES["NVIDIA GeForce RTX 5090"]
    if "3090" in upper:
        return GPU_PROFILES["NVIDIA GeForce RTX 3090"]
    if "A100" in upper:
        return GPU_PROFILES["NVIDIA A100-SXM4-80GB"]
    return None


def resolve_profile(gpu_type_id: str | None = None) -> GpuProfile:
    gpu_name = query_gpu_name()
    detected = match_profile_from_name(gpu_name)
    requested = (gpu_type_id or os.environ.get("PHASE101_GPU_TYPE") or "").strip()
    if detected is not None:
        if requested in GPU_PROFILES and GPU_PROFILES[requested].gpu_type_id != detected.gpu_type_id:
            print(
                "PHASE101_GPU_PROFILE_OVERRIDE="
                f"requested={requested} detected={detected.gpu_type_id} using_detected=1"
            )
        return detected
    if requested in GPU_PROFILES:
        return GPU_PROFILES[requested]
    if requested:
        for profile in GPU_PROFILES.values():
            if profile.gpu_type_id.upper() in requested.upper():
                return profile
    raise RuntimeError(
        f"Unsupported GPU for Phase 10.1: {requested or gpu_name}; "
        f"supported: {', '.join(GPU_PROFILES)}"
    )
