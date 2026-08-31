#!/usr/bin/env python3
"""GPU profile metadata for Phase 10.1 (A40 default, L40S alternate)."""

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
    return None


def resolve_profile(gpu_type_id: str | None = None) -> GpuProfile:
    requested = (gpu_type_id or os.environ.get("PHASE101_GPU_TYPE") or "").strip()
    if requested in GPU_PROFILES:
        return GPU_PROFILES[requested]
    if requested:
        for profile in GPU_PROFILES.values():
            if profile.gpu_type_id.upper() in requested.upper():
                return profile
    detected = match_profile_from_name(query_gpu_name())
    if detected is not None:
        return detected
    raise RuntimeError(
        f"Unsupported GPU for Phase 10.1: {requested or query_gpu_name()}; "
        f"supported: {', '.join(GPU_PROFILES)}"
    )
