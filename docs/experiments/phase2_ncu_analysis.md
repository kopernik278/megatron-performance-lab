# Phase 2.2 Targeted Nsight Compute Analysis

## Status

Targeted Nsight Compute collection is **blocked by RunPod host permissions**. The baseline and environment were validated unchanged, representative kernels were selected from the Phase 2.1 trace, and NCU filters matched the intended attention kernel. However, the host rejected every metric set with `ERR_NVGPUCTRPERM` before hardware-counter collection. No NCU metric is fabricated or replaced with an unrelated proxy.

The Pod was stopped after diagnosis. This report records the failed NCU attempts, all resource data still available from the Phase 2.1 Nsight Systems trace, provisional classifications, and the exact limit on the root-cause conclusion.

## Unchanged Baseline

- Model: 355,919,872-parameter Megatron Core GPT; 24 layers, H=1024, FFN=4096, 16 heads
- Sequence length: 2,048; micro/global batch: 1/1; TP=1, PP=1, DP=1
- Precision: BF16 autocast with FP32 parameters and optimizer state
- Attention: local Megatron specification, unfused backend
- Optimizer: PyTorch AdamW, `foreach=False`, `fused=False`
- Warmup: 20 iterations before `cudaProfilerStart`
- Transformer Engine/CUDA Graph/added fusion: disabled

Environment: NVIDIA A40, Python `3.12.3`, PyTorch `2.8.0+cu128`, CUDA `12.8`, NCCL `2.27.3`, Megatron-LM `09fde85e`, and Nsight Compute `2025.1.1.0`.

## Representative Kernels

Phase 2.1 identified four useful samples:

1. Attention: `cunn_SoftMaxBackwardSmem`, the largest non-copy attention kernel (`506.895 ms`, 6.89% of kernel time).
2. GEMM: CUTLASS BF16 WMMA `32x32 ... tn`, the largest GEMM symbol (`237.407 ms`, 3.23%).
3. GEMM resource case: Ampere BF16 `128x128 ... nt` (`205.788 ms`, 2.80%).
4. Optimizer: AdamW `addcdiv_cuda_kernel` (`154.393 ms`, 2.10%).

The second GEMM was retained because its launch resources expose a distinct register-pressure case that a single GEMM sample would miss.

## NCU Attempts

The first command requested the basic metric set for one steady-state SoftMax backward launch:

```bash
PYTHONPATH=/workspace/Megatron-LM TRANSFORMER_ENGINE_DISABLE=1 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
  /opt/nvidia/nsight-compute/2025.1.1/ncu \
  --set basic \
  --kernel-name 'regex:cunn_SoftMaxBackwardSmem' \
  --kernel-name-base demangled \
  --launch-skip 5 \
  --launch-count 1 \
  --profile-from-start off \
  --target-processes all \
  --export /tmp/phase2_ncu_permission_test \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase2_nsys_profile.py \
  --warmup-iterations 20 --profiled-iterations 1
```

NCU connected to both launcher and rank processes and matched the target, then returned:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

A second run requested only `LaunchStats` and `Occupancy`; it failed with the same error. Host diagnostics showed `RmProfilingAdminOnly: 1`. The container runs as root but its capability bounding set excludes both `CAP_SYS_ADMIN` and `CAP_PERFMON`. Container root is therefore not a host administrator and cannot change the driver policy. NVIDIA documents that performance-counter access must be granted by the host administrator; see [ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM) and the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Static Launch Analysis

The Nsight Systems SQLite trace records kernel duration, grid/block shape, registers, and shared memory. Theoretical occupancy below is a static upper bound computed from the recorded A40 limits (48 warps, 65,536 registers, 102,400 bytes shared memory, and 16 blocks per SM). It is **not achieved occupancy**.

| Kernel | Avg duration | Registers/thread | Shared memory/block | Static occupancy ceiling | Provisional classification |
| --- | ---: | ---: | ---: | ---: | --- |
| SoftMax backward | 1,408.04 us | 22 | 8,224 B | 100.0% | Memory/latency bound |
| CUTLASS WMMA 32x32 TN | 329.73 us | 48 | 5,120 B | 83.3% | Not established |
| Ampere BF16 128x128 NT | 187.93 us | 234 | 32,768 B | 16.7% | Occupancy/resource bound |
| AdamW addcdiv | 35.25 us | 44 | 0 B | 91.7% | Memory/launch-latency bound |

SoftMax has no static occupancy limiter and no Tensor Core path; reduction dependencies and repeated global-memory traffic make memory/latency pressure the likely limit. The CUTLASS symbol proves WMMA BF16 Tensor Core instructions are selected, but does not measure their utilization. The 128x128 GEMM can admit only two 128-thread blocks per SM because of its 234 registers per thread; this is a resource ceiling, though low occupancy alone does not prove low GEMM throughput.

AdamW `addcdiv` is split across heterogeneous tensor shapes. Of 4,380 launches, 2,190 use a one-block grid and average only `1.72 us`, while large parameter tensors form longer bandwidth-oriented launches. The representative is therefore a mixture of memory-bandwidth and launch-latency behavior.

## Unavailable Metrics

Achieved occupancy, SM throughput, DRAM throughput, L2 throughput, arithmetic intensity, roofline position, Tensor Core utilization, warp stall reasons, and global load/store efficiency are unavailable. Kernel symbols and theoretical occupancy must not be reported as those measured quantities.

## Root-Cause Assessment

Phase 2.1 provides strong evidence that the approximately 6% MFU is primarily caused by the unfused attention path, excessive memory traffic, and kernel fragmentation:

- Attention-attributed work consumes `49.41%` of kernel time, while GEMM consumes only `17.16%`.
- The two dominant BF16/FP32 copy kernels consume `26.39%` of kernel time.
- Device-to-device copies consume `861.90 ms`, or `10.35%` of the profile window before overlap.
- `66.63%` of kernels are shorter than `50 us`; `82.82%` are shorter than `100 us`.
- GPU idle is only `1.29%`, so the GPU is busy but spends most of its time outside useful dense-model FLOPs.

The evidence supports this order: unfused attention, memory traffic, excessive small kernels, then GEMM shape/resource pressure. Low Tensor Core utilization cannot be ranked because it was not measured. Consequently, this phase identifies the likely architectural cause but does not complete the requested hardware-counter proof.

## Artifacts And Next Requirement

No `.ncu-rep` was produced because NCU failed before metric collection. The failed temporary outputs were removed, and no large artifact is committed. Completing Phase 2.2 requires an A40 host configured with non-admin NVIDIA performance-counter access or a container launched with host-granted profiling capability. The model and pinned software stack do not need to change.
