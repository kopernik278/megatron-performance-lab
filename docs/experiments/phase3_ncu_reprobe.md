# Phase 3.0 NCU Capability Reprobe

## Summary

Before Phase 3.4 bottleneck re-profiling, this experiment provisioned a **new**
Secure Cloud A40 Pod on a different data center (`US-MO-1`) and retested whether
RunPod exposes NVIDIA GPU performance counters to NCU. **Hardware-counter
collection remains blocked** with `ERR_NVGPUCTRPERM`. Phase 3.4 will proceed with
**Nsight Systems only**; targeted NCU on GEMM, copy, and fused-attention kernels
is not attempted.

The probe Pod was stopped immediately after diagnosis. No full Megatron benchmark
was run.

## Infrastructure

- Pod: `9srvihy8dgwugw`, Secure Cloud, `1x NVIDIA A40 48GB`, `$0.44/hour`
- Data center: `US-MO-1` (prior Phase 2.2/3.x pods used `CA-MTL-1`)
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Host driver: `580.126.16`; host CUDA: `13.0`
- Timestamp: `2026-08-23T13:51:34Z` – `2026-08-23T13:54:00Z` (~2.5 min probe)

## Pinned Stack Verification

| Component | Expected | Measured |
| --- | --- | --- |
| Python | 3.12.3 | 3.12.3 |
| PyTorch | 2.8.0+cu128 | 2.8.0+cu128 |
| CUDA runtime | 12.8 | 12.8 |
| NCCL | 2.27.3 | 2.27.3 |
| Nsight Compute | 2025.1.1.x | 2025.1.1.0 |
| GPU | NVIDIA A40 | NVIDIA A40 (CC 8.6) |

## Host Profiling Diagnostics

```text
RmProfilingAdminOnly: 1
```

Container runs as `root`, but the capability bounding set excludes
`CAP_SYS_ADMIN`, `CAP_PERFMON`, and `CAP_SYS_PTRACE`. Combined with
`RmProfilingAdminOnly: 1`, the driver restricts performance-counter access to
host administrators. This matches the Phase 2.2 diagnosis on a prior A40 host.

No privilege-escalation or host-modification attempts were made.

## NCU Probes

Workload: minimal PyTorch BF16 `4096×4096` matmul (no Megatron, no full
benchmark).

### Probe 1 — `--set LaunchStats`

```bash
/opt/nvidia/nsight-compute/2025.1.1/ncu \
  --set LaunchStats --launch-count 1 --target-processes all \
  python3 -c "import torch; x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize()"
```

- Exit code: `0`
- NCU connected and captured one kernel launch
- **Does not collect hardware performance counters** (launch configuration only)
- Not sufficient for occupancy, SM/DRAM throughput, Tensor Core utilization, or
  roofline analysis

### Probe 2 — `--set basic`

```bash
/opt/nvidia/nsight-compute/2025.1.1/ncu \
  --set basic --launch-count 1 --target-processes all \
  python3 -c "import torch; x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize()"
```

Result:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

Exit code: `1`. No `.ncu-rep` with hardware metrics was produced.

## Comparison to Phase 2.2

| Item | Phase 2.2 | This reprobe |
| --- | --- | --- |
| Error | `ERR_NVGPUCTRPERM` | `ERR_NVGPUCTRPERM` |
| `RmProfilingAdminOnly` | `1` | `1` |
| Data center | `CA-MTL-1` | `US-MO-1` |
| Outcome | NCU blocked | NCU blocked |

A freshly provisioned host in a different data center does **not** change the
restriction.

## Phase 3.4 Implication

**Proceed with Nsight Systems only** on the current best config (fused attention,
micro-batch=8). Do not attempt targeted NCU on:

- dominant GEMM kernels
- `bfloat16_copy` / `direct_copy` kernels
- fused attention (cuDNN SDPA) kernels

Unavailable until RunPod grants non-admin performance-counter access or a host
with `RmProfilingAdminOnly: 0`:

- achieved occupancy, SM throughput, DRAM/L2 throughput
- Tensor Core utilization, warp stall reasons, roofline from hardware counters

Static launch metadata from Nsight Systems SQLite exports (as in Phase 2.2/3.3)
remains the ceiling for kernel-level resource analysis on RunPod A40.

## Artifacts

- JSON summary: `results/phase3_ncu_reprobe.json`
- No `.ncu-rep` committed (collection failed before metric export)
