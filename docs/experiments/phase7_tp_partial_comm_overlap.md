# Phase 7.4b: targeted TP All-Gather communication overlap

FAST ITERATION MODE (5 warmup + 20 measured) plus a formal 20+100 that was
gated by correctness, measured overlap > 0, and B→C1 ≥ 2%. This follow-up
did **not** retry the Phase 7.4 full All-Gather + Reduce-Scatter Userbuffers
configuration. The previous livelock was in

```
userbuffers_fp16_sum_inplace_gpu_rr_rs_oop
```

That kernel launched **0 times** on C1 and C2. CUDA Graph stayed off.
`UB_SKIPMC=1`. NCCL P2P stayed enabled. No TE Linear isolation experiment
(no variant A).

## Source-verified flags (pinned TE 2.17.1 + Megatron `09fde85`)

Inspected Transformer Engine `4329ff84bfbdaa778a33cba02a15fb0807c64689` and
Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`. Flag names below are
from those sources, not invented.

Megatron `TransformerConfig` (`megatron/core/model_parallel_config.py`) and
TE Linear wiring (`megatron/core/extensions/transformer_engine.py`,
`transformer_engine/pytorch/module/linear.py`):

| Megatron flag | Default | TE argument | What it actually does in TE 2.17.1 |
|---|---|---|---|
| `tp_comm_overlap` | False | master switch | Required before any `ub_*` kwargs are set |
| `tp_comm_overlap_ag` | True | `ub_overlap_ag` | Column fprop AG (`ub_overlap_ag_fprop`) and row dgrad AG (`ub_overlap_ag_dgrad`) |
| `tp_comm_overlap_rs` | True | `ub_overlap_rs` | Row fprop RS (`ub_overlap_rs_fprop`) — **this launches the hanging RS kernel** |
| `tp_comm_overlap_rs_dgrad` | False | `ub_overlap_rs_dgrad` | Column dgrad RS — also an RS path; keep off |
| `tp_comm_bulk_dgrad` | True | `ub_bulk_dgrad` | Bulk **All-Gather** vs dgrad GEMM (`CommOverlapType.AG`) |
| `tp_comm_bulk_wgrad` | True | `ub_bulk_wgrad` | Bulk **Reduce-Scatter** vs wgrad GEMM (`CommOverlapType.RS`) — unsafe here |
| `tp_comm_split_ag` / `tp_comm_atomic_ag` | True / False | TE v1.6 deprecated | Not used; TE >= 1.5 maps the current `tp_comm_overlap_ag` flag |
| `tp_comm_split_rs` / `tp_comm_atomic_rs` | True / False | TE v1.6 deprecated | Same for RS; atomic GEMM is FP8-only |
| `tp_comm_bootstrap_backend` | `nccl` | `initialize_ub(..., bootstrap_backend="nccl")` | Keep NCCL |
| `tp_comm_overlap_disable_qkv` / `_fc1` | False | clears AG on those layers | Leave off |

Megatron comments on `tp_comm_bulk_dgrad` / `tp_comm_bulk_wgrad` are swapped
relative to TE. This experiment follows TE semantics.

`initialize_ub(..., ub_cfgs={}, with_cublasmp=False)` still **allocates**
default RS communicators (`proj_fprop`, `fc2_fprop`, `qkv_wgrad`, `fc1_wgrad`).
The hang is from **using** the RS kernel, not from allocating those buffers.
`with_cublasmp` stayed False so bulk dgrad could not fall back onto
`ub_overlap_rs_dgrad`.

## Exact overlap flags used

**B** (reference): Userbuffers off.

```text
tp_comm_overlap=False
```

**C1** (AG-only pipelined overlap):

```text
tp_comm_overlap=True
tp_comm_overlap_ag=True
tp_comm_overlap_rs=False
tp_comm_overlap_rs_dgrad=False
tp_comm_bulk_dgrad=False
tp_comm_bulk_wgrad=False
```

Runtime checks confirmed `ub_overlap_ag_fprop` on QKV/FC1, `ub_overlap_ag_dgrad`
on proj/FC2, and **no** `ub_overlap_rs_fprop`, `ub_overlap_rs_dgrad`, or
`ub_bulk_wgrad`. Mode string: `ag_only`.

**C2** (optional extra safe mode after C1 succeeded with overlap > 0):

```text
same as C1 plus tp_comm_bulk_dgrad=True
```

Mode string: `ag_plus_bulk_dgrad`. Column `ub_bulk_dgrad` was True. RS paths
stayed off.

Collectives targeted: All-Gather. Collectives not overlapped: Reduce-Scatter
fprop, RS dgrad, bulk wgrad RS.

## Infrastructure and topology

- RunPod Pod: `8akpdrt2brqwwm` (deleted after download)
- Data center: CA-MTL-1, public IP `69.30.85.75`, suffix `6441139d`
- Allocation: one Secure Cloud Pod, 2x NVIDIA A40 48GB, $0.88/h (≤ $0.90/h)
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Host CUDA: 12.8; container CUDA runtime: 12.8
- Driver: 570.211.01
- PyTorch: 2.8.0+cu128
- NCCL: 2.27.3
- Transformer Engine: `2.17.1+4329ff84`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Lab commit on the Pod: `ccb5b42dee0c3d63b78ccfdc8b70c334428c21c8`
- Topology: **PXB**, both GPUs NUMA 1, PCI `D1:00.0` / `D6:00.0`, NVLink inactive, roce NICs
- CUDA peer access bidirectional True
- NCCL All-Reduce sanity: 3.0 on both ranks (0.554 ms / 0.648 ms)
- `CUDA_DEVICE_MAX_CONNECTIONS=1`, `UB_SKIPMC=1`, `NCCL_P2P_DISABLE` unset (`"0"`)
- CUDA Graph off; `bias_dropout_fusion=True`; `bias_gelu_fusion=False`

SYS and NCCL-hang hosts were rejected during the hunt and are not in this result.

## Correctness

Smoke (3 steps) on B, C1, and C2: finite losses, finite grads, optimizer
consumed `main_grad`, parameters updated, no deadlock, no NCCL errors, no
NaN/Inf, max rank loss difference 0.0. Forward, backward, and optimizer
completed. Userbuffers initialized on C1/C2 (`communicator_count=12`) and
inactive on B.

C1 and C2 both completed inside the 120 s abort window. The hanging RS kernel
did not appear in logs or nsys traces (`hang_rs_kernel_launches=0`).

## Throughput (reported protocol: formal 20+100)

| Variant | Flags | tokens/s | step ms | MFU | VRAM (smi) |
|---|---|---:|---:|---:|---:|
| B FAST | UB off | 27,224.90 | 601.80 | 22.03% | 17,162 MiB |
| C1 FAST | AG-only | 29,046.88 | 564.05 | 23.51% | 17,650 MiB |
| C2 FAST | AG + bulk dgrad | 28,311.84 | 578.70 | 22.91% | 17,650 MiB |
| B formal | UB off | 27,209.17 | 602.15 | 22.02% | 17,162 MiB |
| C1 formal | AG-only | 29,044.80 | 564.09 | 23.51% | 17,650 MiB |

Formal B→C1 = **+6.75%** (1.067x). FAST B→C1 = +6.69%. FAST B→C2 = +3.99%.
C1 is the better throughput result. C2 is kept as the overlap-evidence result.

## NCCL / exposed communication / overlap (nsys, 5 profiled steps)

| Metric | B | C1 | C2 |
|---|---:|---:|---:|
| NCCL ms/step/GPU | 229.24 | 142.03 | 99.03 |
| Userbuffer ms/step/GPU | 0.00 | 1.44 | 39.60 |
| All-Gather ms/step/GPU | 133.35 | 46.58 | 3.56 |
| Reduce-Scatter ms/step/GPU | 92.28 | 92.25 | 92.03 |
| Exposed comm ms/step | 229.24 | 181.36 | 138.25 |
| Comm/compute overlap % | 0.00 | 11.59 | 31.00 |
| AG-kernel vs GEMM overlap % | 0.00 | 0.00 | **91.47** |
| Hang RS kernel launches | 0 | 0 | 0 |

C1 reduced All-Gather kernel time from 133.35 ms to 46.58 ms and exposed
communication from 229.24 ms to 181.36 ms (−47.88 ms). That is a real
communication reduction, but the remaining C1 All-Gather is still NCCL
`AllGather_RING_LL` and does **not** overlap GEMM (AG-GEMM overlap 0.0%).
The 11.59% communication/compute overlap on C1 comes from short
`kuserbuffers_pushsend` / `kuserbuffers_pushrecv` kernels, not from AG+GEMM
pipelining.

C2 is the configuration that actually overlaps All-Gather with GEMM. Nsight
shows `userbuffers_fp16_sum_inplace_gpu_rw_ag<2>` concurrent with Ampere /
CUTLASS GEMM for 91.47% of AG communication time. Exposed AG drops to
3.57 ms/step. Reduce-Scatter remains fully exposed (~92 ms) because RS
overlap stayed disabled.

## Nsight timeline evidence

C1 AG kernels (no GEMM overlap):

- `ncclDevKernel_AllGather_RING_LL` — 510 launches, 465.81 ms across both GPUs
- `kuserbuffers_pushsend` / `kuserbuffers_pushrecv` — 958 launches each, 3.21 / 11.17 ms

C2 AG kernels (real GEMM overlap):

- `userbuffers_fp16_sum_inplace_gpu_rw_ag<2>` — 480 launches, 382.96 ms, **91.5% overlapped with GEMM**
- leftover NCCL All-Gather — 30 launches, 35.56 ms

Neither C1 nor C2 launched `userbuffers_fp16_sum_inplace_gpu_rr_rs_oop`.

## Did AG-only overlap work?

Yes for **correctness and throughput**. C1 initialized Userbuffers, finished
forward/backward/optimizer with finite loss, and improved formal throughput by
6.75% versus B.

No for **AG NCCL/Userbuffer kernels overlapping GEMM**. C1's pipelined
`tp_comm_overlap_ag` path on this A40 PXB host still runs NCCL All-Gather
sequentially with GEMM. The working AG+GEMM overlap on this SKU is C2's bulk
dgrad All-Gather (`ub_bulk_dgrad` → `userbuffers_fp16_sum_inplace_gpu_rw_ag`).

Another safe overlap mode **was** tested: C2 bulk dgrad. It did not use the
known failing RS kernel. It showed 91.5% AG-GEMM overlap and +4.0% throughput
versus B, but was slower than C1.

Do not enable `tp_comm_overlap_rs` or `tp_comm_bulk_wgrad` next. Remaining
exposed communication is Reduce-Scatter (~92 ms/step) plus leftover NCCL AG.

## Commands

On the Pod, after cloning branch `cursor/phase74b-ag-overlap-3b5c`:

```bash
bash scripts/phase7_partial_overlap_pod.sh 8akpdrt2brqwwm 0.88
```

Raw outputs: `results/phase74b_work/`. Summary:
`results/phase7_tp_partial_comm_overlap.json`.
