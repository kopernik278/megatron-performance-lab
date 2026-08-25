# Phase 7.4b: targeted TP All-Gather communication overlap

FAST ITERATION MODE (5 warmup + 20 measured). This follow-up does **not**
retry the Phase 7.4 full All-Gather + Reduce-Scatter Userbuffers configuration.
The previous livelock was in

```
userbuffers_fp16_sum_inplace_gpu_rr_rs_oop
```

CUDA Graph stays off. `UB_SKIPMC=1`. NCCL P2P stays enabled. No TE Linear
isolation experiment (no variant A).

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
`with_cublasmp` stays False so bulk dgrad cannot fall back onto
`ub_overlap_rs_dgrad`.

Default AG method is `ring_exchange` on `qkv_fprop`, `fc1_fprop`, `proj_dgrad`,
and `fc2_dgrad`. AG kernels are `userbuffers_fp16_sum_inplace_gpu_rr_ag` /
`rw_ag`, not the RS `rr_rs_oop` kernel.

## Experiment

Reference **B**: TP=2 + Sequence Parallel + TE Linear + Userbuffers OFF.

**C1** (this run): enable Userbuffers with All-Gather overlap only.

```text
tp_comm_overlap=True
tp_comm_overlap_ag=True
tp_comm_overlap_rs=False
tp_comm_overlap_rs_dgrad=False
tp_comm_bulk_dgrad=False
tp_comm_bulk_wgrad=False
```

**C2** (optional, only if C1 is correct and shows real AG/GEMM overlap): same
as C1 plus `tp_comm_bulk_dgrad=True`. Still no RS flags.

Formal 20+100 runs only if C1 is correct, measured overlap > 0, and B→C1
throughput improves by at least 2%.

## GPU result

Pending the 2x A40 Pod run. This document will be updated with B vs C1
throughput, NCCL / exposed NCCL time, AG-GEMM overlap percent, nsys
evidence, and Pod final status after the FAST screen.
