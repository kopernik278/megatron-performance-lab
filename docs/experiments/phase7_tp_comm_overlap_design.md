# Phase 7.3: Tensor Parallel communication overlap design

This is a source-and-results-only design phase. No Pod was started, no GPU
experiment was run, and no model or environment code was changed.

The performance baseline is the corrected Phase 7.1 TP=2 result from commit
`709437d`, not the Phase 7.2 Sequence Parallel host.

Phase 7.2 remains a memory-saving experiment. Do not use its P2P-disabled PIX
result as a throughput baseline.

## Decision

The single Phase 7.4 experiment should enable Transformer Engine Userbuffers
TP communication overlap on the Phase 7.1 NODE + P2P topology:

```python
config = TransformerConfig(
    tensor_model_parallel_size=2,
    sequence_parallel=True,          # required by this overlap path
    tp_comm_overlap=True,
    tp_comm_overlap_ag=True,         # already the default
    tp_comm_overlap_rs=True,         # already the default
    tp_comm_bulk_dgrad=True,         # already the default
    tp_comm_bulk_wgrad=True,         # already the default
    tp_comm_overlap_rs_dgrad=False,  # leave off
    tp_comm_bootstrap_backend="nccl",
    cuda_graph_impl="none",
    bias_dropout_fusion=True,
    bias_activation_fusion=False,
)
```

Required launch/runtime companions, not extra optimizations:

- `CUDA_DEVICE_MAX_CONNECTIONS=1` (already required by the Phase 7 harness)
- `UB_SKIPMC=1` (A40 has no CUDA Multicast)
- `transformer_engine.pytorch.module.base.initialize_ub(...)` before the first
  TE Linear forward
- TE Linear modules (`TEColumnParallelLinear` / `TERowParallelLinear`) via
  `get_gpt_layer_with_transformer_engine_spec`
- NCCL P2P left enabled; abort if the host is not same-NUMA with working P2P

Do not enable Sequence Parallel as a standalone speedup. It is a prerequisite
of the Userbuffers AG/RS overlap path. Do not enable CUDA Graph, bias-GELU
fusion, Apex `gradient_accumulation_fusion`, DP overlap, or symmetric-memory
All-Reduce.

## Baseline to beat

Phase 7.1, commit `709437d`, pod `7rpwv95a5j6axg`, same-NUMA **NODE**, P2P
working, Sequence Parallel off:

| Field | Value |
|---|---|
| Throughput | 19,856.48 tokens/s |
| Average step | 825.12 ms |
| MFU | 16.07% |
| Collectives | 101 All-Reduces / step; AG=0; RS=0 |
| NCCL | 259.28 ms/step/GPU (`RING_LL`) |
| Overlap | 0.0% |
| Exposed communication | 259.22 ms/step/GPU |
| NVLink | none |
| CUDA Graph | off |

Approximate NVTX attribution of that 259 ms:

- attention TP forward (row-parallel `proj` All-Reduce): 61.25 ms
- MLP TP forward (row-parallel `fc2` All-Reduce): 61.28 ms
- backward TP (column-parallel QKV/FC1/output dgrad All-Reduce): 125.73 ms
- vocabulary embedding All-Reduce: 10.38 ms

The 97 large activation All-Reduces are 32 MiB BF16 each. They are fully
exposed. The lab already sets `CUDA_DEVICE_MAX_CONNECTIONS=1`, and the pinned
native linear path already launches column-parallel dgrad All-Reduce with
`async_op=True`. Measured overlap is still zero. Hiding this communication
therefore requires a different communication implementation, not another
toggle of the already-on native async All-Reduce.

Phase 7.2 (pod `wtd9cxr3q8obuh`, PIX, `NCCL_P2P_DISABLE=1`) showed Sequence
Parallel is correct, cuts about 2 GiB/GPU, replaces 96 All-Reduces with
All-Gather / Reduce-Scatter, and **loses 8.51% throughput**. That host is not
the 7.4 interconnect.

## Pinned software

- Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine `2.17.1+4329ff84` (`4329ff84bfbdaa778a33cba02a15fb0807c64689`)
- PyTorch 2.8.0+cu128, CUDA 12.8, NCCL 2.27.3
- Current layer spec: `get_gpt_layer_local_spec()` plus TE fused attention
- Linear modules today: Megatron `ColumnParallelLinear` / `RowParallelLinear`
- `TransformerConfig.tp_comm_overlap` currently remains False (the default)

## Available overlap mechanisms

The deprecated config field `async_tensor_model_parallel_allreduce` is **not**
present on this Megatron commit. Column-parallel dgrad All-Reduce overlap is
hard-wired in the native linear autograd function when
`sequence_parallel=False` and TP > 1.

Pipeline-parallel P2P overlap and MoE expert-parallel overlap are out of
scope: this lab is PP=1, dense GPT.

### 1. Native async column-parallel All-Reduce (already on)

1. **Config option:** none. `ModelParallelConfig` no longer exposes
   `async_tensor_model_parallel_allreduce`. The behavior is
   `ColumnParallelLinear.allreduce_dgrad = (tp_size > 1 and not sequence_parallel)`.
2. **Source:**
   [`megatron/core/tensor_parallel/layers.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/tensor_parallel/layers.py)
   `LinearWithGradAccumulationAndAsyncCommunication.backward` launches
   `torch.distributed.all_reduce(..., async_op=True)` then computes the
   weight gradient, then `handle.wait()`.
3. **Collective:** column-parallel **input-dgrad All-Reduce** (QKV, FC1, tied
   output). Not the row-parallel forward All-Reduce.
4. **Overlapped compute:** weight-gradient GEMM of the same linear layer.
5. **Hardware:** any TP group. Relies on `CUDA_DEVICE_MAX_CONNECTIONS=1` so
   the NCCL kernel is issued before the GEMM.
6. **NVLink vs PCIe:** PCIe is supported. NVLink is not required.
7. **A40 TP=2 compatibility:** already used in Phase 7.1. Peer access was
   available. Overlap measured **0%**.
8. **Complexity:** none. Already the default native path.
9. **Numerical risk:** low. Same reduction as a synchronous All-Reduce;
   only the wait is deferred past wgrad.
10. **Expected benefit vs 259 ms:** at most the backward 125.73 ms, and only
    if wgrad actually runs under NCCL. Phase 7.1 already ran this path and
    hid **0 ms**. A40 `RING_LL` All-Reduce of 32 MiB appears to occupy the
    GPU so the subsequent GEMM cannot start. **Do not re-test this as Phase
    7.4.**

Row-parallel `proj` / `fc2` forward All-Reduce is a blocking
`reduce_from_tensor_model_parallel_region` in
[`mappings.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/tensor_parallel/mappings.py).
There is no native overlap switch for those 122 ms.

### 2. Native async Sequence Parallel All-Gather / Reduce-Scatter

1. **Config option:** `TransformerConfig.sequence_parallel=True`.
2. **Source:** the same `LinearWithGradAccumulationAndAsyncCommunication`.
   Forward All-Gather is **synchronous**. Backward All-Gather of activations
   uses `async_op=True` before dgrad GEMM. Backward Reduce-Scatter of dgrad
   uses `async_op=True` before wgrad GEMM. Row-parallel forward Reduce-Scatter
   in `RowParallelLinear.forward` is **synchronous**.
3. **Collective:** SP All-Gather and Reduce-Scatter. Replaces the 96
   per-layer All-Reduces mapped in Phase 7.2.
4. **Overlapped compute:** backward dgrad GEMM (with activation All-Gather)
   and backward wgrad GEMM (with dgrad Reduce-Scatter). Forward GEMM is not
   pipelined with its collective.
5. **Hardware:** TP>=2. Requires a fused LayerNorm (TENorm; Apex LN is
   absent). `CUDA_DEVICE_MAX_CONNECTIONS=1`.
6. **NVLink vs PCIe:** PCIe is supported via NCCL. NVLink is not required.
7. **A40 TP=2 compatibility:** Phase 7.2 proved correctness. On the
   P2P-disabled PIX host, overlap was still 0% and throughput fell 8.51%.
8. **Complexity:** low. The Phase 7.2 harness already exists.
9. **Numerical risk:** moderate. SP changes reduction order; 7.2 losses were
   close but not bitwise identical.
10. **Expected benefit vs 259 ms:** at best the backward half, and only on a
    NODE+P2P host if NCCL no longer saturates the GPU. Forward ~122 ms stays
    exposed. Phase 7.2 already showed this is a memory optimization, not a
    speedup. **Reject as the 7.4 speed experiment.**

### 3. Transformer Engine Userbuffers (`tp_comm_overlap`)

1. **Config option:** `TransformerConfig.tp_comm_overlap=True`, plus the
   sub-flags below. Sub-flags are ignored when the master flag is False.
2. **Source:**
   - Config:
     [`model_parallel_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/model_parallel_config.py)
     (`tp_comm_overlap`, `tp_comm_overlap_ag`, `tp_comm_overlap_rs`,
     `tp_comm_bulk_dgrad`, `tp_comm_bulk_wgrad`, `tp_comm_overlap_rs_dgrad`,
     `tp_comm_bootstrap_backend`).
   - Wiring:
     [`megatron/core/extensions/transformer_engine.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/extensions/transformer_engine.py)
     `TELinear` / `TELayerNormLinear` pass `ub_overlap_ag`, `ub_overlap_rs`,
     `ub_bulk_dgrad`, `ub_bulk_wgrad`, `ub_name`.
   - Init:
     [`megatron/training/initialize.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/training/initialize.py)
     `_initialize_tp_communicators()` →
     `transformer_engine.pytorch.module.base.initialize_ub`.
   - TE 2.17.1 implementation:
     [`transformer_engine/pytorch/module/base.py`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/module/base.py)
     `initialize_ub`, `CommOverlapP2P` (ring exchange), `CommOverlap`
     (pipeline/bulk).
3. **Collective:** Sequence-parallel **All-Gather** and **Reduce-Scatter**,
   implemented as chunked CUDA IPC / P2P ring or pipelined splits instead of
   a single NCCL All-Reduce. It does **not** overlap the native All-Reduce
   path. Sequence Parallel is mandatory.
4. **Overlapped compute:** TE default methods for this TP=2 hidden=1024
   layer:

   | Buffer | Collective | Method | Overlapped GEMM |
   |---|---|---|---|
   | `qkv_fprop`, `fc1_fprop` | All-Gather | ring_exchange | column-parallel forward GEMM |
   | `proj_fprop`, `fc2_fprop` | Reduce-Scatter | pipeline | row-parallel forward GEMM |
   | `qkv_dgrad`, `fc1_dgrad` | All-Gather | bulk | independent backward activation GEMM |
   | `qkv_wgrad`, `fc1_wgrad` | Reduce-Scatter | bulk | independent wgrad GEMM |
   | `proj_dgrad`, `fc2_dgrad` | All-Gather | ring_exchange | row-parallel dgrad path |

   Default `use_ce=True` tries the copy engine for ring-exchange so SMs can
   stay on GEMM. Pipeline methods set `set_sm_margin=True`.
5. **Hardware:** TP>=2, same-node devices with CUDA peer access. CUDA
   Multicast (Hopper + NVLink/NVSwitch) is the default Userbuffers backend.
   A40 is Ampere SM 8.6: `initialize_ub` raises unless `UB_SKIPMC=1`, which
   selects CUDA IPC.
6. **NVLink vs PCIe:** NVLink is the supported high-performance path. TE
   maintainers describe Userbuffers as a single-node NVLink-domain
   interconnect. PCIe is not the design point. IPC over CUDA P2P is the
   documented fallback (`UB_SKIPMC=1`) and is what this A40 pair would use.
   **NVLink is not present on the 7.1 host; PCIe P2P is.**
7. **A40 TP=2 compatibility:** possible only with `UB_SKIPMC=1` and working
   CUDA P2P on a NODE host like 7.1. May fail at `initialize_ub` or hang,
   matching the 7.2 NCCL P2P hang on a different PIX host. Gate the run:
   same-NUMA NODE, `can_device_access_peer` true, `NCCL_P2P_DISABLE` unset.
   If `initialize_ub` fails, stop and record a blocker. Do not fall back to
   the 7.2 P2P-disabled host.
8. **Complexity:** medium. Must switch linears to the TE spec, call
   `initialize_ub` with shape `[seq * micro_batch, hidden] = [16384, 1024]`,
   keep TENorm, and assert `ub_name` in `{qkv, proj, fc1, fc2}`.
9. **Numerical risk:** moderate. Sequence Parallel plus TE Linear GEMMs
   change reduction order and kernel mix. Reuse the Phase 7.2 correctness
   checks (both ranks, fwd/bwd, finite `main_grad`, optimizer, finite loss,
   no deadlock). Exact loss identity with 7.1 is not required.
10. **Expected benefit vs 259 ms:** this is the only pinned mechanism that
    can hide **both** the forward 122 ms and the backward 126 ms, because it
    pipelines dependent AG/RS with their GEMMs and bulk-overlaps independent
    pairs. On NVLink that can hide most of 259 ms. On A40 PCIe IPC a
    conservative bound is hiding **20–40%** of exposed communication
    (**50–100 ms**), or about **6–14%** throughput if SP overhead stays
    small on NODE+P2P. It can also be net negative if IPC is as slow as
    7.2's P2P-disabled NCCL. Treat that as an allowed outcome, not a
    fabricated speedup.

Deprecated TE v1.6 flags `tp_comm_split_ag/rs` and `tp_comm_atomic_ag/rs`
must stay at defaults. Atomic GEMM overlap is FP8-only and untested.

### 4. Userbuffers sub-flags (not separate experiments)

| Flag | Default | Collective | Notes |
|---|---|---|---|
| `tp_comm_overlap_ag` | True | All-Gather | pipelined/ring with GEMM |
| `tp_comm_overlap_rs` | True | Reduce-Scatter | pipelined with GEMM |
| `tp_comm_bulk_dgrad` | True | AG of activations vs dgrad GEMM | independent pair |
| `tp_comm_bulk_wgrad` | True | RS vs wgrad GEMM | independent pair |
| `tp_comm_overlap_rs_dgrad` | False | RS with dgrad GEMM | extra complexity; leave off |
| `tp_comm_overlap_disable_qkv/fc1` | False | disables AG on those layers | leave off |
| `tp_comm_overlap_cfg` | None | YAML Userbuffer presets | leave None; TE defaults |

Phase 7.4 should flip only the master `tp_comm_overlap=True` and keep these
defaults. Do not tune `num_sm` / `num_splits` until the default path is
measured.

### 5. `CUDA_DEVICE_MAX_CONNECTIONS`

1. **Config option:** environment variable, not `TransformerConfig`.
2. **Source:** `linear_with_grad_accumulation_and_async_allreduce` documents
   that `=1` forces kernels to issue in launch order so collectives start
   before GEMM. Megatron requires this on Ampere/Hopper for TP/SP. Blackwell
   dropped the requirement.
3. **Collective:** orders native async AR/AG/RS and Userbuffers launch
   order. Not a collective itself.
4. **Overlapped compute:** whatever GEMM is launched after the collective.
5. **Hardware:** Ampere A40 needs `=1`.
6. **NVLink vs PCIe:** independent of interconnect.
7. **A40 TP=2 compatibility:** already set. The Phase 7 harness fails if it
   is not `1`.
8. **Complexity:** none.
9. **Numerical risk:** none.
10. **Expected benefit:** already applied; 7.1 still measured 0% overlap.
    **Do not A/B this variable.** Raising it to 8 risks reordering TP
    collectives and is not a low-risk overlap experiment.

### 6. `gradient_accumulation_fusion`

1. **Config option:** `TransformerConfig.gradient_accumulation_fusion=True`.
2. **Source:** native path requires Apex `fused_weight_gradient_mlp_cuda`;
   TE Linear uses `fuse_wgrad_accumulation` into `main_grad`.
3. **Collective:** none. Makes the wgrad kernel that native async overlap
   tries to hide communication behind cheaper / fused.
4. **Overlapped compute:** fused wgrad into `main_grad`.
5. **Hardware:** CUDA >= 11 for the Apex extension.
6. **NVLink vs PCIe:** irrelevant.
7. **A40 TP=2 compatibility:** Apex is **not** installed. Native fusion
   raises. TE fusion would require switching to TE Linear first.
8. **Complexity:** installing Apex, or folding it into the TE Linear swap.
9. **Numerical risk:** low-moderate (`main_grad` is already the lab
   gradient).
10. **Expected benefit:** does not hide the 122 ms forward All-Reduce. Not
    the 7.4 experiment. If TE Linear is already required for Userbuffers,
    leave `gradient_accumulation_fusion=False` so the A/B is overlap-only.

### 7. Symmetric-memory All-Reduce (`symmetric_ar_type`)

1. **Config option:** `TransformerConfig.symmetric_ar_type` in
   `{one_shot, two_shot, multimem_all_reduce}`.
2. **Source:** `transformer_config.py`; TE Linear `symmetric_ar_type`.
   Requires PyTorch >= 2.7 (available) and TE >= 2.3 (available).
3. **Collective:** All-Reduce via CUDA symmetric memory / NVLS multimem.
4. **Overlapped compute:** not a GEMM-pipeline overlap; it is a faster AR
   kernel family.
5. **Hardware:** multimem/NVLS needs Hopper NVLink. A40 has neither.
6. **NVLink vs PCIe:** NVLink required for the useful modes.
7. **A40 TP=2 compatibility:** no.
8. **Complexity:** TE Linear swap plus untested AR path.
9. **Numerical risk:** unknown on this stack.
10. **Expected benefit:** not applicable. **Reject.**

### 8. Mechanisms that are not TP overlap

| Mechanism | Why it is out of scope |
|---|---|
| `DistributedDataParallelConfig.overlap_grad_reduce` | DP gradient buckets; DP=1 |
| `overlap_param_gather` | distributed optimizer; disabled |
| `overlap_p2p_comm` | pipeline send/recv; PP=1 |
| `overlap_moe_expert_parallel_comm` | MoE; dense GPT |
| `delay_wgrad_compute` | TE wgrad deferral; extra to Userbuffers |
| CUDA Graph | Phase 6 blocker; must stay off |

## Recommended Phase 7.4 experiment

Exactly one experiment: **TE Userbuffers TP communication overlap** on a
Phase 7.1-class host.

### Variant A

Reproduce Phase 7.1:

- `tensor_model_parallel_size=2`
- `sequence_parallel=False`
- `tp_comm_overlap=False`
- local GPT spec + fused TE attention
- CUDA Graph off
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- NCCL P2P enabled
- same-NUMA NODE, 2x A40, one Pod, <= $0.90/hour

### Variant B

Identical host, software versions, architecture, seq=2048, micro-batch=8,
precision, optimizer, `bias_dropout_fusion=True`, `bias_gelu_fusion=False`,
CUDA Graph off, then the overlap mechanism:

```python
sequence_parallel = True
tp_comm_overlap = True
tp_comm_bootstrap_backend = "nccl"
```

```text
UB_SKIPMC=1
CUDA_DEVICE_MAX_CONNECTIONS=1
```

```python
te.pytorch.module.base.initialize_ub(
    shape=[2048 * 8, 1024],
    tp_size=2,
    use_fp8=False,
    ub_cfgs={},
    bootstrap_backend="nccl",
)
```

Layer spec: `get_gpt_layer_with_transformer_engine_spec()` so QKV/proj/FC1/FC2
are `TEColumnParallelLinear` / `TERowParallelLinear` with `ub_name` in
`{qkv, proj, fc1, fc2}`.

Sequence Parallel and the TE Linear spec are **prerequisites of Userbuffers**,
not additional optimizations. Do not add fusion, graphs, or DP overlap.

### Isolation caveat

B changes three coupled pieces: SP, TE Linear, and Userbuffers. That is the
pinned NVIDIA path; Userbuffers silently does nothing on the current local
linear spec. If B is not at least 2% faster than A, keep the fast screen and
do not add a second isolation run. If B is faster, a later phase may peel SP
and TE Linear apart. Do not do that in 7.4.

### Expected timeline transformation

```text
A (Phase 7.1):
  QKV/FC1 GEMM -> ... -> proj/FC2 GEMM -> blocking All-Reduce -> next op
  backward: dgrad GEMM -> async All-Reduce -> wgrad GEMM (wait); measured overlap 0%

B (Userbuffers + SP):
  scatter sequence -> AG (ring, copy-engine) || QKV/FC1 GEMM
                   -> attention/MLP compute
                   -> proj/FC2 GEMM || pipelined Reduce-Scatter
  backward: bulk AG || independent GEMM; bulk RS || wgrad
```

Nsight must report All-Reduce / All-Gather / Reduce-Scatter counts and
ms/step, total NCCL or Userbuffer comm ms/step, overlap percent, and exposed
communication. Runtime checks must prove `config.tp_comm_overlap=True`, SP
activation shapes `[S/TP, B, H]`, and that Userbuffers communicators exist.

### Topology constraints

- Exactly one Pod, exactly 2x A40, same host, prefer same-NUMA NODE.
- CUDA peer access must be true. Leave NCCL P2P enabled.
- Stop immediately if topology is SYS/cross-NUMA or if NCCL P2P hangs.
- Do not reuse Phase 7.2 pod `wtd9cxr3q8obuh`.
- `UB_SKIPMC=1` is mandatory on A40. CUDA Multicast will not initialize.
- NVLink is absent; treat IPC-over-PCIe as the fallback under test, not as
  an NVLink result.

### Expected performance benefit

This is an estimate, not a measured result.

- Upper bound if all 259 ms were hidden: 825 ms → 566 ms, about 1.46x.
  Unrealistic on PCIe without NVLink.
- Conservative A40 IPC case: hide 50–100 ms of the 259 ms (20–40%), about
  **6–14%** tokens/s, only if SP overhead on NODE+P2P is small.
- Failure case: `initialize_ub` raises, P2P hangs, or B is slower than A.
  Record the blocker and keep the 7.1 number.

Fast screen: 5 warmup + 20 measured. Formal 20+100 only if throughput gain
>= 2%.

### Correctness gates

Same as Phase 7.1/7.2: both ranks, forward/backward, finite `main_grad`,
optimizer consumes `main_grad`, finite loss, no deadlock. Compare A/B loss
trend; do not require bitwise identity.

### Explicitly not Phase 7.4

- Re-benchmarking native async All-Reduce (already on, 0% overlap).
- Sequence Parallel alone on the 7.1 host (memory experiment; already 7.2).
- `symmetric_ar_type`, Apex wgrad fusion, CUDA Graph, DP/PP/MoE overlap.
- Raising `CUDA_DEVICE_MAX_CONNECTIONS`.
- Starting a RunPod during Phase 7.3.

## Phase 7.4 measurement plan (when implemented)

Keep FAST ITERATION MODE. One Pod, stop immediately afterward. Save
`docs/experiments/phase7_tp_comm_overlap.md` and
`results/phase7_tp_comm_overlap.json`. Compare B to the 7.1 `709437d`
baseline and to the reproduced variant A on the same host.
