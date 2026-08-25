# Phase 8.2: Pipeline-parallel P2P communication overlap design

This is a source-and-results-only design phase. No Pod was started, no GPU
experiment was run, and no model or environment code was changed.

The performance baseline is the Phase 8.1 PP=2 result from commit `ebd98618`.
Do not use a BF16 `pipeline_dtype` path. Phase 8.1 established that BF16
pipeline send/recv is numerically invalid for this FP32-param + autocast-BF16
harness (all-NaN last-stage losses). Pipeline communication stays FP32.

Flag names below are the fields on pinned Megatron-LM
`09fde85ea25fb67e9b32019089fae163a3233bd3`. They were read from
`ModelParallelConfig`, `schedules.py`, `p2p_communication.py`, and
`megatron/training/arguments.py`. Names were not assumed from docs.

## Decision

The single Phase 8.3 experiment should enable **native interleaved 1F1B with
`overlap_p2p_comm`** on the Phase 8.1 best cell (TP=1, PP=2, DP=1, global
batch=8, 4 microbatches, CUDA Graph off, FP32 pipeline dtype).

```python
config = TransformerConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=2,  # 6 layers per virtual chunk
    microbatch_group_size_per_vp_stage=2,    # default = PP; valid for M=4
    overlap_p2p_comm=True,
    batch_p2p_comm=False,                    # mutually exclusive with overlap
    overlap_p2p_comm_warmup_flush=False,     # leave off for 8.3
    batch_p2p_sync=True,                     # unused while batch_p2p_comm=False
    use_ring_exchange_p2p=False,
    deallocate_pipeline_outputs=False,
    defer_embedding_wgrad_compute=False,
    pipeline_dtype=torch.float32,            # do not switch to BF16
    cuda_graph_impl="none",
    bias_dropout_fusion=True,
    bias_activation_fusion=False,
    share_embeddings_and_output_weights=False,  # keep embeddings untied
)
```

Required launch/runtime companions, not extra optimizations:

- `parallel_state.initialize_model_parallel(..., virtual_pipeline_model_parallel_size=2)`
- two `GPTModel` chunks per rank with `vp_stage in {0, 1}` and
  `pre_process` / `post_process` set as in Megatron `get_model()`
- `get_forward_backward_func()` must dispatch
  `forward_backward_pipelining_with_interleaving`
- `model` and `data_iterator` passed as length-2 lists
- `CUDA_DEVICE_MAX_CONNECTIONS=8` (Phase 8.1 deadlock with `=1`; do not drop to 1)
- NCCL P2P left enabled; abort if the host is not same-NUMA with working P2P

Do not enable `overlap_p2p_comm_warmup_flush`, `batch_p2p_comm`,
`use_ring_exchange_p2p`, `defer_embedding_wgrad_compute`, CUDA Graph,
bias-GELU fusion, TP, Sequence Parallel, or BF16 `pipeline_dtype`.

On this source, **PP=2 cannot isolate overlap from interleaving**. Non-interleaved
1F1B raises if `overlap_p2p_comm=True`. Interleaved PP=2 without overlap is
rejected by training-args validation and is a deadlock hazard even if the lab
bypasses argparse. Phase 8.3 is therefore one coupled native mechanism, not a
flag flip on the 8.1 schedule.

## Baseline to beat

Phase 8.1, commit `ebd98618`, pod `3i7ehf4o13hbp0` (deleted), CA-MTL-1 NV4
same-NUMA, P2P working, CUDA Graph off, non-interleaved 1F1B:

| Field | Value |
|---|---|
| Best cell | TP=1 PP=2 DP=1, 24 layers 12/12, global batch=8, μbatches=4, μbatch=2 |
| Throughput | 21,183.07 tokens/s |
| Average step | 773.45 ms |
| MFU | 17.14% |
| VRAM/GPU | 15,468 MiB |
| Theoretical fill/drain bubble | 20.0% = `(PP-1)/(M+PP-1)` |
| Mean activation send/recv | 69.04 ms/step/GPU |
| Measured nsys GPU idle | 5.81% (NCCL wait looks busy; **not** the bubble) |
| Schedule | `forward_backward_pipelining_without_interleaving` |
| `overlap_p2p_comm` | False |
| `batch_p2p_comm` | False (batched P2P hung during 8.1 bring-up) |
| `pipeline_dtype` | `torch.float32` |
| Embeddings | untied on PP=2 |
| CUDA Graph | off |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 8 |

Activation payload for that cell: shape `[seq, μbatch, hidden] = [2048, 2, 1024]`,
FP32, **16 MiB** per send/recv. Non-interleaved PP=2 issues one forward hop and
one backward hop per microbatch (4+4 = 8 tensors/GPU/step, 128 MiB).

Closed TP-overlap conclusions stay closed: AG-only Userbuffers is the accepted
TP result; RS Userbuffers stays disabled on A40 PCIe. Phase 8.3 is PP-only.

## Pinned software

- Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine `2.17.1+4329ff84` (`4329ff84bfbdaa778a33cba02a15fb0807c64689`)
- PyTorch 2.8.0+cu128, CUDA 12.8, NCCL 2.27.3
- Lab harness: `scripts/phase8_pp_run.py` builds one `GPTModel` per rank and
  calls `get_forward_backward_func()` with a non-list model

## How Megatron dispatches pipeline schedules

[`megatron/core/pipeline_parallel/schedules.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/pipeline_parallel/schedules.py)
`get_forward_backward_func()`:

| Condition | Function |
|---|---|
| `pp_size == 1` | `forward_backward_no_pipelining` |
| `pp_size > 1` and `vp_size is None` | `forward_backward_pipelining_without_interleaving` |
| `pp_size > 1` and `vp_size is not None` | `forward_backward_pipelining_with_interleaving` |

`vp_size` is `parallel_state.get_virtual_pipeline_model_parallel_world_size()`,
set only when `initialize_model_parallel(..., virtual_pipeline_model_parallel_size=N)`
is called with `N is not None`. Setting the config field alone is not enough.

Training-args validation in
[`megatron/training/arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/training/arguments.py)
lines 992–1010 (the lab does not use argparse, but this is the intended contract):

- If `virtual_pipeline_model_parallel_size is None`: **force**
  `overlap_p2p_comm = False` and warn that non-interleaved 1F1B does not
  support overlapping P2P.
- If VPP is set and `overlap_p2p_comm` is True: require `PP > 1`.
- If VPP is set and `overlap_p2p_comm` is False: require `PP > 2` “to avoid
  having multiple p2p sends and recvs between same 2 ranks per communication
  batch”. **PP=2 + interleaved + overlap off is forbidden.**

CLI name: `--no-overlap-p2p-communication` is `store_false` on dest
`overlap_p2p_comm` (argparse default True, then forced off unless interleaved).
Core config default is `overlap_p2p_comm: bool = False`. The lab must set the
`TransformerConfig` fields explicitly; argparse defaults do not apply.

`argument_utils.py` sets `kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm`.
Overlap and batched P2P are mutually exclusive.

## Non-interleaved 1F1B P2P behavior (Phase 8.1, no compute overlap)

`forward_backward_pipelining_without_interleaving` raises if overlap is requested:

```2181:2184:megatron/core/pipeline_parallel/schedules.py
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )
```

Warmup (`num_warmup = PP - rank - 1`; rank 0 has 1 warmup forward at PP=2):

1. blocking `recv_forward`
2. forward compute
3. blocking `send_forward`

Steady 1F1B:

1. forward compute
2. blocking `send_forward_recv_backward` (activation send + gradient recv with next rank)
3. backward compute
4. blocking `send_backward_recv_forward` except last iteration, which is `send_backward`

Cooldown: blocking `recv_backward` → backward → blocking `send_backward`.

Those send/recv helpers always call `_communicate(..., wait_on_reqs=True)`
(the default). They never pass `overlap_p2p_comm=True`. Communication is
exposed relative to the local stage’s compute.

For PP=2, next rank equals previous rank. `_p2p_ops` then splits one of the
two ops onto `torch.distributed.group.WORLD` so the paired send and recv can
proceed without NCCL same-communicator deadlock. That overlaps **send with
recv of the same exchange**, not P2P with GEMM.

## Interleaved 1F1B / virtual pipeline stages

Config: `ModelParallelConfig.virtual_pipeline_model_parallel_size: Optional[int] = None`.

Each PP rank holds `VPP` model chunks. Layers are interleaved, not contiguous.
Pinned `transformer_block.get_num_layers_to_build` documents the 24-layer
PP=2 VPP=2 assignment analogously to its 8-layer example:

| Rank | Chunk 0 (vp_stage=0) | Chunk 1 (vp_stage=1) |
|---|---|---|
| PP 0 | layers 1–6 | layers 13–18 |
| PP 1 | layers 7–12 | layers 19–24 |

Embedding is only on PP0 / vp0 (`pre_process`). Output/loss is only on PP1 /
vp1 (`post_process`). Intermediate chunks are transformer-only.

`num_layers_per_pipeline_rank` (12) must be divisible by VPP. Legal VPP values
for this 24-layer PP=2 model are therefore `{2, 3, 4, 6, 12}`.
`microbatch_group_size_per_vp_stage` defaults to PP=2, independent of VPP.
Recommend VPP=2 only (6 layers per chunk; fewest extra P2P hops).

Warmup from `get_pp_rank_microbatches` when VPP is set:

```text
total_num_microbatches = M * VPP
num_warmup = (PP - rank - 1) * 2
             + (VPP - 1) * microbatch_group_size_per_vp_stage
```

Default `microbatch_group_size_per_vp_stage = PP = 2`. For M=4, VPP=2:

| Rank | total virtual μb | warmup | remaining 1F1B |
|---|---:|---:|---:|
| 0 | 8 | 4 | 4 |
| 1 | 8 | 2 | 6 |

Constraints in the interleaved function:

- `microbatch_group_size_per_vp_stage` must lie in `[PP, M]` → `[2, 4]`
- remainder `M % microbatch_group_size_per_vp_stage` must be 0 or `>= PP`

M=4, group size=2 satisfies both. Rank 0 still has a 1F1B steady state.
M=2 would make rank-0 warmup=4 and `total=4`, so **all microbatches are
warmup** (`are_all_microbatches_in_warmup=True`) and there is no overlap
steady state. Keep M=4.

Interleaving **increases P2P hops**. Forward path for VPP=2 is
PP0-chunk0 → PP1-chunk0 → PP0-chunk1 → PP1-chunk1: **three** activation
transfers per microbatch instead of one. PP=2 is a ring, so hop 2 is
rank1 → rank0. Backward mirrors that. Payload per tensor stays 16 MiB FP32,
but bytes/step rise by roughly 2–3× if every hop is exposed.

Theoretical fill/drain bubble shrinks from
`(PP-1)/(M+PP-1) = 20%` to about `(PP-1)/(M*VPP+PP-1) = 11.1%`
(Megatron interleaved-1F1B factor-of-VPP reduction). That bubble change is
coupled to the schedule, not to the overlap flag.

## Available overlap mechanisms

### 1. `overlap_p2p_comm` (native; recommended)

1. **Config/API:** `TransformerConfig.overlap_p2p_comm=True`. Requires
   `batch_p2p_comm=False`. Requires VPP (`virtual_pipeline_model_parallel_size`
   not None) or `forward_backward_pipelining_without_interleaving` raises.
   On PP=2, VPP also **requires** this flag (training-args contract; 4-way
   batched send/recv to the same peer otherwise).
   Communicator API: `P2PCommunicator.send_forward_recv_forward(..., overlap_p2p_comm=True)`
   and `send_backward_recv_backward(..., overlap_p2p_comm=True)` pass
   `wait_on_reqs=(not overlap_p2p_comm)` and return wait handles.
2. **Where implemented:**
   - Config: `megatron/core/model_parallel_config.py` (`overlap_p2p_comm`).
   - Schedule: `forward_backward_pipelining_with_interleaving` steady-state
     `pp_pre_forward` / `pp_post_forward` / `pp_pre_backward` / `pp_post_backward`
     (`schedules.py` ~1747–1898).
   - Transport: `p2p_communication.py` `_communicate` + `_p2p_ops`
     (`torch.distributed.isend` / `irecv`, not `batch_isend_irecv`).
3. **What Send/Recv it overlaps:** After a virtual-chunk **forward**, async
   `send_next` of activations and `recv_prev` of the next forward input.
   After a virtual-chunk **backward**, async `send_prev` of input gradients
   and `recv_next` of the next backward grad. Handles are waited at the
   **start of the next** forward or backward that needs that tensor
   (`pp_pre_forward` / `pp_pre_backward`), and previous send handles are
   waited before overwriting the send buffer.
4. **What computation can hide it:** The other virtual chunk on the same GPU
   in the 1F1B pair: 6-layer forward hides the previous backward P2P; 6-layer
   backward hides the previous forward P2P. Warmup/cooldown stay mostly
   synchronous unless mechanism 2 is also on.
5. **Requires interleaved scheduling:** **Yes.** Non-interleaved raises.
6. **PP=2 / 24-layer compatibility:** **Yes**, with VPP=2 (6/6 layers per
   rank) or VPP=4 (3/3/3/3). VPP=2 is the low-risk split. M=4 is enough for
   a rank-0 1F1B window. Harness must build two chunks; today’s
   `phase8_pp_run.py` cannot.
7. **Effect on the ~69 ms P2P cost:** The 69 ms is non-interleaved 1-hop
   traffic. Interleaving adds hops, so **issued** P2P time can grow even if
   **exposed** P2P falls. Steady-state hops can hide behind ~half-stage
   compute (6 layers). Warmup/cooldown hops stay exposed in this
   configuration. Do not expect the nsys send/recv timer to drop to ~0;
   expect some of it to move under compute NVTX.
8. **Effect on pipeline bubble:** Interleaving itself cuts theoretical
   fill/drain from 20% to ~11%. Overlap does not remove fill/drain; it hides
   P2P that currently inflates the wait inside those bubbles and 1F1B.
   Nsys GPU-idle will still under-report bubble.
9. **Implementation complexity:** Medium for this lab (chunked model build,
   list data iterators, DDP/optimizer over two modules, schedule assert),
   low in Megatron (native flags). No custom kernels.
10. **Correctness/deadlock risks:** Moderate. PP=2 same-peer send/recv is
    the reason overlap is mandatory with VPP. `_p2p_ops` uses WORLD vs PP
    group when `group.size()==2`. Phase 8.1 already needed
    `CUDA_DEVICE_MAX_CONNECTIONS=8`; keep it. Handle-wait bugs corrupt the
    next stage if `deallocate_pipeline_outputs` is on (it stays off).
    Finite-loss / `main_grad` / no-deadlock gates from 8.1 still apply.
    Bitwise loss match with 8.1 is not required (layer assignment changes).

### 2. `overlap_p2p_comm_warmup_flush`

1. **Config/API:** `overlap_p2p_comm_warmup_flush=True`. CLI:
   `--overlap-p2p-communication-warmup-flush`. `__post_init__` raises unless
   `overlap_p2p_comm` is True **and** `batch_p2p_comm` is False.
2. **Where implemented:** Same interleaved function, warmup loop ~1576–1728
   and cooldown ~1977–2058. Prefetches `send_forward_recv_forward` /
   `send_backward_recv_backward` with `overlap_p2p_comm=True` and waits the
   recv handle just before the corresponding compute.
3. **What Send/Recv:** Warmup forward send/recv and cooldown backward
   send/recv, which mechanism 1 leaves mostly blocking.
4. **Hidden compute:** Warmup forwards and cooldown backwards of the current
   virtual chunk, with recv prefetched one iteration ahead.
5. **Requires interleaved scheduling:** Yes (and mechanism 1).
6. **PP=2 / 24-layer:** Legal on the same VPP=2 cell. Extra async traffic on
   the two-rank ring.
7. **Effect on ~69 ms:** Could hide the fill/drain portion of P2P, which is
   a larger fraction at PP=2 M=4 than at large M. Second-order after
   mechanism 1.
8. **Effect on bubble:** Does not change the schedule length; may hide P2P
   that currently sits in warmup/cooldown.
9. **Complexity:** Low as a flag, higher deadlock surface (prefetch recv
   buffers, extra wait-handle queues, asserts that handles exist).
10. **Risks:** Higher than mechanism 1. Prefetch + PP=2 same-peer ops is
    exactly the class of bug 8.1 hit with batched P2P. **Leave False in
    8.3.** Only consider later if interleaved+overlap is clean and nsys
    still shows exposed warmup/flush P2P.

### 3. `batch_p2p_comm` (not compute overlap)

1. **Config/API:** `batch_p2p_comm=True` (core default True; 8.1 harness
   False). Must be False if `overlap_p2p_comm` is True. Interleaved path
   raises `ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")`.
2. **Where implemented:** `_communicate` selects `_batched_p2p_ops` →
   `torch.distributed.batch_isend_irecv`. `wait_on_reqs` is asserted True.
   Optional `batch_p2p_sync` then `torch.cuda.synchronize()`.
3. **What Send/Recv:** Packs the isend/irecv of one `_communicate` call into
   one batched group. Still waits before returning to compute.
4. **Hidden compute:** None. If anything, `batch_p2p_sync` **prevents**
   overlap by device-synchronizing after the batch.
5. **Requires interleaved scheduling:** No. Used by non-interleaved 1F1B
   when True.
6. **PP=2 / 24-layer:** 8.1 **hung** with `batch_p2p_comm=True` on this
   two-rank ring. Individual `_p2p_ops` is the working PP=2 path.
7. **Effect on ~69 ms:** Not a hide; possible hang. Do not re-enable.
8. **Effect on bubble:** None if it runs; deadlock if it does not.
9. **Complexity:** Flag only.
10. **Risks:** Known hang on this harness. **Reject.**

### 4. Individual async send/recv (`wait_on_reqs=False`)

This is not a separate user-facing flag. It is the transport used by
mechanism 1/2. Default `_communicate(wait_on_reqs=True)` is synchronous.
Only `send_forward_recv_forward` / `send_backward_recv_backward` expose
`overlap_p2p_comm` and return handle dicts keyed `send_next`, `recv_prev`,
`send_prev`, `recv_next`.

Do not call `_communicate` with `wait_on_reqs=False` from the lab harness.
There is no supported non-interleaved async API.

### 5. Interleaved 1F1B without `overlap_p2p_comm`

1. **Config/API:** `virtual_pipeline_model_parallel_size=2` and
   `overlap_p2p_comm=False`. Steady state uses
   `send_forward_backward_recv_forward_backward` (four-way communicate:
   send_next, send_prev, recv_prev, recv_next in one `_communicate`).
2. **Where implemented:** Interleaved `else:  # No p2p overlap` branch
   ~1900–1949.
3. **What Send/Recv:** Combined forward+backward exchange, waited.
4. **Hidden compute:** None versus local F/B; some send/recv pairing.
5. **Requires interleaved scheduling:** Yes.
6. **PP=2 / 24-layer:** Training args require `PP > 2`. On PP=2, prev==next,
   so one batch contains two sends and two recvs with the **same** peer.
   That is the documented deadlock case.
7. **Effect on ~69 ms:** Extra hops, no hide. Likely hang.
8. **Effect on bubble:** Interleaving still shrinks theoretical bubble, but
   the run is not a valid 8.3 cell on PP=2.
9. **Complexity:** Same chunked harness as mechanism 1.
10. **Risks:** Deadlock. **Cannot be the A/B “overlap off” cell on PP=2.**
    Variant A must remain Phase 8.1 **non-interleaved**.

### 6. `use_ring_exchange_p2p`

1. **Config/API:** `use_ring_exchange_p2p=True`.
2. **Where implemented:** `_communicate` calls `torch.distributed.ring_exchange`.
3. **Send/Recv:** Custom ring kernel, not NCCL isend/irecv.
4. **Hidden compute:** Not the Megatron 1F1B overlap path.
5. **Requires interleaved scheduling:** No.
6. **PP=2 / 24-layer:** Requires a custom PyTorch with `ring_exchange`.
   Stock 2.8.0+cu128 does not provide it.
7–10. **Reject.** Unavailable.

### 7. `defer_embedding_wgrad_compute`

1. **Config/API:** `defer_embedding_wgrad_compute=True` plus
   `wgrad_deferral_limit`. `__post_init__` requires `PP > 1` and
   `gradient_accumulation_fusion=True`.
2. **Where implemented:** Last-stage embedding/output wgrad is buffered
   during 1F1B and drained in `finish_embedding_wgrad_compute` after cooldown
   (`schedules.py`, `gpt_model.py`, `tensor_parallel/layers.py`).
3. **What it overlaps:** Embedding/output **wgrad GEMM** with pipeline
   **flush**, not activation P2P with layer compute.
4. **Hidden compute:** Output-layer wgrad during cooldown. This model’s
   embeddings are **untied**; last stage owns the output layer only.
5. **Requires interleaved scheduling:** No.
6. **PP=2 / 24-layer:** Fusion needs Apex `fused_weight_gradient_mlp_cuda`,
   which this image does not have (Phase 7.3). Config construction would
   raise.
7. **Effect on ~69 ms:** None. Wrong bottleneck.
8. **Effect on bubble:** Might hide some cooldown on the last stage only.
9. **Complexity:** Apex + fusion, outside PP P2P.
10. **Risks:** Unavailable fusion; not the 8.3 question. **Reject.**

### 8. Combined 1F1B / MoE A2A overlap

1. **Config/API:** `overlap_moe_expert_parallel_comm=True` selects
   `combined_1f1b_schedule_for_interleaved_pipelining` (and a no-PP variant).
2. **Where implemented:** `megatron/core/pipeline_parallel/combined_1f1b.py`.
   Docstring: called only if `overlap_moe_expert_parallel_comm` is true.
3. **What it overlaps:** MoE expert-parallel All-to-All with combined F+B,
   not PP activation P2P.
4–10. Dense GPT, EP=1. **Out of scope. Reject.**

### 9. `deallocate_pipeline_outputs` / `variable_seq_lengths` / `batch_p2p_sync`

| Field | Role | 8.3 |
|---|---|---|
| `deallocate_pipeline_outputs` | Frees `.data` after send; memory only. Requires waiting send handles if overlap is on. | Keep False |
| `variable_seq_lengths` | Extra shape send/recv every step | Keep False |
| `batch_p2p_sync` | `cuda.synchronize` after `batch_isend_irecv` | Irrelevant while `batch_p2p_comm=False` |

None of these overlap P2P with compute.

### 10. BF16 `pipeline_dtype`

Out of scope. Phase 8.1: BF16 P2P on this harness is numerically invalid.
Keep `pipeline_dtype=torch.float32`. Do not A/B dtypes in 8.3.

## Recommended Phase 8.3 experiment

Exactly one experiment: **native interleaved 1F1B + `overlap_p2p_comm`**
on a Phase 8.1-class 2x A40 host.

### Variant A (reproduce 8.1 best cell)

- TP=1, PP=2, DP=1, VPP unset
- 24 layers contiguous 12/12
- global batch=8, M=4, μbatch=2
- `overlap_p2p_comm=False`, `batch_p2p_comm=False`
- `pipeline_dtype=torch.float32`
- CUDA Graph off, embeddings untied, `CUDA_DEVICE_MAX_CONNECTIONS=8`
- schedule: `forward_backward_pipelining_without_interleaving`
- same-NUMA NODE, NCCL P2P enabled

### Variant B (the overlap mechanism)

Identical host, software, architecture, sequence=2048, global batch=8, M=4,
precision, optimizer, fusions, CUDA Graph off, FP32 pipeline dtype, then:

```python
virtual_pipeline_model_parallel_size = 2
microbatch_group_size_per_vp_stage = 2
overlap_p2p_comm = True
batch_p2p_comm = False
overlap_p2p_comm_warmup_flush = False
pipeline_dtype = torch.float32
```

```text
CUDA_DEVICE_MAX_CONNECTIONS=8
```

```python
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=2,
)
```

Build two chunks per rank, matching Megatron `training.get_model()`:

```python
for vp_stage in range(2):
    pre_process = is_pp_first_stage(pp_group) and is_vp_first_stage(vp_stage, 2)
    post_process = is_pp_last_stage(pp_group) and is_vp_last_stage(vp_stage, 2)
    GPTModel(..., vp_stage=vp_stage, pre_process=pre_process, post_process=post_process)
```

`get_num_layers_to_build(config, vp_stage=vp_stage)` must see VPP so each
chunk has 6 layers. Pass `model=[chunk0, chunk1]` and a matching list of
microbatch iterators into `forward_backward_pipelining_with_interleaving`.

Assert `get_forward_backward_func().__name__ == "forward_backward_pipelining_with_interleaving"`.
Assert `config.overlap_p2p_comm is True` and `config.batch_p2p_comm is False`.

### Isolation caveat

B changes the **schedule, layer placement, P2P hop count, and overlap**
together. That is the only native P2P-overlap path in this commit. A
non-interleaved overlap cell does not exist. An interleaved-without-overlap
cell is illegal on PP=2. If B is not at least 2% faster than A, keep the
fast screen and do not add `overlap_p2p_comm_warmup_flush` or VPP=4 in 8.3.

### Expected timeline transformation

```text
A (Phase 8.1 non-interleaved 1F1B, 12 layers/stage):
  warmup:  Recv? -> [12-layer F] -> Send          # blocking P2P
  1F1B:    [12-layer F] -> SendF+RecvB -> [12-layer B] -> SendB+RecvF
  cooldown: RecvB -> [12-layer B] -> SendB
  ~8 x 16 MiB tensors/GPU/step, ~69 ms send/recv, 20% theoretical bubble

B (VPP=2 interleaved + overlap_p2p_comm, 6 layers/chunk):
  warmup:  mostly blocking F-send/recv as today (warmup_flush off)
  1F1B:    wait prev RecvF -> [6-layer F] -> isend F / irecv F
           wait prev RecvB -> [6-layer B] -> isend B / irecv B
           (P2P issued at post_forward/post_backward; waited at next pre_*)
  cooldown: blocking B-send/recv (warmup_flush off)
  3 forward hops/μbatch instead of 1; steady-state hops hide under the
  sibling chunk's 6-layer F or B
```

Nsight must report send/recv counts and ms/step, overlap vs compute NVTX,
and that GPU-idle is still not the bubble. Runtime logs must print VPP,
chunk layer counts/offsets, `overlap_p2p_comm`, schedule name, and
`pipeline_dtype`.

### Topology constraints

- Exactly one Pod, exactly 2x A40, same host, prefer same-NUMA NODE (NV4 as in 8.1).
- CUDA peer access must be true. Leave NCCL P2P enabled.
- Stop immediately if topology is SYS/cross-NUMA or if NCCL P2P hangs.
- Do not combine TP. Do not set `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Do not start RunPod during Phase 8.2.

### Expected performance benefit

This is an estimate, not a measured result. Never treat it as a Phase 8.3 number.

Upper bound if the 69 ms were fully hidden **and** fill/drain fell 20% → 11%
on the 773 ms step, ignoring extra hops: about 69 ms + ~0.09×773 ms ≈ 140 ms,
~18% step-time. Unrealistic: warmup/cooldown P2P stay exposed, interleaving
adds hops, and 8.1 nsys idle was only 5.81% so much of the “bubble” is already
NCCL-wait, not SM-idle.

Conservative A40 PCIe case: extra hops cost tens of ms; overlap hides a
fraction of steady-state P2P behind 6-layer compute. Net **−5% to +10%**
tokens/s versus 21,183 tok/s is the honest window. A slowdown is an allowed
outcome if exposed P2P grows more than bubble shrinks.

Failure case: deadlock on the two-rank ring, NaN loss, or hang in
`isend`/`irecv` wait handles. Record the blocker and keep the 8.1 number.
Do not “fix” it by enabling `batch_p2p_comm` or BF16 pipeline dtype.

Fast screen: 5 warmup + 20 measured. Formal 20+100 only if throughput gain
>= 2%.

### Correctness gates

Same as Phase 8.1: both ranks, forward/backward, finite `main_grad`, optimizer
consumes `main_grad`, finite last-stage loss, no deadlock. Compare A/B loss
trend; do not require bitwise identity (chunked layers change reduction order).

Confirm partitioning:

- PP0 vp0: 6 layers, embedding=True
- PP0 vp1: 6 layers, embedding=False, output=False
- PP1 vp0: 6 layers, embedding=False, output=False
- PP1 vp1: 6 layers, output/loss=True

### Explicitly not Phase 8.3

- Enabling `overlap_p2p_comm` on non-interleaved 1F1B (raises).
- Interleaved PP=2 with overlap off (deadlock / args-forbidden).
- `overlap_p2p_comm_warmup_flush` (second experiment only if B is clean).
- `batch_p2p_comm=True` (8.1 hang).
- `use_ring_exchange_p2p`, combined 1F1B, embedding-wgrad deferral.
- VPP=4 (more hops) as the first cell.
- BF16 `pipeline_dtype`.
- TP, Userbuffers, CUDA Graph, bias-GELU fusion.
- Starting a RunPod during Phase 8.2.

## Phase 8.3 measurement plan (when implemented)

Keep FAST ITERATION MODE. One Pod, stop immediately afterward. Save
`docs/experiments/phase8_pipeline_overlap.md` and
`results/phase8_pipeline_overlap.json`. Compare B to the 8.1 `ebd98618`
baseline and to the reproduced variant A on the same host.
