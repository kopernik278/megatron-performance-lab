# Phase 7.2: Sequence Parallel A/B

FAST ITERATION MODE. This run is an A/B of Sequence Parallel on the
corrected Phase 7.1 TP=2 topology (`709437d`), not a replacement of
that NODE+P2P baseline.

No other optimization was added. CUDA Graph remained off.
`bias_dropout_fusion=True` and `bias_gelu_fusion=False` were unchanged.

## Question

On 2x A40 with TP=2, does Megatron Sequence Parallel:

1. replace the ~96 per-layer All-Reduces,
2. replace them with All-Gather + Reduce-Scatter,
3. reduce activation memory,
4. improve communication volume/time,
5. improve throughput by at least 2%?

## Implementation verification (before the GPU run)

Pinned Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`:

- `TransformerConfig.sequence_parallel` is the runtime flag.
- `ColumnParallelLinear` with SP sets `allreduce_dgrad=False` and uses
  `reduce_scatter_to_sequence_parallel_region` on input dgrad.
- `RowParallelLinear` with SP sets `input_is_parallel=True` and uses
  `reduce_scatter_to_sequence_parallel_region` instead of
  `reduce_from_tensor_model_parallel_region` (All-Reduce).
- Hidden states after embedding are scattered along the sequence
  dimension (`scatter_to_sequence_parallel_region`).
- LayerNorm / RMSNorm require a fused CUDA kernel. The local spec
  `megatron/core/fusions/fused_layer_norm.py` asserts
  `not config.sequence_parallel` when Apex is missing and the code
  falls back to `WrappedTorchNorm`.

Apex is not installed in this lab image. Therefore both A and B used
Transformer Engine `TENorm` (`use_te_layernorm=True`) so that Sequence
Parallel can actually activate. This is a prerequisite, not an extra
fusion experiment. Both variants used the same LayerNorm implementation.

Runtime checks on variant B (passed):

- `config.sequence_parallel=True`
- QKV / FC1: `allreduce_dgrad=False`
- proj / FC2: `sequence_parallel=True`, `input_is_parallel=True`
- QKV / FC1 activation shape `[1024, 8, 1024]` = `[S/TP, B, H]`
- proj `[2048, 8, 512]`, FC2 `[2048, 8, 2048]` (full sequence, TP-sharded hidden)

Variant A kept the full-sequence QKV/FC1 shape `[2048, 8, 1024]`.

## Infrastructure

| Field | Value |
|---|---|
| Pod | `wtd9cxr3q8obuh` (`phase72-sequence-parallel`) |
| Host | `194.68.245.18` |
| GPUs | 2x NVIDIA A40 48GB, same host |
| Price | $0.88/hour |
| Datacenter | EU-SE-1 Secure Cloud |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Driver / CUDA | 570.195.03 / 12.8 |
| PyTorch / NCCL / cuDNN | 2.8.0+cu128 / 2.27.3 / 9.10.2 |
| Megatron | `09fde85ea25fb67e9b32019089fae163a3233bd3` |
| Transformer Engine | `2.17.1+4329ff84` |
| Topology | GPU0–GPU1 **PIX**, both NUMA node 0 |
| PCI | `0000:52:00.0`, `0000:53:00.0` |
| NVLink | none |
| CUDA peer access | true, but NCCL P2P hung (see below) |
| NCCL transport | `NCCL_P2P_DISABLE=1` (SHM/socket) |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 |
| Pod status after run | **EXITED** |

NCCL P2P on this PIX host deadlocked during the first `all_reduce`
(with and without `NCCL_CUMEM_ENABLE=0`). Collectives completed only
after `NCCL_P2P_DISABLE=1`. CUDA still reported peer access. This host
is therefore **not** the same interconnect path as Phase 7.1 NODE+P2P
(`7rpwv95a5j6axg`, commit `709437d`). Compare A vs B on this host.
Do **not** treat 7.2 A as a replacement of the 7.1 TP=2 number.

### Phase 7.1 valid TP=2 reference (discard earlier cross-NUMA)

From commit `709437d`, pod `7rpwv95a5j6axg`, same-NUMA **NODE**, P2P working:

| Metric | TP=2 SP=False |
|---|---:|
| Avg step | 825.12 ms |
| Tokens/sec | 19,856.48 |
| MFU | 16.07% |
| nvidia-smi VRAM/GPU | 20,980 MiB |
| All-Reduce / step | 101 |
| All-Gather / Reduce-Scatter | 0 / 0 |
| NCCL ms/step | 259.28 |
| Overlap | 0% |

The discarded earlier 7.1 result (pod `x72b8bn80zqdeg`, SYS/cross-NUMA)
is not used.

## Workload (identical except Sequence Parallel)

| Field | A | B |
|---|---|---|
| tensor_model_parallel_size | 2 | 2 |
| sequence_parallel | False | True |
| pipeline / context / expert parallel | 1 | 1 |
| layers / hidden / heads / seq / micro-batch | 24 / 1024 / 16 / 2048 / 8 | same |
| vocab | 32,000 | same |
| precision | BF16 params + FP32 master + GradBuffer | same |
| fused attention | `fused_attn` | same |
| bias_dropout_fusion | True | same |
| bias_gelu_fusion | False | same |
| CUDA Graph | off | off |
| LayerNorm | TENorm | TENorm |
| optimizer | Dist. Adam, foreach=False, overlap_grad_reduce=False | same |

Fast screen: 5 warmup + 20 measured. Formal 20+100 was not run because
the throughput gain was below 2%.

## Correctness

Passed on both ranks for both variants.

| Check | A | B |
|---|---|---|
| both ranks participated | yes | yes |
| forward + backward | yes | yes |
| `main_grad` finite | yes | yes |
| optimizer consumed `main_grad` | yes | yes |
| parameters updated | yes | yes |
| finite loss | yes | yes |
| deadlock / NCCL error | none | none |
| rank losses identical | yes (max |Δ|=0) | yes (max |Δ|=0) |

Smoke losses (3 steps):

| Step | A | B | \|A−B\| |
|---|---:|---:|---:|
| 1 | 11.034038 | 11.035116 | 0.00108 |
| 2 | 10.852051 | 10.854639 | 0.00259 |
| 3 | 10.718472 | 10.677570 | 0.04090 |

Losses are close and decrease, but they are not identical. Sequence
Parallel changes reduction order (All-Reduce vs Reduce-Scatter /
All-Gather) and activation sharding. Dropout RNG is also sensitive to
tensor layout. This is expected; it is not a bitwise match.

## Fast-screen throughput

Global tokens = `8 * 2048 * 2 GPUs` per step. Peak FLOP/s uses 2x A40
at 149.7 TFLOP/s BF16 each.

| Metric | A TP=2 SP=False | B TP=2 SP=True | Delta |
|---|---:|---:|---|
| Avg step (ms) | 1044.80 | 1142.02 | +97.22 ms |
| Median step (ms) | 1038.47 | 1149.82 | +111.35 ms |
| Tokens/sec | 15,681.52 | 14,346.54 | **−8.51%** |
| MFU | 12.69% | 11.61% | −1.08 pp |
| Allocated VRAM/GPU (MiB) | 16,795.63 | 14,831.82 | −1,963.81 |
| nvidia-smi peak VRAM/GPU (MiB) | 22,934 / 22,934 | 20,884 / 20,884 | **−2,050 (−8.94%)** |
| GPU utilization | 99.85% / 99.85% | 99.90% / 99.90% | ~flat |
| Speedup B/A |  |  | **0.9149x** |

Throughput **did not** improve. Gain −8.51% is below the 2% formal-run
threshold, so the fast-screen result is kept.

This host's TP=2 SP=False (15,682 tok/s) is also slower than the valid
7.1 NODE+P2P number (19,856 tok/s) because NCCL P2P had to be disabled.

## Nsight Systems (5 profiled steps)

Collective counts are NVTX shape events divided by TP=2 and by 5 steps.

| Collective | A count/step | A ms/step | B count/step | B ms/step |
|---|---:|---:|---:|---:|
| All-Reduce | 102 | 514.51 | 6 | 12.15 |
| All-Gather | 0 | 0.00 | 147 | 387.50 |
| Reduce-Scatter | 0 | 0.00 | 145 | 267.84 |
| Total NCCL | 102 | 514.51 | 298 | 667.49 |
| Overlap | 0% |  | 0% |  |
| Exposed communication | ≈514.51 ms |  | ≈667.49 ms |  |

Nsight range duration on this nsys version is 0, so overlap cannot be
measured from range timestamps. NVTX markers sit on the CPU launch
thread and do not overlap each other. The 0% figure is therefore an
instrumentation lower bound, not a proof that kernels never overlap.
`CUDA_DEVICE_MAX_CONNECTIONS=1` still serializes most compute/comm.

## Communication transformation

### Which All-Reduces were replaced

A has **102 All-Reduces/step**:

| Count | Shape | Role |
|---|---:|---|
| 97 | `[2048, 8, 1024]` (16,777,216 elems, 32 MiB) | per-layer TP + output dgrad |
| 1 | `[8, 2048, 1024]` | embedding hidden-state AR (`learned_absolute` position embedding does not scatter) |
| 3 | `[2048, 8]` | vocab-parallel cross-entropy |
| 1 | `[179083264]` | dummy DP GradBuffer (DP=1) |

The 97 large activations are:

- 24 layers × RowParallel `proj` output AR
- 24 layers × RowParallel `fc2` output AR
- 24 layers × ColumnParallel QKV input-dgrad AR (`allreduce_dgrad`)
- 24 layers × ColumnParallel FC1 input-dgrad AR
- 1 output-layer dgrad AR

That is the **96 per-layer All-Reduces** plus the output dgrad AR.

B keeps **6 All-Reduces/step**:

| Count | Shape | Role |
|---|---:|---|
| 3 | `[2048, 8]` | vocab-parallel cross-entropy (unchanged) |
| 1 | `[8, 2048, 1024]` | embedding hidden-state AR (unchanged; `reduce_scatter_embeddings=False`) |
| 1 | `[149504]` | small LayerNorm gradient AR |
| 1 | `[179083264]` | dummy DP GradBuffer |

**Replaced: 96.0 All-Reduces/step** (102 → 6). The 96 per-layer TP
All-Reduces and the output dgrad All-Reduce are gone.

### What they were replaced by

Megatron Sequence Parallel does **not** delete those synchronizations.
It splits each full-hidden All-Reduce into a Reduce-Scatter (forward
or dgrad) plus an All-Gather (the inverse path):

| Original TP=2 All-Reduce | Sequence Parallel replacement |
|---|---|
| RowParallel `proj` / `fc2` output AR | Reduce-Scatter along sequence (forward); All-Gather in backward |
| ColumnParallel QKV / FC1 input-dgrad AR | Reduce-Scatter of dgrad (backward); All-Gather of activations (forward) |
| Output-layer dgrad AR | Reduce-Scatter |
| Hidden-state replication after embedding | `scatter_to_sequence_parallel_region` (no extra AR; backward All-Gathers) |

Measured B collectives:

- All-Gather: **147 / step**, all shape `[1024, 8, 1024]` (local sequence shard, 16 MiB)
- Reduce-Scatter: **145 / step**
  - 97 × `[2048, 8, 1024]` (32 MiB input, scattered to 16 MiB)
  - 48 × `[1024, 8, 1024]` (16 MiB)

The 97 large Reduce-Scatters match the 97 large A All-Reduces
(96 per-layer + output dgrad). The 48 extra Reduce-Scatters of the
sharded shape, and the All-Gathers beyond a 1:1 swap, come from the
SP mappings (forward All-Gather into column-parallel GEMMs, backward
All-Gather out of row-parallel GEMMs, embedding scatter backward, and
TENorm sequence-parallel mappings).

### Volume and time

Payload estimate (sum of NVTX tensor bytes / TP / steps):

| | A | B | B/A |
|---|---:|---:|---:|
| MiB / step / rank | 3,478.29 | 3,486.28 | 1.002 |
| NCCL ms/step | 514.51 | 667.49 | **1.297** |

Algorithmic volume of an All-Reduce on 2 ranks is about the same as
Reduce-Scatter + All-Gather (each moves ~½ the buffer per direction).
The measured payload is essentially unchanged. Time got **worse**
because this host has no NVLink, P2P is disabled, and SP issues ~3×
more collective launches (102 → 298) of smaller messages. Launch
overhead and poorer SHM/socket efficiency dominate.

### Activation memory

QKV/FC1 activations drop from `[2048, 8, 1024]` to `[1024, 8, 1024]`
per rank. nvidia-smi peak VRAM fell **2,050 MiB/GPU (−8.94%)**.
Allocated bytes fell 1,964 MiB. Sequence Parallel did reduce
activation memory, as designed.

### Throughput impact

Tokens/sec **fell 8.51%**. The VRAM win does not help a workload that
already fit in 48 GB. Extra collectives on a PIX + P2P-disabled path
increased step time. `CUDA_DEVICE_MAX_CONNECTIONS=1` also prevents
hiding those extra communications.

## Answers

1. **Which All-Reduces are replaced?** The 96 per-layer TP All-Reduces
   (proj, fc2, QKV dgrad, FC1 dgrad) plus the output dgrad All-Reduce.
   Embedding AR, vocab CE ARs, and the dummy DP AR remain.
2. **Replaced by what?** All-Gather + Reduce-Scatter along the sequence
   dimension (Megatron `mappings.py` / linear layers).
3. **Volume/time?** Volume ≈ unchanged (ratio 1.002). Time **worse**
   (NCCL 514.51 → 667.49 ms/step, +29.7%).
4. **Activation memory?** Yes: −2,050 MiB/GPU nvidia-smi (−8.94%).
5. **Throughput?** **No.** −8.51% (0.9149x). Formal 20+100 not run.

## Limitations

- NCCL P2P was disabled on this PIX host. 7.2 A is not comparable to
  7.1 NODE+P2P as an absolute baseline.
- Nsight overlap is an NVTX lower bound (0 ns ranges).
- Loss A vs B is not bitwise identical.
- Apex fused LayerNorm is unavailable; TENorm was required for SP.
- Traces stayed on the pod (`profiles/phase72_work/`) and were not
  committed.

## Conclusion

Sequence Parallel is **correct and active** on this TP=2 A40 pair. It
replaces the 96 per-layer All-Reduces with All-Gather / Reduce-Scatter,
cuts about 2 GiB of activation memory per GPU, and does **not** improve
throughput here. Keep the fast-screen result. Do not enable Sequence
Parallel on this 2x A40 PIX / P2P-disabled topology as a speedup.
