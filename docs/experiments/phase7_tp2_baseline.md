# Phase 7.1: 2-GPU tensor-parallel baseline

## Outcome

Phase 7.1 established a correct Megatron-Core TP=2, PP=1, DP=1 training
baseline, but it did not speed up this 355.9M-parameter model. TP=2 sustained
15,447.37 tokens/s versus 15,685.31 tokens/s for TP=1, a 0.9848x speedup
(-1.52%) and 49.24% two-GPU scaling efficiency.

The dominant bottleneck is 101 All-Reduce collectives per step over a
cross-NUMA `SYS` PCIe path with no active NVLink. Nsight Systems measured
493.28 ms of NCCL kernel time per step per GPU and no communication/compute
overlap. CUDA Graph remained disabled, and no communication optimization was
attempted.

## Infrastructure and pinned environment

- RunPod Pod: `x72b8bn80zqdeg`
- Allocation: exactly one Secure Cloud Pod with 2x NVIDIA A40 48GB
- Price: $0.88/hour total, below the $0.90/hour target
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Driver: 570.195.03
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- NCCL: 2.27.3
- cuDNN: 9.10.2
- Transformer Engine: 2.17.1+4329ff84
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- CUDA Graph: disabled (`cuda_graph_impl=none`)

The Pod was stopped after the experiment. The Nsight report and SQLite export
remain on its stopped container disk.

## Phase A: topology and NCCL sanity

`nvidia-smi topo -m` reported `SYS` between GPU0 (`57:00.0`, NUMA 0) and GPU1
(`CE:00.0`, NUMA 1). This means peer traffic traverses PCIe and the CPU
inter-socket/NUMA interconnect. `nvidia-smi nvlink --status` reported all NVLink
links inactive.

CUDA peer access was available in both directions:

```text
       GPU0  GPU1
GPU0   true  true
GPU1   true  true
```

A two-rank NCCL scalar All-Reduce returned the expected sum of 3.0 on both
ranks. Observed one-shot latencies were 0.875 ms on rank 0 and 0.764 ms on
rank 1. There were no NCCL errors.

## Phase B: TP implementation and shard verification

The model uses `get_gpt_layer_local_spec`. Runtime type checks verified real
Megatron TP modules rather than two independent replicas:

| Component | Megatron module | Per-rank TP=2 weight shape |
|---|---|---:|
| Word embedding | `VocabParallelEmbedding` | `[25152, 1024]` |
| Attention QKV | `ColumnParallelLinear` | `[1536, 1024]` |
| Attention output projection | `RowParallelLinear` | `[1024, 512]` |
| MLP FC1 | `ColumnParallelLinear` | `[2048, 1024]` |
| MLP FC2 | `RowParallelLinear` | `[1024, 2048]` |
| Tied output weight | `ColumnParallelLinear` using embedding weight | `[25152, 1024]` |

Both ranks reported identical local shapes. The vocabulary/output weight
remained tied to the sharded input embedding.

## Phase C: correctness smoke test

The MCore lifecycle from Phase 6.3 was retained:

1. `DistributedDataParallel.zero_grad_buffer()`
2. `FP32Optimizer.zero_grad()`
3. BF16-autocast forward and loss
4. backward into persistent FP32 `main_grad`
5. `finalize_model_grads`
6. `FP32Optimizer.step()` over `main_grad`

Three smoke steps passed for TP=1 and TP=2:

- both TP ranks initialized;
- forward and backward completed;
- every checked `main_grad` was finite and retained its address;
- the optimizer consumed `main_grad`;
- parameters changed;
- losses remained finite and identical between TP=2 ranks;
- no deadlock or NCCL error occurred.

TP=1 smoke losses were 11.030653, 10.855543, and 10.754660. TP=2 losses were
11.031303, 10.854847, and 10.768642. The maximum absolute TP1/TP2 loss
difference was 0.013983 after three updates. Exact equivalence is not expected
because TP changes floating-point reduction order; the comparison establishes
matching finite behavior rather than bitwise identity.

## Phase D: fast benchmark

Both variants used micro-batch 8, sequence length 2048, 5 warmup steps, and 20
measured steps on the same Pod.

| Metric | TP=1 | TP=2 |
|---|---:|---:|
| Average step time | 1044.54 ms | 1060.63 ms |
| Median step time | 1044.66 ms | 1060.58 ms |
| Global throughput | 15,685.31 tokens/s | 15,447.37 tokens/s |
| Aggregate MFU | 25.39% | 12.50% |
| Peak allocated VRAM/GPU | 32,550.47 MiB | 19,803.63 MiB |
| Peak reserved VRAM/GPU | 33,128 MiB | 20,126 MiB |
| Peak `nvidia-smi` VRAM/GPU | 34,698 MiB | 20,980 / 20,980 MiB |
| Average GPU utilization | 99.90% | 99.87% / 99.91% |

Calculated results:

- TP=2 speedup: **0.9848x**
- throughput delta: **-1.52%**
- scaling efficiency: **49.24%**
- TP=2 VRAM reduction: approximately **39.5% per GPU** by `nvidia-smi`

GPU utilization includes NCCL kernels and therefore does not imply productive
compute utilization. TP=2 halves the major weight shards and VRAM footprint, but
communication more than consumes the saved compute time.

## Phase E: Nsight Systems communication profile

The short profile contained five steady-state TP=2 steps. Profiling
instrumentation increased median step time from 1060.58 ms to 1105.52 ms, so
the trace is used for attribution rather than benchmark throughput.

### Collective inventory

The steady-state trace contained only inter-GPU All-Reduce:

| Collective | Logical calls/step | NCCL ms/step/GPU |
|---|---:|---:|
| All-Reduce | 101 | 493.28 |
| All-Gather | 0 | 0 |
| Reduce-Scatter | 0 | 0 |

All-Gather and Reduce-Scatter are absent because sequence parallelism and the
distributed optimizer are disabled. Initialization-time object All-Gathers were
outside the profiled training window.

NVTX shapes, NCCL INFO counts, and MCore semantics map the 101 All-Reduces:

- 24 attention row-parallel output-projection forward collectives;
- 24 MLP FC2 row-parallel forward collectives;
- 24 QKV column-parallel dgrad collectives;
- 24 MLP FC1 column-parallel dgrad collectives;
- 1 tied output-layer column-parallel dgrad collective;
- 1 FP32 vocabulary-embedding forward collective;
- 3 FP32 vocabulary-parallel cross-entropy collectives.

The first 97 operate on 16,777,216-element activations. The transformer
collectives use BF16, or 32 MiB each; the embedding collective uses FP32, or
64 MiB. Each cross-entropy collective has shape `[2048, 8]`, or 16,384 FP32
elements (64 KiB).

### Time and overlap

- NCCL kernel time: 493.28 ms/step/GPU
- NCCL share of the uninstrumented TP=2 step: approximately 46.5%
- communication/compute overlap: 0.0%
- exposed communication: 493.28 ms/step/GPU
- ring protocol: `RING_LL`

Approximate NVTX launch attribution was:

- attention TP forward: 123.55 ms/step/GPU;
- MLP TP forward: 115.59 ms/step/GPU;
- backward TP: 243.95 ms/step/GPU;
- vocabulary embedding: 9.09 ms/step/GPU;
- output/loss and residual attribution: less than 1 ms/step/GPU.

The forward All-Reduces are synchronous at row-parallel boundaries. The
backward collectives also showed no measurable kernel overlap with compute in
this configuration.

## Bottleneck conclusion

The first TP baseline is communication-bound. This relatively small model has
shorter per-rank GEMMs under TP=2 but still executes 97 large activation
All-Reduces every step. Those collectives run over a `SYS` cross-NUMA PCIe path,
have no NVLink transport, and are fully exposed. The resulting 493 ms/step/GPU
communication cost explains why TP=2 is 1.52% slower than TP=1 despite using
substantially less memory.

This phase establishes the baseline only. Sequence parallelism, collective
overlap, topology-aware placement, communication fusion, and distributed
optimizer changes were intentionally not enabled.

## Artifacts

- Machine-readable result: `results/phase7_tp2_baseline.json`
- Benchmark implementation: `scripts/phase7_tp_run.py`
- Topology/NCCL sanity: `scripts/phase7_topology.py`
- Trace analyzer: `scripts/phase7_analyze_tp.py`
- Nsight report on stopped Pod:
  `profiles/phase71_work/tp2_communication.nsys-rep`
- Nsight SQLite SHA-256:
  `984a8809f0eb89578428e2cf1d01602002be3cec749157764288d172a136adb4`
