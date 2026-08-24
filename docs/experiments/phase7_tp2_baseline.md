# Phase 7.1: 2-GPU tensor-parallel baseline

## Outcome

The corrected Megatron-Core TP=2, PP=1, DP=1 baseline reached 19,856.48
tokens/s versus 15,694.76 tokens/s for TP=1. This is a 1.2652x speedup and
63.26% two-GPU scaling efficiency.

Nsight Systems measured 259.28 ms of NCCL kernel time per step per GPU. The
steady-state workload executes 101 All-Reduces per step, including 97 large
activation collectives, with no measurable communication/compute overlap.
CUDA Graph remained disabled and no communication optimization was attempted.

## Corrected configuration gate

A follow-up pinned-source audit found that the first screening run initialized a
TP=2 process group while leaving
`TransformerConfig.tensor_model_parallel_size` at its default of 1. This is
invalid for `TEDotProductAttention`, which consumes the config value. Those
measurements were discarded.

The final harness explicitly passes the requested TP size into
`TransformerConfig` and fails unless all three values agree:

- distributed world size;
- Megatron tensor-parallel process-group size;
- `TransformerConfig.tensor_model_parallel_size`.

Both final TP=2 ranks reported a config and process-group TP size of 2.

## Infrastructure and environment

- RunPod Pod: `7rpwv95a5j6axg`
- Allocation: one Secure Cloud Pod with 2x NVIDIA A40 48GB
- Price: $0.88/hour total, below the $0.90/hour target
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Driver: 570.211.01
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- NCCL: 2.27.3
- cuDNN: 9.10.2
- Transformer Engine: 2.17.1+4329ff84
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- CUDA Graph: disabled (`cuda_graph_impl=none`)

The pinned framework/runtime stack was unchanged. The replacement physical host
provided a newer patch release of the same 570 driver branch. The Pod was
stopped immediately after artifact download.

## Phase A: topology and NCCL sanity

`nvidia-smi topo -m` reported `NODE` between GPU0 (`98:00.0`) and GPU1
(`D6:00.0`). Both GPUs are attached to NUMA node 1; traffic traverses PCIe and
PCIe host bridges within that NUMA node. No NVLink was active.

CUDA peer access was available in both directions. A two-rank NCCL scalar
All-Reduce returned the expected value 3.0 on both ranks, with one-shot
latencies of 1.061 and 1.080 ms. No NCCL error occurred.

## Phase B: TP implementation and shards

The model uses `get_gpt_layer_local_spec` plus the existing TE fused core
attention replacement. Runtime checks verified real Megatron TP modules:

| Component | Megatron module | Per-rank TP=2 weight shape |
|---|---|---:|
| Word embedding | `VocabParallelEmbedding` | `[25152, 1024]` |
| Attention QKV | `ColumnParallelLinear` | `[1536, 1024]` |
| Attention output projection | `RowParallelLinear` | `[1024, 512]` |
| MLP FC1 | `ColumnParallelLinear` | `[2048, 1024]` |
| MLP FC2 | `RowParallelLinear` | `[1024, 2048]` |
| Tied output weight | `ColumnParallelLinear` using embedding weight | `[25152, 1024]` |

Both ranks reported identical local shapes. The vocabulary and output weights
remained tied.

## Phase C: correctness

Three smoke steps passed for TP=1 and TP=2:

- both TP ranks initialized;
- forward, backward, gradient finalization, and optimizer update completed;
- FP32 `main_grad` remained finite and retained stable addresses;
- `FP32Optimizer` consumed `main_grad`;
- parameters changed;
- TP=2 losses were identical between ranks;
- no deadlock or NCCL error occurred.

TP=1 losses were 11.030653, 10.855515, and 10.757502. TP=2 losses were
11.031303, 10.854897, and 10.767609. The maximum absolute TP1/TP2 difference
was 0.010107 after three updates. Exact identity is not expected because TP
changes reduction and dropout RNG ordering.

## Phase D: 5+20 fast benchmark

Both variants used micro-batch 8 and sequence length 2048.

| Metric | TP=1 | TP=2 |
|---|---:|---:|
| Average step time | 1043.92 ms | 825.12 ms |
| Median step time | 1044.00 ms | 823.68 ms |
| Global throughput | 15,694.76 tokens/s | 19,856.48 tokens/s |
| Aggregate MFU | 25.40% | 16.07% |
| Peak allocated VRAM/GPU | 32,550.47 MiB | 19,803.63 MiB |
| Peak reserved VRAM/GPU | 33,128 MiB | 20,126 MiB |
| Peak `nvidia-smi` VRAM/GPU | 34,698 MiB | 20,980 / 20,980 MiB |
| Average GPU utilization | 99.91% | 99.84% / 99.87% |

- TP=2 speedup: **1.2652x**
- throughput gain: **26.52%**
- scaling efficiency: **63.26%**
- TP=2 `nvidia-smi` VRAM reduction: approximately **39.5% per GPU**

GPU utilization includes NCCL kernels; it is not equivalent to useful compute
utilization.

## Phase E: communication profile

The short trace contained five steady-state TP=2 steps. The exact NVTX
collective count is 101 All-Reduces per step:

| Collective | Logical calls/step | NCCL ms/step/GPU |
|---|---:|---:|
| All-Reduce | 101 | 259.28 |
| All-Gather | 0 | 0 |
| Reduce-Scatter | 0 | 0 |

One raw kernel crossed a profile-window boundary, producing 1,011 GPU kernel
records. The logical count uses the exact NCCL NVTX tensor-shape events instead
of dividing that boundary-contaminated count.

The 101 All-Reduces map to:

- 24 attention output-projection forward collectives;
- 24 MLP FC2 forward collectives;
- 24 QKV dgrad collectives;
- 24 MLP FC1 dgrad collectives;
- 1 tied output-layer dgrad collective;
- 1 vocabulary-embedding forward collective;
- 3 vocabulary-parallel cross-entropy collectives.

The first 97 use 16,777,216-element BF16 activations, or 32 MiB each. The
embedding collective contains the same element count in FP32, or 64 MiB. Each
cross-entropy collective has shape `[2048, 8]`, or 64 KiB in FP32.

Timing and overlap:

- NCCL kernel time: 259.28 ms/step/GPU
- NCCL share of the uninstrumented TP=2 step: approximately 31.4%
- communication/compute overlap: 0.0%
- exposed communication: 259.22 ms/step/GPU
- NCCL protocol: `RING_LL`

Approximate NVTX timing attribution:

- attention TP forward: 61.25 ms/step/GPU;
- MLP TP forward: 61.28 ms/step/GPU;
- backward TP: 125.73 ms/step/GPU;
- vocabulary embedding: 10.38 ms/step/GPU.

## Bottleneck conclusion

TP=2 provides a useful 26.52% throughput improvement and large memory savings,
but scaling is limited to 63.26%. The dominant distributed bottleneck is 97
large, fully exposed activation All-Reduces over the same-NUMA `NODE` PCIe path.
The backward path is the largest contributor. All-Gather and Reduce-Scatter are
absent because sequence parallelism and the distributed optimizer remain
disabled.

## Artifacts

- Machine-readable result: `results/phase7_tp2_baseline.json`
- Benchmark: `scripts/phase7_tp_run.py`
- Topology/NCCL sanity: `scripts/phase7_topology.py`
- Analyzer: `scripts/phase7_analyze_tp.py`
- Nsight report on stopped Pod:
  `profiles/phase71_work/tp2_communication.nsys-rep`
- Nsight report SHA-256:
  `2987dbafe07272ea610f1ca75dc075fc8b661fd6ee1a64c799ce0adfc733bb78`
- Nsight SQLite SHA-256:
  `598412111054fa4022e281920c41af048b56552a01c6fec702a6ce17b996fc1e`
