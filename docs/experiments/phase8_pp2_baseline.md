# Phase 8.1: 2-GPU Pipeline Parallel baseline

FAST ITERATION MODE (5 warmup + 20 measured). CUDA Graph stayed off.
TP=1, PP=2, DP=1. Tensor parallel was not combined with pipeline parallel.

## Tensor-parallel communication-overlap conclusions (closed)

- **AG-only Userbuffers is accepted** and gives **+6.75%** host-local throughput
  (formal 20+100: 27,209 → 29,045 tok/s) versus TE Linear + Sequence Parallel
  without Userbuffers. Flags: `tp_comm_overlap=True`, `tp_comm_overlap_ag=True`,
  `tp_comm_overlap_rs=False`, `tp_comm_overlap_rs_dgrad=False`,
  `tp_comm_bulk_dgrad=False`, `tp_comm_bulk_wgrad=False`.
- **bulk-dgrad demonstrates real AG/GEMM overlap (~91.5%)** via
  `userbuffers_fp16_sum_inplace_gpu_rw_ag`, but is **slower than AG-only**
  (+4.0% vs the same B reference, 28,312 tok/s).
- **Reduce-Scatter Userbuffers remains disabled** because it livelocks on A40
  PCIe in `userbuffers_fp16_sum_inplace_gpu_rr_rs_oop`. This phase did **not**
  continue debugging RS Userbuffers.

## Infrastructure and topology

- RunPod Pod: `3i7ehf4o13hbp0` (deleted)
- Data center / public IP: CA-MTL-1, `69.30.85.97`
- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, $0.88/h (≤ $0.90/h)
- Topology path: **NV4**, same-NUMA=True
- CUDA peer access bidirectional: True
- NCCL All-Reduce sanity: True
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Lab commit on the Pod: `cdcbe4665694ab506250ee20bef45ed55221639e`
- PyTorch: 2.8.0+cu128; CUDA: 12.8; NCCL: 2.27.3
- Driver: 580.159.03
- `NCCL_P2P_DISABLE` unset. No TP+PP mix.

## Pipeline partitioning

24 Transformer layers split evenly across PP=2:

- rank 0 / PP 0: 12 layers (global 1–12), embedding=True, output/loss=False/False, params=204,763,136 (embed 53,608,448, decoder 151,154,688, output 0)
- rank 1 / PP 1: 12 layers (global 13–24), embedding=False, output/loss=True/True, params=202,668,032 (embed 0, decoder 151,156,736, output 51,511,296)

Embedding lives on the first stage. Output layer and loss live on the last stage.
Word embeddings are **untied** on PP=2 (`share_embeddings_and_output_weights=False`) so the embedding group does not all-reduce during 1F1B. Compute stays BF16 autocast; pipeline send/recv uses FP32 (`pipeline_dtype=torch.float32`) because BF16 P2P with this FP32-param model produced all-NaN last-stage losses.

## Correctness

- Smoke (3 steps) on PP=1 and every PP=2 microbatch count: passed
- Forward, backward, `main_grad`, optimizer, finite loss, and no deadlock all passed.
- Schedule: `forward_backward_pipelining_without_interleaving` (Megatron 1F1B without interleaving for PP=2).

## Microbatch sweep (constant global batch = 8, 16384 tokens/step)

| Config | μbatches | μbatch size | tok/s | step ms | MFU | VRAM/GPU (smi) | theoretical bubble | measured idle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PP=1 reference | 1 | 8 | 16198.02 | 1011.48 | 26.22% | 41804 | 0% | n/a |
| PP=2 μb=1 | 1 | 8 | 16387.01 | 999.82 | 13.26% | 26832 | 50.0% | 2.63% |
| PP=2 μb=2 | 2 | 4 | 19963.53 | 820.70 | 16.16% | 21276 | 33.3% | 3.02% |
| PP=2 μb=4 | 4 | 2 | 21183.07 | 773.45 | 17.14% | 15468 | 20.0% | 5.81% |
| PP=2 μb=8 | 8 | 1 | 20570.76 | 796.47 | 16.65% | 12636 | 11.1% | 26.93% |

## Why few microbatches are inefficient

With 1 microbatch, Megatron still uses `forward_backward_pipelining_without_interleaving`, but there is almost no 1F1B steady state. Rank 0 must finish its forward and send activations before rank 1 can start, then rank 1's backward must return before rank 0 can backward. Theoretical fill/drain bubble is 50.0%. Nsys GPU-idle is only 2.63% because NCCL send/recv wait still shows the GPU as busy. Too few microbatches serialize the two stages.

## How more microbatches reduce the bubble

Increasing microbatches (holding global batch=8) fills the pipeline: warmup is still (PP-1) forwards, but a 1F1B steady state appears. Theoretical bubble falls from 50.0% at 1 μb to 11.1% at 8 μb. Throughput moved from 16387 to 20571 tok/s.

## Diminishing returns

Relative tok/s gains between successive doubling of microbatches: 1→2 21.83%, 2→4 6.11%, 4→8 -2.89%. Diminishing returns begin at 8 microbatches (<5% additional gain).

## Does PP=2 help throughput on this small model?

PP=2 improves throughput to 21183 tok/s from PP=1 16198 tok/s (1.308x, 65.4% of ideal 2x). Per-GPU VRAM drops from 41804 to 15468 MiB, so PP still also helps memory capacity.

## Best PP=2 result

- Best throughput: **21183.07 tok/s** (μbatches=4, μbatch=2)
- Step time: 773.45 ms
- MFU: 17.14%
- VRAM/GPU: 15468 MiB
- PP scaling vs same-host PP=1: 1.308x (65.4% of ideal 2x)
- Mean P2P send/recv: 69.04 ms/step/GPU
- Measured nsys GPU idle: 5.81% (NCCL wait often looks busy, so this is **not** the pipeline bubble; theoretical fill/drain at 4 μb is 20.0%)

**Primary bottleneck:** Remaining pipeline idle plus activation P2P. The 355.9M model is small enough that layer compute per stage is only moderately larger than send/recv.

## Commands

```bash
bash scripts/phase8_pp2_pod.sh 3i7ehf4o13hbp0 0.88
```

Raw outputs: `results/phase81_work/`. Summary: `results/phase8_pp2_baseline.json`.

