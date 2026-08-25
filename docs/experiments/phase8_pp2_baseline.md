# Phase 8.1: 2-GPU Pipeline Parallel baseline

FAST ITERATION MODE (5 warmup + 20 measured). CUDA Graph stays off.
TP=1, PP=2, DP=1. Tensor parallel is not combined with pipeline parallel.

GPU measurements are pending. This document will be overwritten by
`scripts/phase8_analyze_pp.py` after the 2x A40 pod run.

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
  PCIe in `userbuffers_fp16_sum_inplace_gpu_rr_rs_oop`. This phase does **not**
  continue debugging RS Userbuffers.

## Planned configuration

- Same 355.9M GPT: 24 layers, hidden 1024, FFN 4096, seq 2048, vocab 50304
- BF16 autocast, fused attention, `bias_dropout_fusion=True`,
  `bias_gelu_fusion=False`, CUDA Graph off
- Even PP split: rank 0 layers 1–12 + embedding; rank 1 layers 13–24 + output/loss
- Megatron `forward_backward_pipelining_without_interleaving` (1F1B)
- Constant global batch 8 (16384 tokens/optimizer step):
  `(num_microbatches, micro_batch_size) = (1,8), (2,4), (4,2), (8,1)`
- Same-host PP=1 reference on one A40 for scaling efficiency

## Commands

```bash
bash scripts/phase8_pp2_pod.sh <pod-id> 0.88
```
