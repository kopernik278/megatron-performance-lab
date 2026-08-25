# Phase 9.1: DP=2 gradient All-Reduce overlap

FAST ITERATION MODE (5 warmup + 20 measured) unless a formal 20+100 cell is noted.
CUDA Graph stayed off. Distributed optimizer stayed off. TP=1, PP=1. Bucket size was
the MCore default (`max(40000000, 1000000 * dp_size)`); buckets were not tuned.

Pinned API (Megatron-LM `09fde85`): `DistributedDataParallelConfig.overlap_grad_reduce`
and CLI `--overlap-grad-reduce`. With `use_distributed_optimizer=False` this issues
gradient **All-Reduce**, not ReduceScatter. `overlap_param_gather` stayed false.

A→B isolates gradient-communication overlap. Weak scaling vs DP=1 is reported separately.

## Outcome

- Status: **success**
- Formal 20+100: **yes**
- A→B throughput: **+5.59%** (27,404.74 → 28,936.38 tok/s)
- Overlap (B, DP comm ∩ compute): **88.2%**
- Exposed DP comm: 49.59 → 11.84 ms/step/GPU
- Dominant remaining bottleneck: non-communication step time (compute / optimizer); remaining exposed DP All-Reduce

## Infrastructure

- RunPod Pod: `h4l1752oob32wv` (deleted)
- Data center / public IP: CA-MTL-1, `69.30.85.75`
- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, $0.88/h
- Topology path: **NODE**, same-NUMA=True
- CUDA peer access bidirectional: True
- NCCL All-Reduce sanity: True
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Lab commit on the Pod: `2d85f174cb7853d829cc63867f4b9cb2a4c89884`
- `CUDA_DEVICE_MAX_CONNECTIONS=8`. `NCCL_P2P_DISABLE` unset.

## Correctness (variant B)

- Both ranks initialized: True
- Different data per DP rank: True
- Forward/backward/`main_grad`/optimizer: True/True/True/True
- Gradients synchronized: True
- Parameters identical after optimizer: True
- Finite loss / no NaN-Inf / no deadlock: True/True/True

## DP=1 vs DP=2 throughput

- Accepted DP=1 (published, Phase 5.2 formal B (bias_dropout_fusion=True, MB=8, DP=1)): **15,801.94 tok/s**
- Same-host DP=1 FAST: **15,052.36 tok/s**, 1,088.47 ms, MFU 24.36%
- DP=2 A (overlap off): **27,404.74 tok/s**
- DP=2 B (overlap on): **28,936.38 tok/s**
- Weak-scaling efficiency vs accepted DP=1 (A): 86.71%
- Weak-scaling efficiency vs accepted DP=1 (B): 91.56%
- Weak-scaling efficiency vs same-host DP=1 (A): 91.03%
- Weak-scaling efficiency vs same-host DP=1 (B): 96.12%

Weak-scaling efficiency = DP2 global tok/s / (2 × DP1 tok/s).

## A/B performance

| Variant | overlap_grad_reduce | tok/s | step ms | per-GPU MFU | VRAM smi |
|---|---|---:|---:|---:|---:|
| A DP=2 overlap off | False | 27,404.74 | 1,195.71 | 22.18% | 39,323 |
| B DP=2 overlap on | True | 28,936.38 | 1,132.42 | 23.42% | 39,323 |

## Gradient communication

- Named All-Reduce launches/step: 1.25 → 8.00
- DP comm launches/step (AllReduce+SendRecv): 1.25 → 8.00
- DP comm time: 49.59 → 99.93 ms/step/GPU
- Comm during backward: 0.00 → 38.33 ms/step/GPU
- Comm during finalize: 0.00 → 0.65 ms/step/GPU
- Overlap %: 0.0% → 88.2%
- Exposed comm: 49.59 → 11.84 ms/step/GPU
- Bucket count A/B: 1 / 8
- Effective bucket size A/B: None / 40000000

### Buckets (B)

- buffer 0 bucket 0: 41986048 unpadded elems, 41986048 padded, 167944192 bytes, 40 params, dtype=torch.float32
- buffer 0 bucket 1: 41987072 unpadded elems, 41987072 padded, 167948288 bytes, 38 params, dtype=torch.float32
- buffer 0 bucket 2: 40939520 unpadded elems, 40939520 padded, 163758080 bytes, 40 params, dtype=torch.float32
- buffer 0 bucket 3: 43035648 unpadded elems, 43035648 padded, 172142592 bytes, 42 params, dtype=torch.float32
- buffer 0 bucket 4: 41987072 unpadded elems, 41987072 padded, 167948288 bytes, 38 params, dtype=torch.float32
- buffer 0 bucket 5: 40939520 unpadded elems, 40939520 padded, 163758080 bytes, 40 params, dtype=torch.float32
- buffer 0 bucket 6: 43035648 unpadded elems, 43035648 padded, 172142592 bytes, 42 params, dtype=torch.float32
- buffer 0 bucket 7: 62009344 unpadded elems, 62009344 padded, 248037376 bytes, 12 params, dtype=torch.float32

## Formal 20+100

- A: 27,329.34 tok/s, 1,199.00 ms
- B: 28,828.92 tok/s, 1,136.64 ms
- A→B: +5.49%

## Decision

Correctness passed. FAST gain +5.59% met the 2% gate; formal 20+100 ran. Overlap_grad_reduce is the isolated A→B variable.

## Commands

```bash
bash scripts/phase9_dp_pod.sh h4l1752oob32wv 0.88
```

Raw outputs: `results/phase91_work/`. Summary: `results/phase9_dp2_grad_overlap.json`.

