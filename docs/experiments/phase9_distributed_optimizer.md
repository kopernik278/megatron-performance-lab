# Phase 9.2: DP=2 Megatron Distributed Optimizer

FAST ITERATION MODE (5 warmup + 20 measured) unless a formal 20+100 cell is noted.
CUDA Graph off. TP=1, PP=1, DP=2. `overlap_grad_reduce=True` on all variants.
Default MCore bucket size; buckets not tuned.

Pinned API (Megatron-LM `09fde85`):
- `DistributedDataParallelConfig.use_distributed_optimizer` / `--use-distributed-optimizer`
- `overlap_grad_reduce` / `--overlap-grad-reduce`
- `overlap_param_gather` / `--overlap-param-gather` (requires distributed optimizer)

A: standard FP32Optimizer + gradient All-Reduce.
B/C: `DistributedOptimizer` + gradient Reduce-Scatter + parameter All-Gather.
C adds `overlap_param_gather=True`.

## Outcome

- Status: **success**
- Formal B/C 20+100: **yes**
- A→B throughput: **-0.39%**
- B→C throughput: **+3.42%**
- Gradient comm transform A→B: All-Reduce 8.2/step → RS 8.0/step + AG 8.0/step
- Param-gather overlap B→C: 28.91 → 5.56 ms exposed

## Infrastructure

- RunPod Pod: `q1bya9tpjltjxg` (deleted)
- Data center / public IP: None, `None`
- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, $0.88/h
- Topology path: **NODE**
- Lab commit: `677d54b2bc6a8e3258552940c01ccfa5d61dab9b`

## Correctness

- Variant A optimizer state sharded: False
- Variant B optimizer state sharded: True
- Variant C optimizer state sharded: True
- B optimizer bytes/rank: [1423679488, 1423679488]
- A optimizer bytes/rank: [2847360144, 2847360144]
- A↔B loss max abs delta (measured): 0.0000
- B↔C loss max abs delta (measured): 0.0000

## A/B/C throughput

| Variant | dist opt | param gather overlap | tok/s | step ms | per-GPU MFU | VRAM smi | opt state B/rank |
|---|---|---:|---:|---:|---:|---:|---:|
| A baseline | off | off | 28,827.25 | 1,136.70 | 23.33% | 39,327 | 2,847,360,144 |
| B dist opt | on | off | 28,714.07 | 1,141.18 | 23.24% | 39,673 | 1,423,679,488 |
| C dist opt + overlap | on | on | 29,696.09 | 1,103.44 | 24.03% | 39,673 | 1,423,679,488 |

## Communication transformation

- A All-Reduce launches/step: 8.25 (RS=0.00, AG=0.00)
- B Reduce-Scatter launches/step: 8.00 (AR=1.00, AG=8.00)
- C Reduce-Scatter launches/step: 8.00 (AG=8.00)

### Exposed communication (ms/step/GPU)

- Gradient exposed: A 14.32 → B 9.18
- Param-gather exposed: B 28.91 → C 5.56
- Overall overlap %: A 87.3% → B 61.8% → C 87.5%

## Memory

- A peak allocated MiB/rank: [32578.58349609375, 32578.58349609375]
- B peak allocated MiB/rank: [31221.15478515625, 31221.13720703125]
- C peak allocated MiB/rank: [31221.15478515625, 31221.13720703125]
- Optimizer state saving B vs A (per rank): +50.0%

## Formal B/C 20+100

- B: 28,581.28 tok/s
- C: 29,631.04 tok/s
- B→C: +3.67%

## Decision

Correctness passed. FAST B→C +3.42% met 2% gate; formal B/C 20+100 ran. A→B dist-opt effect -0.39%.

## Commands

```bash
bash scripts/phase92_distopt_pod.sh q1bya9tpjltjxg 0.88
```

Raw outputs: `results/phase92_work/`. Summary: `results/phase9_distributed_optimizer.json`.

