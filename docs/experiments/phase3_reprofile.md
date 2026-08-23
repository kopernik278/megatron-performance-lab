# Phase 3.4 Deep Nsight Systems Reprofile (Fused Attention MB=8)

## Summary

Phase 3.4 re-profiled the current best configuration (fused TE attention, micro-batch
8) with a 15-step steady-state Nsight Systems capture after a separate 100-step sanity
benchmark. NCU was not attempted (Phase 3.0 confirmed hardware counters unavailable on
RunPod Secure A40).

Sanity throughput and MFU match Phase 3.3 within tolerance. The bottleneck has migrated
from attention-dominated (Phase A, ~49% kernel time) to a GEMM + copy/cast + activation
mix at MB=8. Optimizer kernel share is now ~5% after micro-batch amortization.

**Recommended next optimization (not implemented):** reduce FP32/BF16 cast and layout-copy
overhead at Megatron local linear boundaries.

## Infrastructure

- Pod: `3ixl2btmmwghn5`, Secure Cloud, 1× NVIDIA A40 48GB, `$0.44/hour`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Driver/Python: `570.195.03` / `3.12.3`
- PyTorch/CUDA/NCCL/cuDNN: `2.8.0+cu128` / `12.8` / `2.27.3` / `9.10.2`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine: `2.17.1+4329ff84` (built with `NVTE_WITH_NCCL_EP=0`)
- Nsight Systems: bundled `nsys` under `/opt/nvidia/nsight-compute/2025.1.1/...`
- Pod status after run: **STOPPED**

## Sanity Benchmark (100 steps, no Nsight overhead)

| Metric | Measured | Target | Within ~5% / tolerance |
| --- | ---: | ---: | --- |
| Tokens/s | 15,088.17 | 15,084.70 | Yes (+0.02%) |
| MFU | 24.42% | 24.42% | Yes |
| Avg step (ms) | 1,085.88 | ~1,086.13 (Phase 3.3) | Yes |
| Peak VRAM (MiB) | 32,632 | ~32,632 (Phase 3.3) | Yes |

Artifact: `results/phase34_sanity_mb8_run.json`

## Profile Run (15 steps under Nsight)

| Metric | Value |
| --- | ---: |
| Tokens/s (instrumented) | 15,094.76 |
| MFU | 24.43% |
| Avg / median step (ms) | 1,085.41 / 1,085.29 |
| GPU idle (profile window) | 0.51% |
| Kernels/step | 4,414 |
| Kernels &lt;50 µs (count share) | 51.04% |

Artifacts: `profiles/phase34_mb8_reprofile.nsys-rep` (on pod volume),
`results/phase34_mb8_profile_run.json`, `results/phase3_reprofile.json`

## Current Bottleneck Breakdown (Phase C, kernel-time shares)

Categories are non-exclusive; denominator is total CUDA kernel execution time.

| Category | Share | ms/step |
| --- | ---: | ---: |
| GEMM / matmul | 28.27% | 304.4 |
| Activation / elementwise | 27.47% | 295.8 |
| Copy / cast | 23.78% | 256.0 |
| Attention (incl. cuDNN SDPA) | 13.22% | 142.3 |
| Normalization | 6.62% | 71.3 |
| Optimizer | 4.93% | 53.1 |

GPU-visible idle gaps: **0.51%** of profile window. Async memcpy profile-window share:
**0.0004%** (negligible vs kernel time).

## Top 15 Kernels by Total GPU Time

| Rank | Share | ms/step | Kernel (abbreviated) |
| ---: | ---: | ---: | --- |
| 1 | 15.00% | 161.6 | `bfloat16_copy` vectorized elementwise |
| 2 | 9.44% | 101.7 | cuDNN SDPA bprop (WMMA) |
| 3 | 7.48% | 80.5 | `direct_copy` / load-store with cast |
| 4 | 6.17% | 66.5 | CUTLASS BF16 GEMM 256×128 TN + ReLU |
| 5 | 5.36% | 57.7 | Ampere BF16 GEMM 128×128 NN |
| 6 | 4.92% | 53.0 | CUTLASS BF16 GEMM 128×256 NT + ReLU |
| 7 | 3.32% | 35.8 | vectorized `add` (residual) |
| 8 | 3.28% | 35.3 | Ampere BF16 GEMM 128×128 TN |
| 9 | 3.19% | 34.4 | GELU backward |
| 10 | 2.82% | 30.4 | elementwise `add` |
| 11 | 2.63% | 28.4 | Ampere BF16 GEMM 128×256 NT |
| 12 | 2.53% | 27.2 | cuDNN SDPA fprop (WMMA) |
| 13 | 2.45% | 26.4 | reduce (sum) |
| 14 | 2.45% | 26.3 | layer norm grad input |
| 15 | 2.16% | 23.3 | elementwise `add` (nocast) |

## Forward / Backward / Optimizer (NVTX)

GPU-projected NVTX ranges (`nvtx_gpu_proj_sum`):

| Range | GPU-projected ms/step |
| --- | ---: |
| Forward | 348.5 |
| Optimizer step | 54.8 |
| Train step (avg projected) | 1,083.7 |

The explicit `backward` NVTX range shows negligible GPU projection because autograd
work is attributed to nested `train_step_*` and operator ranges rather than the outer
`backward` push/pop. Wall-clock NVTX summary (`nvtx_sum`) reports backward at
~559 ms/step — most backward GPU work is real but not top-level tagged.

Synchronization APIs in the capture window: `cudaDeviceSynchronize` (32 calls,
0.40 ms total), `cudaStreamSynchronize` (15 calls, 0.07 ms total). CPU launch API
median: **6.8 µs**; GPU-visible inter-kernel idle median: **0.77 µs**.

## Phase Comparison (A → B → C)

| Phase | Config | tok/s | MFU | Attn | GEMM | Optimizer | Copy/cast | Kernels/step | &lt;50 µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Unfused MB=1 | — | 6.01% | 49.4% | 17.2% | 10.8% | — | 4,768 | 66.6% |
| B | Fused MB=1 | 9,763 | 15.80% | 10.9% | 20.9% | 26.1% | — | 4,361 | 73.0% |
| C | Fused MB=8 | 15,095 | 24.43% | 13.2% | 28.3% | 4.9% | 23.8% | 4,414 | 51.0% |

## Bottleneck Migration

1. **Attention** fell from ~49% (A) to ~11–13% (B/C) after cuDNN FusedAttention.
2. **GEMM** rose to ~28% at MB=8 as matmuls became larger and better utilized.
3. **Copy/cast** remains ~24% combined (`bfloat16_copy` alone is still the #1 kernel).
4. **Optimizer** dropped from ~26% (B, MB=1) to ~5% (C) — fixed ~53 ms/step AdamW
   amortized over 8× tokens.
5. **GPU idle** stayed low (&lt;1.3% across phases); the limiter is kernel mix, not
   launch gaps.

## Top 3 Remaining Bottlenecks

1. **GEMM / matmul (28.3%)** — dominant compute; CUTLASS/Ampere symbols fill ranks 4–8
   and 11 after copy kernels.
2. **Copy / cast (23.8%)** — `bfloat16_copy` + `direct_copy` are ranks 1 and 3; FP32
   params with BF16 autocast drive dtype traffic at linear boundaries.
3. **Attention (13.2%)** — cuDNN SDPA fprop/bprop still visible but no longer dominant;
   further attention swaps have diminishing returns vs copy + GEMM.

(Activation/elementwise at 27.5% overlaps GEMM/backward work in classification; it
is reported separately but is not the primary optimization target.)

## Recommended Next Optimization

**Test next:** Eliminate FP32/BF16 cast and layout-copy overhead at Megatron local
linear boundaries (keep FP32 parameters; cast Q/K/V once or use BF16-native linears
where correctness allows).

**Why highest expected payoff:** Copy/cast kernels still hold the largest single-kernel
share (~15% for `bfloat16_copy` alone). Phase A copy-related work was ~26% of kernel
time; Phase C still shows ~24% in the `copy_cast` category. A measured 30–50% reduction
in copy time could yield high single-digit to low double-digit throughput gains before
hitting pure GEMM roofline limits — without changing the mathematical model and without
duplicating completed attention fusion.

**Not recommended next:**

- MB=16 (predicted OOM ~60 GB)
- FlashAttention before copy/GEMM fixes
- Fused AdamW alone (optimizer already ~5% at MB=8)

## Commands

Sanity (100 steps):

```bash
PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase3_microbatch_run.py --micro-batch-size 8 \
  --warmup-iterations 20 --measured-iterations 100 \
  --output-json results/phase34_sanity_mb8_run.json
```

15-step Nsight profile:

```bash
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
"$NSYS" profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=process-tree --cpuctxsw=process-tree \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=true --force-overwrite=true \
  --output=profiles/phase34_mb8_reprofile \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase3_microbatch_run.py --micro-batch-size 8 \
  --warmup-iterations 20 --measured-iterations 15 \
  --profile-mode --output-json results/phase34_mb8_profile_run.json

"$NSYS" export --type sqlite --force-export=true \
  --output profiles/phase34_mb8_reprofile.sqlite profiles/phase34_mb8_reprofile.nsys-rep
"$NSYS" stats --report nvtx_gpu_proj_sum --format csv --force-export=true \
  --output profiles/phase34_nvtx_projection profiles/phase34_mb8_reprofile.nsys-rep

python scripts/phase3_analyze_reprofile.py \
  --sqlite profiles/phase34_mb8_reprofile.sqlite \
  --run-metrics results/phase34_mb8_profile_run.json \
  --nvtx-projection-csv profiles/phase34_nvtx_projection_nvtx_gpu_proj_sum.csv \
  --trace profiles/phase34_mb8_reprofile.nsys-rep \
  --pod-id 3ixl2btmmwghn5 \
  --output results/phase3_reprofile.json
```

## Artifacts

- `results/phase3_reprofile.json` — full structured analysis
- `results/phase34_sanity_mb8_run.json` — 100-step sanity benchmark
- `results/phase34_mb8_profile_run.json` — 15-step instrumented profile metrics
- `profiles/phase34_mb8_reprofile.nsys-rep` — trace on pod volume (not committed)
