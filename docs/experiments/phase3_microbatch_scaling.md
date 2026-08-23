# Phase 3.3 Micro-Batch Scaling

## Summary

Increasing the fused-attention reference from micro-batch (MB) 1 to 8 raised
throughput from `9,763.07` to `15,084.70 tok/s` and MFU from `15.80%` to
`24.42%`. MB=8 was the best measured point. The gain is real but diminishing:
GPU utilization was already `98.04%` at MB=1, while larger batches mainly
improved GEMM efficiency and amortized an almost fixed kernel count and optimizer
cost over more tokens.

The Phase 3.2 Pod could not be rescheduled because its host had no available A40.
After two failed recovery attempts, one replacement Secure Cloud Pod
(`t1voho50r1euaw`) was created with 1x A40 at `$0.44/hour`. No additional GPU or
Pod was created.

## Controlled Configuration

- Model: `355,919,872` parameters; 24 layers; H=1024; FFN=4096; 16 heads;
  head dimension 64; vocabulary 50,304; learned positions; tied embeddings.
- Workload: sequence length 2,048; fixed synthetic token IDs; TP/PP/DP=1/1/1;
  20 warmup and 100 measured steps per point.
- Attention: `TEDotProductAttention`, forced `AttnBackend.fused`, cuDNN
  `FusedAttention` sub-backend 1, dropout `0.1`.
- Precision: BF16 forward/backward autocast, FP32 parameters and optimizer state.
- Optimizer: PyTorch AdamW, `lr=1e-4`, `foreach=False`, `fused=False`.
- Disabled: FlashAttention, CUDA Graphs, full TE layer spec, and every unrelated
  optimization. Only micro-batch and its equal global batch changed.

## Environment

- Pod/image: `t1voho50r1euaw`, 1x NVIDIA A40 48GB,
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Driver/Python: `570.195.03` / `3.12.3`
- PyTorch/CUDA/NCCL/cuDNN: `2.8.0+cu128` / `12.8` / `2.27.3` / `9.10.2`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine: `2.17.1+4329ff84`, source commit
  `4329ff84bfbdaa778a33cba02a15fb0807c64689`
- Nsight Systems: `2025.1.1.0`; measurement commit: `f9694dd`

The replacement volume was rebuilt from the same pinned sources. TE was built
with `--no-deps`, and `pip check` passed. PyTorch, CUDA, NCCL, Megatron, TE, and
the optimizer were not upgraded or replaced. Runtime logs explicitly confirmed
`FusedAttention backend (sub-backend 1)` for every run.

## Commands

Each `${MB}` in `1 2 4 8` used this full benchmark command:

```bash
PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase3_microbatch_run.py --micro-batch-size "${MB}" \
  --warmup-iterations 20 --measured-iterations 100 \
  --output-json "results/phase3_mb${MB}_run.json"
```

Separate 10-step steady-state samples used the same command under `nsys profile`
with `--trace=cuda,nvtx,osrt,cublas,cudnn`, CUDA profiler API capture, and output
`profiles/phase3_mb${MB}.nsys-rep`. Nsight overhead is therefore excluded from
the 100-step performance metrics.

## Results

| MB | Tokens/step | Avg / median step (ms) | Tokens/s | ms/token | MFU | Peak VRAM (MiB) | GPU util. | Final loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 209.770 / 208.289 | 9,763.07 | 0.10243 | 15.80% | 8,410 | 98.04% | 0.036531 |
| 2 | 4,096 | 337.203 / 337.306 | 12,147.00 | 0.08232 | 19.66% | 11,828 | 99.40% | 0.049633 |
| 4 | 8,192 | 589.827 / 589.851 | 13,888.82 | 0.07200 | 22.48% | 18,664 | 99.69% | 1.603506 |
| 8 | 16,384 | 1,086.133 / 1,087.107 | 15,084.70 | 0.06629 | 24.42% | 32,632 | 99.85% | 3.557202 |

| MB | Throughput gain vs MB1 | Scaling efficiency | Kernels/step | Kernels/token | Attention share | GEMM share | Optimizer share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00% | 100.00% | 4,361 | 2.1294 | 10.86% | 20.89% | 26.12% |
| 2 | 24.42% | 62.21% | 4,406 | 1.0757 | 11.72% | 25.23% | 15.98% |
| 4 | 42.26% | 35.56% | 4,406 | 0.5378 | 12.50% | 27.61% | 9.10% |
| 8 | 54.51% | 19.31% | 4,414 | 0.2694 | 13.15% | 28.28% | 4.93% |

Scaling efficiency is
`(throughput_MB / throughput_MB1) / (MB / MB1) * 100`; 100% would mean the
larger batch completed in the MB=1 step time. MFU uses the unchanged Phase 1.2
formula:

```text
F_iter = 72*B*S*L*H^2 + 6*B*L*S^2*H + 6*B*S*H*V
MFU = (F_iter / step_seconds) / 149.7e12
```

The A40 denominator is the `149.7 TFLOP/s` dense BF16 Tensor Core peak. Nsight
shares use total CUDA kernel execution time; categories are non-exclusive.
Finite losses verify training execution, but are not cross-MB correctness data
because each optimizer step processes a different token count.

## Interpretation

GEMM share increased from `20.89%` to `28.28%`, and kernels under 50 us fell
from `72.96%` to `51.04%`, consistent with larger and more efficient matrix
operations. Kernel count per step rose only `1.22%`, so kernels per token fell
almost exactly with batch size. The fixed optimizer work remained about
`53 ms/step`, shrinking from `26.12%` to `4.93%` of GPU kernel time.

MB=8 used `70.83%` of visible VRAM and was not memory-limited. A least-squares
fit to MB=1/2/4 predicted `32,336 MiB` for MB=8, below the 90% safety limit of
`41,461 MiB`; measured use was `32,632 MiB`. The same fit predicts about
`59,680 MiB` for MB=16, so the next doubling is unsafe without changing another
variable and was not attempted. Throughput gains taper because attention and
activation/data-movement work scale with batch while the already-amortized
optimizer and launch overhead cannot provide another linear gain.

## Profiling Notes And Artifacts

The first MB=1 capture used a missing output directory and fell back to `/tmp`;
it was rejected, the directory was created, and the capture was rerun unchanged.
The initial analyzer also omitted cuDNN kernels named `sdpa` from attention;
adding `sdpa` and `fused_attn` symbol matching corrected classification without
changing or rerunning the workload.

The four 5.7-6.1 MiB traces and 19-21 MiB SQLite exports remain on the stopped
Pod and are not committed. Their paths, sizes, and SHA-256 fingerprints are
preserved in `results/phase3_microbatch_scaling.json`.
