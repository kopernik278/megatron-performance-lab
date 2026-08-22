# Phase 2.1 Nsight Systems Baseline Profile

## Summary

This experiment profiles the unchanged Phase 1.2 single-GPU baseline after 20 warmup steps. Nsight Systems captured 15 steady-state training steps on one NVIDIA A40. Transformer Engine, CUDA Graphs, added kernel fusion, precision changes, and batch-size changes remained disabled.

The Phase 1.2 Pod could not be scheduled after repeated restart attempts because its host had no free A40. Per the recovery condition, one replacement Secure Cloud Pod (`nrk1bdpgmo1ej3`) was created with the same image, one A40, and the same `$0.44/hour` GPU price. No pinned software component or training configuration changed.

## Exact Command

```bash
cd /workspace/megatron-performance-lab
PYTHONPATH=/workspace/Megatron-LM TRANSFORMER_ENGINE_DISABLE=1 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
  /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --output=profiles/phase2_nsys_baseline \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase2_nsys_profile.py \
  --warmup-iterations 20 \
  --profiled-iterations 15 \
  --output-json=results/phase2_nsys_run_metrics.json
```

The script starts the CUDA profiler only after warmup. It adds NVTX ranges for the profile window, each train step, forward, backward, optimizer zeroing, and optimizer step. NVTX and profiling do not alter model computation.

## Baseline Configuration

- Parameters: `355,919,872`
- GPT: 24 layers, hidden size 1024, FFN size 4096, 16 heads, head dimension 64
- Vocabulary/positions: 50,304 tokens, learned absolute positions, tied embeddings
- Sequence length: 2,048; micro/global batch: 1/1
- Parallelism: TP=1, PP=1, DP=1
- Precision: BF16 forward/backward autocast; FP32 parameters and optimizer state
- Attention: local Megatron spec, unfused backend
- Optimizer: PyTorch AdamW, `lr=1e-4`, `foreach=False`, `fused=False`
- Data: fixed synthetic random token IDs

## MFU Validation

The pinned Megatron-LM training FLOP accounting for this dense GPT is:

```text
F_iter = 72*B*S*L*H^2 + 6*B*L*S^2*H + 6*B*S*H*V
MFU = (F_iter / step_seconds) / GPU_dense_BF16_peak
```

The first term covers the token-linear transformer work for forward and backward; the second covers causal attention with half of the full S-by-S matrix; the third covers the logits projection. Embedding lookup, normalization, activation, optimizer, and other elementwise FLOPs are excluded.

For B=1, S=2048, L=24, H=1024, and V=50304, this gives `4,962,297,839,616 FLOPs/iteration`. The A40 dense BF16 Tensor Core peak is `149.7 TFLOP/s`; the `299.4 TFLOP/s` sparsity figure is not used. The Phase 1.2 average of `551.8516 ms` therefore gives `8.9921 TFLOP/s` and `6.0067% MFU`. The instrumented profile timing gives `5.9733%`; its `0.56%` timing difference is profiling overhead, not a baseline change.

Sources: [Megatron-LM FLOP accounting](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/training/training.py) and [NVIDIA A40 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/proviz-print-nvidia-a40-datasheet-us-nvidia-1469711-r8-web.pdf).

## Timeline Results

- Profile window: `8,324.645 ms`; projected time for 15 train steps: `8,291.251 ms`
- Average projected/instrumented step: `552.750 / 554.940 ms`; median instrumented step: `554.935 ms`
- GPU active union: `8,217.347 ms`; GPU idle: `107.299 ms` (`1.2889%`)
- Largest GPU-visible idle gap: `3.042 ms`; p95 gap: `1.280 us`
- CUDA kernels: `71,520`, or `4,768/step`; total kernel execution time: `7,352.310 ms`
- CUDA GPU activities including copies/memsets: `75,900`, or `5,060/step`
- Final loss after 20 warmup and 15 profiled steps: `7.640325`

GPU idle is the complement of the merged kernel, memcpy, and memset intervals inside `profile_window`. Concurrent activities are counted once.

## Top CUDA Kernels

| Rank | Kernel identifier | Calls | Total ms | Kernel time |
| ---: | --- | ---: | ---: | ---: |
| 1 | `bfloat16_copy_kernel_cuda` vectorized | 9,465 | 1,210.087 | 16.46% |
| 2 | `direct_copy_kernel_cuda` unrolled | 5,835 | 730.100 | 9.93% |
| 3 | `BinaryFunctor Mul<float>` vectorized | 375 | 518.800 | 7.06% |
| 4 | `cunn_SoftMaxBackwardSmem` | 360 | 506.895 | 6.89% |
| 5 | `masked_fill_kernel` | 720 | 423.843 | 5.76% |
| 6 | `fused_dropout_kernel_vec` | 1,095 | 417.607 | 5.68% |
| 7 | `masked_scale_kernel` | 1,095 | 417.315 | 5.68% |
| 8 | `softmax_warp_forward` | 360 | 349.498 | 4.75% |
| 9 | `cutlass_80_wmma...gemm...32x32...tn` | 720 | 237.407 | 3.23% |
| 10 | `ampere_bf16...gemm...128x128...nt` | 1,095 | 205.788 | 2.80% |
| 11 | `ampere_bf16...gemm...64x64...nn` | 720 | 191.870 | 2.61% |
| 12 | `AUnaryFunctor Mul<float>` vectorized | 8,760 | 163.561 | 2.22% |
| 13 | `addcdiv_cuda_kernel` vectorized | 4,380 | 154.393 | 2.10% |
| 14 | `ampere_bf16...gemm...256x128...nn` | 1,095 | 144.688 | 1.97% |
| 15 | `ampere_bf16...gemm...128x128...tn` | 720 | 122.276 | 1.66% |

Full demangled kernel names are retained in `results/phase2_nsys_summary.json`.

## Workload Attribution

Shares below use total CUDA kernel execution time as the denominator and are non-exclusive. A kernel can be both attention and GEMM.

- Attention, including NVTX-correlated BMM, softmax, and masking: `3,632.651 ms` (`49.4083%`)
- GEMM/matmul by kernel symbol: `1,261.834 ms` (`17.1624%`)
- Normalization by kernel/NVTX attribution: `131.558 ms` (`1.7893%`)
- Optimizer kernels: `795.472 ms` (`10.8194%`)
- Optimizer GPU projection: `822.543 ms`, or `9.9206%` of projected train-step time

Memory copies total `861.929 ms`, equivalent to `10.3539%` of the profile window before accounting for overlap. This is almost entirely 1,830 device-to-device copies totaling `241.59 GB`; 15 four-byte loss transfers account for only `0.025 ms`.

## Launches And Synchronization

Kernel launch API duration has a `6.815 us` median and `468.346 us` p95 under profiling. The per-thread gap between kernel-launch API calls has a `16.745 us` median and `105.094 us` p95. Its `322.865 ms` maximum occurs across an inactive launch thread and is not a GPU stall; the largest GPU-visible gap is only `3.042 ms`.

Small kernels are clearly present: 47,655 kernels (`66.63%`) are shorter than `50 us`, and 59,235 (`82.82%`) are shorter than `100 us`. These groups account for `10.23%` and `22.00%` of total kernel time, respectively.

The trace contains 32 `cudaDeviceSynchronize` and 15 `cudaStreamSynchronize` calls. They delimit timed steps and transfer one loss value per step. Combined with only `1.29%` GPU idle time, there is no obvious long synchronization-driven GPU starvation in the steady-state window. This observation does not imply that the explicit synchronization is desirable in an optimized benchmark.

## Environment And Artifacts

- Pod: `nrk1bdpgmo1ej3`, Secure Cloud, `1x NVIDIA A40 48GB`, `$0.44/hour`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Driver: `570.195.03`; Python: `3.12.3`
- PyTorch: `2.8.0+cu128`; CUDA runtime: `12.8`; NCCL: `2.27.3`
- Megatron Core: `0.20.0+09fde85ea`; Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Nsight Systems: `2025.1.1.0`
- Transformer Engine: not installed and disabled; CUDA Graph: disabled
- Timestamp: `2026-08-22T16:43:02Z`

The trace is preserved on the stopped Pod at `profiles/phase2_nsys_baseline.nsys-rep` (`10,863,586` bytes, SHA-256 `638eac5ed36629d297a7a52e44ed29fc1ccd1c326a990d0aa6e525ad2667ac0e`). Its SQLite export is also preserved for reproducible analysis. Neither binary artifact is committed to Git because the JSON summary retains all reported measurements.

The first profiler invocation used an unsupported `--cuda-trace-all-apis` option and exited before launching the workload. Removing only that profiler option resolved the issue; no training or environment setting changed.
