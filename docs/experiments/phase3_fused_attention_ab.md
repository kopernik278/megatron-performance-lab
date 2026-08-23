# Phase 3.2 Fused Attention A/B Benchmark

## Summary

This controlled experiment replaced only Megatron's local core
`DotProductAttention` with `TEDotProductAttention`, forced to Transformer
Engine's cuDNN `FusedAttention` sub-backend 1. On one A40, fused attention cut
average step time from `550.3701 ms` to `215.7446 ms` (`2.5510x`) and reduced
peak device memory from `20,218 MiB` to `8,442 MiB`. The Phase 3.1 hypothesis is
supported.

The stopped Phase 2 Pod could not be rescheduled because its host had no free
A40. One replacement Secure Cloud Pod (`ivhbchlt526g67`) was created with one
A40 at `$0.44/hour`; no additional Pod or GPU was created.

## Controls

- Model: `355,919,872` parameters; 24 layers; H=1024; FFN=4096; 16 heads;
  head dimension 64; vocabulary 50,304; learned positions; tied embeddings.
- Workload: sequence length 2,048; micro/global batch 1; fixed synthetic token
  batch; TP=1, PP=1, DP=1; 20 warmup and 100 measured steps.
- Precision: BF16 forward/backward autocast with FP32 parameters and optimizer
  state. Benchmark dropout remained `0.1`.
- Optimizer: PyTorch AdamW, `lr=1e-4`, `foreach=False`, `fused=False`.
- Disabled: FlashAttention, CUDA Graphs, full TE layer spec, added kernel fusion,
  and every unrelated optimization.

The local Megatron QKV projection emits FP32 tensors even inside autocast; its
`baddbmm` and `bmm` operations are then autocast to BF16. TE requires BF16 Q/K/V
at its API boundary. The fused-only adapter therefore casts Q/K/V to the active
autocast dtype before calling the unmodified `TEDotProductAttention`. It adds no
new algorithm and preserves FP32 parameter storage.

## Environment

- Pod/image: `ivhbchlt526g67`, Secure Cloud, 1x NVIDIA A40 48GB,
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Driver/Python: `580.159.04` / `3.12.3`
- PyTorch/CUDA/NCCL/cuDNN: `2.8.0+cu128` / `12.8` / `2.27.3` / `9.10.2`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine: `2.17.1+4329ff84`, source commit
  `4329ff84bfbdaa778a33cba02a15fb0807c64689`
- Nsight Systems: `2025.1.1.0`; project measurement commit: `b76cf4b`

TE was built with `NVTE_FRAMEWORK=pytorch`, `NVTE_CUDA_ARCHS=86`,
`MAX_JOBS=8`, `--no-build-isolation`, and `--no-deps`. The wheel SHA-256 was
`7f7b4e0dadd4c63e77e1e964126ddf1ea5cd7008d7b6febbdb354cc68097ada5`.
Runtime dependencies were pinned from Megatron's lock file. `pip check` passed,
and PyTorch, CUDA, and NCCL versions were unchanged.

## Correctness Gate

The dropout-free comparison used a 2-layer GPT (H=128, FFN=512, 4 heads,
sequence 128, batch 1) with identical weights and inputs. TE logs explicitly
reported `FusedAttention backend (sub-backend 1)` with FlashAttention and the
unfused TE backend disabled.

- Forward max/mean absolute error: `0.0039310 / 0.0009537`
- Forward max/mean relative error: `0.0006005 / 0.0001379`
- Gradients compared: 28 parameter tensors
- Worst gradient max/mean absolute error: `0.0009166 / 0.0002582`
- Acceptance: `torch.testing.assert_close(atol=0.05, rtol=0.05)` passed for the
  forward output and every gradient tensor.

The large worst relative gradient error (`4.1407`) is caused by near-zero
reference elements; the combined absolute/relative criterion passed.

```bash
PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase3_attention_correctness.py \
  --output-json results/phase3_attention_correctness.json
```

## Exact Benchmark Commands

Both commands used the same Nsight options and measurement script. The local
run added `TRANSFORMER_ENGINE_DISABLE=1`:

```bash
PYTHONPATH=/workspace/Megatron-LM TRANSFORMER_ENGINE_DISABLE=1 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=process-tree --cpuctxsw=process-tree \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=true --force-overwrite=true \
  --output=profiles/phase3_ab_local \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase3_attention_profile.py \
  --attention-implementation local-unfused \
  --warmup-iterations 20 --measured-iterations 100 \
  --output-json results/phase3_ab_local_run.json
```

The fused run was:

```bash
PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=process-tree --cpuctxsw=process-tree \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=true --force-overwrite=true \
  --output=profiles/phase3_ab_fused \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase3_attention_profile.py \
  --attention-implementation te-fused \
  --warmup-iterations 20 --measured-iterations 100 \
  --output-json results/phase3_ab_fused_run.json
```

## Results

| Metric | Local A | Fused B | Change |
| --- | ---: | ---: | ---: |
| Average step time | 550.3701 ms | 215.7446 ms | -60.8001% |
| Median step time | 550.3266 ms | 215.8606 ms | -60.77% |
| Throughput | 3,721.1324 tok/s | 9,492.7058 tok/s | +155.1026% |
| MFU | 6.0229% | 15.3646% | +9.3417 points |
| Peak allocated VRAM | 18,622.3 MiB | 7,481.8 MiB | -59.82% |
| Peak `nvidia-smi` VRAM | 20,218 MiB | 8,442 MiB | -11,776 MiB |
| Average GPU utilization | 64.6469% | 43.9915% | -20.6554 points |
| Final synthetic-batch loss | 2.521301 | 0.021534 | not parity data |

MFU uses `F_iter = 4,962,297,839,616` and the A40 dense BF16 peak of
`149.7 TFLOP/s`: `MFU = (F_iter / step_seconds) / 149.7e12`. The FLOP estimate
is held constant because the mathematical model is unchanged.

The lower sampled GPU utilization does not indicate less work completed. The
200 ms `nvidia-smi` sampler is coarse relative to the `216 ms` fused step and
includes synchronized CPU launch/optimizer intervals. Step time, throughput,
and the CUDA timeline are the primary A/B measures.

## Nsight Systems Analysis

- Attention GPU time: `241.8847` to `23.8019 ms/step` (`-90.1598%`)
- Attention kernel-time share: `49.7581%` to `11.6999%`
- GEMM kernel-time share: `17.1339%` to `21.0072%` (categories overlap)
- Kernels: `4,769` to `4,361/step` (`-8.5553%`)
- Kernels below 50 us: `66.7226%` to `72.9649%`; the absolute count remained
  `3,182/step`, so the higher percentage reflects removal of longer attention
  kernels rather than creation of more small kernels.
- Explicit memcpy time: `57.4603` to `0.00394 ms/step`

The local masked fill, softmax, dropout, scale, and BMM sequence was replaced by
cuDNN SDPA forward/backward kernels. Avoiding materialized S-by-S scores and
probabilities removed most attention memory traffic, copies, and long kernels.
The earlier 1.98x planning bound used a non-exclusive kernel-time category and
excluded memcpy and secondary cast/layout costs, so it was not a valid wall-time
upper bound; the measured speedup is `2.5510x`.

Training losses are finite but not expected to match because fused and local
dropout consume RNG differently. Loss parity is established only by the
dropout-free correctness gate; these final values merely verify training
progress on a repeatedly reused synthetic batch.

## Problems And Artifacts

The image contains system cuDNN 9.8 and PyTorch's cuDNN 9.10.2. TE initially
loaded 9.8 and the sequence-2048 training graph failed with
`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. Preloading the already-installed
PyTorch cuDNN 9.10.2 main library aligned the main library and sublibraries; a
1-step training probe, the correctness gate, and the full run then passed. No
package was installed, upgraded, or replaced for this fix.

The traces remain on the stopped Pod and are not committed: local
`phase3_ab_local.nsys-rep` is `73,316,577` bytes (SHA-256
`639b2fec2396223ec9ab8dab6636dfaa56a59cdee68e0e673c2db107635e711d`), and
fused `phase3_ab_fused.nsys-rep` is `59,196,453` bytes (SHA-256
`ee2c25e6fc1b626106e10312c9a67b37909faf9b6f85dfad114d0cf9c226c8a9`).
The committed JSON preserves the complete summary and trace fingerprints.
