# Phase 5.2 Bias-Dropout-Add Fusion A/B

## Outcome

Enabling Megatron's existing Bias-Dropout-Add (BDA) fusion improved the formal
MB=8 endpoint by **4.27% throughput** while changing only
`bias_dropout_fusion`:

```python
# A
bias_dropout_fusion = False

# B
bias_dropout_fusion = True
```

The 15-step screen passed the decision gate, so the planned 20-warmup,
100-measured-step validation was run. Nsight Systems confirmed the expected
mechanism: B removed exactly **96 total and BDA-attributed kernel launches per
step**, and forward BDA GPU time fell by **34.790 ms/step (68.9%)**.

## Controlled configuration

Both timing variants used the same A40 Pod and the current best fused-attention
MB=8 configuration:

- 24 layers, hidden size 1,024, FFN size 4,096, 16 attention heads;
- sequence length 2,048, vocabulary 50,304, micro-batch size 8;
- BF16 autocast with FP32 parameters, residual stream, and AdamW state;
- local Megatron GPT layer spec with `TEDotProductAttention` forced to cuDNN
  FusedAttention;
- unfused `torch.optim.AdamW` with `foreach=False`;
- hidden and attention dropout 0.1 for performance runs;
- Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`,
  Transformer Engine `2.17.1+4329ff84`, PyTorch `2.8.0+cu128`;
- NVIDIA A40 48 GB, driver 570.195.03, CUDA 12.8.

The harness asserted that `bias_activation_fusion`,
`masked_softmax_fusion`, and `cross_entropy_loss_fusion` remained disabled,
CUDA Graph remained disabled, parameter storage remained FP32, and each
variant selected the intended BDA function. No optimizer, dtype, module, or
other optimization changed.

## Correctness

Correctness **passed** using identical weights and seeds with hidden and
attention dropout set to zero for this comparison only. The check used a
reduced two-layer model to compare full tensors and gradients economically.

| Check | Result |
| --- | --- |
| Scalar loss | 6.9515428543 for both; absolute error 0 |
| Per-token loss | Max/mean absolute error 0; all-close |
| Forward output | Max/mean absolute error 0; all-close |
| Gradients | All 28 parameter gradients close |
| Worst gradient max absolute error | 2.9802e-8 |
| Global gradient cosine similarity | 0.9999999999999992 |
| NaN/Inf | None in outputs, losses, or gradients |

The timing runs retained dropout 0.1. They are performance measurements, not
bitwise loss comparisons, because compiled and eager dropout need not produce
identical masks.

## Fast screen

The screen used 3 warmup and 15 measured steps in A-then-B order.

| Metric | A: fusion off | B: fusion on | B vs A |
| --- | ---: | ---: | ---: |
| Average step time | 1,075.595 ms | 1,031.506 ms | **-44.088 ms (-4.10%)** |
| Median step time | 1,075.487 ms | 1,031.472 ms | **-44.015 ms** |
| Throughput | 15,232.503 tok/s | 15,883.566 tok/s | **+651.063 (+4.274%)** |
| MFU | 24.655% | 25.709% | **+1.054 percentage points** |
| Peak allocated VRAM | 31,192.5 MiB | 31,192.5 MiB | 0 MiB |
| Peak `nvidia-smi` VRAM | 32,632 MiB | 32,700 MiB | +68 MiB |

The 4.274% throughput gain exceeded the 2% threshold and the profile confirmed
the intended fusion, requiring formal validation.

## Nsight Systems mechanism check

Each short trace covered 3 warmup steps followed by 5 profiled steps. Identical
manual NVTX ranges attributed kernels to the 24 attention and 24 MLP forward
BDA sites per step. BDA values below are therefore forward-only; total kernel
and idle values cover the full forward, backward, and optimizer step.

| Metric per step | A: fusion off | B: fusion on | B vs A |
| --- | ---: | ---: | ---: |
| All CUDA kernels | 4,414 | 4,318 | **-96 (-2.17%)** |
| Total CUDA-kernel time | 1,071.004 ms | 1,026.249 ms | **-44.755 ms** |
| BDA NVTX ranges | 48 | 48 | 0 |
| BDA-attributed kernels | 192 | 96 | **-96** |
| BDA-attributed GPU time | 50.476 ms | 15.686 ms | **-34.790 ms (-68.9%)** |
| Kernels shorter than 50 us | 2,253 | 2,349 | +96 |
| Time in kernels shorter than 50 us | 21.397 ms | 21.636 ms | +0.239 ms |
| GPU idle time | 6.897 ms | 6.502 ms | **-0.395 ms** |
| GPU idle share | 0.6380% | 0.6278% | -0.0103 percentage points |

The under-50-us count increased rather than decreased, so that threshold is
not evidence for the fusion by itself. The direct BDA attribution is
unambiguous: the unfused path launched four BDA kernels per site, while the
compiled path launched two, removing 96 launches/step overall and within the
BDA ranges. The fused trace's dominant BDA kernel was
`triton_poi_fused__to_copy_add_native_dropout_0`.

This analysis does not use Phase 3.4's 295.82 ms
`activation_elementwise` category as pure elementwise time; that category has
known GEMM and optimizer overlap.

## Formal validation

The formal run used 20 warmup and 100 measured steps per variant on the same
Pod.

| Metric | A: fusion off | B: fusion on | B vs A |
| --- | ---: | ---: | ---: |
| Average step time | 1,081.104 ms | 1,036.835 ms | **-44.269 ms (-4.09%)** |
| Median step time | 1,081.285 ms | 1,037.352 ms | **-43.933 ms** |
| Throughput | 15,154.885 tok/s | 15,801.942 tok/s | **+647.057 (+4.270%)** |
| MFU | 24.529% | 25.577% | **+1.047 percentage points** |
| Peak allocated VRAM | 31,192.5 MiB | 31,192.5 MiB | 0 MiB |
| Peak `nvidia-smi` VRAM | 32,632 MiB | 32,700 MiB | +68 MiB |

The formal result is a **1.0427x throughput speedup**. It reproduced the
screen within 0.005 percentage points of throughput gain, while the average
step-time saving remained about 44.3 ms.

## Decision

Accept `bias_dropout_fusion=True` as the Phase 5.2 optimization for this
endpoint. It passed correctness, delivered a stable formal 4.27% throughput
gain, and produced the predicted 96-launch reduction with a measured
34.790 ms/step drop in forward BDA GPU time.

The complete machine-readable result, including per-step timing samples,
environment metadata, trace hashes, and detailed kernel families, is in
`results/phase5_bias_dropout_fusion.json`. The Nsight reports remain on the
stopped Pod `zg2rcz9362h3p4`.
