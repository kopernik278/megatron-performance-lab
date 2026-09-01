# Phase 4.4 Persistent BF16 Residual-Stream Screen

## Summary

The Phase 4.3 mechanism works, but the performance hypothesis is **not supported**.

- All 24 Transformer layer residual/output boundaries remained BF16.
- Correctness remained acceptable: no NaN/Inf and minimum gradient cosine `0.999752`.
- B improved throughput by `3.29%`, below the `5%` promising gate.
- Copy/cast was unchanged: `255.91` → `256.12 ms/step` (`−0.08%` reduction).
- No 20-warmup + 100-step benchmark was run.

Preventing FP32 bias from promoting BDA output is necessary to persist BF16 residuals, but it does not remove the dominant copy traffic. LayerNorm/autocast, FC1-bias/GELU-to-FC2 autocast, QKV adapter, backward dtype restoration, logits, and GEMM-layout copies remain.

## Change

Only the lab repository changed. Upstream Megatron-LM remained untouched.

`scripts/bf16_hidden_residual.py` retains the decoder-entry cast and replaces each layer's attention/MLP BDA factory with a local equivalent:

```python
if x.dtype != residual.dtype:
    x = x.to(residual.dtype)

if bias is not None and bias.dtype != residual.dtype:
    bias = bias.to(residual.dtype)
```

Unchanged:

- FP32 parameters
- FP32 AdamW state (`foreach=False`, `fused=False`)
- LayerNorm implementation and precision policy
- FP32 loss reduction
- TE cuDNN FusedAttention sub-backend 1
- 355.9M GPT architecture, sequence length 2048, micro-batch 8
- TP/PP/DP = 1/1/1
- PyTorch `2.8.0+cu128`, CUDA 12.8, NCCL 2.27.3, cuDNN 9.10.2
- Megatron-LM `09fde85e`, Transformer Engine `2.17.1+4329ff84`

## Lifecycle Gate

One instrumented iteration ran before correctness or benchmarking.

| Boundary | Layers observed | Result |
| --- | ---: | --- |
| Layer input | 24/24 | BF16 |
| Attention BDA `x` / residual / output | 24/24 | BF16 / BF16 / BF16 |
| MLP BDA `x` / residual / output | 24/24 | BF16 / BF16 / BF16 |
| Layer output | 24/24 | BF16 |
| Final LayerNorm input | 1 | BF16 |

All 24 layers passed. Parameters were FP32 and loss reduction was FP32.

The BDA trace later confirmed, per training step:

- BDA calls: `48`
- residual/`x` casts required: `0`
- BF16 → FP32 residual casts: `0`
- FP32 bias → BF16 casts required: `48`

The exact Phase 4.2 reversion mechanism is fixed.

## Correctness

Baseline and candidate used seed 1234, identical weights, and dropout 0 only for this comparison.

| Check | A | B | Difference |
| --- | ---: | ---: | ---: |
| Loss | 11.033495 | 11.033475 | abs `2.00e-5`, rel `1.82e-6` |
| Forward max abs error | — | — | `0.03900` |
| Forward mean abs error | — | — | `0.006847` |
| Forward max relative error | — | — | `0.003535` |
| Forward mean relative error | — | — | `0.000623` |
| Gradient tensors compared | 292 | 292 | — |
| Gradient minimum cosine | — | — | `0.999752` |
| Gradient mean cosine | — | — | `1.000005` (rounding above 1) |
| Worst gradient max abs error | — | — | `4.45e-4` (`layers.0.linear_proj.bias`) |
| NaN/Inf | none | none | pass |

The numerical screen is acceptable. This is not evidence about long-horizon BF16 residual accumulation.

## A/B Performance

Both variants ran on the same replacement A40 with 3 warmup and 10 measured steps.

| Metric | A: current FP32 residual | B: persistent BF16 residual | Delta |
| --- | ---: | ---: | ---: |
| Average step time | 1072.63 ms | 1038.43 ms | `−34.20 ms` (`3.19%` faster) |
| Tokens/sec | 15,274.63 | 15,777.68 | **`+3.29%`** |
| MFU | 24.72% | 25.54% | `+0.81` points |
| Peak allocated VRAM | 31,192.47 MiB | 31,256.47 MiB | `+64 MiB` |
| Peak reserved VRAM | 31,830 MiB | 32,028 MiB | `+198 MiB` |

The short timing screen is stable within each variant, but `+3.29%` does not satisfy the required `>=5%` threshold.

## Copy/Cast Profile

B used 3 warmup and 5 profiled steps. Values are CUDA kernel self-time.

| Metric | Phase 4.1 | Phase 4.4 B | Change |
| --- | ---: | ---: | ---: |
| Total copy/cast | 255.91 ms | 256.12 ms | `+0.20 ms` (`−0.08%` reduction) |
| `bfloat16_copy` | 174.72 ms / 632 calls | 174.75 ms / 680 calls | `+0.03 ms`, `+48` calls |
| `direct_copy` | 81.19 ms / 350 calls | 81.36 ms / 398 calls | `+0.17 ms`, `+48` calls |
| `aten::copy_` | 255.92 ms / 1275 calls | 256.12 ms / 1371 calls | `+0.20 ms`, `+96` calls |

The 48 corrected BDA bias casts cost only `0.098 ms/step`, but introduce 48 calls in each copy-kernel family. Eliminating large residual promotions did not reduce aggregate copy time because the remaining paths dominate.

Phase 4.1 and Phase 4.4 profiles used different A40 hosts and driver revisions because both stopped hosts had no free A40. That limits sub-millisecond cross-phase conclusions, but not the conclusion that there was no material reduction.

## Remaining Copy Sources

Profiler module ranges, shapes, and the Python `Tensor.to` tracer resolve the requested sources:

### LayerNorm → Linear autocast

- BF16 residual → FP32-policy normalization:
  - `[2048, 8, 1024]`
  - `48 calls/step`
  - `8.54 ms/step`
- FP32 normalized activation → BF16 QKV/FC1 operand:
  - `24` QKV + `24` FC1 calls/step
  - `8.57 ms/step`

Keeping only the residual boundary BF16 does not change LayerNorm's FP32 policy. Each layer still performs the BF16 → FP32 → BF16 round trip before QKV and FC1.

### FC1 bias/GELU promotion and FC2 autocast

- FC1 output combines with FP32 bias, so the GELU/FC2 input path is FP32.
- FC2 autocast copy `[2048, 8, 4096]`:
  - `24 calls/step`
  - `17.09 ms/step`
- The module-range total including FC2 weight casts is `18.20 ms/step`, 48 calls.

### QKV adapter

`AutocastTEDotProductAttention` still explicitly casts Q, K, and V:

- Forward: `72 calls/step`, `13.15 ms/step`
- Matching backward: `72 calls/step`, `12.86 ms/step`
- Shape: `[2048, 8, 16, 64]`

### Corrected BDA bias

- FP32 bias → BF16: `48 calls/step`
- Module-attributed time: `0.098 ms/step`
- Residual BF16 → FP32 casts: zero

### Other

The profiler assigns `214.16 ms/step` and 1107 `aten::copy_` calls to unparented or other work. Dominant groups are:

- `[2048, 8, 4096]`: `67.90 ms`, 96 calls — forward/backward FFN activation and gradient copies
- `[2048, 8, 1024]`: `25.59 ms`, 146 calls — hidden activation/gradient copies
- `[2048, 8, 50304]`: `17.36 ms`, 2 calls — logits
- FC1/FC2/QKV GEMM operand/layout copies
- `LinearFunction.backward` BF16 → FP32 gradient restoration:
  - `[2048, 8, 1024]`, 24 calls
  - `[2048, 8, 4096]`, 24 calls

## Decision

| Gate | Result |
| --- | --- |
| All 24 residual/output boundaries BF16 | pass |
| No NaN/Inf | pass |
| Gradients strongly aligned | pass (`min cosine 0.999752`) |
| Material copy/cast reduction | **fail** (`−0.08%`) |
| Throughput improvement >=5% | **fail** (`+3.29%`) |

**Not promising under the Phase 4.4 rule.** The mechanism is validated, but the hypothesis that BDA residual promotion causes a large fraction of the measured 256 ms/step copy overhead is not supported. The full 20+100 benchmark was not run.

The next optimization, if pursued, should target a measured remaining boundary rather than residual BDA: either the LayerNorm FP32 round trip, FC1-bias/GELU-to-FC2 promotion, or the explicit QKV adapter.

## Infrastructure and Limitations

- Pod: `kooj4fxq9zqn93`, Secure Cloud, 1x A40 48GB, `$0.44/hour`
- Prior stopped Pods `x6gc433o9vu7sx` and `3ixl2btmmwghn5` could not restart because their hosts had no free A40.
- Replacement image was unchanged: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- Host driver was `580.159.03`, versus `570.195.03` in Phase 4.1/4.2. A/B controls were on the same replacement Pod.
- GPU experiment runtime after setup remained a short screen; no Nsight Systems or Nsight Compute was used.
- Pod final status: `EXITED`.
