# Phase 4.2 BF16 Hidden/Residual Stream Screen

## Summary

This fast-iteration screen tests the Phase 4.1 recommendation: keep Transformer hidden/residual activations in BF16 while leaving FP32 master weights and FP32 AdamW unchanged.

The change is numerically safe. It is **not promising** as a performance optimization.

- No NaN/Inf
- Gradient min cosine similarity `0.99984`
- Copy/cast stayed **256.74 ms/step** vs Phase 4.1 **255.91 ms/step**
- Throughput **−0.45%** (A 15,199 tok/s → B 15,131 tok/s)
- Full 20-warmup + 100-step A/B was **not** run

A single decoder-entry cast does not stop per-layer FP32 residual recapture. LayerNorm still feeds FP32 into the residual path, so `x.to(residual.dtype)` and Linear activation autocast remain.

## Exact Code / Dtype Change

Lab repository only. Megatron-LM was not modified. Master weights and optimizer states stayed FP32 (`params_dtype=float32`, `TransformerConfig.bf16=False`).

`scripts/bf16_hidden_residual.py` wraps `model.decoder.forward` and casts `hidden_states` to BF16 once at `TransformerBlock` entry. Residual connections copy that incoming hidden state, so the intent was to make `fused_bias_dropout.py` `x.to(residual.dtype)` a BF16 no-op. Loss still upcasts in `masked_language_model_loss(...).float()`.

Observed B dtype boundaries:

| Site | Input | Output |
| --- | --- | --- |
| embedding | — | FP32 `[2048, 8, 1024]` |
| decoder (module hook sees pre-wrap args) | FP32 `[2048, 8, 1024]` | BF16 `[2048, 8, 1024]` |
| final_layernorm | **FP32** `[2048, 8, 1024]` | BF16 `[2048, 8, 1024]` |
| output_layer | BF16 `[2048, 8, 1024]` | BF16 `[2048, 8, 50304]` |

The entry cast took effect at the block boundary. Final LayerNorm input remaining FP32 shows the intra-block residual/hidden stream was promoted back to FP32.

Unchanged: A40, 355.9M GPT, S=2048, MB=8, TE cuDNN FusedAttention sub-backend 1, FP32 AdamW (`foreach=False`, `fused=False`), TP/PP/DP=1/1/1, PyTorch `2.8.0+cu128`, CUDA 12.8, NCCL 2.27.3, Megatron-LM `09fde85e`, Transformer Engine `2.17.1+4329ff84`.

## Infrastructure

The Phase 3.4 / 4.1 Pod `3ixl2btmmwghn5` could not restart (host had no free A40). One replacement Secure Cloud Pod `x6gc433o9vu7sx` was created: 1x A40 48GB, `$0.44/hour`, EU-SE-1, same image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

PyTorch was not rebuilt. TE was compiled on the replacement Pod from pinned commit `4329ff84` so the version string matches Phase 3.4 (`2.17.1+4329ff84`). The first PyPI `transformer-engine-cu12` wheel failed fused-attention backward (`CUDNN_STATUS_BAD_PARAM`) and was replaced by that source build.

## Method

```bash
PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0 \
LD_LIBRARY_PATH=.../nvidia/nccl/lib:.../nvidia/cudnn/lib:/usr/local/cuda/lib64 \
LD_PRELOAD=.../nvidia/cudnn/lib/libcudnn.so.9 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase4_bf16_residual_screen.py \
  --warmup-iterations 3 \
  --measured-iterations 10 \
  --profile-iterations 5 \
  --output-json results/phase4_bf16_residual_screen.json
```

- A = current fused MB=8, FP32 residual (no wrap)
- B = identical + decoder-entry BF16 wrap
- Correctness first: seed 1234, dropout 0, same weights
- Performance: 3 warmup + 10 measured on the same Pod
- Short PyTorch Profiler on B only (3 warmup + 5 profiled)
- No Nsight Systems, no Nsight Compute

## Correctness

Dropout disabled only for this comparison. Bitwise equality was not required.

| Check | Result |
| --- | --- |
| Loss A / B | `11.033495` / `11.033464` |
| Loss abs / rel diff | `3.05e-5` / `2.77e-6` |
| Forward max / mean abs error | `0.02353` / `0.004068` |
| Forward max / mean rel error | `0.002006` / `0.000370` |
| Gradients compared | 292 parameter tensors |
| Gradient min / mean cosine | `0.999836` / `1.000071` |
| Worst grad max abs error | `0.000243` (`linear_proj.bias` layer 0) |
| NaN / Inf | none in loss, forward, or gradients |

Gradients remain strongly aligned. Numerical behavior is acceptable.

## A vs B Performance

| Metric | A (FP32 residual) | B (BF16 entry cast) | Delta |
| --- | ---: | ---: | ---: |
| Average step time | 1077.95 ms | 1082.79 ms | +4.84 ms (−0.45% faster? no, slower) |
| Tokens/sec | 15,199.25 | 15,131.32 | **−0.45%** |
| MFU | 24.60% | 24.49% | −0.11 points |
| Peak allocated VRAM | 31,192 MiB | 31,256 MiB | +64 MiB |
| Peak reserved VRAM | 31,826 MiB | 31,920 MiB | +94 MiB |

Throughput did not improve by 5%.

## Copy/Cast vs Phase 4.1

B profiler, 5 steps, kernel self-time:

| Metric | Phase 4.1 | B | Delta |
| --- | ---: | ---: | ---: |
| copy/cast total | 255.91 ms/step | 256.74 ms/step | +0.82 ms |
| `bfloat16_copy` | 174.72 ms / 632 calls | 175.29 ms / 635 calls | +0.57 ms |
| `direct_copy` | 81.19 ms / 350 calls | 81.44 ms / 351 calls | +0.25 ms |
| `aten::copy_` | 255.92 ms / 1275 calls | 256.74 ms / 1279 calls | +0.82 ms |

Copy/cast did not decrease. Call counts are the same work.

## Decision

**Not promising.** Hypothesis is **not supported** for this implementation.

| Gate | Result |
| --- | --- |
| No NaN/Inf | pass |
| Gradients strongly aligned (min cosine ≥ 0.99) | pass (`0.99984`) |
| Copy/cast down substantially | **fail** (unchanged) |
| Throughput ≥ +5% | **fail** (−0.45%) |

No 20+100 benchmark.

## Diagnosis

The Phase 4.1 bulk copies are per-layer activation/residual FP32↔BF16 conversions. Casting once at `TransformerBlock` entry does not keep those residuals in BF16:

1. Local LayerNorm still presents an FP32 hidden/residual to the next Linear / BDA (final LN input is FP32).
2. `x.to(residual.dtype)` therefore still promotes BF16 layer outputs to FP32.
3. Autocast Linear still casts those FP32 activations back to BF16.

A later attempt, if pursued, would need to keep residual BF16 after each LayerNorm / residual snapshot inside `TransformerLayer`, without converting master weights. That is a narrower Megatron residual-path change and was not implemented here.

## Git Commit and Pod Status

- Branch: `cursor/phase42-bf16-residual-3b5c`
- Implementation commit: `4e9a8ad` (plus this results commit)
- Pod `x6gc433o9vu7sx` final status: **EXITED**
- Original Pod `3ixl2btmmwghn5` remains **EXITED** (unstartable host)
