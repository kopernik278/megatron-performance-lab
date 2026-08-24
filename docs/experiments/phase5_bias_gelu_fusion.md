# Phase 5.3 Bias + GELU Fusion A/B

## Outcome

Megatron's fused bias-plus-GELU path was selected and produced a
**2.53% fast-screen throughput gain**, but it did **not** preserve the current
numerical function. The pinned fused implementation uses tanh-approximate
GELU, while the baseline uses `torch.nn.functional.gelu`'s exact/erf path.
Same-weight output, loss, and gradient comparisons failed the established
`atol=rtol=1e-5` check.

The optimization is therefore rejected for the current numerical
configuration. Formal 20-warmup + 100-measured validation was not run even
though the performance and profiler thresholds passed: correctness is a
prerequisite for long performance validation.

## Controlled change and runtime path

The requested user-facing variants map to Megatron's actual configuration
field:

```python
# A
bias_activation_fusion = False  # bias_gelu_fusion=False

# B
bias_activation_fusion = True   # bias_gelu_fusion=True
```

Both retained:

- cuDNN FusedAttention, sub-backend 1;
- micro-batch size 8 and sequence length 2,048;
- `bias_dropout_fusion=True`;
- 24 layers, hidden size 1,024, FFN size 4,096, 16 attention heads;
- BF16 autocast with FP32 parameters, residual stream, and AdamW state;
- TP/PP/DP 1/1/1 and unfused `torch.optim.AdamW`;
- CUDA Graph, fused cross entropy, optimizer fusion, and dtype changes off.

Source inspection showed that `MLP.forward` calls `bias_gelu_impl` only when
`bias_activation_fusion=True`, the activation is `F.gelu`,
`gated_linear_unit=False`, and `add_bias_linear=True`. Runtime probes observed
zero `bias_gelu_impl` calls in A and exactly 24 calls in B for one full-model
forward, confirming that the requested path was active.

## Correctness

Correctness used identical weights and seeds, with hidden and attention
dropout set to zero for the comparison only. All outputs and gradients were
finite, but strict equivalence failed:

| Check | A | B | Difference |
| --- | ---: | ---: | ---: |
| Scalar loss | 6.9515428543 | 6.9514288902 | 1.1396e-4 absolute |
| Forward output | — | — | 0.0078125 max abs; 0.0006500 mean abs |
| Per-token loss | — | — | 0.0027227 max abs; 0.0007213 mean abs |
| Gradients | — | — | Not all-close; 0.99999155 global cosine |
| Worst gradient max abs | — | — | 0.0009060 |
| NaN/Inf | None | None | All finite |

This is not random variation or an unselected path. The pinned
`megatron/core/fusions/fused_bias_gelu.py` explicitly implements
tanh-approximate GELU and its matching approximate derivative, whereas A uses
the exact/erf GELU path.

## Fast A/B screen

The same replacement A40 Pod ran A then B with 3 warmup and 15 measured
steps.

| Metric | A: fusion off | B: fusion on | B vs A |
| --- | ---: | ---: | ---: |
| Average step time | 1,034.879 ms | 1,009.366 ms | **-25.513 ms (-2.47%)** |
| Median step time | 1,034.556 ms | 1,009.643 ms | **-24.913 ms** |
| Throughput | 15,831.799 tok/s | 16,231.971 tok/s | **+400.173 (+2.528%)** |
| MFU | 25.625% | 26.273% | **+0.648 percentage points** |
| Peak allocated VRAM | 31,192.5 MiB | 28,120.5 MiB | -3,072 MiB |
| Peak `nvidia-smi` VRAM | 32,572 MiB | 29,500 MiB | -3,072 MiB |

The measured fast-screen speedup was **1.0253x**.

## Nsight Systems

Each short trace used 3 warmup and 5 profiled steps. Identical Megatron
`MLP.forward.activation` NVTX ranges attributed the forward bias/GELU work.
The eager backward kernel has an explicit `GeluBackward` name; the compiled
backward was enclosed in a manual fused-backward range.

| Metric per step | A: fusion off | B: fusion on | B vs A |
| --- | ---: | ---: | ---: |
| All CUDA kernels | 4,318 | 4,294 | **-24 (-0.56%)** |
| Total CUDA-kernel time | 1,032.615 ms | 1,003.236 ms | **-29.379 ms** |
| Bias + GELU forward kernels | 48 | 24 | **-24** |
| Bias + GELU forward GPU time | 40.435 ms | 17.040 ms | **-23.395 ms (-57.9%)** |
| GELU backward kernels | 24 | 24 | 0 |
| GELU backward GPU time | 34.412 ms | 28.197 ms | **-6.215 ms (-18.1%)** |
| GELU-related kernels | 72 | 48 | **-24** |
| GELU-related GPU time | 74.847 ms | 45.236 ms | **-29.611 ms** |
| Kernels shorter than 50 us | 2,349 | 2,349 | 0 |
| Time in kernels shorter than 50 us | 21.733 ms | 21.731 ms | -0.002 ms |

The forward path changed from one separate bias-add kernel plus one exact GELU
kernel per layer to one `triton_poi_fused_add_mul_tanh_0` kernel. The fused
backward used `triton_poi_fused_add_mul_rsub_tanh_0`.

Profiler evidence therefore confirms the expected mechanism: 24 forward
launches were removed, forward bias/GELU time fell by 23.395 ms/step, and
combined forward/backward GELU-related time fell by 29.611 ms/step.

## Decision

Do not adopt `bias_activation_fusion=True` under the current requirement to
preserve the numerical configuration. The fast performance result is
promising and the fusion mechanism is genuine, but it accelerates a different
GELU approximation. Formal validation was skipped because the correctness
gate failed.

## Environment note

The stopped Phase 5.2 Pod could not restart because its host had no free A40.
A replacement Secure Cloud A40 Pod was used for the entire Phase 5.3 A/B.
PyTorch 2.8.0+cu128, CUDA runtime 12.8, NCCL 2.27.3, Megatron
`09fde85ea25fb67e9b32019089fae163a3233bd3`, and Transformer Engine
`2.17.1+4329ff84` were unchanged. The replacement host supplied NVIDIA driver
580.159.03 rather than Phase 5.2's 570.195.03. This does not affect the
internal same-Pod A/B control, but it limits direct cross-phase timing
comparison.

The complete result, per-step samples, environment metadata, kernel families,
and trace hashes are saved in `results/phase5_bias_gelu_fusion.json`. Nsight
artifacts remain on stopped Pod `dcyms83qyyo3lb`.
