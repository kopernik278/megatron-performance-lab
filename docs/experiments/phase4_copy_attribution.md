# Phase 4.1 Copy/Cast Root-Cause Attribution

## Summary

This diagnostic attributes the Phase 3.4 `bfloat16_copy` and `direct_copy` kernels to PyTorch operators, dtypes, shapes, phases, and source locations. No model, optimizer, attention backend, batch size, or software-stack change was applied.

Ground-truth GPU copy/cast time is **255.91 ms/step** (kernel self-time). That matches Phase 3.4 `copy_cast` **256.03 ms/step** (23.78% of kernel time). The work is almost entirely **activation dtype conversion** caused by an **FP32 residual/hidden stream** mixed with **BF16 autocast compute**.

| Kernel family | ms/step | calls/step | Role |
| --- | ---: | ---: | --- |
| `bfloat16_copy` | 174.72 | 632 | Contiguous FP32↔BF16 copies (`aten::copy_` → `bfloat16_copy_kernel_cuda`) |
| `direct_copy` | 81.19 | 350 | Non-vectorizable dtype casts (`LoadWithCast` / `StoreWithCast`) |
| `aten::contiguous` / `clone` | 1.48 | 15 | Layout copies; not the bottleneck |

`aten::to`, `aten::_to_copy`, and `aten::copy_` are nested. Their CUDA times (~255 ms each) are the same work and are **not additive**.

Nsight Compute was not used. Nsight Systems was not required: PyTorch Profiler shapes plus a Python `Tensor.to` tracer resolved the dominant sites. The Pod was stopped immediately after the capture.

## Controls

Unchanged from the Phase 3.3 / 3.4 best configuration:

- Hardware: reused stopped Secure Cloud Pod `3ixl2btmmwghn5`, 1x NVIDIA A40 48GB, `$0.44/hour`, CA-MTL-1
- Model: 355,919,872-parameter Megatron Core GPT; 24 layers; H=1024; FFN=4096; 16 heads; head dim 64
- Workload: sequence length 2048; micro-batch 8; TP/PP/DP = 1/1/1
- Precision: BF16 autocast, FP32 parameters, FP32 AdamW (`foreach=False`, `fused=False`)
- Attention: `TEDotProductAttention`, cuDNN FusedAttention sub-backend 1
- Software: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; PyTorch `2.8.0+cu128`; CUDA 12.8; NCCL 2.27.3; Megatron-LM `09fde85e`; Transformer Engine `2.17.1+4329ff84`
- PyTorch, Megatron, and Transformer Engine were **not** rebuilt
- The 100-step sanity benchmark was **not** rerun; Phase 3.4 already validated ~15,085 tok/s and 24.42% MFU

## Method

Fast-iteration capture only: **3 warmup + 5 profiled steps**. GPU experiment runtime was about one minute excluding Pod startup.

```bash
cd /workspace/megatron-performance-lab
PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase4_copy_attribution.py \
  --warmup-iterations 3 \
  --measured-iterations 5 \
  --output-json results/phase4_copy_attribution.raw.json
```

The script uses:

- PyTorch Profiler CPU + CUDA, `record_shapes=True`, `with_stack=True`, `with_modules=True`
- `record_function` ranges for `phase/{forward,backward,optimizer}` and named modules
- NVTX ranges around the same phases (no Nsight Systems export)
- A Python tracer on `Tensor.to` / `copy_` / `contiguous` for src→dst dtype and file:line stacks

Profiler limitations that affect the raw tables:

1. CUDA kernels have empty stacks and no CPU parent, so automatic phase/module on `bfloat16_copy` / `direct_copy` is `unknown` / `unattributed`.
2. `phase/backward` does not parent autograd launches, so backward copies show as `phase=unknown`.
3. Nested `aten::to` → `aten::_to_copy` → `aten::copy_` CUDA times must not be summed.

Attribution therefore uses **kernel self-time** as GPU ground truth, **unique `aten::copy_` (phase, module, shape)** groups for operator sites, and the **Python tracer** for source-code roots.

## Top Copy/Cast Operators

| Operator | calls/step | CUDA ms/step | How to read |
| --- | ---: | ---: | --- |
| `aten::copy_` | 1275 | 255.92 | Ground-truth operator time; launches the copy kernels |
| `aten::to` | 1365 | 255.17 | Parent of `_to_copy`; same GPU work |
| `aten::_to_copy` | 1266 | 255.17 | Same GPU work as `copy_` |
| `aten::contiguous` | 6 | 0.74 | Layout only |
| `aten::clone` | 9 | 0.74 | Layout only |

## Dominant Unique `aten::copy_` Sites

One row per site. Nested `aten::to` / `_to_copy` rows for the same site are omitted.

| Shape | Phase / module | calls/step | ms/step | Kind | Origin |
| --- | --- | ---: | ---: | --- | --- |
| `[2048, 8, 4096]` | unknown / unattributed | 96 | 67.88 | B + C | MLP FFN-width activations; includes Linear autocast and `grad_output.to(input_dtype)` |
| `[2048, 8, 1024]` | unknown / unattributed | 145 | 25.51 | B | Hidden / residual stream (`24*6+1`) |
| `[16384, 4096]` | unknown / unattributed | 24 | 17.11 | B | Flattened `B*S × FFN` FC2 input |
| `[4096, 16384]` | unknown / unattributed | 24 | 17.09 | C | Transpose of that operand (dgrad / wgrad layout) |
| `[2048, 8, 4096]` | forward / `mlp_fc2` | 24 | 17.08 | B | FC2 input autocast, parented under `linear_fc2` |
| `[2048, 8, 16, 64]` | forward / attention | 72 | 13.14 | B | QKV adapter `query/key/value.to(bf16)` |
| `[2048, 8, 16, 64]` | unknown | 72 | 12.85 | C | Backward of the QKV adapter casts |
| `[3072, 16384]` | unknown | 24 | 12.82 | B/C | Fused QKV `3H × B*S` GEMM operand |
| `[2048, 8, 3072]` | unknown | 48 | 12.81 | B/C | Fused QKV output; 2 per layer |

These listed sites cover about **196 ms/step**. The remaining ~60 ms of `aten::copy_` is smaller weight, bias, norm, and loss copies.

## Forward / Backward / Optimizer

| Phase | GPU copy/cast | Evidence |
| --- | --- | --- |
| Forward | Parent of residual upcasts, Linear activation autocast, and the QKV adapter | Unique parented sites: FC2 17.08 ms + QKV adapter 13.14 ms. Python: 72/step residual `x.to(residual.dtype)`, 72/step QKV `.to(bf16)`. Many forward Linear casts lack a kernel CPU parent and appear as `unknown`. |
| Backward | Linear dgrad casts and autograd of the forward `.to()`s | `LinearFunction.backward`: `grad_output.to(ctx.input_dtype)` BF16→FP32 `[2048,8,4096]` 24/step. QKV adapter backward: 72/step, 12.85 ms. `[4096,16384]` 17.09 ms is a dgrad layout cast. |
| Optimizer | **0.0 ms GPU copy** | 876 CPU copy ops/step, 6.50 ms CPU. Phase 3.4 optimizer GPU time (53.06 ms/step) is Adam `addcmul` math, not copy/cast. |

## Module / Source-Code Attribution

### QKV adapter (kind B, 13.14 ms/step) — resolved

`AutocastTEDotProductAttention` in `scripts/phase1_baseline.py:180` casts local FP32 Q/K/V to the autocast dtype before `TEDotProductAttention`:

```180:183:scripts/phase1_baseline.py
                    query = query.to(autocast_dtype)
                    key = key.to(autocast_dtype)
                    value = value.to(autocast_dtype)
```

72 calls/step (24 layers × Q,K,V), FP32→BF16, `[2048, 8, 16, 64]`. This is exactly the 72-launch **elementwise** `bfloat16_copy` (13.14 ms/step). It is **not** the 161.6 ms vectorized bulk.

### Residual stream (kind B) — resolved

Megatron `_bias_dropout_add_func` upcasts the BF16 attention/MLP output to the FP32 residual:

```text
megatron/core/fusions/fused_bias_dropout.py
    # For fp32 residual connections: upcast x (and bias) to residual's dtype
    x = x.to(residual.dtype)
```

72 calls/step, BF16→FP32, `[2048, 8, 1024]`. That is 3 per layer. `self_attn_bda` and `mlp_bda` are two of the sites; the tracer groups every same-shape BF16→FP32 `Tensor.to` with this stack.

Embedding and LayerNorm stay on an FP32 hidden stream. Autocast Linear/attention then emit BF16, and the residual add casts back to FP32 every layer.

### MLP FC2 (kind B, 17.08 ms/step parented)

Forward `aten::copy_` of `[2048, 8, 4096]` under `linear_fc2` (24/step). This is C++ Linear autocast of the FP32 FC2 input to BF16. Flattened twin: `[16384, 4096]` 17.11 ms, 24/step.

### MLP FC1 / Linear backward (kind C)

```text
megatron/core/tensor_parallel/layers.py  LinearFunction.backward
    # Cast dY to the input dtype to match the legacy FP32-logit-cast backward path.
    grad_output = grad_output.to(ctx.input_dtype)
```

24/step, BF16→FP32, `[2048, 8, 4096]`. FC1 `grad_output` is FFN-width; input was FP32.

### QKV projection / attention output / MLP FC1 forward

C++ `aten::linear` autocast is not visible to the Python tracer. Shapes `[2048,8,3072]`, `[3072,16384]`, and hidden `[2048,8,1024]` are the activation-side evidence. Stored parameter shapes such as `[3072, 1024]` are **not** in the top copy groups.

### Normalization / loss / embedding — small

| Site | calls/step | Kind | Source |
| --- | ---: | --- | --- |
| TE LayerNorm weight `.to(bf16)` | 3 | A | `maybe_dequantize(self.weight, dtype)` |
| Final LN input `.to(bf16)` | 1 | B | `maybe_dequantize(input_.contiguous(), dtype)` |
| Embedding `transpose().contiguous()` | 1 | D | `language_model_embedding.py` |
| Loss label/loss `contiguous()` | 2 | D | `compute_language_model_loss` |

### Optimizer — not a copy/cast problem

No GPU `bfloat16_copy` / `direct_copy` under `phase/optimizer`.

## Kind Scorecard

| Kind | Dominant? | Finding |
| --- | --- | --- |
| A. FP32 parameter → BF16 | No | Only tiny LN weight casts in the Python tracer. Top shapes are activations, not stored weights. |
| B. Activation dtype conversion | **Yes** | Residual upcast, Linear input autocast, QKV adapter, FFN-width activation copies. This is the 161.6 ms bulk plus most of `direct_copy`. |
| C. Gradient dtype conversion | Secondary | FC1 `grad_output.to(input_dtype)`, QKV-adapter backward, FFN dgrad transpose. |
| D. Layout / contiguous | No | 0.74 ms/step. |
| E. Optimizer-related | No | 0 ms GPU copy. |
| F. Other | No | Raw profiler `F_other` is an empty-stack artifact, not a real class. |

## Root Cause of `bfloat16_copy`

PyTorch operator chain:

```text
Tensor.to / C++ autocast
  → aten::to
  → aten::_to_copy
  → aten::copy_
  → at::native::bfloat16_copy_kernel_cuda
```

Two launch shapes:

1. **Vectorized** (560/step, **161.58 ms/step**): contiguous FP32↔BF16 of hidden and FFN activations. Phase 3.4's top kernel (161.57 ms/step) is this variant. Sources: Linear autocast of FP32 residual-stream activations, Megatron residual `x.to(residual.dtype)`, and the matching backward conversions.
2. **Elementwise** (72/step, **13.14 ms/step**): the explicit QKV adapter in `scripts/phase1_baseline.py:180`.

`bfloat16_copy` is therefore **not** primarily an FP32 parameter cast and **not** optimizer traffic.

## Root Cause of `direct_copy`

Same `aten::to` → `_to_copy` → `copy_` chain, but TensorIterator selects `direct_copy_kernel_cuda` with `LoadWithCast<1>` / `StoreWithCast<1>` when the copy is not a contiguous vectorized BF16 path. The dominant variant is 343/step, **80.45 ms/step**, matching Phase 3.4's 80.53 ms/step kernel.

Typical shapes are the **transposed / flattened GEMM operands** (`[4096, 16384]`, `[3072, 16384]`) produced by the same Linear boundary casts. This is still dtype conversion, not `memcpy` and not AdamW.

## Category Accounting Overlap

Phase 3.4 matchers were applied to this profiler's CUDA kernel names:

| Bucket | Kernel events |
| --- | ---: |
| copy/cast only | 4,910 |
| activation/elementwise only | 13,305 |
| both | **0** |
| other CUDA kernels | 9,445 |

**No overlap.** Phase 3.4's 27.47% activation/elementwise share is GELU / add / dropout / `addcmul` functors. It does not double-count `bfloat16_copy` or `direct_copy`. The 23.78% copy/cast share stands.

## Recommended Single Optimization for Phase 4.2

**Keep the transformer hidden/residual stream in BF16** (cast once after embedding; stay BF16 through the layers; upcast only at loss). **Leave FP32 parameter storage and the FP32 AdamW optimizer unchanged.** Do not change the attention backend, micro-batch, or software versions.

This is the single lever that hits the residual `x.to(residual.dtype)` path, the Linear activation autocasts at hidden/FFN boundaries, and the 161.6 ms vectorized `bfloat16_copy` bulk.

Do **not** start with the QKV adapter (only 13.14 ms/step) or with a BF16 weight cache (parameter shapes are not the top copy sites).

No A/B 20-warmup + 100-step validation until that change is implemented and looks promising.

## What Was Not Collected

- No 100-step throughput/MFU rerun
- No Nsight Systems timeline
- No Nsight Compute
- No redundant Phase 3.4 category totals beyond the overlap check and the kernel-time cross-check

The Pod (`3ixl2btmmwghn5`) was stopped immediately after the profiler JSON was collected.
