# Phase 5.1 Kernel Fusion Design (Existing Components Only)

## Decision

This is a **design-only** analysis. No Pod was started, no GPU work was run, and
no model/config code was changed for an experiment.

At the pinned stack (Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`,
Transformer Engine `4329ff84bfbdaa778a33cba02a15fb0807c64689` /
`2.17.1+4329ff84`), the highest expected endpoint gain under the Phase 5
freeze—**precision, optimizer, attention, architecture, and MB=8 unchanged**—comes
from **replacing local Torch LayerNorm + local TP Linear boundaries with TE
`LayerNormLinear` / TE Linear modules already wired by Megatron**, not from
enabling isolated JIT bias/GELU or cross-entropy fusion alone.

**Recommended Phase 5.2 screen (in order):**

1. **Primary:** Hybrid local GPT layer → TE fused LN+Linear on QKV and MLP FC1
   (`TELayerNormColumnParallelLinear`), TE row/column linears for proj/FC2,
   keep `TEDotProductAttention` (cuDNN FusedAttention sub-backend 1).
2. **Cheap secondary:** Turn on Megatron `bias_dropout_fusion=True` (and
   optionally `bias_activation_fusion=True` only if tanh-approx GELU is
   accepted).
3. **Do not pursue next:** TE residual RMSNorm fusion, TE `LayerNormMLP` as a
   direct Megatron GPT submodule, TEFusedMLPGrouped/CuTe SwiGLU path, Apex
   persist LN, TE CE fusion as primary, or FP32→BF16 parameter storage.

Phase 4.4 already showed that fixing residual BF16 persistence alone yields
only ~3% tokens/s and does **not** shrink the ~256 ms/step copy/cast class.
Phase 5 must attack LN/Linear dtype boundaries and elementwise launch
families that remain after attention fusion.

## Frozen controls

Unchanged relative to Phase 3.4 / 4.x best config:

| Control | Value |
| --- | --- |
| Hardware target | 1× NVIDIA A40, Secure Cloud image used in Phase 3–4 |
| Model | 355,919,872-param GPT; 24× H=1024 / FFN=4096 / 16 heads / head dim 64 |
| Sequence / batch | S=2048, **MB=8**, TP/PP/DP=1/1/1 |
| Precision | BF16 CUDA autocast; **FP32** parameter storage; **FP32** AdamW state |
| Optimizer | `torch.optim.AdamW` (`foreach=False`, `fused=False`) |
| Attention | TE cuDNN FusedAttention sub-backend 1 (`TEDotProductAttention`) |
| Architecture | Pre-norm GPT, **LayerNorm** (not RMSNorm), non-GLU **GELU**, `add_bias_linear=True` |
| Software pins | PyTorch `2.8.0+cu128`, CUDA 12.8, NCCL 2.27.3, cuDNN 9.10.2, Megatron `09fde85e`, TE `4329ff84` |

## Current local GPT layer (lab baseline)

`scripts/phase1_baseline.py` builds:

```python
transformer_layer_spec = get_gpt_layer_local_spec()
# then only:
layer_spec.submodules.self_attention.submodules.core_attention = AutocastTEDotProductAttention
```

`TransformerConfig` keeps `params_dtype=pipeline_dtype=torch.float32`,
`bf16=False`, `fp16=False`, `add_bias_linear=True`, `gated_linear_unit=False`,
and leaves fusion flags at Megatron defaults (`bias_activation_fusion=False`,
`bias_dropout_fusion=False`, `cross_entropy_loss_fusion=False`,
`fused_residual_rmsnorm=False`, `gradient_accumulation_fusion=False`).

At Megatron `09fde85e`, `get_gpt_layer_local_submodules` selects
([`gpt_layer_specs.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/gpt/gpt_layer_specs.py#L359-L463)):

| Site | Local module |
| --- | --- |
| `input_layernorm` / `pre_mlp_layernorm` | `WrappedTorchNorm` → `torch.nn.LayerNorm` (Apex absent) |
| `linear_qkv` | local `ColumnParallelLinear` (`skip_bias_add=False`) |
| `core_attention` | lab override → `TEDotProductAttention` |
| `linear_proj` | local `RowParallelLinear` (`skip_bias_add=True`) |
| `self_attn_bda` / `mlp_bda` | `get_bias_dropout_add` (unfused because `bias_dropout_fusion=False`) |
| MLP `linear_fc1` / `linear_fc2` | local Column / Row (`skip_bias_add=True`) |
| Activation | separate `bias + gelu` (no `bias_activation_fusion`) |
| Block final norm | `TENorm` (because TE is installed; `TransformerBlock` sets `LayerNormImpl=TENorm`) |

`LocalSpecProvider.fuse_layernorm_and_linear()` returns `False` and
`column_parallel_layer_norm_linear()` returns `None`
([`backends.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/backends.py#L99-L127)).

Phase 3.4 MB=8 kernel mix (denominator = CUDA kernel time): GEMM 28.3%,
activation/elementwise 27.5%, **copy/cast 23.8% (256 ms/step)**, attention
13.2%, norm 6.6%, optimizer 4.9%; **4,414 kernels/step**. Top kernels include
`bfloat16_copy` (161.6 ms), `direct_copy` (80.5 ms), GELU backward (34.4 ms),
vectorized residual `add` (35.8 ms), and Torch layer-norm grad (~26.3 ms).

Phase 4.1 attributed copy/cast to FP32 residual/hidden mixing with BF16
autocast at Linear/LN boundaries plus the QKV `.to(bf16)` adapter
(72 calls/step). Phase 4.4 fixed residual persistence (+3.3% tokens/s) without
reducing copy/cast.

## Inventory at pinned TE / Megatron

### 1. Fused bias + GELU

| Source | What exists | Lab relevance |
| --- | --- | --- |
| Megatron `fusions/fused_bias_gelu.py` | TorchScript/`jit_fuser` **tanh-approx** GELU: `bias_gelu` / `GeLUFunction` | Enabled by `TransformerConfig.bias_activation_fusion=True` in `MLP.forward` when `activation_func=F.gelu` and `add_bias_linear=True` ([`mlp.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/mlp.py#L275-L307)). **Compatible** with local spec; config-only. |
| TE `pytorch/jit.py` | Same family: `bias_gelu_fused` / `bgrad_dgelu_fused` (NVFuser/JIT) | Used inside TE `LayerNormMLP` when `NVTE_BIAS_GELU_NVFUSION=1` and activation is `gelu`. |
| TE CUDA `tex.gelu` / `tex.dbias_dgelu` | Device GELU (+ fused dbias in non-FP8 BF16 recipes via LayerNormMLP) | Available in TE `4329ff84`; **not** selected by local Megatron MLP unless TE activation / LayerNormMLP path is used. |
| TE op-fuser `ForwardLinearBiasActivation` | Registers GEMM+bias fusion; **`activation` always passed as `None`** in the matcher; requires `weight.dtype in {fp16,bf16}` ([`forward_linear_bias_activation.py`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/ops/fused/forward_linear_bias_activation.py#L185-L196)) | **Incompatible with FP32 `params_dtype`:** fusion matcher rejects FP32 weights. Even under TEFusedMLP, this path does not fire for the lab precision policy. |

**Numerical note:** Megatron/TE JIT bias-GELU is the OpenAI tanh approximation, not
`erf` GELU. Enabling `bias_activation_fusion` is therefore a **numerics change**,
not a pure scheduling change.

### 2. LayerNormLinear / LayerNormMLP

| Component | TE `4329ff84` | Megatron `09fde85e` GPT wiring |
| --- | --- | --- |
| `te.pytorch.LayerNormLinear` | Full fused LN→Linear module | Wrapped as `TELayerNormColumnParallelLinear` ([`transformer_engine.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/extensions/transformer_engine.py#L1378-L1541)); selected by `TESpecProvider.column_parallel_layer_norm_linear()` |
| `te.pytorch.LayerNormMLP` | Fuses LN + FC1 + activation + FC2; supports BF16 `bias_gelu_fusion` / optional `NVTE_GEMM_GELU_FUSION` | **Not referenced** by Megatron GPT layer specs. TE's own `TransformerLayer` uses it ([`transformer.py`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/transformer.py#L497-L501)); Megatron GPT does **not**. |
| TE op-fuser `TEFusedMLP` | LN + BasicLinear + Bias + GELU + BasicLinear Sequential | Available when `use_te_op_fuser=True` and TE ≥ 1.13 ([`gpt_layer_specs.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/gpt/gpt_layer_specs.py#L521-L551)). Requires FC1 to already be `TELayerNormColumnParallelLinear`. **GEMM+bias fuse still blocked by FP32 weights** (above). |

Full TE GPT dense submodules
([`get_gpt_layer_with_transformer_engine_submodules`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/gpt/gpt_layer_specs.py#L318-L347)):

- No separate `input_layernorm` (folded into `linear_qkv=TELayerNormColumnParallelLinear`)
- `linear_proj=TERowParallelLinear`
- `pre_mlp_layernorm=IdentityOp` for dense MLP (folded into FC1 LNLinear)
- `mlp` uses `TELayerNormColumnParallelLinear` FC1 + `TERowParallelLinear` FC2
- Still uses Megatron `get_bias_dropout_add` for both residual BDA sites
- Default `core_attention=TEDotProductAttention` (already matches lab attention)

### 3. Bias-dropout-add

| Source | Mechanism | Lab status |
| --- | --- | --- |
| Megatron `fused_bias_dropout.py` | Unfused Python path vs `@jit_fuser` train/infer kernels selected by `get_bias_dropout_add(training, fused)` | Spec already passes `get_bias_dropout_add`; **fusion off** because `bias_dropout_fusion` defaults `False`. Config-only enable. |
| TE `pytorch/jit.py` | Parallel JIT BDA helpers; TE `TransformerLayer` defaults `NVTE_BIAS_DROPOUT_FUSION=1` | Not used by Megatron GPT residual path; Megatron keeps its own BDA. |

Phase 4.3 documented the BDA bias dtype guard bug; Phase 4.4 fixed persistence
in-lab. Enabling JIT BDA is orthogonal and still valid under BF16 residuals.

### 4. Residual fusion

Megatron `TENorm` residual fusion is a two-level opt-in
([`transformer_engine.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/extensions/transformer_engine.py#L1048-L1105)):

1. `config.fused_residual_rmsnorm=True`
2. `has_residual=True` at the build site

Fusion constructs `TEFusedResidualRMSNorm` and **raises if normalization ≠
RMSNorm**. Config validation repeats that constraint
([`transformer_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/transformer_config.py#L2431-L2435)).

**Unavailable under frozen architecture** (lab is LayerNorm GPT). Switching to
RMSNorm would violate the architecture freeze.

### 5. Fused cross entropy / masking

| Path | Pin support | Lab status |
| --- | --- | --- |
| Default `tensor_parallel.vocab_parallel_cross_entropy` | Always | Current: `cross_entropy_loss_fusion=False` |
| Megatron native fused CE | `cross_entropy_loss_fusion=True`, `cross_entropy_fusion_impl='native'` → `fused_vocab_parallel_cross_entropy` | Available; JIT-fuses CE helper stages; TP=1 so all-reduce fusion benefit is small |
| TE `parallel_cross_entropy` (Triton) | Wired as `te_parallel_cross_entropy` when TE import succeeds ([`transformer_engine.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/extensions/transformer_engine.py#L3596-L3617)) | Megatron **warns of known stability issues** for `impl='te'` ([`model_parallel_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/model_parallel_config.py#L536-L542)) |
| Attention masking fusion | Local `masked_softmax_fusion` / Apex softmax | **Irrelevant**—core attention is already TE cuDNN fused; local softmax path is gone |

CE is not among Phase 3.4 top kernels; expect **low** endpoint gain.

### 6. TE module vs local tensor-parallel Linear

| Aspect | Local `ColumnParallelLinear` / `RowParallelLinear` | TE `TEColumnParallelLinear` / `TERowParallelLinear` / `TELayerNormColumnParallelLinear` |
| --- | --- | --- |
| Weight storage | FP32 (`params_dtype`) | FP32 via `_get_extra_te_kwargs(config)` → `params_dtype` |
| Compute | `torch.matmul` under BF16 autocast | TE `general_gemm` with `cast_if_needed(..., activation_dtype)` |
| Bias return | `skip_bias_add` returns separate FP32 bias → BDA/GELU promotion issues (Phase 4) | Same Megatron wrapper contract (`te_return_bias`); bias cast toward `activation_dtype` inside TE |
| LN fusion | None | LNLinear fuses LN stats/affine with QKV or FC1 GEMM preparation |
| Grad accum fusion | Needs Apex `fused_weight_gradient_mlp_cuda` | `fuse_wgrad_accumulation` flag exists, but lab AdamW has no Megatron `main_grad` path—**not useful** without optimizer/stack change |
| Op-fuser GEMM+bias | N/A | Requires BF16/FP16 **parameter** dtype; blocked under FP32 storage |

TE Linears are the supported way to reduce LN↔Linear and autocast boundary
traffic **without** changing the mathematical GPT shape. They do **not** by
themselves remove all FP32↔BF16 copies while master weights stay FP32; they
change *where* casts happen and can eliminate Torch LN's forced FP32 output
policy and the lab QKV adapter when Q/K/V already match autocast dtype.

## Comparison to the lab layer

```text
Current local + TE attention only
  Torch LN  →  local Col(QKV)+FP32 bias  →  .to(bf16) adapter  →  TE FA
            →  local Row(proj)  →  unfused BDA
  Torch LN  →  local Col(FC1)  →  bias+GELU  →  local Row(FC2)  →  unfused BDA
  TENorm (final)

Supported TE GPT dense (same attention backend)
  TE LNLinear(QKV)  →  TE FA  →  TE Row(proj)  →  BDA (still Megatron)
  TE LNLinear(FC1)+act  →  TE Row(FC2)  →  BDA
  TENorm (final)
```

Gaps the TE path closes relative to Phase 3.4/4.x bottlenecks:

- Removes per-layer Torch LN FP32 autocast policy (Phase 4.3).
- Folds LN output → QKV/FC1 into one TE module (fewer launches + less
  intermediate traffic).
- Makes QKV compute dtype TE-native, allowing removal of
  `AutocastTEDotProductAttention` casts if Q/K/V already BF16.
- Optionally folds FC1 bias+GELU via TE kernels / Megatron
  `bias_activation_fusion` depending on path.

Gaps it does **not** close under the freeze:

- FP32 master weights still need BF16 compute views (some cast traffic remains).
- Residual BDA remains a separate Megatron op (unless JIT-fused).
- Attention is already fused; no further attention gain expected.
- True LayerNormMLP / residual-add fusion either unwired or RMSNorm-only.

## Ranked viable existing-component changes

Rank key: **endpoint gain / implementation complexity**, subject to freeze.
Gains are **planning estimates** from Phase 3.4/4.x attribution, not measured
Phase 5 results.

| Rank | Candidate | Complexity | Expected endpoint effect | Expected launch / kernel removal | Risks |
| ---: | --- | --- | --- | --- | --- |
| 1 | **TE LNLinear hybrid / full TE GPT layer spec** (keep TE FA, FP32 params, AdamW) | Medium (spec swap + correctness gate) | **Highest.** Targets copy/cast (~24%) and norm (~7%) plus some elementwise; plausible **high-single-digit to low-double-digit** tokens/s if 30–50% of copy/cast and part of LN/GELU boundary work disappear | Remove 48 Torch LN fwd sites folded into LNLinear; remove or no-op 72 QKV adapter casts/step; fold FC1 LN; reduce intermediate elementwise between LN and GEMM. Directionally hundreds of launches/step from 4,414 baseline | Weight-name / init differences vs local; TE vs Torch LN numerics; possible new TE weight-cast traffic; must keep BF16 autocast + FP32 params |
| 2 | **`bias_dropout_fusion=True`** (+ optional `bias_activation_fusion=True`) | Very low (config flags) | **Moderate alone.** Attacks activation/elementwise (GELU bwd 34 ms, residual adds ~35–30 ms) but **not** the #1 copy kernel. Plausible **low- to mid-single-digit** tokens/s | Fuse 48 BDA sites (bias+dropout+add); with bias-GELU, fuse 24 FC1 bias+GELU fwd (+ paired bwd). JIT may still emit multiple kernels | Tanh-approx GELU if bias-activation on; RNG/order differences for dropout; must re-check BF16 residual + FP32 bias interaction |
| 3 | **Per-layer `TENorm` only** (keep local Linears) | Low | **Moderate.** Removes Torch LN FP32 output policy on 48 norms; helps copy/cast and norm categories without TE Linear | Replace Torch LN kernels with TE LN; fewer BF16↔FP32 transitions on norm outputs | Norm numerical delta vs Torch; does not fuse into GEMM; QKV adapter may remain |
| 4 | **Native `cross_entropy_loss_fusion=True`** | Very low | **Low.** CE not in Phase 3.4 top kernels | Fuses CE helper stages; tiny launch win at TP=1 | Small; prefer `impl='native'` over TE |
| 5 | **`use_te_op_fuser=True` / `TEFusedMLP`** on top of TE MLP | High | **Low under FP32 params**—op-fuser GEMM+bias matcher rejects FP32 weights | Little extra vs TE LNLinear MLP | Complexity without payoff under freeze |
| — | TE residual RMSNorm fusion | — | — | — | **Incompatible** (architecture) |
| — | Direct TE `LayerNormMLP` GPT submodule | — | — | — | **Unavailable** in Megatron GPT specs at `09fde85e` |
| — | `TEFusedMLPWithGroupedLinear` | — | — | — | **Incompatible** (SwiGLU + `add_bias_linear=False`, TE≥2.14 CuTe path) |
| — | Apex `FusedLayerNorm` / `persist_layer_norm` | — | — | — | **Unavailable** (Apex not installed; lab historically avoided adding it) |
| — | `gradient_accumulation_fusion` | — | — | — | **Incompatible** with torch AdamW / no Apex wgrad ext / no `main_grad` |
| — | TE CE `cross_entropy_fusion_impl='te'` as primary | — | — | — | Available but Megatron-flagged **stability** risk; low gain |
| — | BF16 parameter storage / fused AdamW | — | — | — | Violates **precision** / **optimizer** freeze |

## Recommended candidate(s) with exact substitutions

### Candidate A — Primary: TE fused LN+Linear GPT layer (preferred)

**Intent:** One controlled module-spec change that hits the Phase 3.4 copy/cast
and LN boundaries while freezing attention backend, MB=8, FP32 params/AdamW,
LayerNorm+GELU GPT shape.

**Exact substitution:**

```python
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.extensions.transformer_engine import TEDotProductAttention

layer_spec = get_gpt_layer_with_transformer_engine_spec()
# Attention already TEDotProductAttention; keep lab autocast adapter only if
# Q/K/V dtype still mismatches under FP32 params + BF16 autocast:
layer_spec.submodules.self_attention.submodules.core_attention = (
    AutocastTEDotProductAttention  # delete if TE LNLinear already emits BF16
)

config = TransformerConfig(
    # unchanged shapes / dropouts / eps / add_bias_linear=True /
    # gated_linear_unit=False / params_dtype=fp32 / pipeline_dtype=fp32 /
    # bf16=False / fp16=False / attention_backend=AttnBackend.fused
    bias_dropout_fusion=True,          # optional same-change bundling
    # bias_activation_fusion=False until tanh-GELU policy is accepted
)
```

Module mapping versus current local spec:

| Current | Replacement |
| --- | --- |
| `input_layernorm=WrappedTorchNorm` + `ColumnParallelLinear` QKV | `TELayerNormColumnParallelLinear` (no separate input LN) |
| `RowParallelLinear` proj | `TERowParallelLinear` |
| `pre_mlp_layernorm=WrappedTorchNorm` + `ColumnParallelLinear` FC1 | `TELayerNormColumnParallelLinear` FC1 |
| `RowParallelLinear` FC2 | `TERowParallelLinear` |
| `TEDotProductAttention` | unchanged backend |
| BDA factories | unchanged type; enable `bias_dropout_fusion` |

**Expected launch / time removal (planning):**

- Eliminate folded Torch LN launches (24×2 fwd + matching bwd family).
- Eliminate or no-op QKV adapter `bfloat16_copy` elementwise family (**72/step,
  ~13 ms** in Phase 4.1).
- Reduce Linear-boundary `bfloat16_copy` / `direct_copy` traffic that Phase 4.1
  tied to FFN-width and hidden-stream casts (largest remaining copy share after
  residual fix).
- Net: target **material drop from ~256 ms/step copy/cast** and some of
  **~71 ms/step norm**; endpoint **~8–20% tokens/s** if half of copy/cast plus
  modest LN/elementwise savings realize—**not a guarantee**.

**Risks:** TE vs Torch LN numerics; state_dict key layout
(`layer_norm_weight` inside LNLinear); CPU init path differences; verify fused
attention sub-backend 1 still selected; do not enable FP8.

### Candidate B — Minimal config screen (optional precursor)

```python
TransformerConfig(
    ...,
    bias_dropout_fusion=True,
    # bias_activation_fusion=True  # only if tanh-approx GELU accepted
)
```

Keep `get_gpt_layer_local_spec()` + current TE attention override.

**Expected:** fuse 48 BDA graphs; optionally fuse 24 bias+GELU; **low-single-digit**
tokens/s; does not fix dominant copy/cast (Phase 4.4 lesson).

**Risks:** tanh GELU if activation fusion enabled; dropout RNG divergence vs
unfused train.

### Candidate C — Native CE fusion (optional, last)

```python
TransformerConfig(
    ...,
    cross_entropy_loss_fusion=True,
    cross_entropy_fusion_impl="native",  # not "te"
)
```

**Expected:** negligible endpoint gain; useful only as zero-cost secondary once
Candidate A lands.

## Explicitly unavailable / incompatible at pinned versions

| Option | Status at TE `4329ff84` / Megatron `09fde85e` | Why blocked for Phase 5.1 |
| --- | --- | --- |
| `fused_residual_rmsnorm` | Implemented for RMSNorm only | Architecture freeze = LayerNorm |
| TE `LayerNormMLP` as Megatron GPT submodule | Exists in TE; **not** in Megatron GPT specs | Would be a new integration, not an existing GPT substitution |
| `TEFusedMLPWithGroupedLinear` / CuTe GEMM-SwiGLU | Requires SwiGLU, no bias, TE≥2.14 patterns | Architecture (`GELU`, `add_bias_linear=True`) incompatible |
| TE op-fuser `ForwardLinearBiasActivation` under FP32 weights | Matcher requires BF16/FP16 `weight.dtype` | Precision freeze = FP32 parameter storage |
| Apex persist / fused LN | Code present; Apex missing in lab | Dependency not installed; prior phases avoided Apex |
| Local `gradient_accumulation_fusion` | Needs Apex CUDA ext + `main_grad` | Optimizer freeze = torch AdamW |
| TE CE fusion as default | Importable; Megatron stability warning | Low gain + known risk |
| Kitchen / inference-optimized specs | Stubs or different product path | Out of scope / unavailable in this lab image |
| FlashAttention package swap | Supported by TE selector but not required | Attention freeze: keep cuDNN fused sub-backend 1 |

## Validation plan for Phase 5.2 (not executed here)

1. Dropout-off weight-tied forward/backward compare vs current MB=8 TE-attention
   baseline; record max/mean abs/rel errors and gradient cosines.
2. Assert attention backend log still shows FusedAttention sub-backend 1.
3. Assert parameter and AdamW state dtypes remain FP32; masked loss reduction
   remains FP32.
4. Short screen: 3 warmup + 10 timed steps; compare tokens/s, MFU, step time,
   peak VRAM vs Phase 3.4 (~15,085 tok/s, 24.42% MFU).
5. Profiler: confirm reduction in `bfloat16_copy` / `direct_copy` and disappearance
   of per-layer Torch LN symbols; count kernels/step vs 4,414.
6. Promote to 20+100 only if short screen clears a **≥5%** tokens/s gate (same
   bar used in Phase 4.4) with acceptable numerics.

## Sources (pinned)

- Megatron local / TE GPT specs:
  [`gpt_layer_specs.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/gpt/gpt_layer_specs.py)
- Megatron TE wrappers:
  [`transformer_engine.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/extensions/transformer_engine.py)
- Megatron TE backend provider:
  [`transformer_engine_spec_provider.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/extensions/transformer_engine_spec_provider.py)
- Megatron BDA / bias-GELU / CE fusions:
  [`fused_bias_dropout.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/fusions/fused_bias_dropout.py),
  [`fused_bias_gelu.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/fusions/fused_bias_gelu.py),
  [`fused_cross_entropy.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/fusions/fused_cross_entropy.py),
  [`language_module.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/common/language_module/language_module.py)
- TE LayerNormLinear / LayerNormMLP / op-fuser / CE:
  [`layernorm_linear.py`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/module/layernorm_linear.py),
  [`layernorm_mlp.py`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/module/layernorm_mlp.py),
  [`forward_linear_bias_activation.py`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/ops/fused/forward_linear_bias_activation.py),
  [`cross_entropy.py`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/cross_entropy.py)
- Lab evidence: Phase 3.4 reprofile, Phase 4.1 copy attribution, Phase 4.3/4.4
  residual design/screens (branches
  `cursor/phase34-reprofile-3b5c`,
  `cursor/phase41-copy-attribution-3b5c`,
  `cursor/phase43-residual-dtype-design-3b5c`,
  `cursor/phase44-persistent-bf16-residual-3b5c`).
