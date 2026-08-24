# Phase 5.1 Activation/Elementwise Kernel-Fusion Design

## Decision

This is a source-and-results-only design phase. No Pod was started, no GPU
experiment was run, and no model or environment code was changed.

The single Phase 5.2 experiment should enable Megatron's existing compiled
**bias + dropout + residual-add (BDA) fusion**:

```python
config = TransformerConfig(
    ...,
    bias_dropout_fusion=True,
)
```

The local GPT layer spec already sends both `self_attn_bda` and `mlp_bda`
through `get_bias_dropout_add`, so no module replacement or custom CUDA kernel
is required. This changes only the implementation of the existing BDA
expression. It retains the current FP32 residual stream, BF16 autocast, FP32
parameters, FP32 AdamW state, cuDNN FusedAttention, model architecture, and
micro-batch size.

Expected Phase 5.2 outcome: remove about **96 forward kernel launches/step**
(two eliminated launches for each of 48 BDA calls) and save an estimated
**20-40 ms/step**, corresponding to roughly **1.9-3.8% throughput** at the
Phase 3.4 `1085.41 ms` endpoint. This is an estimate, not a measured result.

## Evidence and baseline

The baseline is the Phase 3.4 fused-attention MB=8 configuration:

- 24 layers, hidden size 1024, FFN size 4096, sequence length 2048,
  micro-batch 8, vocabulary 50,304, TP/PP/DP 1/1/1.
- BF16 forward/backward autocast; FP32 parameter storage and FP32 AdamW state.
- Local Megatron GPT layer spec with only `core_attention` replaced by
  `TEDotProductAttention`, forced to cuDNN FusedAttention.
- Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`;
  Transformer Engine `4329ff84bfbdaa778a33cba02a15fb0807c64689`.
- Phase 3.4: `1085.41 ms/step`, `15,094.76 tokens/s`, `24.43%` MFU,
  `1076.82 ms/step` total CUDA-kernel time, and 4,414 kernels/step.

Evidence comes from:

1. Phase 3.4's 15-step Nsight Systems result
   (`ae19800:results/phase3_reprofile.json`);
2. Phase 4.1's unchanged-configuration five-step PyTorch Profiler capture,
   used only for category event counts and the copy/category overlap check;
3. the pinned Megatron-LM and Transformer Engine source.

The Phase 3.4 trace itself is on the stopped Pod rather than in Git. The
committed result retains only the top 15 kernel families and broad category
totals. Consequently, this document reports exact values where those artifacts
support them, bounds where the top-15 cutoff permits them, and leaves the
remaining time unresolved rather than inventing attribution.

## Correction to the 27.5% category

The reported `activation_elementwise` total is **295.819 ms/step**, or
**27.471% of CUDA kernel time**. It is not an exclusive or complete
activation total.

The Phase 3.4 matcher is:

```text
gelu, dropout, lerp, addcmul, mul_functor, binaryfunctor,
unaryfunctor, silu, relu
```

It has three important consequences:

1. It falsely includes GEMMs whose CUTLASS symbol contains `gemm_relu`.
   The top-15 list alone proves at least **119.459 ms/step and 97
   calls/step** of GEMM overlap. The configured model uses GELU, not ReLU.
2. It includes AdamW `mul`, `lerp`, and `addcmul` kernels. The unchanged
   optimizer's MB=1 profile gives an approximately fixed **32.227 ms/step
   and 1,460 calls/step** lower-bound proxy for this overlap; total AdamW
   time stays essentially constant from MB=1 to MB=8.
3. It does **not** match generic `CUDAFunctor_add`, subtraction, division,
   reduction, `masked_scale`, or most cross-entropy symbols. Three generic
   add families in the Phase 3.4 top 15 therefore contribute another exact
   **89.483 ms/step and 198 calls/step outside** the 27.5% bucket.

Phase 4.1 applied the same matcher to the unchanged workload and observed
2,661 activation-matched kernel events/step. It also proved zero overlap with
the separate copy/cast matcher. Thus copy/cast is not double-counted, but GEMM
and optimizer work are.

### What can be separated exactly

| Component | ms/step | Kernel calls/step | Accounting status |
| --- | ---: | ---: | --- |
| CUTLASS `gemm_relu` name collision | **at least 119.459** | **at least 97** | Inside 295.819; also GEMM; not a standalone activation |
| AdamW matched functors | **about 32.227** | **about 1,460** | Inside 295.819; also optimizer; MB=1 proxy |
| GELU backward | **34.369** | **24** | Inside 295.819; exact Phase 3.4 top-15 value |
| Remaining matched pool | **at most 109.764** | **at most 1,080** | Forward GELU, forward dropout, loss/model functors, and additional false positives |
| Top three generic add families | **89.483** | **198** | Outside 295.819 because the matcher omits plain add |
| Generic reduction family | **26.391** | **96** | Outside 295.819; mixed loss and gradient-reduction sources |

The remaining-pool bound subtracts the two measured/proxied overlaps and
GELU backward from 295.819 ms/step. Additional non-top-15 `gemm_relu`
families would make the true pool smaller.

## Operation-level breakdown and available fusions

### GELU and bias + GELU

1. **Time:** GELU backward is exactly `34.369 ms/step`. The forward GELU
   family is not in the committed top 15, so any one omitted kernel family is
   at most `23.287 ms/step`. The separate FC1 bias-add time is mixed into the
   generic add families and cannot be isolated.
2. **Calls:** 24 forward GELUs and 24 backward GELUs per step; one MLP per
   layer. FC1 also performs 24 separate forward bias additions.
3. **Source:** `megatron/core/transformer/mlp.py::MLP.forward`; with the
   current flag it executes `intermediate_parallel + bias_parallel` followed
   by `F.gelu`.
4. **Current fusion:** not fused. `TransformerConfig.bias_activation_fusion`
   defaults to `False`, and the lab does not override it.
5. **Available fusion:** pinned Megatron provides
   `fusions/fused_bias_gelu.py::bias_gelu_impl`, selected by
   `bias_activation_fusion=True`.
6. **Exact change:** add only `bias_activation_fusion=True` to
   `TransformerConfig`.
7. **Expected launch removal:** at least 24 forward launches/step by merging
   FC1 bias add with GELU. Backward may also compile the GELU derivative and
   bias gradient expression more efficiently, but no backward launch saving
   is credited without a profile.
8. **Expected benefit:** approximately `15-30 ms/step` (`1.4-2.8%` of the
   endpoint), with medium confidence.
9. **Risk:** medium. Megatron's fused function uses tanh-approximate GELU,
   whereas the current default `F.gelu` path is exact GELU. It therefore needs
   an explicit forward/loss/gradient comparison and is not the first
   recommendation despite the easy configuration change.

### Dropout, bias + dropout + add, and residual add

1. **Time:** the committed result does not separately retain dropout time.
   A standalone dropout kernel family omitted from the top 15 is bounded by
   `23.287 ms/step`; multiple symbol variants could sum above that. The two
   adds in BDA contribute to the exact `89.483 ms/step`/198-call generic-add
   aggregate, but that aggregate also contains non-BDA adds.
2. **Calls:** 48 BDA calls/step (attention and MLP BDA in each of 24 layers).
   There is also one embedding dropout. Source structure therefore predicts
   49 standalone dropout forward calls and 49 dropout backward calls. The 24
   attention-dropout operations are internal to cuDNN FusedAttention.
3. **Source:** `TransformerLayer._apply_self_attn_bda_step`,
   `TransformerLayer._apply_mlp_bda_step`, and
   `fusions/fused_bias_dropout.py::_bias_dropout_add_func`.
4. **Current fusion:** BDA is unfused. Each training call executes bias add,
   `torch.nn.functional.dropout`, and residual add separately because
   `bias_dropout_fusion=False`.
5. **Available fusion:** pinned Megatron already provides
   `bias_dropout_add_fused_train` and `bias_dropout_add_fused_inference`.
   On PyTorch 2.8, Megatron's `jit_fuser` resolves to `torch.compile`.
6. **Exact change:** set `bias_dropout_fusion=True`. The current local layer
   spec already uses `get_bias_dropout_add` at both BDA sites.
7. **Expected launch removal:** 48 calls × (3 current forward kernels - 1
   compiled fused kernel) = **about 96 forward launches/step**. No backward
   saving is assumed in the estimate.
8. **Expected benefit:** **20-40 ms/step**, or about **1.9-3.8% throughput**.
   The estimate is anchored by the measured 89.483 ms generic-add aggregate
   and discounts both non-BDA adds and work that remains inside the fused
   kernel.
9. **Risk:** low. The fused and unfused entry points call the same
   `_bias_dropout_add_func`; parameter dtypes, residual dtype, dropout
   probability, and mathematical expression do not change. The screen must
   exclude first-use compilation and verify dropout behavior because compiled
   RNG need not reproduce eager masks bit-for-bit.

There is no separate higher-value residual-add switch in the pinned local
layer. BDA fusion is the supported way to combine residual addition with its
producer-side bias and dropout.

### Elementwise multiply/add

1. **Time/calls:** approximately `32.227 ms/step` and 1,460 calls/step of the
   reported category are fixed AdamW `mul`/`lerp`/`addcmul` kernels. Generic
   adds contribute `89.483 ms/step` and 198 calls/step outside the category.
2. **Source:** `torch.optim.AdamW` for the fixed optimizer functors; BDA,
   MLP bias, loss, and gradient paths for the mixed generic adds.
3. **Fusion state:** AdamW is intentionally `foreach=False, fused=False`.
   BDA and FC1 bias add are unfused as described above.
4. **Available implementation:** PyTorch has fused optimizer variants, but
   optimizer work is only `53.062 ms/step`, optimizer implementation is not
   the Phase 5 activation target, and changing it would confound this study.
5. **Change/launch/benefit/risk:** no optimizer change is proposed. The
   actionable model-side add launches are covered by BDA and bias+GELU.

### Masking and attention elementwise work

1. **Time/calls:** no standalone attention `masked_fill`, attention softmax,
   or attention-dropout kernel is visible in the Phase 3.4 top 15. Their work
   is inside the cuDNN SDPA kernels counted as attention.
2. **Source:** `TEDotProductAttention` with `AttnBackend.fused`; the causal
   mask is passed into cuDNN FusedAttention.
3. **Fusion state:** already fused.
4. **More-fused option:** `masked_softmax_fusion=True` applies to Megatron's
   local attention path, not the active TE cuDNN path. Enabling it would not
   further fuse the active attention implementation.
5. **Change/launch/benefit/risk:** no change; expected endpoint gain is zero.

Loss target masking remains part of the cross-entropy implementation discussed
next. The all-ones lab `loss_mask` is applied afterward by the lab script.

### Loss-related elementwise kernels

1. **Time:** not separately retained. It is part of the at-most
   `109.764 ms/step` matched remainder, while several loss `sub`, `exp`,
   `div`, mask, and reduction kernels are outside the original matcher.
   The `26.391 ms/step` reduction family is mixed and is only an upper bound
   for loss reductions.
2. **Calls:** one cross-entropy call/step. Source inspection predicts roughly
   18-22 eager tensor/reduction launches across forward and backward, but this
   launch count is source-derived rather than measured.
3. **Source:** `LanguageModule.compute_language_model_loss` currently falls
   through to `tensor_parallel.vocab_parallel_cross_entropy`; the lab then
   applies `scripts/phase1_baseline.py::masked_language_model_loss`.
4. **Current fusion:** not fused because
   `cross_entropy_loss_fusion=False`.
5. **Available fusion:** pinned Megatron provides
   `fused_vocab_parallel_cross_entropy`; pinned TE also provides
   `parallel_cross_entropy`.
6. **Exact safe candidate change:** set
   `cross_entropy_loss_fusion=True` and explicitly retain
   `cross_entropy_fusion_impl="native"`.
7. **Expected launch removal:** about 8-12 launches/step by compiling the
   max/predicted-logit/loss/gradient groups and batching two collectives.
   At TP=1, collective batching has little value; reduced full-logit memory
   passes matter more than launch count.
8. **Expected benefit:** plausibly `20-50 ms/step` (`1.9-4.8%` throughput),
   but confidence is lower than for BDA because the committed trace did not
   retain a loss-only subtotal. The logits contain 824,180,736 values
   (`[2048, 8, 50304]`), so each avoided full FP32 read/write pass moves about
   6.59 GB less data.
9. **Risk:** medium for the native path and high for the TE path. The native
   implementation is the pinned Megatron-recommended option but explicitly
   returns a BF16 logits gradient and therefore requires a gradient-equivalence
   screen. The pinned TE 2.17.1 Triton kernel writes FP32-computed gradients
   back into the input buffer's dtype; that is the numerical issue fixed only
   after this pin. TE cross-entropy fusion is therefore rejected for Phase
   5.2.

The outer lab loss mask still performs a multiply and reductions after either
cross-entropy implementation. No pinned Megatron/TE switch fuses that lab-local
reduction, and a custom kernel is not justified before profiling the native
cross-entropy option.

### Other activation kernels

The unresolved matched pool is an upper bound, not an extra additive category.
It contains forward GELU, forward dropout, loss/model unary/binary functors,
and any additional `relu`-named GEMM false positives below the top-15 cutoff.
The existing artifacts cannot split it further without reopening the stopped
Phase 3.4 trace or collecting a new profile, both outside this design-only
phase.

## Existing fused versus bypassed paths

| Path | Current state | Finding |
| --- | --- | --- |
| cuDNN attention + causal masking + attention dropout | Fused | Active and confirmed; no accidental bypass |
| Bias-dropout-add | **Bypassed** | Local spec is wired correctly, but the default `bias_dropout_fusion=False` selects eager operations |
| Bias-GELU | **Bypassed** | `bias_activation_fusion=False` selects separate bias add and exact GELU |
| Native fused cross entropy | **Bypassed** | `cross_entropy_loss_fusion=False` selects eager vocab-parallel CE |
| TE fused cross entropy | Bypassed, intentionally reject | Available but unsafe at pinned TE 2.17.1 for this BF16 training path |
| TE activation op | Bypassed | `use_te_activation_func=False`; setting it alone does not alter the local spec |
| Full TE lower-level layer spec / op-fused MLP | Bypassed intentionally | Phase 3 replaced only core attention; full TE would also replace linears and norms |
| Local masked-softmax fusion | Inapplicable | Active core attention is already cuDNN fused |

The lab is therefore accidentally leaving three existing fusion switches off,
but it is not accidentally bypassing attention fusion. The full TE layer spec
was deliberately excluded in Phase 3 to isolate attention and remains a
larger, confounded change.

## Candidate ranking

Ranking uses expected endpoint gain divided by implementation complexity, with
confidence and correctness risk used as tie-breakers.

| Rank | Candidate | Exact change | Expected endpoint gain | Complexity / risk | Reason |
| ---: | --- | --- | --- | --- | --- |
| **1** | Megatron compiled BDA | `bias_dropout_fusion=True` | `20-40 ms`, about `1.9-3.8%` throughput | One config field / low | 48 repeated sites, about 96 removable forward launches, direct measured add evidence, same expression |
| 2 | Megatron native fused CE | `cross_entropy_loss_fusion=True`, impl `native` | `20-50 ms`, about `1.9-4.8%` throughput, low confidence | Two fields / medium | Very large logits and fewer memory passes, but loss-only time was not retained and gradient equivalence needs care |
| 3 | Megatron bias+GELU | `bias_activation_fusion=True` | `15-30 ms`, about `1.4-2.8%` | One field / medium | At least 24 launches removed, but changes exact GELU to tanh approximation |
| 4 | Full TE layer spec with op-fused MLP | replace local layer spec; request `use_te_op_fuser=True` | Potentially `30-70 ms`, unmeasured | Broad module replacement / high | Could fuse LayerNorm-linear and GEMM/GELU paths, but changes linears and norms and is not an isolated fusion screen |

Optimizer fusion, precision conversion, BF16 residual changes, parameter dtype,
and optimizer-state dtype are intentionally excluded.

## Exactly one Phase 5.2 experiment

Run **current baseline versus `bias_dropout_fusion=True` only**.

The implementation should add the flag at model construction and assert that:

- the local layer spec and cuDNN FusedAttention backend are unchanged;
- FP32 parameters, FP32 residual stream, FP32 AdamW state, and BF16 autocast
  are unchanged;
- both 24 self-attention BDA and 24 MLP BDA sites select
  `bias_dropout_add_fused_train`;
- no BF16-residual helper or Phase 4 dtype patch is enabled.

Before timing, compare A/B with identical weights and dropout disabled:

- aggregate and per-token loss;
- final hidden/logit error;
- representative attention, MLP, and embedding/output gradients;
- gradient cosine similarity and NaN/Inf.

Then use a short same-Pod screen and profile after compile warmup. The mechanism
check is a reduction of approximately 96 forward BDA launches, accompanied by
lower `CUDAFunctor_add` and standalone dropout traffic. Do not require identical
dropout masks between eager and compiled implementations; with dropout enabled,
verify finite behavior and reproducibility within each variant instead.

This is the only recommended Phase 5.2 change. Native fused cross entropy should
remain the next design candidate only if BDA fusion does not deliver the
expected launch and endpoint reduction.
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
