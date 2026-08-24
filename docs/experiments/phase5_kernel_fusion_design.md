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

This matcher has three important consequences:

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

### Separable measured components

| Component | ms/step | Kernel calls/step | Accounting status |
| --- | ---: | ---: | --- |
| CUTLASS `gemm_relu` name collision | **at least 119.459** | **at least 97** | Inside 295.819; also GEMM; not a standalone activation |
| AdamW matched functors | **about 32.227** | **about 1,460** | Inside 295.819; also optimizer; MB=1 proxy |
| GELU backward | **34.369** | **24** | Inside 295.819; exact Phase 3.4 top-15 value |
| Remaining matched pool | **at most 109.764** | **at most 1,080** | Forward GELU/dropout, loss/model functors, further false positives |
| Top three generic add families | **89.483** | **198** | Outside 295.819 because the matcher omits plain add |
| Generic reduction family | **26.391** | **96** | Outside 295.819; mixed loss and gradient-reduction sources |

The remaining-pool bound subtracts the measured/proxied overlaps and GELU
backward from 295.819 ms/step. Additional non-top-15 `gemm_relu` families
would make the true pool smaller.

## Operation-level breakdown and available fusions

### GELU and bias + GELU

1. **Time:** GELU backward is exactly `34.369 ms/step`. The forward GELU
   family is not in the committed top 15, so any one omitted family is at
   most `23.287 ms/step`. FC1 bias-add time is mixed into generic adds.
2. **Calls:** 24 forward GELUs, 24 backward GELUs, and 24 separate FC1
   forward bias additions per step.
3. **Source:** `megatron/core/transformer/mlp.py::MLP.forward`.
4. **Current fusion:** off; `bias_activation_fusion` defaults to `False`.
5. **Available fusion:** pinned Megatron
   `fusions/fused_bias_gelu.py::bias_gelu_impl`.
6. **Exact change:** `bias_activation_fusion=True`.
7. **Expected removable launches:** at least 24 forward launches/step by
   merging FC1 bias add with GELU.
8. **Expected benefit:** approximately `15-30 ms/step` (`1.4-2.8%` of the
   endpoint), with medium confidence.
9. **Risk:** medium. Megatron's fused function uses tanh-approximate GELU,
   whereas the current `F.gelu` path is exact GELU. It is not the first
   recommendation because this is a numerical-function change.

### Dropout, bias + dropout + add, and residual add

1. **Time:** the committed result does not separately retain dropout time.
   A standalone dropout family omitted from the top 15 is bounded by
   `23.287 ms/step`; multiple variants could sum above that. BDA's two adds
   are part of the exact `89.483 ms/step`/198-call generic-add aggregate,
   which also contains non-BDA adds.
2. **Calls:** 48 BDA calls/step: attention and MLP BDA in each of 24 layers.
   Source structure predicts 49 standalone dropout forward and backward
   calls after adding embedding dropout. The 24 attention-dropout operations
   are internal to cuDNN FusedAttention.
3. **Source:** `TransformerLayer._apply_self_attn_bda_step`,
   `TransformerLayer._apply_mlp_bda_step`, and
   `fusions/fused_bias_dropout.py::_bias_dropout_add_func`.
4. **Current fusion:** off. Each BDA training call separately executes bias
   add, `torch.nn.functional.dropout`, and residual add.
5. **Available fusion:** pinned Megatron
   `bias_dropout_add_fused_train`/`bias_dropout_add_fused_inference`.
   With PyTorch 2.8, Megatron's `jit_fuser` resolves to `torch.compile`.
6. **Exact change:** `bias_dropout_fusion=True`; no module swap is needed.
7. **Expected removable launches:** 48 × (3 eager forward kernels - 1
   compiled kernel) = **about 96 forward launches/step**. No backward saving
   is credited without measurement.
8. **Expected benefit:** **20-40 ms/step**, or **1.9-3.8% throughput**.
   This discounts non-BDA adds and work that remains in the fused kernel.
9. **Risk:** low. Fused and unfused entry points call the same
   `_bias_dropout_add_func`; dtypes, dropout probability, and expression are
   unchanged. Compiled dropout masks need not match eager masks bit-for-bit.

There is no separate supported residual-add switch in the pinned local layer;
BDA fusion is the existing way to fuse the residual addition with producer
bias and dropout.

### Elementwise multiply/add

1. **Time/calls:** approximately `32.227 ms/step` and 1,460 calls/step of
   the reported category are fixed AdamW `mul`/`lerp`/`addcmul` kernels.
   Generic adds contribute `89.483 ms/step` and 198 calls/step outside it.
2. **Source:** `torch.optim.AdamW` for the fixed optimizer functors; BDA,
   MLP bias, loss, and gradient paths for the mixed generic adds.
3. **Fusion state:** AdamW is intentionally `foreach=False, fused=False`;
   BDA and FC1 bias add are unfused as described above.
4. **Available implementation:** fused optimizer variants exist, but
   optimizer work is only `53.062 ms/step` and is not this activation study.
5. **Change/launch/benefit/risk:** no optimizer change is proposed. The
   actionable model-side adds are covered by BDA and bias+GELU.

### Masking and attention elementwise work

1. **Time/calls:** no standalone attention `masked_fill`, softmax, or
   attention-dropout family is visible in the Phase 3.4 top 15.
2. **Source:** `TEDotProductAttention` with `AttnBackend.fused`; causal mask,
   softmax, and attention dropout are handled by cuDNN FusedAttention.
3. **Fusion state:** already fused.
4. **More-fused option:** `masked_softmax_fusion=True` targets Megatron's
   local attention, not the active TE cuDNN path.
5. **Change/launch/benefit/risk:** no change; expected gain is zero.

Loss target masking is part of cross entropy below. The lab applies its
all-ones `loss_mask` afterward in `masked_language_model_loss`.

### Loss-related elementwise kernels

1. **Time:** not separately retained. It is within the at-most
   `109.764 ms/step` matched remainder, while several loss `sub`, `exp`,
   `div`, mask, and reduction kernels are outside the original matcher.
   The mixed `26.391 ms/step` reduction family is only an upper bound.
2. **Calls:** one cross-entropy call/step; roughly 18-22 eager tensor and
   reduction launches are predicted from source, not measured.
3. **Source:** `LanguageModule.compute_language_model_loss` currently falls
   through to `tensor_parallel.vocab_parallel_cross_entropy`; the lab then
   applies `scripts/phase1_baseline.py::masked_language_model_loss`.
4. **Current fusion:** off; `cross_entropy_loss_fusion=False`.
5. **Available fusion:** pinned Megatron native
   `fused_vocab_parallel_cross_entropy`; pinned TE also has
   `parallel_cross_entropy`.
6. **Exact safe candidate:** `cross_entropy_loss_fusion=True` with
   `cross_entropy_fusion_impl="native"`.
7. **Expected removable launches:** about 8-12/step from compiled
   max/predicted-logit/loss/gradient groups. At TP=1, collective batching has
   little value; reduced full-logit memory passes matter more.
8. **Expected benefit:** plausibly `20-50 ms/step` (`1.9-4.8%` throughput),
   but with lower confidence because there is no measured loss subtotal.
   Logits contain 824,180,736 values (`[2048, 8, 50304]`); each avoided full
   FP32 read/write pass moves about 6.59 GB less data.
9. **Risk:** medium for native and high for TE. Native explicitly returns a
   BF16 logits gradient and needs an equivalence screen. Pinned TE 2.17.1
   writes FP32-computed gradients into the input buffer's dtype, the numerical
   problem fixed only after this pin. TE CE is rejected for Phase 5.2.

The outer lab loss mask still performs multiply/reduction after either CE
path. No pinned switch fuses that lab-local reduction.

### Other activation kernels

The unresolved matched pool is an upper bound, not an additive category. It
contains forward GELU/dropout, loss/model functors, and additional
`relu`-named GEMM false positives below the top-15 cutoff. Existing artifacts
cannot split it further without reopening the stopped trace or reprofiling.

## Existing fused versus bypassed paths

| Path | Current state | Finding |
| --- | --- | --- |
| cuDNN attention, causal mask, attention dropout | Fused | Active and confirmed; no accidental bypass |
| Bias-dropout-add | **Bypassed** | Correct factory is wired, but `bias_dropout_fusion=False` selects eager ops |
| Bias-GELU | **Bypassed** | `bias_activation_fusion=False` selects bias add + exact GELU |
| Native fused cross entropy | **Bypassed** | `cross_entropy_loss_fusion=False` selects eager CE |
| TE fused cross entropy | Intentionally reject | Available but unsafe at pinned TE 2.17.1 for this BF16 path |
| TE activation op | Bypassed | `use_te_activation_func=False`; setting it alone does not change the local spec |
| Full TE lower-level layer spec | Bypassed intentionally | Phase 3 replaced only attention; full TE also replaces linears and norms |
| Local masked-softmax fusion | Inapplicable | Active attention is already cuDNN fused |

The pinned TE op fuser is not a hidden drop-in GELU solution here:
`ForwardLinearBiasActivation` says activations are not yet supported, and its
linear+bias matcher requires FP16/BF16 stored weights. FP32 parameter storage
therefore blocks that fusion. Full TE `LayerNormLinear` may still alter
normalization/linear execution, but it is a broad module and copy-boundary
experiment rather than an isolated activation fusion.

## Candidate ranking

Ranking is expected endpoint gain divided by implementation complexity, with
confidence and numerical risk as tie-breakers.

| Rank | Candidate | Exact change | Expected endpoint gain | Complexity / risk | Reason |
| ---: | --- | --- | --- | --- | --- |
| **1** | Megatron compiled BDA | `bias_dropout_fusion=True` | `20-40 ms`; `1.9-3.8%` throughput | One field / low | 48 repeated sites, ~96 removable launches, direct add evidence, same expression |
| 2 | Megatron native fused CE | Fusion on, impl `native` | `20-50 ms`; `1.9-4.8%`, low confidence | Two fields / medium | Huge logits, but no measured loss subtotal and gradient equivalence needs care |
| 3 | Megatron bias+GELU | `bias_activation_fusion=True` | `15-30 ms`; `1.4-2.8%` | One field / medium | At least 24 launches, but changes exact GELU to tanh approximation |
| 4 | Full TE layer/op-fused MLP | Replace local spec; request op fuser | Low/uncertain for activation under FP32 stored weights | Broad / high | Activation fusion unsupported and linear+bias fusion's dtype gate fails |

Optimizer fusion, precision conversion, BF16 residual changes, parameter dtype,
and optimizer-state dtype are intentionally excluded.

## Exactly one Phase 5.2 experiment

Run **current baseline versus `bias_dropout_fusion=True` only**.

The implementation should add the flag at model construction and assert:

- local layer spec and cuDNN FusedAttention remain unchanged;
- FP32 parameters, FP32 residual stream, FP32 AdamW state, and BF16 autocast
  remain unchanged;
- all 24 self-attention and 24 MLP BDA sites select the fused train function;
- no BF16-residual helper or Phase 4 dtype patch is enabled.

Before timing, compare A/B with identical weights and dropout disabled:

- aggregate and per-token loss;
- final hidden/logit error;
- representative attention, MLP, and embedding/output gradients;
- gradient cosine similarity and NaN/Inf.

Then run a short same-Pod screen after compile warmup. The mechanism check is
approximately 96 fewer forward BDA launches with lower `CUDAFunctor_add` and
standalone dropout traffic. Do not require identical dropout masks between
eager and compiled implementations; with dropout enabled, verify finite
behavior and reproducibility within each variant.

This is the only recommended Phase 5.2 change.

## Pinned sources

- Megatron [GPT layer specs](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/gpt/gpt_layer_specs.py)
  and [MLP](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/mlp.py).
- Megatron [BDA](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/fusions/fused_bias_dropout.py),
  [bias-GELU](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/fusions/fused_bias_gelu.py),
  and [native CE](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/fusions/fused_cross_entropy.py).
- Megatron [language-model loss selection](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/common/language_module/language_module.py)
  and [fusion defaults](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/model_parallel_config.py).
- TE [op-fuser linear/bias matcher](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/ops/fused/forward_linear_bias_activation.py)
  and [pinned cross entropy](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/cross_entropy.py).
