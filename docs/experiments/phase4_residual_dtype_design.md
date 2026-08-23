# Phase 4.3 Residual Dtype Design

## Scope and conclusion

This is a source-only design analysis. No Pod, GPU experiment, package change, or
pinned-environment change was made.

The pinned stack is Megatron-LM `09fde85ea25fb67e9b32019089fae163a3233bd3`,
PyTorch `2.8.0+cu128`, and Transformer Engine
`4329ff84bfbdaa778a33cba02a15fb0807c64689`. The model uses the local GPT layer
spec, FP32 parameters, BF16 CUDA autocast, and a local
`AutocastTEDotProductAttention` substitution for cuDNN FusedAttention.

**The first FP32 reversion in Phase 4.2 is the first layer's self-attention
bias-dropout-add (BDA), not the residual snapshot in LayerNorm.** The input
LayerNorm does return FP32, but `TransformerLayer._run_input_layernorm` saves
`residual = hidden_states`, i.e. the pre-norm BF16 tensor. The BDA receives BF16
attention output, BF16 residual, and an FP32 projection bias. Its dtype guard
only casts the bias when `x.dtype != residual.dtype`. Because BF16 `x` already
matches the BF16 residual, the guard is skipped; `x + bias` then promotes the
result to FP32. All later residual snapshots are consequently FP32.

The minimum semantic fix is to cast `bias` to the residual dtype independently
of whether `x` needs a cast:

```diff
 if x.dtype != residual.dtype:
     x = x.to(residual.dtype)
-    if bias is not None:
-        bias = bias.to(residual.dtype)
+if bias is not None and bias.dtype != residual.dtype:
+    bias = bias.to(residual.dtype)
```

Together with the existing one-time decoder-entry BF16 cast, that change keeps
both residual additions and every Transformer-layer output in BF16 across all
24 layers. It does not change FP32 parameter storage, optimizer state, or loss
accumulation.

## Configuration that determines the dtypes

`scripts/phase1_baseline.py` builds `TransformerConfig` with
`params_dtype=torch.float32`, `pipeline_dtype=torch.float32`, `bf16=False`, and
`add_bias_linear=True`; the whole forward and loss run inside
`torch.autocast(..., dtype=torch.bfloat16)`.

The local layer spec in
[`gpt_layer_specs.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/gpt/gpt_layer_specs.py#L353-L463)
selects:

- `WrappedTorchNorm` for both per-layer norms because Apex is absent;
- local `ColumnParallelLinear` / `RowParallelLinear`;
- `get_bias_dropout_add` after attention and MLP;
- local `SelfAttention`, with only `core_attention` replaced by the lab's
  `AutocastTEDotProductAttention`.

The per-layer norm is ordinary `torch.nn.LayerNorm`, selected by
[`WrappedTorchNorm`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/torch_norm.py).
CUDA autocast registers `layer_norm` and `native_layer_norm` in the FP32 policy
in PyTorch 2.8
[`autocast_mode.h`](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/autocast_mode.h#L854-L875).
Thus a BF16 layer input produces an FP32 normalized branch.

The block-final norm is different: because TE is installed,
`TransformerBlock` selects `TENorm` as `LayerNormImpl`. TE chooses the active
autocast dtype and casts its input and affine parameters to that dtype in
[`LayerNorm.op_forward`](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/ops/basic/layer_norm.py).
This explains the Phase 4.2 observation that final LayerNorm received FP32 and
returned BF16.

## Exact lifecycle: first layer in Phase 4.2

This trace separates the residual stream from the normalized/compute branch.
“Linear output” below means the activation tensor; projection biases are listed
separately because `skip_bias_add=True` returns them for later fusion.

| # | Site | Dtype in Phase 4.2 | Source operation and reason |
| ---: | --- | --- | --- |
| 1 | Layer input `hidden_states` | **BF16** | `scripts/bf16_hidden_residual.py::_cast_hidden`: the decoder-entry FP32 embedding is cast once with `hidden_states.to(torch.bfloat16)`. |
| 2 | Input LayerNorm output | BF16 → **FP32** | `torch_norm.py::WrappedTorchNorm` creates `torch.nn.LayerNorm`; CUDA autocast's FP32 policy for `aten::layer_norm` casts the BF16 input and computes/returns FP32. |
| 3 | QKV linear input/output | input FP32; matmul result BF16; final QKV output **FP32** | `layers.py::_linear_forward`: autocast runs `torch.matmul(input, weight.t())` in BF16. `SelfAttention` constructs QKV with `skip_bias_add=False`, so `_linear_forward` then executes `output = output + bias`; BF16 output plus FP32 stored bias promotes to FP32. |
| 4 | Attention output | Q/K/V FP32 → BF16; core output BF16; output-projection tensor **BF16**, separate bias FP32 | `scripts/phase1_baseline.py::AutocastTEDotProductAttention.forward` explicitly applies `.to(autocast_dtype)` to Q, K, and V. TE cuDNN FusedAttention preserves BF16. Local `RowParallelLinear` runs its matmul under BF16 autocast and, with `skip_bias_add=True`, returns the FP32 bias separately. |
| 5 | First residual snapshot | **BF16** | `transformer_layer.py::_run_input_layernorm`: because Torch LayerNorm does not return a residual tuple, `residual = hidden_states`. This is the pre-LayerNorm BF16 input, not the FP32 LayerNorm result. `config.fp32_residual_connection` is false, so `residual.float()` is not executed. |
| 6 | Self-attention BDA | BF16 `x` + FP32 bias + BF16 residual → **FP32** | `fused_bias_dropout.py::_bias_dropout_add_func`: `x.dtype == residual.dtype`, so the `if x.dtype != residual.dtype` block skips both `x.to(...)` and the incorrectly nested `bias.to(...)`. `x = x + bias` promotes to FP32; dropout stays FP32 and `residual + out` remains FP32. **This is the first reversion.** |
| 7 | Post-attention LayerNorm output | FP32 → **FP32** | `transformer_layer.py::_forward_pre_mlp_layernorm` invokes the same Torch LayerNorm. The input is already FP32, so there is no dtype transition. |
| 8 | MLP FC1 input/output | input FP32; FC1 tensor **BF16**, separate bias FP32 | `mlp.py::MLP.forward` calls local `ColumnParallelLinear` with `skip_bias_add=True`; `_linear_forward` matmul uses BF16 autocast and returns FP32 bias separately. |
| 9 | Activation | BF16 FC1 tensor + FP32 bias → **FP32** | `mlp.py::MLP.forward` either calls `bias_gelu_impl` or explicitly executes `intermediate_parallel + bias_parallel`. In both paths, the FP32 stored bias promotes the BF16 tensor before GELU, so the activation and FC2 input are FP32. |
| 10 | MLP FC2 input/output | input FP32; FC2 tensor **BF16**, separate bias FP32 | Local `RowParallelLinear` matmul runs in BF16 under autocast and returns its FP32 bias separately because `skip_bias_add=True`. |
| 11 | Second residual snapshot | **FP32** | `transformer_layer.py::_pre_mlp_layernorm_and_residual`: `residual = hidden_states`, where `hidden_states` is the FP32 result of the first BDA. |
| 12 | MLP BDA | BF16 `x` → FP32; FP32 bias/residual; output **FP32** | `_bias_dropout_add_func` now sees `x.dtype != residual.dtype` and explicitly executes `x = x.to(residual.dtype)`. Bias is already FP32. Bias, dropout, and residual addition therefore stay FP32. |
| 13 | Layer output | **FP32** | `TransformerLayer._apply_mlp_bda_step` only makes the BDA result viewless; it does not change dtype. Layer 2 consequently starts in FP32, as do layers 3–24. |

The execution path is
`TransformerLayer.forward` → `_forward_attention` → `_run_input_layernorm` →
`SelfAttention.forward` → `_apply_self_attn_bda_step` → `_forward_mlp` →
`_pre_mlp_layernorm_and_residual` → `MLP.forward` →
`_apply_mlp_bda_step`. The relevant pinned implementation is
[`transformer_layer.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/transformer_layer.py#L604-L1120).

### Steady state in layers 2–24

After the first BDA, every layer receives FP32. Its first residual snapshot is
therefore FP32. Attention still returns a BF16 projection tensor, so the
self-attention BDA takes the explicit `x.to(residual.dtype)` BF16→FP32 path.
The post-attention hidden state, second residual snapshot, second BDA output,
and layer output all remain FP32. The same pattern repeats through layer 24,
which is why final LayerNorm still saw FP32 and why the Phase 4.1 copy profile
was unchanged.

## Complete FP32/BF16 transition inventory

| Direction | File / function | Exact operation | Why it occurs |
| --- | --- | --- | --- |
| FP32 → BF16 | `scripts/bf16_hidden_residual.py::_cast_hidden` | `hidden_states.to(dtype=torch.bfloat16)` | Phase 4.2's one-time block-entry cast. |
| BF16 → FP32 | PyTorch `aten::layer_norm`, selected by `megatron/core/transformer/torch_norm.py::WrappedTorchNorm` | CUDA autocast FP32 policy | LayerNorm keeps statistics and affine work in FP32. It changes the compute branch, not the saved residual. |
| FP32 operands → BF16 GEMM/output | `megatron/core/tensor_parallel/layers.py::_linear_forward` | `torch.matmul(input, weight.t())` inside BF16 autocast | `matmul` has the CUDA lower-precision autocast policy. FP32 parameter storage is unchanged; autocast uses transient/cached BF16 views. |
| BF16 → FP32 | `layers.py::_linear_forward` for QKV | `output = output + bias` | QKV uses `skip_bias_add=False`; the matmul result is BF16 and the parameter bias is FP32, so normal promotion produces FP32. |
| FP32 → BF16 | `scripts/phase1_baseline.py::AutocastTEDotProductAttention.forward` | `query/key/value.to(autocast_dtype)` | TE FusedAttention requires Q/K/V in the active BF16 compute dtype. This is 3 explicit casts per layer. |
| BF16 → FP32 | `megatron/core/fusions/fused_bias_dropout.py::_bias_dropout_add_func` | `x = x + bias` when `x` and residual are BF16 but bias is FP32 | The bias cast is nested under the unrelated `x/residual` mismatch guard. This is the first Phase 4.2 reversion. |
| BF16 → FP32 | `megatron/core/transformer/mlp.py::MLP.forward` and `fusions/fused_bias_gelu.py::bias_gelu` | `intermediate_parallel + bias_parallel` / `x = bias + y` | FC1 returns a BF16 tensor and separate FP32 bias; type promotion makes the activation FP32. |
| FP32 operands → BF16 GEMM/output | `layers.py::_linear_forward` for FC2 | `torch.matmul(input, weight.t())` in autocast | FP32 activation and weight are consumed by BF16 GEMM. |
| BF16 → FP32 | `fused_bias_dropout.py::_bias_dropout_add_func` in the second BDA | `x = x.to(residual.dtype)` | The first BDA made the residual stream FP32, while FC2 returns BF16. |
| FP32 → BF16 | TE block-final `LayerNorm.op_forward` | `maybe_dequantize(input_.contiguous(), dtype)` where autocast `dtype` is BF16 | `TransformerBlock` globally chooses `TENorm` when TE is available; this is distinct from per-layer Torch LayerNorm. |
| BF16 loss values → FP32 | `scripts/phase1_baseline.py::masked_language_model_loss` | `losses = output_tensor.float()` | Explicit FP32 masked loss reduction; this boundary must remain. |

### Linear backward boundary

The forward lifecycle above is sufficient to locate the residual reversion, but
the profiler also saw matching backward copies. Local
`LinearWithGradAccumulationAndAsyncCommunication.forward` records
`ctx.input_dtype`. Its backward then executes:

```python
grad_output = grad_output.to(ctx.input_dtype)
```

QKV, FC1, and FC2 inputs in the current path are FP32, while their GEMM outputs
and incoming gradients can be BF16, so this operation restores FP32 gradients
for the custom Linear contract. The QKV adapter's three forward `.to(BF16)`
operations also induce three matching casts back to their FP32 source dtype in
autograd. These backward boundaries explain copy cost but do not cause the
forward residual stream to revert. The minimum BDA patch does not remove all of
them because per-layer norms and the FC1 bias/GELU path intentionally remain
FP32.

## Minimum patch design

### Required semantic change

Retain the Phase 4.2 decoder-entry cast and change BDA's bias handling so each
operand is independently aligned to the residual dtype:

```python
if x.dtype != residual.dtype:
    x = x.to(residual.dtype)
if bias is not None and bias.dtype != residual.dtype:
    bias = bias.to(residual.dtype)
```

The smallest textual patch is in the pinned
`megatron/core/fusions/fused_bias_dropout.py::_bias_dropout_add_func`. A
lab-only implementation can preserve the pinned checkout by putting the same
corrected, JIT-fused BDA factory in `scripts/bf16_hidden_residual.py` and
installing it as `self_attn_bda` and `mlp_bda` for every decoder layer inside
`enable_bf16_hidden_residual_stream`. That requires changing only that lab file
when Phase 4.3 is implemented; `TransformerLayer`, the model architecture, and
the local layer spec remain unchanged.

No cast should be added after LayerNorm. The normalized branch may remain FP32:
the residual snapshot is already the pre-norm tensor. Likewise, no master
parameter or AdamW state should be converted.

### Lifecycle after the patch

1. Decoder entry and every layer input: BF16.
2. Per-layer Torch LayerNorm output: FP32.
3. QKV output after bias: FP32; adapter: BF16; fused attention and projection
   tensor: BF16; projection bias: FP32.
4. First residual snapshot: BF16.
5. Corrected first BDA: cast only the small FP32 bias to BF16; result BF16.
6. Pre-MLP LayerNorm output: FP32; second residual snapshot: BF16.
7. FC1 tensor BF16; bias/GELU activation FP32; FC2 tensor BF16; FC2 bias FP32.
8. Corrected second BDA: cast only the small FP32 bias to BF16; result and layer
   output BF16.
9. Repeat through layer 24; block-final TE LayerNorm receives BF16 instead of
   FP32.
10. FP32 parameters, parameter gradients, AdamW states, and explicit FP32 loss
    reduction remain unchanged.

This patch fixes persistence of the hidden/residual stream, but it is not
expected to remove every copy. Torch LayerNorm, QKV's in-linear FP32 bias add,
the QKV adapter, FC1 bias/GELU, FC2 autocast, and corresponding backward casts
remain. They should be measured before considering a broader bias/branch-dtype
change.

## Numerical risk

BF16 has the same 8-bit exponent range as FP32, so overflow/underflow range is
not the main concern. The risk is precision: BF16 has 7 explicit fraction bits.
Its spacing near one is `2^-7 = 0.78125%` and unit roundoff is approximately
`2^-8 = 0.390625%`.

With BF16 residual accumulation:

- each of the 48 residual additions (attention and MLP across 24 layers) rounds
  the accumulated state to BF16;
- a branch update smaller than about half an ulp of the current residual can be
  partially or completely lost;
- rounding errors can accumulate or correlate rather than cancel;
- pre-norm Transformers depend on the residual path as the high-fidelity
  information highway, so long training may show convergence or final-quality
  degradation even when one-step loss and gradient cosine look good;
- casting FP32 biases to BF16 also quantizes their forward contribution, though
  autograd through the cast still returns gradients to the FP32 bias parameter.

Risk mitigations retained by the design are FP32 LayerNorm computation for the
per-layer normalized branch, FP32 master parameters, FP32 parameter gradients
and AdamW states, FP32 loss reduction, and BF16's FP32-like exponent range.
Phase 4.2's dropout-free one-step comparison (no NaN/Inf and minimum gradient
cosine `0.99984`) is encouraging but does not validate 48 BF16 residual
accumulations or training convergence.

## Is a GPU screen justified?

**Yes, but only a short correctness and mechanism screen—not a full
benchmark.** The source audit identifies a precise guard bug and predicts an
unambiguous runtime invariant: all 24 layer inputs/outputs, both residual
snapshots, and the block-final LayerNorm input must be BF16.

The screen should:

1. use the same Pod and pinned stack for A/B;
2. disable dropout only for fixed-seed numerical comparison;
3. hook every layer and both BDA sites to assert BF16 residual/input/output
   persistence;
4. assert all parameters and AdamW states remain FP32 and the masked loss
   reduction is FP32;
5. compare loss, forward errors, representative gradients, cosine similarity,
   and NaN/Inf;
6. use only 3 warmup + 10 measured steps and a 3–5 step profiler capture;
7. compare `aten::copy_`, `bfloat16_copy`, `direct_copy`, throughput, MFU, and
   peak VRAM against Phase 4.2.

If final LayerNorm still receives FP32, the mechanism failed. If persistence is
confirmed but copy/cast and throughput improve only marginally, stop: the
remaining targets are the QKV bias add, FC1 bias/GELU promotion, LayerNorm
boundary, and Linear backward casts. Do not run 20+100 validation unless the
short screen shows both acceptable numerics and a substantial copy/throughput
improvement.

## Files/functions that would change

Design phase: only this document.

Proposed implementation phase, preferred lab-only form:

- `scripts/bf16_hidden_residual.py`
  - add corrected JIT-fused BDA train/inference functions and factory;
  - extend `enable_bf16_hidden_residual_stream` to install the factory on each
    layer's `self_attn_bda` and `mlp_bda`;
  - retain the existing one-time decoder-input BF16 cast and FP32 parameter
    assertion.

Equivalent smallest upstream form, not applied:

- `megatron/core/fusions/fused_bias_dropout.py::_bias_dropout_add_func`
  - move the bias cast outside the `x.dtype != residual.dtype` condition and
    guard it by its own dtype comparison.

No change is required in `TransformerLayer.forward`, the LayerNorm
implementation, local Linear, attention backend, optimizer, model architecture,
or loss function for the minimum residual-persistence patch.
