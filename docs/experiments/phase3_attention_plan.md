# Phase 3.1 Attention Optimization Experiment Plan

## Decision

Run one controlled experiment that replaces only Megatron's local core attention
with Transformer Engine (TE) `TEDotProductAttention`, forced to the cuDNN
`FusedAttention` backend. Do not use the full TE transformer-layer spec. This is
the smallest supported change because it preserves the local QKV/output linears,
norms, MLP, bias-dropout-add, optimizer, precision, data, and model shape.

This phase is design-only. No Pod was started and no performance result is claimed.

## Current Attention Path

The Phase 1.2 script constructs `get_gpt_layer_local_spec()` and sets
`attention_backend=AttnBackend.unfused`. It also launches with
`TRANSFORMER_ENGINE_DISABLE=1`. The resulting core is Megatron's local
`DotProductAttention`: QK uses `torch.baddbmm`, softmax/masking and dropout are
separate operations, and AV uses `torch.bmm`. It materializes the quadratic
attention scores/probabilities.

Changing only the enum to `AttnBackend.fused` or `AttnBackend.flash` is
insufficient. The enum controls TE backend environment variables, while the local
layer spec still instantiates local `DotProductAttention`. The layer spec must
also replace `self_attention.core_attention`.

Sources: [local GPT spec](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/models/gpt/gpt_layer_specs.py),
[local attention](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/dot_product_attention.py),
and [TE wrapper](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/extensions/transformer_engine.py#L2041-L2323).

## Supported Backends At The Pinned Commit

- **TE cuDNN FusedAttention:** Supported for this A40 (`sm86`) BF16 training
  shape: causal self-attention, `sbhd`, sequence length 2048, head dimension 64,
  and dropout 0.1. It requires TE plus a compatible cuDNN runtime, but not the
  standalone `flash-attn` package. This is the recommended experiment.
- **TE FlashAttention:** Supported on A40 through FlashAttention 2 for this
  shape. It requires both TE and external `flash-attn`; pinned TE accepts
  `flash-attn` 2.1.1 through 2.8.3. This adds a second compiled dependency and is
  therefore not the minimal first experiment.
- **TE auto selection:** Supported, but rejected because it does not hold the
  attention implementation constant across hosts.
- **PyTorch SDPA:** PyTorch 2.8 provides SDPA, but pinned Megatron does not wire it
  into this GPT local training spec. Using it requires a new adapter and
  correctness work, so it is a larger code change.
- **Full TE GPT layer spec:** Supported, but rejected because it also replaces
  linears, norms, and MLP modules.
- **Megatron Kitchen SDPA/FA:** Present only as an unavailable integration stub at
  this commit. Local masked-softmax fusion is not a fused attention algorithm and
  does not remove the score/probability matrices.

TE documents FlashAttention, cuDNN FusedAttention, and native unfused attention
as its three PyTorch backends; both optimized families use a flash-style tiled,
recomputation-based algorithm. See the [TE attention guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html)
and the [pinned selector](https://github.com/NVIDIA/TransformerEngine/blob/4329ff84bfbdaa778a33cba02a15fb0807c64689/transformer_engine/pytorch/attention/dot_product_attention/utils.py#L327-L1530).

## Exact Dependency Requirement

- Transformer Engine `2.17.1+4329ff84`, source commit
  `4329ff84bfbdaa778a33cba02a15fb0807c64689`, built for PyTorch and against the
  existing CUDA 12.8 toolchain.
- Existing cuDNN runtime `>=8.9.3` for BF16 fused attention on A40 `sm86`; record
  the exact runtime version before execution.
- No standalone FlashAttention, PyTorch SDPA adapter, Apex, or other attention
  package is required for the selected experiment.

Do not install Megatron's current `te` convenience extra because it requests a
CUDA 13 TE core package. Install the exact TE source with build isolation and
dependency resolution disabled, then verify that PyTorch remains `2.8.0+cu128`,
CUDA remains 12.8, NCCL remains 2.27.3, and Megatron remains at `09fde85e`.

## Exact Code And Configuration Change

Add an opt-in `te-fused` attention choice to `scripts/phase1_baseline.py`, leaving
the existing default unchanged. For that choice, construct the local spec and
replace only its core attention module:

```python
from megatron.core.extensions.transformer_engine import TEDotProductAttention

layer_spec = get_gpt_layer_local_spec()
layer_spec.submodules.self_attention.submodules.core_attention = TEDotProductAttention

config = TransformerConfig(
    # All existing Phase 1.2 arguments remain byte-for-byte unchanged.
    attention_backend=AttnBackend.fused,
    ...
)
```

Run without `TRANSFORMER_ENGINE_DISABLE=1`. `AttnBackend.fused` sets
`NVTE_FLASH_ATTN=0`, `NVTE_FUSED_ATTN=1`, and `NVTE_UNFUSED_ATTN=0`. Enable TE
backend logging and require the runtime message `FusedAttention backend
(sub-backend 1)`; abort instead of falling back. Keep the explicit causal mask:
TE ignores the tensor for a causal mask and implements the same causal semantics
inside the fused kernel.

Everything else remains the Phase 1.2 baseline: 24 layers, H=1024, FFN=4096, 16
heads, 355,919,872 parameters, sequence length 2048, micro/global batch 1,
TP/PP/DP=1, BF16 autocast with FP32 parameters and optimizer state, dropout 0.1,
standard AdamW, fixed synthetic batch and seeds, 20 warmup plus 100 measured
steps, no CUDA Graph, and no unrelated fusion.

## Validation And Measurement

Before timing, load identical weights into local and fused variants, switch both
to evaluation mode, and compare token losses while dropout is inactive. Record
maximum absolute, mean absolute, and relative error; require finite outputs and
start with BF16 tolerances `atol=0.05, rtol=0.05`. Do not loosen a failed gate
without diagnosis. Then verify a training forward pass, backward pass, finite
gradients, optimizer step, and checkpoint round trip with the fused variant.

Use the unchanged 20/100 timing protocol and report average/median step time,
tokens/s, MFU, peak allocated/reserved/device memory, GPU utilization, loss, and
all software versions. A short Nsight Systems comparison should confirm that the
separate masked-fill, softmax, dropout, scale, and BMM path has been replaced.

## Expected Mechanism And Range

Phase 2.1 attributed 49.41% of kernel time to attention, found 4,768 kernels per
step, and measured 10.35% of the profile window in memory copies before overlap.
FusedAttention tiles QK/softmax/dropout/AV and recomputes normalization data in
backward, avoiding full score/probability traffic. Expected improvements are:

- 15-35% lower average step time, implying roughly 18-54% higher tokens/s and MFU;
- materially lower peak VRAM, plausibly several GiB at 24 x 2048-token layers;
- hundreds fewer launches per step and sharply lower attention-related D2D traffic;
- disappearance of the current standalone softmax/mask/dropout kernels from the
  dominant-kernel list.

These are planning estimates, not acceptance guarantees. GPU utilization is
already 99.62%, so it may remain flat; useful work per busy cycle is the target.
The impossible upper bound from removing all 49.41% attention-attributed time is
1.98x overall speedup, but that category includes necessary attention compute and
overlaps GEMM, so it must not be treated as a prediction.

## Risks And Stop Conditions

- The image's exact cuDNN version is not yet recorded. Stop if TE does not select
  fused sub-backend 1; never permit silent unfused fallback.
- Building TE can accidentally resolve a different Torch/CUDA package. Use the
  exact source commit with `--no-build-isolation --no-deps` and compare versions
  before and after installation.
- Fused dropout uses a different RNG consumption order, so training losses need
  not be bitwise identical even with the same seed. Use the dropout-free parity
  gate and compare finite training behavior rather than exact loss trajectories.
- TE/cuDNN workspace allocation can offset some memory savings. Measure device,
  allocated, and reserved peaks rather than assuming improvement.
- NVIDIA notes that FlashAttention often outperforms cuDNN attention on Ampere.
  That is a possible later experiment, but adding `flash-attn` now would violate
  the smallest-change objective.
