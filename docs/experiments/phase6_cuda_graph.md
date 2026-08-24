# Phase 6.1: CUDA Graph feasibility and fast-screen gate

## Outcome

CUDA Graph capture and replay are technically possible with the pinned
Megatron-LM implementation, but the current direct `GPTModel` +
`torch.optim.AdamW` harness is not correctness-safe with the existing MCore
training graph path.

The correctness gate failed on gradients before the production-size fast A/B
screen. Per-token loss and scalar loss were bit-identical and all checked
values were finite, but the graphed gradients did not meet `atol=rtol=1e-5`.
Per the stop rule, no 3+15 timing run, Nsight Systems comparison, or 20+100
formal validation was performed.

## Pinned configuration

- GPU: 1x NVIDIA A40 48 GB
- Driver: 570.195.03
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- NCCL: 2.27.3
- cuDNN: 9.10.2
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine: `2.17.1+4329ff84`
- Model: 355.9M GPT, sequence length 2048, micro-batch 8
- Precision: BF16 autocast, FP32 parameters/residuals/AdamW
- Attention: cuDNN FusedAttention, sub-backend 1
- Retained optimization: `bias_dropout_fusion=True`
- Disabled optimization: `bias_activation_fusion=False`
- TP/PP/DP: 1/1/1

## Source inspection

The pinned `TransformerConfig` exposes three relevant training mechanisms:

1. `cuda_graph_impl="local"`: MCore `CudaGraphManager`, with per-layer capture.
2. `cuda_graph_impl="transformer_engine"`: Transformer Engine
   `make_graphed_callables`, requiring `TECudaGraphHelper` integration.
3. `cuda_graph_impl="full_iteration"`: `FullCudaGraphWrapper`, capturing the
   forward/backward iteration while leaving the optimizer outside.

The safest existing path for this harness was the local MCore implementation:

```python
TransformerConfig(
    cuda_graph_impl="local",
    cuda_graph_modules=[],
    cuda_graph_warmup_steps=5,
)
```

An empty `cuda_graph_modules` list captures the complete `TransformerLayer`.
The embeddings, final normalization, language-model loss, `zero_grad`, and
AdamW step remain eager.

The local manager was preferred because it already:

- allocates graph-owned input/output buffers;
- validates input shape, dtype, device, and `requires_grad`;
- copies changing runtime layer inputs into static buffers;
- shares a CUDA Graph memory pool across the layer graphs;
- records a forward and backward graph for each layer;
- registers graph-safe RNG states for dropout;
- supports one reusable runner per layer when PP=1.

The Transformer Engine helper was not selected because it assumes the
higher-level Megatron training integration. Full-iteration capture was not
selected because its wrapper requires Megatron's `forward_backward_func` and
static data-iterator interface. Adapting either path to the direct model loop
would be more invasive than the requested feasibility screen. A custom
`torch.cuda.CUDAGraph` system was intentionally not written.

## Feasibility and blocker analysis

### Fixed and dynamic shapes

The production workload has fixed batch and sequence dimensions, no MoE
routing, no variable sequence lengths, and no pipeline schedule. Shape
stability is therefore satisfied.

### Tensor and parameter addresses

MCore copies layer inputs into graph-owned static buffers before replay.
Parameter storage is stable because AdamW updates values in place. These are
not blockers.

### Dynamic allocation

Captured layer intermediates are allocated in MCore's CUDA Graph pool.
Allocation outside the captured layers remains eager. The two-layer
correctness capture reserved an additional 23,068,672 bytes and completed in
0.4003 seconds, so capture-time allocation itself was not the observed blocker.

### Dropout and RNG

The first capture attempt used MCore's graph-safe RNG tracker. cuDNN
FusedAttention rejected it during capture warmup with:

```text
AssertionError: Unsupported RNG states tracker.
```

The pinned Transformer Engine attention implementation requires its own
`CudaRNGStatesTracker` whenever `is_graph_capturing()` is true. Both A and B
were therefore changed to use the same Transformer Engine graph-safe tracker.
After that correction, graph capture and replay succeeded. Production dropout
settings remained 0.1; dropout was set to zero only for the strict correctness
comparison.

### Optimizer, gradients, and `zero_grad`

This is the blocking incompatibility.

The local MCore backward graph does not return ordinary parameter gradients for
captured layer parameters. It executes:

```python
param.main_grad.add_(wgrad)
```

MCore normally provides and manages `main_grad` through its DDP/training stack.
The accepted benchmark instead instantiates a raw `GPTModel` and raw
`torch.optim.AdamW`, whose contract is `parameter.grad`.

The feasibility harness used the smallest adapter that preserved the optimizer
configuration:

1. allocate persistent FP32 `main_grad` buffers only for captured layer
   parameters;
2. zero those buffers outside the graph each step;
3. let graph replay accumulate into `main_grad`;
4. temporarily expose each buffer as `parameter.grad` for the unchanged AdamW
   step;
5. clear the aliases after the optimizer step.

Capture and replay worked with this adapter, but the resulting gradients did
not reproduce eager execution. Further work would require changing gradient
buffer ownership, DDP integration, or the training-loop contract. That exceeds
the requested non-invasive screen.

## Correctness gate

The gate used a two-layer reduced model, identical initial weights and seed,
BF16 autocast, FP32 parameter/gradient storage, cuDNN FusedAttention, and
dropout disabled only for comparison.

| Check | Eager | CUDA Graph | Result |
|---|---:|---:|---|
| Scalar loss | 6.9515428543 | 6.9515428543 | exact |
| Per-token max absolute error | — | 0.0 | exact |
| NaN count | 0 | 0 | pass |
| Inf count | 0 | 0 | pass |
| Gradient global cosine | — | 0.9998964442 | fail |
| Gradients all-close, 1e-5/1e-5 | — | false | fail |

Worst gradient evidence:

- Maximum absolute error: `0.0042900946` in
  `embedding.word_embeddings.weight`.
- The QKV weight had maximum absolute error `0.00048828125`.
- All 544,256 checked gradient elements were finite.

Capture/replay evidence:

- 2 TransformerLayers
- 2 `CudaGraphManager` instances
- 2 graph runners
- 2 forward graphs
- 2 backward graphs
- global graph-created state: true
- replay-ready state after execution: true

Thus this is a correctness rejection, not a failure to enter the replay path.

## Performance gate

The planned fast screen was:

- 5 graph/capture warmup iterations
- 3 benchmark warmup iterations
- 15 measured iterations
- same A40 Pod for A and B
- Nsight Systems node-level CUDA Graph tracing

It was not run because correctness failed first.

| Metric | A: eager | B: CUDA Graph |
|---|---:|---:|
| Average step time | not run | not run |
| Median step time | not run | not run |
| Tokens/sec | not run | not run |
| MFU | not run | not run |
| Peak VRAM | not run | not run |
| CPU time/step | not run | not run |
| Kernel count/step | not run | not run |
| CUDA API launches/step | not run | not run |
| CPU launch gaps | not run | not run |
| GPU idle time/step | not run | not run |
| Speedup | not run | not run |

The >=2% formal-validation threshold was therefore not evaluated. Formal
20+100 validation was not run.

## Decision

Do not adopt CUDA Graph for the current raw-`GPTModel`/raw-AdamW benchmark
harness. The existing MCore local mechanism is designed around MCore-managed
`main_grad`, and the minimal compatibility adapter failed strict gradient
equivalence.

Reconsider CUDA Graph only after migrating the workload to MCore DDP and its
gradient-buffer lifecycle, or to the complete Megatron training stack where
the Transformer Engine or full-iteration helpers are already integrated.

## Infrastructure cleanup

- Experiment Pod: `9c001avwsjbibr`
- Final Pod status: `EXITED`
- GPU billing stopped immediately after the correctness blocker.
- An earlier candidate Pod was deleted before setup because its driver was
  550.127.05 rather than the pinned 570.195.03.

Machine-readable evidence is in `results/phase6_cuda_graph.json`.
