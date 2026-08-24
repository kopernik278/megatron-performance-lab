# Phase 6.2: Megatron DDP lifecycle for local CUDA Graphs

## Decision

Migrate the Phase 6 harness to MCore `DistributedDataParallel` (DDP) and
MCore's `FP32Optimizer` lifecycle before attempting another local CUDA Graph
measurement.

The minimum Phase 6.3 path is:

```text
GPTModel (FP32 parameters, BF16 autocast)
  -> MCore DistributedDataParallel
  -> DDP contiguous grad_data buffers and per-parameter main_grad views
  -> loss.backward()
  -> DDP finalizes main_grad
  -> MCore FP32Optimizer exposes main_grad as param.grad
  -> unchanged torch.optim.AdamW step
```

For the graph variant, MCore's per-layer backward graphs accumulate captured
weight gradients directly into the same DDP-owned `main_grad` views. This
removes the unsupported hand-written bridge used in Phase 6.1.

This is a design only. No implementation or GPU run is part of Phase 6.2.

## Fixed scope

The migration must preserve:

- Megatron-LM commit `09fde85ea25fb67e9b32019089fae163a3233bd3`;
- PyTorch `2.8.0+cu128`, CUDA 12.8, and Transformer Engine
  `2.17.1+4329ff84`;
- one A40 and `TP=PP=DP=1`;
- the existing 355.9M `GPTModel`, sequence length 2048, and micro-batch 8;
- FP32 model parameters and optimizer state with BF16 autocast;
- cuDNN FusedAttention and the Transformer Engine CUDA-graph-safe RNG tracker;
- `bias_dropout_fusion=True`;
- `bias_gelu_fusion=False`;
- the existing AdamW hyperparameters and parameter grouping;
- `use_distributed_optimizer=False`.

The optimizer step itself is not captured. Phase 6.3 tests MCore local
per-TransformerLayer graphs only.

## Pinned-source findings

### 1. Required classes and functions

The minimum required public or stable MCore interfaces are:

1. [`DistributedDataParallelConfig`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/distributed_data_parallel_config.py)
   from `megatron.core.distributed`;
2. [`DistributedDataParallel`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/distributed_data_parallel.py#L85-L115)
   from `megatron.core.distributed`;
3. [`finalize_model_grads`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/finalize_model_grads.py#L543-L696)
   from `megatron.core.distributed`;
4. [`OptimizerConfig`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/optimizer/optimizer_config.py)
   from `megatron.core.optimizer`;
5. [`FP32Optimizer`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/optimizer/optimizer.py#L1250-L1354)
   from `megatron.core.optimizer.optimizer`;
6. [`create_cudagraphs`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/cuda_graphs.py#L703-L713)
   from `megatron.core.transformer.cuda_graphs`;
7. the existing `model_parallel_cuda_manual_seed(..., te_rng_tracker=True)`
   initialization.

MCore internally constructs `_ParamAndGradBuffer` objects. The lab must not
instantiate or manipulate that private class. DDP allocates the buffers and
assigns each trainable parameter a persistent `main_grad` view.

The full Megatron training entry point normally calls
`wrap_model_chunks_with_ddp`, `get_megatron_optimizer`, and a pipeline
forward/backward schedule. Those are not required for the initial
single-model, single-microbatch lab harness:

- direct DDP construction performs the same core buffer and hook setup;
- direct `loss.backward()` is sufficient at `PP=1`;
- `finalize_model_grads([ddp_model])` performs the standard finalization;
- the existing fixed-shape harness already records and creates local graphs
  successfully.

### 2. How gradients reach `main_grad`

DDP performs two related setup actions:

1. It groups parameters by parameter/gradient dtype and allocates contiguous
   `_ParamAndGradBuffer.grad_data` storage.
2. It maps every trainable parameter's `param.main_grad` to a shaped view into
   that storage. The assignment is in
   [`param_and_grad_buffer.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/param_and_grad_buffer.py#L1357-L1382).

There are then two valid accumulation paths.

#### Eager DDP path

DDP registers a post-hook on each parameter's autograd `AccumulateGrad` node.
After autograd produces `param.grad`, the hook executes:

```python
param.main_grad.add_(param.grad.data)
param.grad = None
```

It skips the add when the parameter has already accumulated directly into
`main_grad`. See
[`DistributedDataParallel._make_backward_post_hook`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/distributed_data_parallel.py#L551-L584).

#### MCore local CUDA Graph path

During backward graph capture, `_CudaGraphRunner` uses `torch.autograd.grad`
and records the weight-gradient accumulation directly in the graph:

```python
param.main_grad.add_(wgrad)
param.grad_added_to_main_grad = True
```

See
[`cuda_graphs.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/cuda_graphs.py#L1488-L1508).
On replay, MCore marks the weight gradient ready and sets
`grad_added_to_main_grad=True`, so the outer DDP hook does not add the same
gradient a second time. See
[`cuda_graphs.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/transformer/cuda_graphs.py#L861-L889).

Parameters outside captured TransformerLayers, such as embeddings and the
output layer, continue through the ordinary DDP hook. Both paths therefore
land in one DDP-owned gradient buffer covering the complete model.

### 3. Gradient synchronization and finalization

`finalize_model_grads([ddp_model])` calls `finish_grad_sync()` on each model
chunk before handling any TP/PP-specific gradient reductions. With
`overlap_grad_reduce=False`, `finish_grad_sync()` starts and completes a
synchronous all-reduce. At `DP=1` it is semantically an identity, but retaining
the call keeps the lifecycle identical between Phase 6.3 variants and ready
for later DP scaling.

The pinned DDP implementation explicitly waits for each local CUDA Graph
weight-gradient completion event before reading, scaling, or reducing its
bucket. See
[`param_and_grad_buffer.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/param_and_grad_buffer.py#L629-L647).

Do not enable overlap in Phase 6.3. It would add asynchronous communication and
bucket-readiness behavior as another changed variable.

### 4. Correct zeroing lifecycle

The order at the beginning of every iteration must match Megatron training:

```python
ddp_model.zero_grad_buffer()
optimizer.zero_grad(set_to_none=True)
```

The pinned training loop uses exactly this order in
[`training.py`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/training/training.py#L2987-L2994).

`ddp_model.zero_grad_buffer()`:

- resets `grad_added_to_main_grad=False` for local CUDA Graph mode;
- zeros each contiguous `grad_data` buffer;
- resets bucket readiness metadata.

The buffer reset implementation zeros storage in place, preserving every
`main_grad` address. See
[`DistributedDataParallel.zero_grad_buffer`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/distributed_data_parallel.py#L688-L702)
and
[`_ParamAndGradBuffer.reset`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/distributed/param_and_grad_buffer.py#L1608-L1614).

`FP32Optimizer.zero_grad(set_to_none=True)` clears `param.grad`. This matters
because `FP32Optimizer.prepare_grads()` aliases `param.grad` to `main_grad`
before the previous optimizer step. Zeroing only `param.grad`, or replacing
`main_grad` tensors instead of zeroing their storage, is incorrect.

### 5. How the optimizer consumes `main_grad`

The current model owns FP32 parameters even though its forward executes under
BF16 autocast. Therefore the matching Megatron wrapper is `FP32Optimizer`, not
`Float16OptimizerWithFloat16Params`.

On `optimizer.step()`, `FP32Optimizer.prepare_grads()` assigns:

```python
param.grad = param.main_grad
```

for each managed parameter. The underlying optimizer then consumes the
ordinary `.grad` interface. No gradient copy, cast, master parameter, or loss
scaler is introduced. See
[`FP32Optimizer.prepare_grads`](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/core/optimizer/optimizer.py#L1281-L1300).

For Phase 6.3, wrap the existing optimizer rather than using
`get_megatron_optimizer`:

```python
base_optimizer = torch.optim.AdamW(
    ddp_model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1.0e-8,
    foreach=False,
    fused=False,
)
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=learning_rate,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1.0e-8,
    decoupled_weight_decay=True,
    params_dtype=torch.float32,
    fp16=False,
    bf16=False,
    clip_grad=0.0,
    use_distributed_optimizer=False,
)
optimizer = FP32Optimizer(
    base_optimizer,
    optimizer_config,
    init_state_fn=lambda *_args, **_kwargs: None,
)
```

This is a Megatron-compatible optimizer lifecycle while preserving the exact
PyTorch AdamW implementation used by the accepted baseline.

Using `get_megatron_optimizer` in the pinned environment is not the minimum
controlled migration. Transformer Engine is installed, so the factory selects
TE `FusedAdam`; its default parameter-group helper also excludes biases and
one-dimensional parameters from weight decay unless
`config_overrides={}` is passed. Either change would confound the lifecycle
migration. A later optimizer-fusion phase can evaluate that factory path
separately.

Also set `clip_grad=0.0`: `OptimizerConfig` defaults to clipping at 1.0, while
the current raw AdamW loop does not clip gradients.

## Proposed construction and step order

The implementation should mirror Megatron's dedicated DDP initialization
stream, then wait on it from the current stream:

```python
ddp_config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=False,
    overlap_grad_reduce=False,
    overlap_param_gather=False,
    use_distributed_optimizer=False,
    check_for_nan_in_grad=False,
)

ddp_stream = torch.cuda.Stream()
ddp_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(ddp_stream):
    ddp_model = DistributedDataParallel(
        config=model.config,
        ddp_config=ddp_config,
        module=model,
        disable_bucketing=True,
    )
torch.cuda.current_stream().wait_stream(ddp_stream)
```

`disable_bucketing=True` is appropriate for the initial non-overlapped,
single-rank test. DDP also forces `bucket_size=None` when gradient overlap is
disabled.

The common step for both Phase 6.3 variants is:

```python
ddp_model.zero_grad_buffer()
optimizer.zero_grad(set_to_none=True)

with torch.autocast("cuda", dtype=torch.bfloat16):
    output = ddp_model(tokens, position_ids, attention_mask, labels=labels)
    loss = masked_language_model_loss(output, loss_mask)

loss.backward()
finalize_model_grads([ddp_model])

if create_local_graphs_this_step:
    create_cudagraphs()

update_successful, grad_norm, num_zeros = optimizer.step()
```

For the graph variant, call `create_cudagraphs()` only after the first eager
recording forward/backward and `finalize_model_grads`, but before that
iteration's optimizer step. This matches the pinned no-pipeline schedule,
which finalizes gradients, creates local graphs at the end of the schedule,
and only then returns to the outer optimizer step. The graph implementation
backs up and restores finalized `main_grad` values around capture.

There is no manual `main_grad` allocation, selective list of graph-owned
parameters, `main_grad -> grad` bridge, or post-step alias cleanup.

## CUDA Graph lifecycle requirements

MCore local graphs require all of the following:

1. Every captured trainable parameter already has a persistent
   `param.main_grad` tensor before the first recording iteration.
2. `main_grad` addresses and DDP `grad_data` storage remain stable across
   capture and replay.
3. Gradients are zeroed in place through `DDP.zero_grad_buffer()`; the code
   must never replace `main_grad`.
4. `grad_added_to_main_grad` is owned by DDP/MCore so eager hooks and graph
   replay cannot double-accumulate.
5. `finish_grad_sync()` runs before the optimizer reads gradients. It performs
   the graph-replay event wait in the pinned implementation.
6. Model shapes, module path, training/evaluation state, and argument structure
   are static for replay.
7. The Transformer Engine RNG tracker remains enabled for cuDNN
   FusedAttention and dropout.
8. `create_cudagraphs()` runs after the first complete recording backward in
   MCore execution order.
9. `cuda_graph_impl="local"` and `cuda_graph_modules=[]` remain the only graph
   configuration changes.
10. The optimizer is outside the graph; `optimizer_cuda_graph=False`.

## Minimum lab changes for Phase 6.3

Do not alter the accepted Phase 1 baseline path. Add the DDP lifecycle only to
the Phase 6 experiment:

1. Add one small shared helper, for example
   `scripts/phase6_megatron_ddp_lifecycle.py`, containing DDP construction,
   FP32 optimizer wrapping, the common step order, and lifecycle assertions.
2. Update `scripts/phase6_cuda_graph_correctness.py` to:
   - construct both candidates with DDP before optimizer creation;
   - remove `prepare_main_grad_buffers` and selective graph-parameter logic;
   - compare all DDP `main_grad` tensors;
   - compare post-step parameters and Adam states.
3. Update `scripts/phase6_cuda_graph_run.py` to:
   - remove the manual gradient bridge and alias cleanup;
   - call the shared DDP step in both variants;
   - create graphs after finalized gradients and before the first optimizer
     step;
   - unwrap `ddp_model.module` only for graph-runner inspection.
4. Update `scripts/phase6_analyze_cuda_graph.py` only if result field names
   change. Its timing and Nsight analysis can otherwise be reused.

Expected implementation size is small-to-moderate: one lifecycle helper and
localized changes in two Phase 6 scripts, approximately 150-250 lines
including assertions and result metadata. No model, kernel, scheduler,
checkpoint format, or framework patch is required.

## Correctness risks and controls

| Risk | Consequence | Required control |
|---|---|---|
| `OptimizerConfig.bf16=True` because autocast is BF16 | Selects a float16-parameter optimizer path and introduces master parameters/copies | Keep `bf16=False`, `fp16=False`, and `params_dtype=float32` |
| Default `clip_grad=1.0` | Changes updates relative to the accepted baseline | Set `clip_grad=0.0` |
| `get_megatron_optimizer` factory defaults | May switch to TE FusedAdam and change weight-decay grouping | Wrap the unchanged PyTorch AdamW with `FP32Optimizer` for Phase 6.3 |
| Optimizer built before DDP | Parameter/buffer lifecycle can be initialized in the wrong order | Build model, wrap DDP, then build optimizer from DDP parameters |
| Only `optimizer.zero_grad()` is called | Stale `main_grad` accumulates across steps | Call DDP buffer zero first, then optimizer zero |
| `main_grad` is reallocated | Captured graph writes to stale addresses | Zero DDP buffers in place only |
| Manual graph-parameter filtering | Embedding/output gradients use a different lifecycle | Let DDP own every trainable parameter |
| Missing `finalize_model_grads()` | Optimizer may read graph gradients before replay completion | Finalize before every optimizer step |
| Wrong graph creation point | Capture warmup can corrupt the current update or record incomplete order | Create after first finalized backward and before its optimizer step |
| Stale/double accumulation | Gradients are counted twice | Assert DDP owns `grad_added_to_main_grad`; remove all manual bridge code |
| Tied embedding/output weight | Duplicate or missing accumulation | Compare parameter identities and one `main_grad` per unique parameter |
| Different RNG setup between A and B | Dropout changes become a second variable | Use TE RNG tracker for both variants |
| DDP all-reduce overhead at `DP=1` | Absolute result is not directly comparable to pre-DDP Phase 5 timing | Compare Phase 6.3 A/B only; report DDP overhead separately if a legacy reference is run |
| Graph capture global state leaks between candidates | False replay evidence or allocator interference | Run each candidate in a fresh process |

Checkpoint keys remain unchanged because MCore DDP delegates `state_dict()` to
the wrapped module. Optimizer state remains the underlying AdamW state dict,
through `FP32Optimizer.state_dict()`.

## Phase 6.3 fast GPU test

### Test objective

Compare:

- **A — DDP eager:** MCore DDP/`main_grad`/`FP32Optimizer`,
  `cuda_graph_impl="none"`;
- **B — DDP local graph:** the identical lifecycle,
  `cuda_graph_impl="local"`, `cuda_graph_modules=[]`.

The graph setting must be the only A/B variable.

### Stage 1: lifecycle and numerical correctness gate

Run each candidate in a fresh process with identical initial weights, fixed
input tensors, and the same seed.

First use the existing two-layer reduced model with attention and hidden
dropout set to zero. Validate three consecutive optimizer steps, not just one
backward:

1. per-token output and scalar loss before each update;
2. the complete set of named `main_grad` tensors after
   `finalize_model_grads`;
3. all post-step model parameters;
4. AdamW `exp_avg`, `exp_avg_sq`, and step state;
5. absence of NaN/Inf in outputs, loss, gradients, parameters, and optimizer
   state;
6. matching trainable-parameter and gradient counts;
7. no unexpected `param.grad` before `FP32Optimizer.prepare_grads`;
8. stable `main_grad.data_ptr()` values across zero, backward, finalize, and
   optimizer steps;
9. one forward and one backward graph per TransformerLayer, with replay
   observed after capture.

Use the existing `atol=1e-5`, `rtol=1e-5` gate and report max/mean absolute
and relative errors plus global gradient cosine similarity. Any missing
gradient, non-finite value, stale pointer, or tolerance failure blocks the
performance test.

Then run a single-step full-model correctness check at a memory-safe
micro-batch with dropout zero. This catches tied embedding/output and
non-TransformerLayer gradient issues that the reduced model may miss.

As a migration diagnostic, one additional non-graph comparison may check the
old raw loop against the new DDP eager loop before the optimizer step. Their
losses and gradients should agree. This diagnostic is not part of the A/B
timing and must not substitute for the A-versus-B gate.

### Stage 2: fast A/B screen

Only after Stage 1 passes:

- same A40 Pod and pinned software stack;
- accepted 355.9M model, sequence length 2048, micro-batch 8;
- production attention and hidden dropout 0.1;
- cuDNN FusedAttention;
- `bias_dropout_fusion=True`, `bias_gelu_fusion=False`;
- FP32 parameters and unchanged PyTorch AdamW;
- `TP=PP=DP=1`;
- DDP overlap and distributed optimizer disabled;
- 5 internal graph warmup iterations for B;
- 3 benchmark warmup steps;
- 15 measured steps;
- alternating independent A/B process runs if memory permits multiple repeats.

Measure:

- average, median, minimum, and standard deviation of step time;
- tokens/second and MFU;
- peak allocated/reserved VRAM and `nvidia-smi` peak;
- CPU process time per step;
- graph capture time and memory overhead;
- loss sequence and finite-gradient status.

Use short Nsight Systems captures for both variants and report per step:

- CUDA kernel count;
- CUDA API launch count;
- CPU launch/API time;
- CPU launch-gap distribution;
- kernels shorter than 50 microseconds;
- total GPU idle time;
- graph replay API count and replay confirmation.

The A and B step ranges must include the same lifecycle:
DDP zeroing, forward, backward, gradient finalization, and optimizer step.
Graph capture time is reported separately and excluded from steady-state step
timing.

### Decision rule

- If correctness fails, stop and record the first lifecycle invariant that
  failed. Do not profile performance.
- If throughput improves by at least 2% with confirmed replay and reduced
  launch/API overhead, proceed to a separate formal 20-warmup + 100-measured
  validation.
- If the gain is below 1%, stop without formal validation.
- If the gain is 1-2%, decide from repeat stability and profiler evidence.

No distributed optimizer, optimizer graph, optimizer fusion, dtype change,
new activation fusion, model change, or software-version change is allowed in
Phase 6.3.

## Final assessment

The migration is technically straightforward and does not require invasive
model changes. The critical change is ownership: DDP must allocate and retain
all `main_grad` buffers before MCore records local graphs, and optimizer
zero/step operations must use Megatron's wrapper lifecycle.

The highest-risk mistakes are not CUDA Graph capture itself, but silently
changing optimizer semantics or zeroing only `param.grad`. Keeping the
underlying PyTorch AdamW and wrapping it with `FP32Optimizer` isolates the
gradient-lifecycle correction and makes the proposed Phase 6.3 A/B a valid
one-variable test.
