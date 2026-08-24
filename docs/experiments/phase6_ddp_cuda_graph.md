# Phase 6.3: MCore DDP lifecycle and local CUDA Graph

## Outcome

The MCore DDP migration is correct, but MCore local CUDA Graph replay is not
numerically equivalent to eager DDP with the pinned stack and accepted numerical
configuration.

Phase A matched the previous raw-PyTorch lifecycle exactly for three consecutive
steps. Phase B successfully captured, replayed, and reused both forward and
backward graphs, but the first replay produced a deterministic gradient mismatch.
That mismatch changed the AdamW update and compounded into the next forward pass.

The correctness gate therefore rejected CUDA Graph before the 355.9M-parameter
3-warmup + 15-measured fast A/B. No performance result is reported, and no formal
20+100 validation was run.

## Pinned environment

- GPU: 1x NVIDIA A40 48 GB
- Driver: 570.195.03
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- NCCL: 2.27.3
- cuDNN: 9.10.2
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine: `2.17.1+4329ff84`
- Production target: 355.9M GPT, sequence length 2048, micro-batch 8
- Precision: BF16 autocast, FP32 parameters/residuals/gradients/AdamW state
- Attention: cuDNN FusedAttention, sub-backend 1
- `bias_dropout_fusion=True`
- `bias_activation_fusion=False`
- TP/PP/DP: 1/1/1
- Distributed optimizer: disabled

Dropout was zero only in the reduced correctness gate. The planned production
timing configuration retained dropout 0.1.

## Implemented lifecycle

The harness now builds the model in this order:

1. Construct the existing `GPTModel`.
2. Wrap it with MCore `DistributedDataParallel`.
3. Let DDP allocate contiguous gradient buffers and assign every trainable
   parameter a persistent FP32 `main_grad` view.
4. Construct the unchanged `torch.optim.AdamW` over the wrapped parameters.
5. Wrap AdamW with MCore `FP32Optimizer`, with clipping disabled and no
   distributed optimizer.

Every step uses:

```text
DistributedDataParallel.zero_grad_buffer()
FP32Optimizer.zero_grad(set_to_none=True)
forward
loss.backward()
finalize_model_grads([ddp_model])
FP32Optimizer.step()
```

`FP32Optimizer.prepare_grads()` aliases each `param.grad` to its DDP-owned
`param.main_grad` before calling AdamW. The optimizer remains outside CUDA Graph.

DDP communication overlap, parameter-gather overlap, and the distributed
optimizer are disabled. At DP=1, `finalize_model_grads` preserves the same
gradient values while exercising the standard lifecycle.

## Phase A: DDP lifecycle correctness

The gate used a two-layer model, sequence length 128, micro-batch 1, identical
initial state and input, cuDNN FusedAttention, and dropout 0. It compared the old
raw `param.grad` + AdamW path against eager MCore DDP + `main_grad` +
`FP32Optimizer` for three updates.

| Check | Result |
|---|---:|
| Loss maximum absolute difference | 0.0 |
| Output maximum absolute difference | 0.0 |
| Gradient global cosine similarity | 1.0 |
| Gradient maximum absolute difference | 0.0 |
| Parameter-update maximum absolute difference | 0.0 |
| AdamW-state maximum absolute difference | 0.0 |
| `main_grad` tensor addresses stable | yes |
| `main_grad` zero before every step | yes |
| `param.grad` cleared before every step | yes |
| Optimizer consumed the `main_grad` storage | yes |
| Parameters updated | yes |
| NaN/Inf | none |

There was no stale accumulation: all three steps, including optimizer state,
matched bit-for-bit. This validates the Phase 6.2 DDP lifecycle design.

## Phase B: CUDA Graph correctness

The graph candidate used:

```python
cuda_graph_impl = "local"
cuda_graph_modules = []
cuda_graph_warmup_steps = 5
```

The first eager recording iteration was exact. Capture then created one forward
and one backward graph for each of the two TransformerLayers:

- graph managers: 2
- graph runners: 2
- forward graphs: 2
- backward graphs: 2
- capture time: 0.3895 seconds
- additional allocated bytes during capture: 17,043,456
- additional reserved bytes during capture: 25,165,824

Runner and graph object identities remained unchanged over subsequent iterations,
so this was real graph reuse rather than repeated capture.

### Numerical comparison

| Metric | Recording step | First replay | Second replay |
|---|---:|---:|---:|
| Loss absolute difference | 0.0 | 0.0 | 3.67165e-5 |
| Output maximum absolute difference | 0.0 | 0.0 | 0.00389814 |
| Gradient global cosine | 1.0 | 0.9998457250 | 0.9997978663 |
| Gradient maximum absolute difference | 0.0 | 0.00518620 | 0.00482816 |
| Update maximum absolute difference | 0.0 | 0.000164179 | 0.000177053 |
| Parameters-after maximum difference | 0.0 | 0.000164179 | 0.000336707 |
| All finite | yes | yes | yes |
| Strict `atol=rtol=1e-5` pass | yes | no | no |

The largest gradient error was again in
`embedding.word_embeddings.weight`. The first replay's forward output and loss
were exact, which localizes the initial divergence to backward replay. The
resulting AdamW update then caused the second replay's forward output to diverge.

### Capture-flag diagnostic

The pinned backward capture evaluates `grad_added_to_main_grad` in Python before
recording `main_grad.add_(wgrad)`. A diagnostic run temporarily forced the flag
false around capture while preserving finalized gradient buffers. It produced
the same replay errors bit-for-bit and was removed from the final code.

Source inspection explains why: the eager DDP post-hook moves `param.grad` into
`main_grad` but does not set that flag true; local graph replay sets it after the
captured backward completes. The normal first capture therefore already records
the intended `main_grad` accumulation. The flag is not the remaining cause.

The evidence establishes that DDP buffer ownership fixes the raw optimizer
lifecycle, but it does not make this pinned local backward replay strictly
equivalent to eager DDP. The exact lower-level operation responsible was not
proven; forcing additional changes would violate the pinned-version and
numerical-control requirements.

## Phase C and decision

The required decision sequence is correctness before performance. Because Phase
B failed:

| Metric | A: eager DDP | B: local CUDA Graph |
|---|---:|---:|
| Average/median step time | not run | not run |
| Tokens/sec | not run | not run |
| MFU | not run | not run |
| Peak VRAM | not run | not run |
| CPU time/step | not run | not run |
| CUDA API launches/step | not run | not run |
| GPU idle/gaps | not run | not run |
| Speedup | not run | not run |

No Nsight Systems timing capture was taken because it cannot make an incorrect
candidate acceptable. The >=2% throughput threshold was not evaluated, and the
formal 20-warmup + 100-measured A/B was not run.

Do not adopt MCore local CUDA Graph for this workload yet. Keep the validated
MCore DDP lifecycle, and isolate the local graph backward numerical divergence
before any renewed performance screen.

## Infrastructure cleanup

- Pod: `9ibu90ocv6eulw`
- Data center: EU-SE-1
- Final status: `EXITED`
- GPU billing stopped immediately after the repeated correctness rejection.

Machine-readable evidence is in `results/phase6_ddp_cuda_graph.json`.
