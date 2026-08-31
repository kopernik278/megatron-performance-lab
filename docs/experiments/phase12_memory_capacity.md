# Phase 12: Training Memory and Capacity Engineering

Status: **harness ready** — awaiting 2×A40 pod run.

## Scope change

After Phase 12, the only remaining phase is **Phase 15** final packaging.
Do not start Phase 10/11/13/14.

## Pinned Megatron activation recompute (commit `09fde85`)

Verified flags (do not invent names):

| Field | Value | Role |
|---|---|---|
| `recompute_granularity` | `'full'` | Checkpoint entire Transformer layer |
| `recompute_method` | `'uniform'` | Uniform layer chunks |
| `recompute_num_layers` | `1` | One layer per recompute unit (24 units for 24 layers) |

Sources:
- `megatron/core/transformer/transformer_config.py` — config fields + validation
- `megatron/core/transformer/transformer_block.py` — `recompute_granularity == 'full'` → `checkpointed_forward`
- `megatron/core/recompute.py` — uniform `tensor_parallel.checkpoint` chunks

Deprecated CLI `--checkpoint-activations` is rejected; `--recompute-activations` maps to **selective** (core_attn only). Phase 12 uses programmatic **full** layer recompute, not selective.

Expected behavior: layer inputs saved; activations discarded after forward; backward recomputes each layer before local gradients.

## Variants (DP=2, TP=1, PP=1)

| Variant | DistOpt | overlap_grad_reduce | overlap_param_gather | Activation CKPT |
|---|---|---|---|---|
| A | OFF | ON | OFF | OFF |
| B | ON | ON | ON | OFF |
| C | OFF | ON | OFF | ON (full/uniform/1) |
| D | ON | ON | ON | ON (full/uniform/1) |

## Workload

Fixed: ~355.9M GPT, 24L, h=1024, FFN=4096, heads=16, seq=2048, MB=8/GPU, BF16 autocast, fused attn, BDA on, bias-GELU off, CUDA Graph off.

Capacity: bounded exponential+binary search on microbatch @ seq=2048 (cap MB≤64).

## Harness

- `scripts/phase12_memory_run.py`
- `scripts/phase12_capacity_search.py`
- `scripts/phase12_analyze_memory.py`
- `scripts/phase12_memory_pod.sh`
- `build_model(..., recompute_*)` wired in `scripts/phase1_baseline.py`

```bash
bash scripts/phase12_memory_pod.sh <pod_id> 0.88
```
