# Phase 12: Training Memory and Capacity Engineering

Status: **retrying** — EU-SE-1 topology **passed** (SYS + NCCL); smoke aborted due to premature TE backend check (fixed). Redeploying.

## Latest attempts

| Pod | DC | Host | Outcome |
|---|---|---|---|
| `4e5cg8xbus1gb5` | EU-SE-1 | 64411731 | SYS gate bug (fixed `a742cb8`) |
| `ymmbcftxbayqyy` | CA-MTL-1 | 644110db | NCCL P2P hang |
| `1cjjx4oc2l2uvr` | CA-MTL-1 | 64411856 | NCCL P2P hang (TE built OK) |
| `x73moln06loei1` | EU-SE-1 | 64411267 | Topology OK (`SYS`, NCCL pass); smoke failed: `fused_backend_status` before first forward |

Known-bad CA-MTL NCCL hosts: `644110db`, `64411856`. Prefer EU-SE-1 (SYS+NCCL works with `--allow-sys-topology`).

## Fix after `x73moln06loei1`

TE populates `_attention_backends` on first `DotProductAttention` forward. Phase 12 called `fused_backend_status()` before smoke (unlike Phase 9.2/8.1). Moved the check to after smoke.

## Scope change

After Phase 12, the only remaining phase is **Phase 15** final packaging.
Do not start Phase 10/11/13/14.

## Pinned Megatron activation recompute (commit `09fde85`)

| Field | Value | Role |
|---|---|---|
| `recompute_granularity` | `'full'` | Checkpoint entire Transformer layer |
| `recompute_method` | `'uniform'` | Uniform layer chunks |
| `recompute_num_layers` | `1` | One layer per recompute unit |

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

```bash
bash scripts/phase12_memory_pod.sh <pod_id> 0.88
```
