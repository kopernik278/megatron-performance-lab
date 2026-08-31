# Phase 12: Training Memory and Capacity Engineering

Status: **retrying** — topology OK on EU-SE `64411267`; variant C failed on recompute outside BF16 autocast (fixed). Redeploying.

## Latest attempts

| Pod | DC | Host | Outcome |
|---|---|---|---|
| `4e5cg8xbus1gb5` | EU-SE-1 | 64411731 | SYS gate bug (fixed `a742cb8`) |
| `ymmbcftxbayqyy` | CA-MTL-1 | 644110db | NCCL P2P hang |
| `1cjjx4oc2l2uvr` | CA-MTL-1 | 64411856 | NCCL P2P hang |
| `x73moln06loei1` | EU-SE-1 | 64411267 | Topology OK; premature `fused_backend_status` |
| `idk2hckrkrbejr` | EU-SE-1 | 64411267 | Topology OK; A/B smoke OK; **C** failed TE backend on recompute |

## Fixes

1. `e249c1d` — call `fused_backend_status()` after first smoke forward.
2. Autocast must wrap `loss.backward()`: full recompute re-runs TE attention during backward; without autocast QKV stay FP32 and fused attn is disabled (`NVTE_UNFUSED_ATTN` off → hard fail).

## Scope change

After Phase 12, only **Phase 15** final packaging remains.

## Variants (DP=2)

| Variant | DistOpt | Activation CKPT |
|---|---|---|
| A | OFF | OFF |
| B | ON | OFF |
| C | OFF | ON (full/uniform/1) |
| D | ON | ON |

```bash
bash scripts/phase12_memory_pod.sh <pod_id> 0.88
```
