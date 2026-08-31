# Phase 12: Training Memory and Capacity Engineering

Status: **final GPU retry running** — pod `gyisoehezdhyiv` on EU-SE-1 host `64411137` (CUDA 13.0, $0.88/h, harness `70e9aab`).

## Attempts

| Pod | DC | Host | Outcome |
|---|---|---|---|
| `4e5cg8xbus1gb5` | EU-SE-1 | 64411731 | SYS gate bug (fixed) |
| `ymmbcftxbayqyy` | CA-MTL-1 | 644110db | NCCL hang |
| `1cjjx4oc2l2uvr` | CA-MTL-1 | 64411856 | NCCL hang |
| `x73moln06loei1` | EU-SE-1 | 64411267 | Topology OK; premature fused check |
| `idk2hckrkrbejr` | EU-SE-1 | 64411267 | Topology OK; A/B OK; C autocast bug |
| `8iishiy8k49pf8` | CA-MTL-1 | 64411133 | NCCL hang |
| `gyisoehezdhyiv` | EU-SE-1 | 64411137 | **running** (CUDA 13.0) |

Known-good topology: EU-SE `64411267`. Known-bad NCCL: CA-MTL `644110db`, `64411856`, `64411133`.

## Fixes landed

- `e249c1d`: fused backend check after smoke
- `70e9aab`: BF16 autocast wraps backward for TE recompute

After Phase 12 → **Phase 15 only**. If this EU-SE attempt fails → stop GPU loops (infrastructure-blocked).
