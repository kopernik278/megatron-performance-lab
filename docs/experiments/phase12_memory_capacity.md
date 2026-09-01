# Phase 12: Training Memory and Capacity Engineering

Status: **running** — user-directed rerun on EU-SE-1. Target known-good topology host `64411267`; delete immediately on NCCL hang.

## Attempts

| Pod | DC | Host | CUDA | Outcome |
|---|---|---|---|---|
| `4e5cg8xbus1gb5` | EU-SE-1 | 64411731 | — | SYS gate bug (fixed) |
| `ymmbcftxbayqyy` | CA-MTL-1 | 644110db | — | NCCL hang |
| `1cjjx4oc2l2uvr` | CA-MTL-1 | 64411856 | — | NCCL hang |
| `x73moln06loei1` | EU-SE-1 | 64411267 | — | Topology OK; premature fused check |
| `idk2hckrkrbejr` | EU-SE-1 | 64411267 | — | Topology OK; A/B smoke OK; C autocast bug |
| `8iishiy8k49pf8` | CA-MTL-1 | 64411133 | 13.0 | NCCL hang |
| `gyisoehezdhyiv` | EU-SE-1 | 64411137 | 13.0 | NCCL hang |
| `mebfqubiyk8agy` | EU-SE-1 | 6441127d | 13.0 | NCCL hang |
| `tsp6vzihrfsajk` | EU-SE-1 | 6441120d | 13.0 | NCCL hang (restart loop) |
| `ita9rvo5y7jbps` | EU-SE-1 | 64411279 | 12.8 | **running** |

## Host classes

- Known-good topology: EU-SE **`64411267` only**
- Known-bad NCCL: `644110db`, `64411856`, `64411133`, `64411137`, `6441127d`, `6441120d`

## Harness fixes

- `e249c1d`: TE fused backend check after smoke forward
- `70e9aab`: BF16 autocast wraps forward+backward for TE full recompute
