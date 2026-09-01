# Phase 12: Training Memory and Capacity Engineering

Status: **aborted** — host `64411be7` NCCL hang; pod `f2jnkr58vn06lq` deleted. Resuming stock watch.

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
| `tsp6vzihrfsajk` | EU-SE-1 | 6441120d | 13.0 | NCCL hang |
| `ita9rvo5y7jbps` | EU-SE-1 | 64411279 | 12.8 | NCCL hang |
| — | EU-SE-1 | 64411279 | 12.8 | **stock saturated** on bad host (3× recreate) |
| — | EU-SE-1 | 64411137 | 13.0 | **stock watch 04:31** — 3× recreate, all bad host |
| `cnsmk11q8r4gzj` | EU-SE-1 | 64411be4 | 13.0 | NCCL hang |
| `f2jnkr58vn06lq` | EU-SE-1 | 64411be7 | 12.8 | NCCL hang (topology 90s timeout); pod deleted |

## Host classes

- Known-good topology: EU-SE **`64411267` only**
- Known-bad NCCL: `644110db`, `64411856`, `64411133`, `64411137`, `6441127d`, `6441120d`, `64411279`, `64411be4`, **`64411be7`**
- **Current blocker:** EU-SE-1 2×A40 stock rotating bad hosts; waiting for `64411267` or new good host

## Harness fixes

- `e249c1d`: TE fused backend check after smoke forward
- `70e9aab`: BF16 autocast wraps forward+backward for TE full recompute
- `7bcc107`: persistent abort marker (stop restart billing loops)
- pending: check `ABORT_MARKER` before git restore on container restart
