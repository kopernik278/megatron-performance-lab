# Phase 12: Training Memory and Capacity Engineering

Status: **retrying on EU-SE-1** — user-directed rerun after infrastructure-blocked pause. Harness `70e9aab` + `e249c1d` ready. Prefer topology-good host class `64411267`; delete immediately on NCCL hang.

After Phase 12 → **Phase 15 only** (packaging).

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
| `mebfqubiyk8agy` | EU-SE-1 | 6441127d | 13.0 | **running** (rerun) |

## Host classes

- Known-good topology (prior): EU-SE `64411267`
- Known-bad NCCL: CA-MTL `644110db`, `64411856`, `64411133`; EU-SE `64411137`

## Harness fixes

- `e249c1d`: TE fused backend check after smoke forward
- `70e9aab`: BF16 autocast wraps forward+backward for TE full recompute

## Experiment design

Variants A/B/C/D on DP=2, TP=1, PP=1; no SP/VPP/Userbuffers/CUDA Graph. Workload ~355.9M GPT, MB=8 fixed + capacity search MB≤64. Megatron recompute: full/uniform/num_layers=1.
