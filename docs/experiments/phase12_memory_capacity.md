# Phase 12: Training Memory and Capacity Engineering

Status: **infrastructure-blocked** — final EU-SE retry (`gyisoehezdhyiv`) NCCL P2P-hung on topology sanity. GPU retry loops stopped. Harness fixes remain ready (`e249c1d`, `70e9aab`).

After Phase 12 → **Phase 15 only** (packaging). Do not start Phases 10/11/13/14.

## Attempts

| Pod | DC | Host | CUDA | Outcome |
|---|---|---|---|---|
| `4e5cg8xbus1gb5` | EU-SE-1 | 64411731 | — | SYS gate bug (fixed) |
| `ymmbcftxbayqyy` | CA-MTL-1 | 644110db | — | NCCL hang |
| `1cjjx4oc2l2uvr` | CA-MTL-1 | 64411856 | — | NCCL hang |
| `x73moln06loei1` | EU-SE-1 | 64411267 | — | Topology OK; premature fused check |
| `idk2hckrkrbejr` | EU-SE-1 | 64411267 | — | Topology OK; A/B smoke OK; C autocast bug |
| `8iishiy8k49pf8` | CA-MTL-1 | 64411133 | 13.0 | NCCL hang |
| `gyisoehezdhyiv` | EU-SE-1 | 64411137 | 13.0 | NCCL hang (final allowed retry) |

## Host classes

- Known-good topology (prior): EU-SE `64411267` (SYS+NCCL OK; harness bugs blocked A–D)
- Known-bad NCCL: CA-MTL `644110db`, `64411856`, `64411133`; EU-SE `64411137`

## Harness fixes landed (not validated end-to-end on A–D)

- `e249c1d`: TE fused backend check after smoke forward
- `70e9aab`: BF16 autocast wraps forward+backward for TE full recompute

## Experiment design (unchanged; not measured)

Variants A/B/C/D on DP=2, TP=1, PP=1; no SP/VPP/Userbuffers/CUDA Graph. Workload ~355.9M GPT, MB=8 fixed + capacity search MB≤64. Megatron recompute: full/uniform/num_layers=1.

## Decision

Stop GPU loops. Phase 12 results are incomplete due to multi-host NCCL P2P topology failures on available 2×A40 Secure stock. Re-run later only when a known-good topology host (e.g. `64411267`) is available or NCCL/host placement is otherwise reliable.
