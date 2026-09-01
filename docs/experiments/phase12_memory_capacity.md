# Phase 12: Training Memory and Capacity Engineering

Status: **terminated** — Runpod 2×A40 infrastructure cannot provide a reliable NCCL DP=2 experiment environment. Per user directive (2026-09-01): stop Phase 12; no further pod retries or stock watch.

## Termination summary

Phase 12 harness (A/B/C/D variants + capacity search) is implemented and merged on branch `cursor/phase12-memory-capacity-3b5c`, but **no complete measured dataset** was collected.

**Root cause:** Most EU-SE (and CA-MTL) 2×A40 hosts hang on NCCL P2P topology sanity (90s timeout). Runpod does not allow pinning a specific host; stock repeatedly lands on known-bad machines. Only host `64411267` passed topology and partial smoke tests; re-landing that host is not controllable.

**Partial evidence on `64411267`:**

| Step | Result |
|---|---|
| Topology | OK |
| Smoke A/B | OK (before later harness fixes) |
| Smoke C | Failed (TE autocast; fixed in `70e9aab`, not re-run on good host) |
| Fixed workload A/B/C/D | Not collected |
| Capacity search | Not collected |

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
| — | EU-SE-1 | 64411279 | 12.8 | stock saturated on bad host |
| — | EU-SE-1 | 64411137 | 13.0 | stock watch — all bad hosts |
| `cnsmk11q8r4gzj` | EU-SE-1 | 64411be4 | 13.0 | NCCL hang |
| `f2jnkr58vn06lq` | EU-SE-1 | 64411be7 | 12.8 | NCCL hang |
| `aegwyeert35umd` | EU-SE-1 | 64411730 | 12.8 | NCCL hang (final attempt) |

## Host inventory (final)

- **Known-good topology:** `64411267` only (unreachable on demand)
- **Known-bad NCCL (10):** `644110db`, `64411856`, `64411133`, `64411137`, `6441127d`, `6441120d`, `64411279`, `64411be4`, `64411be7`, `64411730`

## Harness deliverables (usable without new pods)

- `scripts/phase12_memory_pod.sh` — pod entry + topology gate + full pipeline
- `scripts/phase12_memory_run.py` — variant runner
- `scripts/phase12_capacity_search.py` — MB search
- `scripts/phase12_analyze_memory.py` — results aggregation
- Runpod template `y8zwfexxki`

## Harness fixes landed

- `e249c1d`: TE fused backend check after smoke forward
- `70e9aab`: BF16 autocast wraps forward+backward for TE full recompute
- `7bcc107`: persistent abort marker
- `ba06424`: check `ABORT_MARKER` before git restore on container restart

## Next

Phase 12 closed. Resume only if a **pin-able** 2×A40 host with working NCCL P2P becomes available (e.g. dedicated bare-metal or host-locked cloud).
