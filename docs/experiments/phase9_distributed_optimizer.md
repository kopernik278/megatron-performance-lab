# Phase 9.2: DP=2 Megatron Distributed Optimizer — ABORTED (host pool)

FAST ITERATION MODE harness is implemented on branch `cursor/phase92-distributed-optimizer-3b5c` (commit `c4d82d5`) but **GPU benchmarks did not complete** because the available Secure 2×A40 CA-MTL-1 pool repeatedly allocated hosts that failed topology/NCCL sanity.

## Harness (ready)

Pinned Megatron-LM `09fde85` flags (verified, not invented):

- `DistributedDataParallelConfig.use_distributed_optimizer` / `--use-distributed-optimizer`
- `overlap_grad_reduce` / `--overlap-grad-reduce`
- `overlap_param_gather` / `--overlap-param-gather` (requires distributed optimizer + overlap_grad_reduce)

Scripts:

- `scripts/phase92_distopt_run.py` — A/B/C benchmark + dist-opt-aware correctness
- `scripts/phase92_analyze_distopt.py` — All-Reduce / Reduce-Scatter / All-Gather trace analysis
- `scripts/phase92_distopt_pod.sh` — pod orchestration (5+20 FAST; formal B/C if C≥2% over B)

## Planned variants

| Variant | dist opt | overlap_param_gather | Gradient comm | Param comm |
|---------|----------|----------------------|---------------|------------|
| A | off | off | All-Reduce | — |
| B | on | off | Reduce-Scatter | All-Gather (sync) |
| C | on | on | Reduce-Scatter | All-Gather (overlap) |

## Host hunt summary

| Host suffix | DC | Outcome |
|-------------|-----|---------|
| `64411397` | CA-MTL-1 | **NODE** topology OK; smoke failed on loss check (fixed in `68703e9`) |
| `64411b62` | CA-MTL-1 | SYS topology; NCCL may pass with `--allow-sys-topology` |
| `64411136` | CA-MTL-1 | NCCL P2P hang (90s topology timeout) × repeated allocations |
| `64411275` | EU-SE-1 | NCCL P2P hang |
| `64411914` | CA-MTL-1 | NCCL P2P hang |

Phase 9.1 reference host: suffix `6441139d`, public IP `69.30.85.75`, NODE path — not re-allocated during this hunt.

## Correctness fixes applied during hunt

1. **DP=2 smoke:** do not require identical per-rank loss (different micro-batches per rank).
2. **SYS topology:** allow when NCCL All-Reduce sanity and bidirectional P2P pass (`--allow-sys-topology`).

## Re-run command

When a good host is available:

```bash
bash scripts/phase92_distopt_pod.sh <pod_id> 0.88
```

Target: CA-MTL-1 Secure 2×A40 ≤$0.90/h, NODE/PIX topology, NCCL sanity pass.

## PR

https://github.com/kopernik278/megatron-performance-lab/pull/21
