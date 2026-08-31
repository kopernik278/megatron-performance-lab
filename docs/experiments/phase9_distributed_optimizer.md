# Phase 9.2: DP=2 Megatron Distributed Optimizer — INFRASTRUCTURE BLOCKED

**Status:** infrastructure blocked (retry 2026-08-31)  
**Harness:** `cursor/phase92-distributed-optimizer-3b5c` (reuse existing scripts, no redesign)

## Retry outcome

Host-search budget: **3 pods / 20 minutes**. Search stopped after **0 pods provisioned** because RunPod reported **A40 stock Out** for both Secure and Community clouds at search time. No topology probe or A/B/C benchmark ran.

| Attempt | Cloud | DC | Result |
|---------|-------|-----|--------|
| 1 | Secure | CA-MTL-1 | No stock |
| 2 | Secure | any | No stock |
| 3 | Community | any | No stock |

Relaxed topology policy was in effect (SYS allowed when P2P + NCCL sanity pass); never reached host validation.

## Planned variants (not executed)

| Variant | dist opt | overlap_param_gather | Grad comm | Param comm |
|---------|----------|----------------------|-----------|------------|
| A | off | off | All-Reduce | — |
| B | on | off | Reduce-Scatter | All-Gather sync |
| C | on | on | Reduce-Scatter | All-Gather overlap |

All variants: `overlap_grad_reduce=True`, TP=1, PP=1, DP=2, FAST 5+20.

## Harness (unchanged)

- `scripts/phase92_distopt_run.py`
- `scripts/phase92_analyze_distopt.py`
- `scripts/phase92_distopt_pod.sh`
- Megatron-LM `09fde85`: `use_distributed_optimizer`, `overlap_grad_reduce`, `overlap_param_gather`

## Re-run

When 2×A40 stock returns:

```bash
bash scripts/phase92_distopt_pod.sh <pod_id> 0.88
```

## PR

https://github.com/kopernik278/megatron-performance-lab/pull/21
