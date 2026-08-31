# Phase 10.1: TP=2 + PP=2 hybrid baseline (DP=1)

FAST ITERATION MODE planned (5 warmup + 20 measured). CUDA Graph off.

## Outcome

- Status: **aborted (infrastructure)**
- Reason: 4-GPU NCCL/P2P topology sanity hung on all attempted RunPod hosts before hybrid runs could start.
- Harness: **ready** on branch `cursor/phase101-tp2-pp2-hybrid-3b5c`

## Planned experiment

- Model: ~355.9M GPT, 24 layers, hidden=1024, FFN=4096, heads=16, seq=2048
- Parallel: TP=2, PP=2, DP=1, world_size=4
- Pipeline: 12 layers/stage, non-interleaved 1F1B, `pipeline_dtype=torch.float32`
- Phase A: microbatch sweep M=2/4/8 at global batch 8
- Phase B: Nsight profile on best M (TP vs PP comm separation)
- Phase C: same-host references R1–R4 (1/2/2/4 GPUs)

## Infrastructure attempts

```
pod_id          | DC        | host       | price/h | failure
----------------|-----------|------------|---------|------------------------------------------
c2xo8wckih83bz  | CA-MTL-1  | 644113db   | 1.76    | template git clone loop (fixed)
f45lmcr4ssphq6  | CA-MTL-1  | 644113db   | 1.76    | NCCL 4-GPU topology timeout 120s
adkaupj01r70h7  | EU-SE-1   | 644112a8   | 1.76    | NCCL 4-GPU topology timeout ~240s loop
```

Budget target was $3.00 at $1.76/h (4x A40 Secure). Three pods were provisioned and deleted; no benchmark data collected.

## Harness files

- `scripts/phase10_topology.py` — full 4-GPU topo matrix + NCCL/P2P sanity
- `scripts/phase10_tp_pp_run.py` — hybrid runner with explicit rank/group/layer/sharding reports
- `scripts/phase10_analyze_tp_pp.py` — microbatch sweep, TP/PP comm split, scaling
- `scripts/phase10_tp_pp_pod.sh` — pod orchestration

## Retry command

```bash
bash scripts/phase10_tp_pp_pod.sh <pod_id> 1.76
```

Requires a 4x A40 host where `torch.distributed.run --nproc_per_node=4 scripts/phase10_topology.py` completes within 240s.
