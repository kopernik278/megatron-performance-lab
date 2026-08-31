# Phase 10.1: TP=2 + PP=2 hybrid baseline (DP=1)

FAST ITERATION MODE planned (5 warmup + 20 measured). CUDA Graph off.

## Outcome

- Status: **blocked (capacity)**
- Harness: **ready** on branch `cursor/phase101-tp2-pp2-hybrid-3b5c`
- GPU: **NVIDIA A40 48GB ×4** (user request; aligns with prior Phase 7–9 baselines)
- Blocker: RunPod has no available **4×A40 SECURE** instances (global stockout as of 15:10 UTC)

## Latest action (2026-08-31 15:10 UTC)

User requested **A40 only** for Phase 10.1:

- Deleted 1×A40 probe pod `rvy5biypcapl1c` (cannot run TP=2+PP=2 on one GPU)
- RunPod template `zh7yn78wii` reverted to `PHASE101_GPU_TYPE=NVIDIA A40`, price arg `1.76` ($0.44/h×4 SECURE)
- `create-pod` 4×A40 (EU-SE-1, CA-MTL-1, global) → **no instances available**
- Timer `phase101-a40-retry` retries every 5 minutes until 4×A40 capacity appears

Harness GPU profiles (`scripts/phase10_gpu_profile.py`) still support L40S/4090/etc. for auto-detect, but deployment is pinned to A40.

Prior fix `ddd32e6`: preflight → topology `--preflight-json` wiring + one-shot abort guard.

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
09lguqnpdx9c2e  | CA-MTL-1  | 6441153f   | 1.76    | missing --preflight-json (fixed ddd32e6)
```

After `ddd32e6`, repeated create-pod calls (SECURE/COMMUNITY, CA-MTL-1/global, 4-GPU and 1-GPU) return **no instances available**.

## Harness files

- `scripts/phase10_gpu_profile.py` — GPU peak TFLOPS + NVTE CUDA arch selection
- `scripts/phase10_preflight.py` — single-process topo/P2P/pairwise NCCL gate
- `scripts/phase10_topology.py` — 4-rank NCCL all-reduce sanity
- `scripts/phase10_tp_pp_run.py` — hybrid runner with rank/group/layer/sharding reports
- `scripts/phase10_analyze_tp_pp.py` — microbatch sweep, TP/PP comm split, scaling
- `scripts/phase10_tp_pp_pod.sh` — pod orchestration

## Retry

Template `zh7yn78wii` auto-deploys A40 with `PHASE101_GPU_TYPE=NVIDIA A40`:

```bash
PHASE101_GPU_TYPE="NVIDIA A40" bash scripts/phase10_tp_pp_pod.sh <pod_id> 1.76
```

Cloud Agent timer `phase101-a40-retry` retries every 5 minutes until 4×A40 SECURE capacity appears (CUDA 12.8 host preferred for `cu128` image).
