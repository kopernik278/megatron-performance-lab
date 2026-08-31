# Phase 10.1: TP=2 + PP=2 hybrid baseline (DP=1)

FAST ITERATION MODE planned (5 warmup + 20 measured). CUDA Graph off.

## Outcome

- Status: **aborted (NCCL preflight)**
- Harness: **ready** on branch `cursor/phase101-tp2-pp2-hybrid-3b5c`
- GPU: **NVIDIA A40 48GB ×4** (user request)
- Blocker: Preflight NCCL pairwise sanity (GPU0↔GPU1) timed out after 90s on CA-MTL-1 host `64410fe7`

## Latest run (2026-08-31 15:50–16:01 UTC)

Pod `nlg0ojcni8i1xk` (4×A40 SECURE, CA-MTL-1, CUDA 12.8, $1.76/h):

1. Repo / Megatron / TE clone OK
2. `PHASE101_GPU_PROFILE=NVIDIA A40` / `NVTE_CUDA_ARCHS=86`
3. TransformerEngine build OK (~8 min)
4. `PHASE101_CUDA_READY elapsed=0s`
5. **Preflight failed**: `nccl_pair_sanity(0,1)` → `TimeoutExpired` after 90s
6. `PHASE101_ABORT=preflight failed` → container restart loop → **pod deleted** (~$0.32)

This matches earlier NCCL hangs on CA-MTL-1 (`644113db`) and EU-SE-1 (`644112a8`). Preflight now catches the failure before the expensive hybrid sweeps.

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
nlg0ojcni8i1xk  | CA-MTL-1  | 64410fe7   | 1.76    | preflight NCCL pair 0-1 timeout 90s
```

## Harness files

- `scripts/phase10_gpu_profile.py` — GPU peak TFLOPS + NVTE CUDA arch selection
- `scripts/phase10_preflight.py` — single-process topo/P2P/pairwise NCCL gate
- `scripts/phase10_topology.py` — 4-rank NCCL all-reduce sanity
- `scripts/phase10_tp_pp_run.py` — hybrid runner with rank/group/layer/sharding reports
- `scripts/phase10_analyze_tp_pp.py` — microbatch sweep, TP/PP comm split, scaling
- `scripts/phase10_tp_pp_pod.sh` — pod orchestration

## Next

Retry 4×A40 on a **different host** (prefer EU-SE-1; avoid known bad suffixes `644113db`, `644112a8`, `64410fe7` when possible). Do **not** disable NCCL P2P for this baseline — TP/PP needs working GPU interconnect.

```bash
PHASE101_GPU_TYPE="NVIDIA A40" bash scripts/phase10_tp_pp_pod.sh <pod_id> 1.76
```
