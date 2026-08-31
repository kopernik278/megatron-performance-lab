# Phase 10.1: TP=2 + PP=2 hybrid baseline (DP=1)

FAST ITERATION MODE planned (5 warmup + 20 measured). CUDA Graph off.

## Outcome

- Status: **blocked (capacity)**
- Harness: **ready** on branch `cursor/phase101-tp2-pp2-hybrid-3b5c` at commit `f79e4f3`
- GPU: **NVIDIA L40S 48GB ×4** (user-approved alternate; A40 stockout)
- Blocker: RunPod has no available **4×L40S** instances (SECURE/COMMUNITY/global as of 13:10 UTC)

## Latest fix (`f79e4f3`)

User approved **L40S** as alternate 48GB GPU after A40 stockout:

- `scripts/phase10_gpu_profile.py` — A40 (149.7 TFLOPS) / L40S (181.0 TFLOPS) peaks + NVTE archs (86/89)
- Pod script auto-detects GPU, rebuilds TransformerEngine when CUDA arch changes
- RunPod template `zh7yn78wii` updated: `PHASE101_GPU_TYPE=NVIDIA L40S`, price arg `3.96` ($0.99/h×4 SECURE)

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

Template `zh7yn78wii` auto-deploys L40S branch with `PHASE101_GPU_TYPE=NVIDIA L40S`:

```bash
PHASE101_GPU_TYPE="NVIDIA L40S" bash scripts/phase10_tp_pp_pod.sh <pod_id> 3.96
```

Cloud Agent timer `phase101-l40s-retry` retries every 5 minutes until 4×L40S capacity appears.
