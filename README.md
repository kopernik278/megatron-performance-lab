# Megatron-LM Distributed Training Performance Optimization Lab

A reproducible performance-engineering portfolio built on **Megatron-Core** (~356M GPT, BF16, 2× NVIDIA A40). Every headline number below is **measured** in this repository unless labeled *Derived* or *Analytical*.

**Stack:** PyTorch · Megatron-LM · Transformer Engine · NCCL · Nsight Systems · RunPod (2× A40)

---

## Headline Results (Measured)

| Result | Before → After | Context |
|--------|----------------|---------|
| **Single-GPU throughput** | **~3,711 → ~15,802 tokens/s** (~4.26×) | Cumulative arc: Phase 1 unfused MB=1 → Phase 5 formal BDA+fused+MB=8 on **1× A40** ([`results/phase1_baseline.json`](results/phase1_baseline.json), [`results/phase5_bias_dropout_fusion.json`](results/phase5_bias_dropout_fusion.json)) |
| **MFU** | **~6.01% → ~25.58%** | Same single-GPU arc (A40 dense BF16 peak 149.7 TFLOP/s) |
| **Fused attention (controlled A/B)** | **2.55×** throughput; attention GPU time **−90.2%** | Phase 3.2, same host, only attention backend changed ([`results/phase3_fused_attention_ab.json`](results/phase3_fused_attention_ab.json)) |
| **DP=2 gradient overlap** | Exposed comm **~49.6 → ~11.8 ms/step**; weak scaling **~91.0% → ~96.1%** | Phase 9.1 formal, same-host DP=1 ref ([`results/phase9_dp2_grad_overlap.json`](results/phase9_dp2_grad_overlap.json)) |
| **Distributed Optimizer** | Optimizer state/rank **~2.85 GB → ~1.42 GB** (−50%) | Phase 9.2 variant B vs A ([`results/phase9_distributed_optimizer.json`](results/phase9_distributed_optimizer.json)) |
| **Parameter-gather overlap** | Formal B→C throughput **+3.67%** | Phase 9.2, DistOpt + `overlap_param_gather` ([`results/phase9_distributed_optimizer.json`](results/phase9_distributed_optimizer.json)) |

> **Do not** treat numbers from different hosts or parallelism modes as a single controlled experiment. See [docs/FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md) for protocol labels (FAST vs Formal).

---

## What This Project Covers

| Area | Phases | Status |
|------|--------|--------|
| Baseline + profiling | 1–2 | Measured |
| Kernel fusion (attention, BDA) | 3, 5 | Measured, accepted |
| Microbatch / MFU scaling | 3 | Measured |
| CUDA Graph | 6 | Blocked (correctness) |
| Tensor Parallel + comm overlap | 7 | Measured; AG-only overlap accepted |
| Pipeline Parallel + VPP | 8 | Measured; PP2 MB=4 best; VPP rejected |
| Data Parallel + DistOpt | 9 | Measured, accepted |
| Memory / activation checkpointing | 12 | **Analytical study only** — harness built, **no benchmark data** ([`docs/experiments/phase12_memory_capacity.md`](docs/experiments/phase12_memory_capacity.md)) |

---

## Performance Journey

Accepted optimizations only (failed paths → [Negative Results](#engineering-findings--negative-results)):

| Stage | Hypothesis | Profiler evidence | Change | Measured effect | Conclusion |
|-------|------------|-------------------|--------|-----------------|------------|
| **Baseline** | Establish reproducible harness | Nsight: attention ~50% GPU time | Unfused local attention, MB=1 | 3,711 tok/s, 6.01% MFU | Bottleneck = attention + copies |
| **Profiling** | Find dominant kernels | Nsys timeline + kernel stats | Target softmax/GEMM/copy | — | Justified fused-attention A/B |
| **Fused attention** | cuDNN fused SDPA cuts attention time | Attention 242→24 ms/step | TE `FusedAttention` | **2.55×** tok/s (controlled A/B) | Accepted |
| **Microbatch scaling** | Larger MB improves MFU | GEMM share rises with MB | MB 1→8 | 9,763→15,085 tok/s | MB=8 pinned |
| **BDA fusion** | Fuse bias-dropout-add residual | −96 launches/step, −35 ms BDA | `bias_dropout_fusion=true` | Formal **+4.27%** tok/s → **15,802** | Accepted |
| **Tensor Parallel** | TP=2 splits weights, adds AR | 259 ms NCCL/step, 0% overlap | TP=2 | 1.27× aggregate tok/s (FAST) | Baseline for comm work |
| **TP comm overlap** | Overlap AG with GEMM | Exposed comm −48 ms | Userbuffers AG-only (C1) | Formal **+6.75%** vs TE-linear SP | Accepted; RS path livelocked |
| **Pipeline Parallel** | PP splits layers, trades bubble for VRAM | Bubble vs MB sweep | PP=2, MB=4 | **21,183** tok/s (FAST); VRAM −63% | Best PP operating point |
| **DP grad overlap** | Bucketed AR during backward | Exposed comm 49.6→11.8 ms | `overlap_grad_reduce` | Formal **+5.49%**; weak scale 91→96% | Accepted |
| **Distributed Optimizer** | Shard optimizer state | 2.85→1.42 GB state/rank | DistOpt + param-gather overlap | **+3.67%** formal vs DistOpt alone | Accepted |

---

## Distributed Training (Measured Summaries)

### Tensor Parallel (TP=2, 2× A40)

- **Scaling:** FAST TP=1 15,695 → TP=2 19,856 tok/s (**1.27×**, 63% of ideal 2×) — [`results/phase7_tp2_baseline.json`](results/phase7_tp2_baseline.json)
- **NCCL:** 101 All-Reduces/step; **259 ms** NCCL GPU time/step/GPU; **0%** compute–comm overlap at baseline
- **Sequence Parallel:** −8.5% throughput but ~2 GiB activation memory saved — memory enabler, not throughput win — [`results/phase7_sequence_parallel.json`](results/phase7_sequence_parallel.json)
- **Comm overlap:** AG-only Userbuffers **+6.75%** formal; bulk dgrad path showed **91.5%** AG–GEMM overlap but **lower** net throughput than AG-only — [`results/phase7_tp_partial_comm_overlap.json`](results/phase7_tp_partial_comm_overlap.json)
- **Lesson:** Max overlap ≠ max throughput (extra RS/dgrad paths add latency and risk livelock on A40 PCIe)

### Pipeline Parallel (PP=2, 2× A40 NVLink)

- **Microbatch sweep:** Best **MB=4** at **21,183 tok/s** (FAST); MB=8 regresses −2.9% — [`results/phase8_pp2_baseline.json`](results/phase8_pp2_baseline.json)
- **Bubble:** Theoretical 20% (MB=4) vs measured idle ~5.8% — scheduler + overlap differ from textbook
- **VRAM:** PP=1 peak 41.8 GiB → PP=2 best 15.5 GiB per GPU
- **VPP / P2P overlap:** +2.34% only (below 3% gate); 38% P2P overlap but extra hops — rejected — [`results/phase8_interleaved_pp_overlap.json`](results/phase8_interleaved_pp_overlap.json)

### Data Parallel (DP=2, 2× A40)

- **Gradient buckets:** 8 buckets × 40 MB; `overlap_grad_reduce` overlaps backward with NCCL
- **Exposed communication:** **49.6 → 11.8 ms/step** (formal traces); overlap **0% → 88%**
- **Weak scaling:** **91.0% → 96.1%** vs same-host DP=1 ref (15,052 tok/s) — [`results/phase9_dp2_grad_overlap.json`](results/phase9_dp2_grad_overlap.json)

### Distributed Optimizer (DP=2)

- **Mechanism:** Gradient **AR** replaced by **RS** + parameter **AG**; optimizer state sharded per rank
- **Memory:** **2,847,360,144 → 1,423,679,488 B/rank** (−50%)
- **Throughput:** DistOpt alone **−0.39%**; with `overlap_param_gather` formal **+3.67%** (29,631 tok/s) — [`results/phase9_distributed_optimizer.json`](results/phase9_distributed_optimizer.json)

---

## Engineering Findings / Negative Results

| Finding | Evidence |
|---------|----------|
| **NCU hardware counters blocked** on RunPod Secure A40 (`ERR_NVGPUCTRPERM`) | [`results/phase2_ncu_summary.json`](results/phase2_ncu_summary.json) |
| **BF16 residual-direction** did not reduce copy traffic | Phase 4 design — not in headline table |
| **CUDA Graph** correctness mismatch (`main_grad` lifecycle) | [`results/phase6_cuda_graph.json`](results/phase6_cuda_graph.json), [`results/phase6_ddp_cuda_graph.json`](results/phase6_ddp_cuda_graph.json) |
| **TP Userbuffers RS path livelock** on A40 PCIe | [`results/phase7_tp_userbuffers_overlap.json`](results/phase7_tp_userbuffers_overlap.json) |
| **91.5% AG–GEMM overlap slower than 11.6% AG-only config** | [`results/phase7_tp_partial_comm_overlap.json`](results/phase7_tp_partial_comm_overlap.json) |
| **VPP reduced bubble but added P2P hops** — net +2.3% only | [`results/phase8_interleaved_pp_overlap.json`](results/phase8_interleaved_pp_overlap.json) |
| **Phase 12 activation checkpointing** — harness only; **no measured A/B/C/D** (infra terminated) | [`results/phase12_memory_capacity.json`](results/phase12_memory_capacity.json) |

---

## Repository Map

| Document | Description |
|----------|-------------|
| [docs/FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md) | Full technical report |
| [docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md) | Interview Q&A grounded in project data |
| [docs/RESUME.md](docs/RESUME.md) | Chinese + English resume bullets |
| [docs/experiments/](docs/experiments/) | Per-phase experiment writeups |
| [results/](results/) | Machine-readable benchmark JSON |
| [profiles/](profiles/) | Nsight Systems / Compute traces (not committed in full) |
| [AI_INFRA_CONTEXT.md](AI_INFRA_CONTEXT.md) | Learning goals and measurement standards |

### Key result files

- [`results/phase1_baseline.json`](results/phase1_baseline.json) — unfused baseline
- [`results/phase3_fused_attention_ab.json`](results/phase3_fused_attention_ab.json) — fused attention A/B
- [`results/phase3_microbatch_scaling.json`](results/phase3_microbatch_scaling.json) — MB sweep
- [`results/phase5_bias_dropout_fusion.json`](results/phase5_bias_dropout_fusion.json) — BDA fusion
- [`results/phase7_tp2_baseline.json`](results/phase7_tp2_baseline.json) — TP=2
- [`results/phase7_tp_partial_comm_overlap.json`](results/phase7_tp_partial_comm_overlap.json) — TP comm overlap
- [`results/phase8_pp2_baseline.json`](results/phase8_pp2_baseline.json) — PP=2 sweep
- [`results/phase9_dp2_grad_overlap.json`](results/phase9_dp2_grad_overlap.json) — DP overlap
- [`results/phase9_distributed_optimizer.json`](results/phase9_distributed_optimizer.json) — DistOpt
- [`results/phase12_memory_capacity.json`](results/phase12_memory_capacity.json) — *Analytical / terminated*

---

## Reproducing (no GPU required to browse)

```bash
# Inspect a result
python -m json.tool results/phase3_fused_attention_ab.json

# Run harness (requires 1–2× A40 + Megatron/TE setup)
bash scripts/phase12_memory_pod.sh <pod_id> 0.88   # Phase 12 harness (infra-dependent)
```

See [AGENTS.md](AGENTS.md) for repository conventions.

---

## Methodology

- **Correctness before optimization** — numerical checks before benchmarks
- **FAST screen** (5 warmup + 20 measured) then **Formal** (20 + 100) for accepted changes
- **Nsight Systems** for timeline, exposed communication, overlap %
- **Never fabricate** — hardware, driver, commit hash, and timestamps recorded in each JSON
