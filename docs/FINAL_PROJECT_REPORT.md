# Final Project Report

**Megatron-LM Distributed Training Performance Optimization Lab**

| Field | Value |
|-------|-------|
| Model | ~356M GPT (24L, H=1024, FFN=4096, 16 heads, seq=2048) |
| Framework | Megatron-Core `09fde85`, Transformer Engine 2.17.1 |
| Primary hardware | NVIDIA A40 48GB (RunPod Secure Cloud) |
| Precision | BF16 autocast, FP32 params + optimizer state |
| Status | Experimentally closed; portfolio packaging complete |

---

## 1. Project Objective

Build a **reproducible, measured** understanding of Megatron-LM training performance from single-GPU kernel fusion through **tensor**, **pipeline**, and **data** parallelism — including communication overlap and the **Distributed Optimizer** — using profilers and controlled A/B benchmarks.

Secondary goal: document **negative results** and infrastructure limits with the same rigor as wins.

---

## 2. Environment

| Component | Typical value |
|-----------|---------------|
| GPU | NVIDIA A40 48GB (SM 8.6) |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| NCCL | 2.27.3 |
| Driver | 570.x / 580.x (varies by pod) |
| Profiler | Nsight Systems 2025.1; Nsight Compute blocked on Secure Cloud |
| Cloud | RunPod EU-SE / CA-MTL, 1–2× A40 |

Each `results/*.json` records hostname, driver, commit hash, and timestamp.

---

## 3. Methodology

1. **Correctness gate** — forward/backward/grad checks vs reference or paired arms
2. **FAST iteration** — 5 warmup + 20 measured iterations for screening
3. **Formal validation** — 20 warmup + 100 measured for accepted optimizations
4. **Profiling** — Nsight Systems traces; overlap and exposed-communication metrics from CUDA+NCCL timelines
5. **Documentation** — Before/After/Delta in JSON + markdown experiment notes

**Labels used in this report:**

- **Measured** — direct benchmark output in `results/`
- **Derived** — computed from measured values (e.g. speedup ratio, weak-scaling %)
- **Analytical** — design/capacity reasoning without completed benchmarks (Phase 12 only)

---

## 4. Accepted Measured Results

### 4.1 Single-GPU stack (1× A40, DP=TP=PP=1)

| Milestone | tokens/s | MFU | step time | Source |
|-----------|----------|-----|-----------|--------|
| Phase 1 baseline (unfused, MB=1) | **3,711** | **6.01%** | 552 ms | `phase1_baseline.json` |
| Phase 3 fused attention A/B (fused arm) | 9,493 | 15.36% | 216 ms | `phase3_fused_attention_ab.json` |
| Phase 3 MB=8 (fused) | 15,085 | 24.42% | 1,086 ms | `phase3_microbatch_scaling.json` |
| Phase 5 formal B (BDA + fused + MB=8) | **15,802** | **25.58%** | 1,037 ms | `phase5_bias_dropout_fusion.json` |

*Derived:* cumulative **4.26×** throughput (3,711 → 15,802) spans multiple optimization stages, not one knob.

### 4.2 Fused attention controlled A/B (Phase 3.2)

| Metric | Unfused | Fused | Delta |
|--------|---------|-------|-------|
| tokens/s | 3,721 | 9,493 | **+155%** (2.55×) |
| Attention GPU time/step | 241.9 ms | 23.8 ms | **−90.2%** |
| Kernel launches/step | 4,769 | 4,361 | −408 |

### 4.3 Tensor Parallel (2× A40, TP=2)

| Config | tokens/s (aggregate) | TP scaling | Notes |
|--------|---------------------|------------|-------|
| TP=1 FAST | 15,695 | 1.00× | Same-host NODE topo |
| TP=2 FAST | 19,856 | **1.27×** (63% of 2×) | 259 ms NCCL/step, 0% overlap |
| TE-linear + SP + AG-only UB (C1 formal) | **29,045** | — | +6.75% vs B formal |

### 4.4 Pipeline Parallel (2× A40 NV4, PP=2)

| MB | tokens/s | idle % | VRAM peak/GPU |
|----|----------|--------|---------------|
| 1 | 16,387 | 2.6% | — |
| 4 | **21,183** | 5.8% | 15.5 GiB |
| 8 | 20,571 | 26.9% | — |

PP=1 reference: 16,198 tok/s, 41.8 GiB peak.

### 4.5 Data Parallel (2× A40, DP=2, global batch 16)

| Variant | tokens/s (formal) | exposed DP comm ms/step | weak scaling vs same-host DP1 |
|---------|-------------------|-------------------------|-------------------------------|
| A (overlap off) | 27,329 | 49.6 | 91.0% |
| B (overlap on) | **28,829** | **11.8** | **96.1%** |

Formal A→B: **+5.49%** throughput.

### 4.6 Distributed Optimizer (DP=2)

| Variant | tokens/s (formal) | opt state / rank | vs prior |
|---------|-------------------|------------------|----------|
| A (no DistOpt) | — | 2.85 GB | baseline |
| B (+ DistOpt) | 28,581 | **1.42 GB** | −0.39% FAST vs A |
| C (+ overlap param gather) | **29,631** | 1.42 GB | **+3.67%** formal vs B |

---

## 5. Performance Evolution

```text
Phase 1  Unfused baseline          3,711 tok/s   6.0% MFU
    ↓ profiling (Nsys)
Phase 3  Fused attention           9,493 tok/s  15.4% MFU   [controlled 2.55×]
    ↓ microbatch → MB=8
Phase 3  MB=8                      15,085 tok/s 24.4% MFU
    ↓ BDA fusion
Phase 5  Formal single-GPU pin     15,802 tok/s 25.6% MFU  [accepted stack]
    ↓ parallelism tracks (separate hardware configs)
Phase 7  TP=2 + comm overlap       up to 29,045 tok/s (2×GPU aggregate)
Phase 8  PP=2 best                 21,183 tok/s (2×GPU aggregate)
Phase 9  DP=2 + overlap + DistOpt  29,631 tok/s (2×GPU global batch 16)
```

Parallelism results are **not additive** — they use different world sizes and batch semantics.

---

## 6. Tensor Parallel Analysis

**Hypothesis:** Splitting attention/MLP across 2 GPUs reduces per-GPU compute but introduces All-Reduce latency.

**Profiler evidence (Phase 7.1):** 101 All-Reduces per step; 259 ms NCCL GPU time per GPU per step; 0% overlap with compute at baseline.

**Measured:** 1.27× aggregate throughput (63% of ideal).

**Sequence Parallel:** Reduced activation memory (~2 GiB) but increased collective time (514→668 ms/step); **−8.5%** throughput — accepted only as enabler for TE-linear path.

**Communication overlap (Phase 7.4b):**

- **C1 AG-only Userbuffers:** exposed comm 229→181 ms; formal **+6.75%**
- **C2 bulk dgrad:** 91.5% AG–GEMM overlap but **slower** than C1
- **RS Userbuffers (Phase 7.4):** livelock — never benchmarked to completion

**Conclusion:** On A40 PCIe, partial AG overlap is the safe win; chasing maximum overlap via RS/dgrad paths hurt or hung.

---

## 7. Pipeline Parallel Analysis

**Hypothesis:** PP=2 fits larger models per GPU by splitting layers; throughput depends on microbatch count vs bubble.

**Profiler evidence:** MB sweep shows idle fraction rising sharply at MB=8 (26.9%).

**Measured best:** PP=2, MB=4 — **21,183 tok/s**, 5.8% idle, 15.5 GiB VRAM/GPU.

**VPP + interleaved 1F1B (Phase 8.3):** Theoretical bubble 20%→11%, P2P overlap 38%, but only **+2.34%** throughput — rejected (<3% gate).

**Conclusion:** PP helps memory capacity; MB must balance bubble vs utilization; VPP overhead dominated on 2-stage PP.

---

## 8. Data Parallel Analysis

**Mechanism:** `DistributedDataParallel` with 8×40 MB gradient buckets; `overlap_grad_reduce` issues NCCL during backward while later layers still compute.

**Exposed communication** (time NCCL waits while GPU could compute): **49.6 → 11.8 ms/step**.

**Weak scaling** (vs same-host DP=1 at 15,052 tok/s): **91.0% → 96.1%**.

**Note:** Total NCCL kernel time can **increase** after bucketing (more, smaller collectives) while exposed time **decreases** — bucketing enables overlap, not fewer bytes moved.

---

## 9. Distributed Optimizer Analysis

**Mechanism:** Optimizer state sharded via Reduce-Scatter on gradients + All-Gather on updated parameters (`use_distributed_optimizer`).

**Memory (measured):** 5.69 GB total optimizer state → 2.85 GB total (−50%); **2.85 GB → 1.42 GB per rank**.

**Throughput trade-off:**

- DistOpt alone: slight regression (−0.39% FAST) — extra RS/AG without overlap
- `overlap_param_gather`: exposed param comm 28.9 → 5.6 ms; formal **+3.67%**

**Conclusion:** DistOpt is a memory optimization that needs param-gather overlap to recover and exceed baseline throughput.

---

## 10. Capacity Engineering Study (Phase 12) — Analytical Only

**Status:** `terminated` — infrastructure could not provide reliable 2×A40 NCCL topology.

**Design (not measured):** Variants A/B/C/D crossing Distributed Optimizer × activation checkpointing (`recompute_granularity=full`, `uniform`, 1 layer); capacity search MB≤64.

**Harness delivered:** `phase12_memory_pod.sh`, `phase12_memory_run.py`, `phase12_capacity_search.py`, `phase12_analyze_memory.py`.

**Partial smoke on host `64411267` only:** topology OK; A/B smoke passed before infra blockers. **No fixed-workload or capacity-search JSON exists.**

> **Do not cite Phase 12 throughput, MFU, or memory numbers as measured.** Any capacity predictions in design docs are *Analytical*, not benchmarked.

---

## 11. Negative / Failed Experiments

| Experiment | Outcome | Lesson |
|------------|---------|--------|
| NCU on RunPod Secure | `ERR_NVGPUCTRPERM` | Use Nsys + kernel naming; NCU needs bare-metal or admin caps |
| CUDA Graph (Phase 6) | Gradient mismatch after replay | MCore `main_grad` lifecycle incompatible with naive graph capture |
| BF16 residual direction (Phase 4) | Copy traffic unchanged | Dtype alone doesn't fix layout/copy patterns |
| Userbuffers RS overlap | Livelock | A40 PCIe topology sensitive; partial overlap only |
| SP for throughput | −8.5% | Memory/throughput trade-off; use when needed for TE-linear |
| VPP interleaved PP | +2.3% | Extra P2P hops ate bubble savings |
| Phase 12 benchmarks | 10/10 bad NCCL hosts on stock | Cloud GPU placement is part of experimental design |

---

## 12. Limitations

1. **Hardware:** A40 only; no H100/NVLink cluster; topology varies by Runpod host suffix
2. **Scale:** Max 2 GPUs; no 8/64/1000 GPU measurements
3. **NCU:** Hardware counters unavailable on Secure Cloud
4. **Cross-phase comparison:** Different hosts, FAST vs formal, aggregate vs per-GPU MFU
5. **Phase 12:** No activation-checkpointing benchmark data
6. **Model size:** ~356M — not representative of multi-billion training at scale

---

## 13. Final Engineering Conclusions

1. **Profile before optimizing** — unfused attention was ~50% of GPU time; fusion yielded 2.55× in a controlled experiment.
2. **MFU rises with microbatch** until memory-bound — optimizer share falls as GEMM dominates.
3. **Parallelism adds communication** — TP/PP/DP wins require overlap or acceptance of sub-linear scaling.
4. **Exposed communication > total NCCL time** for overlap engineering decisions.
5. **Memory optimizations (DistOpt, SP, PP) need overlap companions** to avoid throughput regression.
6. **Maximum overlap % is not the objective** — 91% AG–GEMM overlap lost to C1's simpler AG-only path.
7. **Cloud GPU infra is a variable** — NCCL topology failures terminated Phase 12; document host classes.

---

## Appendix: Result Index

| Phase | JSON | Experiment doc |
|-------|------|----------------|
| 1 | `phase1_baseline.json` | `phase1_baseline.md` |
| 2 | `phase2_nsys_summary.json`, `phase2_ncu_summary.json` | `phase2_nsys_baseline.md` |
| 3 | `phase3_fused_attention_ab.json`, `phase3_microbatch_scaling.json` | `phase3_*.md` |
| 5 | `phase5_bias_dropout_fusion.json` | `phase5_bias_dropout_fusion.md` |
| 6 | `phase6_cuda_graph.json`, `phase6_ddp_cuda_graph.json` | `phase6_*.md` |
| 7 | `phase7_*.json` | `phase7_*.md` |
| 8 | `phase8_*.json` | `phase8_*.md` |
| 9 | `phase9_*.json` | `phase9_*.md` |
| 12 | `phase12_memory_capacity.json` | `phase12_memory_capacity.md` |
