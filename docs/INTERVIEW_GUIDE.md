# Interview Guide

Grounded in measured results from this repository. When citing numbers, prefer the JSON sources listed in [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md).

---

## A. 30-Second Introduction

> I built a Megatron-LM performance lab on RunPod A40s: profiled a 356M GPT baseline, fused attention for a **2.55×** controlled speedup, scaled microbatch to raise MFU from **6% to 26%**, then studied **tensor, pipeline, and data parallelism** with Nsight Systems. On DP=2 I cut **exposed gradient communication from 50 ms to 12 ms** with overlap, and combined the **Distributed Optimizer** with param-gather overlap for **+3.67%** formal throughput while halving optimizer state per GPU. I document negative results too — CUDA Graph correctness failures, NCU blocked on cloud, and TP overlap paths that livelocked.

---

## B. 3-Minute Explanation

**Problem:** Training throughput is limited by kernels, memory bandwidth, and distributed communication — you can't optimize what you don't measure.

**Approach:** Megatron-Core harness with reproducible synthetic data, correctness gates, FAST then formal benchmarks, and Nsight Systems for every major change.

**Single-GPU arc:** Baseline unfused attention at **3,711 tokens/s** and **6% MFU**. Profiling showed attention dominated GPU time. A controlled A/B switching only the attention backend to Transformer Engine cuDNN FusedAttention yielded **2.55×** throughput and **90% less attention GPU time**. Microbatch scaling to 8 improved MFU to **24%**; BDA fusion added **4%** for a pinned stack of **15,802 tokens/s**.

**Parallelism:** On 2×A40, TP=2 gave **1.27×** aggregate throughput but added **259 ms/step** of All-Reduce with zero overlap. AG-only Userbuffers overlap recovered **6.75%** formally — but a higher-overlap dgrad path was slower. PP=2 with 4 microbatches best balanced bubble vs throughput at **21k tokens/s** while cutting VRAM **63%**. DP=2 with `overlap_grad_reduce` reduced **exposed** grad comm from **50 to 12 ms** and improved weak scaling from **91% to 96%**. Distributed Optimizer halved optimizer state (**2.85→1.42 GB/rank**) and needed `overlap_param_gather` to beat baseline by **3.67%**.

**What I'd highlight as an infra engineer:** I treat exposed communication and overlap percentage as first-class metrics, not just total NCCL time. I also stopped Phase 12 when cloud NCCL topology made experiments invalid — correctness of the measurement environment matters.

---

## C. Deep-Dive Q&A

### Why was GPU Util ≈100% while MFU was only 6%?

**Measured:** Phase 1 averaged **99.6% GPU utilization** but **6.01% MFU** (`phase1_baseline.json`).

**Answer:** GPU utilization counts whether *any* kernel is running on the SM, not whether those kernels use Tensor Cores efficiently. The baseline used unfused attention (memory-bound softmax and elementwise ops), BF16 autocast without TE fusion, MB=1 (tiny GEMMs), and many small kernels (66% under 50 µs in Phase 3 profile). MFU divides achieved FLOPs by **peak dense BF16 Tensor Core throughput (149.7 TFLOP/s on A40)** — a much stricter metric. High util + low MFU means the GPU is busy but not doing efficient dense math.

---

### Why did fused attention help so much?

**Measured:** 2.55× throughput; attention GPU time 242→24 ms/step (`phase3_fused_attention_ab.json`).

**Answer:** Unfused path spent ~50% of GPU time in attention kernels (softmax backward alone was top-5 by time). TE cuDNN FusedAttention fuses softmax, dropout, and GEMM tiles into fewer, larger kernels with better memory access. Memory copies dropped from 57 ms/step to ~0.004 ms/step. This was a **controlled A/B** — same model, batch, precision; only attention implementation changed.

---

### Why does kernel-count reduction not equal performance gain?

**Measured:** Fused arm removed 408 launches/step (−8.6%) but gained 155% throughput.

**Answer:** Launch overhead matters, but the dominant win was **per-kernel work fusion** — attention time fell 90%, not 8%. Fewer launches help tail latency; kernel fusion changes arithmetic intensity and memory traffic. Don't use launch count as a proxy for speedup.

---

### Why does increasing microbatch improve MFU?

**Measured:** MB 1→8: 15.8%→24.4% MFU; optimizer share 26%→4.9% of GPU time (`phase3_microbatch_scaling.json`).

**Answer:** Fixed overhead (optimizer, layer norm, launches) is amortized over more tokens per step. GEMM dimensions grow, improving Tensor Core utilization. Step time grows sub-linearly with batch (209→1086 ms for 8× tokens), so tokens/sec rises. Memory caps the maximum MB (MB=4 OOM'd in Phase 1; MB=8 predicted safe in Phase 3).

---

### Why did TP=2 not achieve 2× throughput?

**Measured:** 15,695→19,856 tok/s aggregate (1.27×), 63% of ideal (`phase7_tp2_baseline.json`).

**Answer:** TP adds **All-Reduce communication** every layer (101 ARs/step, 259 ms NCCL/step/GPU) with **0% overlap** at baseline. You trade per-GPU compute for sync latency. Ideal 2× assumes compute halves and comm is free — neither holds. MFU per GPU also drops because the same global batch is split across GPUs.

---

### AR vs AG vs RS?

**Answer (project context):**

- **All-Reduce (AR):** Each rank ends with the full reduced tensor. Classic DDP gradient sync.
- **All-Gather (AG):** Each rank has a shard; gather builds full tensor. Used in TP column-parallel forward and DistOpt param gather.
- **Reduce-Scatter (RS):** Reduce across ranks, each rank keeps one shard. DistOpt uses RS on gradients so each rank updates only its optimizer-state shard.

**Measured:** Phase 9.2 DistOpt replaces dense grad AR with RS + param AG; optimizer state halved.

---

### What is exposed communication?

**Answer:** Time per training step where NCCL kernels run on the GPU timeline **without** overlapping compute kernels — the communication visible to the critical path. Phase 9.1: **49.6→11.8 ms/step** exposed DP comm when enabling `overlap_grad_reduce`.

Distinct from **total NCCL GPU time**, which can rise when bucketing splits one large AR into many overlapped smaller ones.

---

### Async communication vs real overlap?

**Answer:** NCCL is asynchronous at the API level, but overlap only matters if **compute kernels run concurrently on the same GPU** during communication. Nsight shows this as overlapping CUDA ranges. Phase 7.1 had async NCCL but **0% overlap** — comm and compute serialized. Phase 9.1 reached **88% overlap** with bucketing + `overlap_grad_reduce`.

---

### Why can 91% overlap be slower than a lower-overlap configuration?

**Measured:** Phase 7.4b C2 had **91.5%** AG–GEMM overlap but **lower throughput** than C1 (11.6% overlap, +6.75% formal winner).

**Answer:** Overlap % measures timeline concurrency, not critical-path length. C2's bulk dgrad path added synchronization points, extra buffer management, and less favorable kernel ordering. Higher overlap can hide work that still extends the dependency chain. **Throughput is the acceptance metric**, not overlap percentage.

---

### Why does pipeline parallelism have bubbles?

**Answer:** With PP>1, each GPU owns a subset of layers. Early ranks finish their forward wave before late ranks consume activations — GPUs idle waiting for P2P transfers. Bubble fraction grows with pipeline depth and shrinks with more microbatches (1F1B schedules).

**Measured:** PP=2 MB=8 had **26.9% idle** vs **5.8%** at MB=4 (`phase8_pp2_baseline.json`).

---

### Why did VPP add communication hops?

**Answer:** Virtual pipeline parallelism splits each physical stage into multiple model chunks, interleaving forward/backward waves. More chunks → more P2P transfers between ranks per step, even if theoretical bubble drops.

**Measured:** Phase 8.3 — bubble 20%→11% theoretical, 38% P2P overlap achieved, but only **+2.34%** throughput. Extra hops + scheduling overhead dominated.

---

### How does `overlap_grad_reduce` work?

**Answer:** MCore DDP divides gradients into buckets (~40 MB). When backward finishes a bucket's layers, NCCL reduce starts **while earlier buckets' layers may still be computing** on other streams. Requires `CUDA_DEVICE_MAX_CONNECTIONS>1` and bucket boundaries aligned to backward order.

**Measured:** Exposed comm −37.8 ms/step; overlap 0%→88%.

---

### Why did total NCCL kernel time increase after bucketing?

**Answer:** One monolithic All-Reduce becomes multiple smaller collectives — more launch overhead and synchronization, but each can start earlier in backward. **Total NCCL time is not the optimization target**; exposed comm on the critical path is.

---

### Why is exposed communication more useful than total NCCL time?

**Answer:** Total NCCL counts all collective GPU time including portions hidden under compute. Exposed comm approximates what actually extends step time. Phase 9.1 improved throughput while total NCCL could rise — because more comm ran under backward compute.

---

### How does Distributed Optimizer differ from normal DDP?

**Answer:** Standard DDP: All-Reduce full gradients; every rank holds **full optimizer state** (2× m,v for Adam). DistOpt: Reduce-Scatter gradients so each rank updates **1/DP shard** of optimizer state; All-Gather updated params before next forward.

**Measured:** Optimizer state **2.85→1.42 GB/rank** (−50%).

---

### Why RS + AG instead of AR for DistOpt?

**Answer:** RS produces the per-rank gradient shard needed for sharded optimizer updates in one step. AG broadcasts updated parameter shards back to all ranks for the next forward. AR would leave every rank with full gradients — defeating state sharding.

---

### Why is `overlap_param_gather` important?

**Measured:** DistOpt alone **−0.39%** throughput; with overlap **+3.67%** formal.

**Answer:** Without overlap, parameter All-Gather is on the critical path before forward. `overlap_param_gather` pipelines param gather with optimizer step and/or previous compute — exposed param comm **28.9→5.6 ms**.

---

### Why does PyTorch allocated memory drop while nvidia-smi memory may rise?

**Answer:** `torch.cuda.memory_allocated` tracks active tensors; `reserved` includes caching allocator pools. `nvidia-smi` includes driver allocations, NCCL buffers, cudnn workspaces, and fragmentation. Phase 3 fused run showed lower allocated memory but nvidia-smi can still report high usage from pools and libraries.

---

### Throughput-optimal vs capacity-optimal configuration?

**Answer:**

- **Throughput-optimal (measured):** Single-GPU MB=8 + fused + BDA (**15,802 tok/s**). DP=2 global batch 16 + overlap + DistOpt C (**29,631 tok/s formal**).
- **Capacity-optimal:** PP=2 MB=4 — **21,183 tok/s** but **15.5 GiB** peak vs **41.8 GiB** at PP=1. SP trades −8.5% throughput for ~2 GiB activations.

Phase 12 *designed* capacity search (MB≤64, DistOpt × recompute) but **no measurements** were collected.

---

### What would you change at 8 / 64 / 1000 GPUs?

**Analytical (not measured in this project):**

- **8 GPUs:** Combine TP×DP explicitly; tune bucket sizes for NCCL NVLink topology; enable hierarchical AR; validate DistOpt + overlap at scale.
- **64 GPUs:** Add PP for memory; interleaved schedules worth revisiting on NVLink fat-tree; profile stragglers and NIC contention.
- **1000 GPUs:** Full 3D parallelism; overlap all three comm types; checkpointing mandatory; invest in custom NCCL groups and deterministic host placement — lessons from Phase 12 infra failures apply at scale.

---

## Quick Reference: Numbers to Remember

| Topic | Number | Label |
|-------|--------|-------|
| Baseline tok/s | 3,711 | Measured |
| Pinned single-GPU tok/s | 15,802 | Measured formal |
| Fused attention speedup | 2.55× | Measured A/B |
| DP exposed comm | 49.6→11.8 ms | Measured |
| Weak scaling | 91.0%→96.1% | Derived |
| DistOpt state/rank | 2.85→1.42 GB | Measured |
| DistOpt + overlap | +3.67% formal | Measured |
| Phase 12 checkpointing | — | **Not measured** |
