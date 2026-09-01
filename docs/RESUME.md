# Resume Material

All metrics below are **measured** in this repository unless noted. Phase 12 activation checkpointing is **not** included — that study was analytical/infrastructure-blocked only.

---

## 中文简历

**项目名称：** Megatron-LM 大模型分布式训练性能优化与通信分析

**技术栈：** PyTorch / Megatron-LM / CUDA / Transformer Engine / NCCL / Nsight Systems / RunPod

**项目描述（四条）：**

- 在 NVIDIA A40 上搭建可复现的 Megatron-Core 训练性能实验平台，对 356M GPT 基线进行 Nsight 剖析，通过 **Transformer Engine 融合注意力** 在控制变量 A/B 实验中实现 **2.55×** 吞吐提升、注意力 GPU 耗时降低 **90%**，并将单卡吞吐从 **~3,711 优化至 ~15,802 tokens/s（约 4.26×）**、MFU 从 **~6% 提升至 ~26%**（融合注意力 + microbatch 扩展 + BDA 融合）。

- 系统研究 **Tensor Parallel（TP=2）** 通信行为：基线 **259 ms/step** All-Reduce、**0%** 计算通信重叠；实现 **AG-only Userbuffers 重叠**，正式验证吞吐 **+6.75%**；分析 **91.5% AG-GEMM 重叠** 反而慢于低重叠方案的工程原因，并记录 RS 路径 **livelock** 负结果。

- 完成 **Pipeline Parallel（PP=2）** microbatch 扫描，在 **NVLink 2×A40** 上确定最优 **M=4（21,183 tokens/s）**，显存峰值从 **41.8 GiB 降至 15.5 GiB**；评估 **VPP + P2P 重叠** 仅 **+2.3%** 而拒绝采纳。

- 在 **DP=2** 上启用 **gradient bucket All-Reduce 与 overlap_grad_reduce**，将 **暴露通信从 ~50 ms 降至 ~12 ms/step**，弱扩展效率从 **~91% 提升至 ~96%**；部署 **Megatron Distributed Optimizer** 使优化器状态 **~2.85 GB → ~1.42 GB/卡（−50%）**，配合 **overlap_param_gather** 正式验证吞吐 **+3.67%**。

---

## English Resume

**Project:** Megatron-LM Distributed Training Performance Optimization & Communication Analysis

**Stack:** PyTorch · Megatron-LM · CUDA · Transformer Engine · NCCL · Nsight Systems · RunPod

**Bullets:**

- Built a reproducible Megatron-Core performance lab on NVIDIA A40; profiled a 356M GPT baseline and delivered a **2.55×** controlled fused-attention speedup (**−90%** attention GPU time), improving single-GPU throughput **~3,711 → ~15,802 tokens/s (~4.26×)** and MFU **~6% → ~26%** via kernel fusion, microbatch scaling, and bias-dropout-add fusion.

- Characterized **Tensor Parallel (TP=2)** NCCL behavior (**259 ms/step** All-Reduce, **0%** overlap at baseline); shipped **AG-only Userbuffers overlap** with **+6.75%** formal throughput gain; documented why **91.5% AG–GEMM overlap** underperformed a simpler config and why RS overlap **livelocked** on A40 PCIe.

- Ran **Pipeline Parallel (PP=2)** microbatch sweep on **NVLink 2×A40**, selecting **M=4 (21,183 tokens/s)** with **63%** VRAM reduction; rejected **VPP + P2P overlap** after only **+2.3%** measured gain.

- On **DP=2**, reduced **exposed gradient communication ~50 → ~12 ms/step** via bucketed **`overlap_grad_reduce`**, improving weak-scaling efficiency **~91% → ~96%**; deployed **Megatron Distributed Optimizer** cutting optimizer state **~2.85 → ~1.42 GB/GPU (−50%)**, with **`overlap_param_gather`** yielding **+3.67%** formal throughput.

---

## Optional One-Liner (LinkedIn / GitHub)

> Megatron-LM performance lab: **4.26×** single-GPU throughput, **2.55×** fused attention, TP/PP/DP communication overlap & Distributed Optimizer — all measured on A40 with Nsight Systems.

---

## What NOT to Claim

- Phase 12 activation-checkpointing throughput or memory numbers (harness only; infra terminated)
- NCU hardware-counter analysis (blocked on RunPod Secure Cloud)
- Multi-node or >2 GPU scaling numbers
- CUDA Graph speedups (correctness blocked before benchmark)
