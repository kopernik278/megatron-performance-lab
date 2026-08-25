# Phase 8.3: interleaved 1F1B + VPP + P2P overlap

FAST ITERATION MODE (5 warmup + 20 measured) unless a formal 20+100 cell is noted.
CUDA Graph stayed off. Pipeline dtype stayed FP32. TP was not combined with PP.

This A/B is the **combined** effect of interleaved 1F1B, virtual pipeline
stages (VPP=2), and `overlap_p2p_comm`. On PP=2 those pieces cannot be isolated.
Do not attribute the full throughput change to communication overlap alone.

## Outcome

- Status: **success**
- Formal 20+100: **no**
- A→B throughput: **+2.34%** (19652.75 → 20112.63 tok/s)
- P2P overlap (B, kernel vs compute): **38.2%**
- Exposed P2P: 78.32 → 65.98 ms/step/GPU
- Dominant remaining bottleneck: exposed pipeline activation P2P (extra interleaved hops and/or warmup/cooldown, because overlap_p2p_comm_warmup_flush stayed off)

## Infrastructure

- RunPod Pod: `o2xtfls95kpxey` (deleted)
- Data center / public IP: CA-MTL-1, `69.30.85.9`
- Allocation: one Secure Cloud Pod, 2x NVIDIA A40, $0.88/h
- Topology path: **PIX**, same-NUMA=True
- CUDA peer access bidirectional: True
- NCCL All-Reduce sanity: True
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Lab commit on the Pod: `d582ebf744655de2ca80ca91d6a1b7bb2b226b5d`
- PyTorch: 2.8.0+cu128; CUDA: 12.8; NCCL: 2.27.3; Transformer Engine: 2.17.1+4329ff84
- Driver: 580.159.04
- PCI: `00000000:52:00.0` + `00000000:53:00.0`
- Precision: compute BF16 autocast; `pipeline_dtype=torch.float32` (BF16 P2P was NaN in Phase 8.1)
- `CUDA_DEVICE_MAX_CONNECTIONS=8`. `NCCL_P2P_DISABLE` unset. No TP+PP mix.

Phase 8.1 reference on NV4 host `69.30.85.97` was 21,183 tok/s / 773 ms. This 8.3 A/B ran on a different PIX host, so do not treat 8.3 A as matching that absolute number.

## Correctness (variant B)

- Forward/backward/`main_grad`/optimizer: True/True/True/True
- Finite loss / no NaN-Inf / no deadlock: True/True/True
- Interleaved schedule: `forward_backward_pipelining_with_interleaving`
- Async P2P issued: True

## Layer / chunk mapping (variant B)

Expected and verified:

- PP0 vp0: layers 1–6, embedding
- PP0 vp1: layers 13–18
- PP1 vp0: layers 7–12
- PP1 vp1: layers 19–24, output/loss

## A/B performance

| Variant | Schedule | tok/s | step ms | MFU | VRAM smi | theoretical bubble |
|---|---|---:|---:|---:|---:|---:|
| A non-interleaved | `forward_backward_pipelining_without_interleaving` | 19652.75 | 833.67 | 15.90% | 15443 | 20.0% |
| B interleaved+VPP+overlap | `forward_backward_pipelining_with_interleaving` | 20112.63 | 814.61 | 16.28% | 17077 | 11.1% |

## Communication and bubble

- Theoretical bubble: 20.0% → 11.1% (delta 8.9 pp)
- Measured nsys GPU idle: 7.41% → 11.82% (NCCL wait still looks busy; this is not the pipeline bubble)
- Activation send/recv: 78.32 → 119.61 ms/step/GPU
- Exposed P2P: 78.32 → 65.98 ms/step/GPU (reduction 12.34 ms)
- Extra hop cost (issued P2P union): +41.29 ms/step/GPU
- Kernel P2P/compute overlap: 0.0% → 38.2%
- NVTX chunk-compute vs async send/recv overlap (B): 1.27 ms/step
- Forward/backward P2P transfers per step (B NVTX): 8.0 / 9.0
- Warmup / steady / cooldown (B): 119.25 / 361.32 / 175.81 ms/step
- Stage imbalance ratio: A 0.17380942326903964 / B 0.17472243074575058

Timeline evidence for B: `pp_chunkN_forward` NVTX on the compute stream with
`pp_async_send_recv_forward/backward` issued at chunk post-forward/post-backward.
Kernel overlap percent is P2P NCCL send/recv intersecting non-NCCL compute.

## Decision

Correctness passed but fast-screen gain +2.34% is below 3%, so formal 20+100 did not run. Keep the FAST screen. Combined interleaved+VPP+overlap is not accepted as a throughput win on this host.

## Commands

```bash
bash scripts/phase8_interleaved_pp_pod.sh o2xtfls95kpxey 0.88
```

Raw outputs: `results/phase83_work/`. Summary: `results/phase8_interleaved_pp_overlap.json`.

