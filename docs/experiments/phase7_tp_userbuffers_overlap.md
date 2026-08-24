# Phase 7.4: TE Userbuffers TP communication overlap

## Outcome

FAST ITERATION MODE (5 warmup + 20 measured) ran on one 2x A40 Pod with
working NCCL P2P. Variants A and B completed. Variant C created the
Userbuffers communicator and then livelocked in the A40 reduce-scatter
kernel, so B→C overlap was not measured. Formal 20+100 did not run.
NCCL P2P was never disabled. The Pod was deleted after artifact download.

| Variant | TP | sequence_parallel | TE Linear | tp_comm_overlap | tokens/s | step ms | MFU | VRAM (smi) |
|---|---:|---|---|---|---:|---:|---:|---:|
| A | 2 | False | False | False | 20,751.63 | 789.53 | 16.79% | 21,828 MiB |
| B | 2 | True | True | False | 23,438.42 | 699.02 | 18.97% | 16,938 MiB |
| C | 2 | True | True | True | n/a | n/a | n/a | hung at 6,684 MiB |

B vs A is +12.95% throughput (1.129x). That is TE Linear + Sequence Parallel
on this host, not Userbuffers. C never produced a step time.

Phase 7.1 valid TP=2 baseline (commit `709437d`, pod `7rpwv95a5j6axg`, NODE,
P2P on) was 19,856.48 tokens/s at 825.12 ms. This host's variant A is the
same model/config family and is slightly faster; do not mix it with the
Phase 7.2 P2P-disabled Sequence Parallel host.

## Infrastructure and environment

- RunPod Pod: `lzsg0odj8y3kyw`
- Data center: CA-MTL-1, public IP `69.30.85.51`
- Allocation: one Secure Cloud Pod, 2x NVIDIA A40 48GB, $0.88/h (≤ $0.90/h)
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Host CUDA: 13.0; container CUDA runtime: 12.8
- Driver: 580.126.09
- PyTorch: 2.8.0+cu128
- NCCL: 2.27.3
- cuDNN: 9.10.2
- Transformer Engine: `2.17.1+4329ff84`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Lab commit on the Pod: `e0d08d67320a51291f5bfe6d98d6718428aa7d75`
- CUDA Graph: disabled (`cuda_graph_impl=none`)
- `bias_dropout_fusion=True`, `bias_gelu_fusion=False`
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `UB_SKIPMC=1`
- `NCCL_P2P_DISABLE` unset (recorded as `"0"`)

## Topology and NCCL P2P

`nvidia-smi topo -m` reported **NODE** between GPU0 (`98:00.0`) and GPU1
(`D2:00.0`). Both GPUs are NUMA 1. NVLink was inactive. NICs were `mlx5_0`
and `mlx5_1`.

CUDA `can_device_access_peer` was True in both directions. A two-rank NCCL
scalar All-Reduce returned 3.0 on both ranks (0.475 ms / 0.482 ms). The
harness topology probe also passed (`PHASE74_TOPOLOGY_OK path=NODE`).

This is the required same-NUMA, P2P-working topology. SYS and P2P-disabled
hosts were rejected during the hunt and are not in this result.

## Variant A (TP=2, no SP, local spec + TE attention/TENorm)

Smoke (3 steps): finite losses, finite grads, no deadlock, no NCCL errors.
Userbuffers inactive. Linears were Megatron `ColumnParallelLinear` /
`RowParallelLinear`.

Measured 20 steps after 5 warmup: 20,751.63 tokens/s, 789.53 ms/step,
MFU 16.79%, 21,828 MiB/GPU nvidia-smi.

## Variant B (TP=2, SP, TE Linear, overlap off)

Smoke passed. Sequence Parallel and TE Linear were active. Userbuffers
inactive (`communicator_count=0`). Linears were
`TELayerNormColumnParallelLinear` / `TERowParallelLinear`.

Measured 20 steps after 5 warmup: 23,438.42 tokens/s, 699.02 ms/step,
MFU 18.97%, 16,938 MiB/GPU nvidia-smi. VRAM drop vs A matches Sequence
Parallel activations.

## Variant C (same as B plus Userbuffers)

`initialize_ub(shape=[16384, 1024], tp_size=2, use_fp8=False, ub_cfgs={},
bootstrap_backend="nccl")` ran. Logs showed
`!!! [UB] Create Userbuffers Communicator`, then both ranks spun in

```
userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop:352
Reduce-scatter: SM N [peer]: expecting 1 got 0
```

For ~6 minutes both GPUs sat at 100% util and 6,684 MiB. No step JSON was
written. The 1800 s harness timeout had not fired; the hung ranks were
killed so the Pod could be released. That produced
`PHASE74_ABORT=Variant C Userbuffers run failed`.

No nsys traces were collected (A/B short profiles and C profile run after
C's timed run). Formal B vs C (20+100) requires C correct and B→C ≥ 3%;
neither gate was met.

## Bottleneck

Userbuffers overlap did not become a measurable communication hide on this
A40 NODE/PCIe pair. The failure is the GPU-side reduce-scatter handshake
(`expecting 1 got 0`), not missing `UB_SKIPMC`, not disabled P2P, and not
SYS topology. Remaining work is to make TE Userbuffers RS complete on A40
without NVLink, or to accept that this overlap path is not usable on this
SKU/topology.

## Commands

On the Pod, after cloning branch `cursor/phase74-userbuffers-overlap-3b5c`:

```bash
bash scripts/phase7_userbuffers_pod.sh lzsg0odj8y3kyw 0.88
```

Raw outputs: `results/phase74_work/topology.json`,
`results/phase74_work/A_tp2_baseline.json`,
`results/phase74_work/B_te_linear_sp.json`.
Summary: `results/phase7_tp_userbuffers_overlap.json`.
