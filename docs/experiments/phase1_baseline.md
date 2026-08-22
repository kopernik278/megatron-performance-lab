# Phase 1.2 Single-GPU Megatron Baseline

## Summary

This experiment establishes the untouched single-GPU baseline for a 355.9M-parameter Megatron Core GPT model on one NVIDIA A40. It uses fixed synthetic token IDs, BF16 autocast, local Megatron layers, unfused attention, and standard unfused PyTorch AdamW. Transformer Engine and CUDA Graphs are disabled.

The original validated Pod could not restart after five attempts because its host had no free A40. One replacement Secure Cloud A40 Pod was provisioned with the same image and pinned software stack at `$0.44/hour`; no additional GPU was used.

## Exact Command

```bash
cd /workspace/megatron-performance-lab
PYTHONPATH=/workspace/Megatron-LM TRANSFORMER_ENGINE_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase1_baseline.py \
  --warmup-iterations 20 \
  --measured-iterations 100 \
  --micro-batch-candidates 4,2,1 \
  --memory-safety-fraction 0.90 \
  --sequence-length 2048 \
  --vocab-size 50304 \
  --num-layers 24 \
  --hidden-size 1024 \
  --ffn-hidden-size 4096 \
  --num-attention-heads 16 \
  --learning-rate 1.0e-4 \
  --seed 1234 \
  --gpu-sample-interval-ms 200 \
  --output-json results/phase1_baseline.json
```

## Model And Training Configuration

- Parameters: `355,919,872`, all trainable
- Architecture: 24 layers, hidden size 1024, FFN size 4096, 16 heads, head dimension 64
- Vocabulary: 50,304; learned absolute positions; tied input/output embeddings
- Sequence length: 2,048
- Parallelism: TP=1, PP=1, DP=1
- Selected micro-batch/global batch: 1/1
- Precision: BF16 forward/backward autocast with FP32 parameter and optimizer storage
- Optimizer: `torch.optim.AdamW`, learning rate `1e-4`, `foreach=False`, `fused=False`
- Data: one fixed synthetic random-token batch, seed 1235
- Warmup/measured iterations: 20/100

Micro-batch 4 OOMed during warmup. Micro-batch 2 reserved 98.2% of VRAM and was rejected by the 90% safety threshold. Micro-batch 1 reserved 59.9% during warmup and was selected.

## Results

- Average step time: `551.8516 ms`
- Median step time: `551.7120 ms`
- Throughput: `3,711.1427 tokens/s`
- Average GPU utilization: `99.6231%` from 268 samples at 200 ms intervals
- Peak device memory from `nvidia-smi`: `28,313 MiB`
- Peak PyTorch allocated/reserved memory: `19,663.9 / 27,260 MiB`
- Warmup final loss: `8.892344`
- Measured final loss: `0.053028`

The loss reflects repeated fitting of a fixed synthetic batch and is only a training-loop correctness signal, not a model-quality result.

## MFU

The calculation follows Megatron-LM commit `09fde85e` for a dense GPT with a 4H MLP, standard multi-head causal attention, and one logits projection:

```text
F_iter = 72*B*S*L*H^2 + 6*B*L*S^2*H + 6*B*S*H*V
MFU = (F_iter / step_seconds) / GPU_dense_BF16_peak
```

For B=1, S=2048, L=24, H=1024, and V=50304, the estimate is `4.9623e12 FLOPs/iteration`, or `8.9921 TFLOP/s`. Using the NVIDIA A40 dense BF16 peak of `149.7 TFLOP/s` gives `6.0067% MFU`. The peak value excludes the datasheet's structured-sparsity multiplier.

Sources: [Megatron-LM FLOP accounting](https://github.com/NVIDIA/Megatron-LM/blob/09fde85ea25fb67e9b32019089fae163a3233bd3/megatron/training/training.py) and [NVIDIA A40 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/proviz-print-nvidia-a40-datasheet-us-nvidia-1469711-r8-web.pdf).

## Environment

- Pod: `4xrckm3r6yh5dc`, Secure Cloud, `1x NVIDIA A40 48GB`, `$0.44/hour`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Driver: `570.211.01`
- Python: `3.12.3`
- PyTorch: `2.8.0+cu128`
- CUDA runtime: `12.8`
- NCCL: `2.27.3`
- Megatron Core: `0.20.0+09fde85ea`
- Megatron-LM: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine: not installed and explicitly disabled
- CUDA Graph: disabled
- Timestamp: `2026-08-22T16:00:56Z`
