# Phase 1.1 Megatron Single-GPU Smoke Test

## Summary

The Phase 1.1 smoke test validates that the existing A40 RunPod environment can run a minimal Megatron Core GPT training loop on one GPU. The test uses synthetic random data and verifies forward pass, backward pass, optimizer step, loss production, and checkpoint save/load.

## Exact Command

```bash
cd /workspace/megatron-performance-lab
source .venv/bin/activate
PYTHONPATH=/workspace/Megatron-LM TRANSFORMER_ENGINE_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/phase1_smoke_test.py \
  --iterations 5 \
  --micro-batch-size 2 \
  --sequence-length 64 \
  --vocab-size 128 \
  --num-layers 2 \
  --hidden-size 64 \
  --ffn-hidden-size 256 \
  --num-attention-heads 4 \
  --checkpoint-dir results/phase1_smoke_test/checkpoint \
  --output-json results/phase1_smoke_test/metrics.json
```

## Environment

- Pod: `megatron-performance-lab-a40-migration`
- GPU: `1x NVIDIA A40`
- Driver: `580.159.04`
- Python: `3.12.3`
- PyTorch: `2.8.0+cu128`
- CUDA runtime: `12.8`
- NCCL: `2.27.3`
- Megatron Core: `0.20.0+09fde85ea`
- Megatron-LM commit: `09fde85ea25fb67e9b32019089fae163a3233bd3`
- Transformer Engine: disabled/not installed; Megatron used Torch fallback implementations.
- `nvcc`: unavailable in the container; this test only required the CUDA runtime.

## Model And Run Configuration

- Model: Megatron Core `GPTModel`
- Transformer layer spec: `get_gpt_layer_local_spec()`
- Attention backend: `AttnBackend.unfused`
- Layers: `2`
- Hidden size: `64`
- FFN hidden size: `256`
- Attention heads: `4`
- Vocabulary size: `128`
- Batch size: `2`
- Sequence length: `64`
- Iterations: `5`
- Precision path: BF16 autocast with FP32 parameters
- Optimizer: `torch.optim.AdamW`, learning rate `1.0e-3`
- Data: synthetic random token IDs

## Results

- Status: success
- Losses: `4.8618059158`, `4.6573410034`, `4.4836206436`, `4.3465747833`, `4.2347960472`
- Final loss: `4.2347960472`
- Step times, ms: `629.5021604747`, `12.5749092549`, `13.2447835058`, `9.6689704806`, `9.4864023849`
- Average step time: `134.8954452202 ms`
- Peak allocated GPU memory: `20.3579101562 MiB`
- Checkpoint: saved to `results/phase1_smoke_test/checkpoint/single_gpu_smoke.pt` and loaded successfully

## Compatibility Notes

The initial `torchrun` invocation resolved to the system Python and could not import Megatron. The final command uses `.venv/bin/python -m torch.distributed.run` with `PYTHONPATH=/workspace/Megatron-LM`.

Using BF16 parameters with the no-Apex local LayerNorm fallback produced a dtype mismatch. The final test keeps parameters in FP32 and exercises BF16 through CUDA autocast, without changing PyTorch, CUDA, NCCL, or Megatron versions.
