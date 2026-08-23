#!/usr/bin/env bash
# Reproduce pinned Megatron + TE environment on RunPod pytorch image.
set -euo pipefail
export LC_ALL=C

MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
REPO_URL="${REPO_URL:-https://github.com/kopernik278/megatron-performance-lab.git}"
BRANCH="${BRANCH:-cursor/phase34-reprofile-3b5c}"

cd /workspace
if [ ! -d Megatron-LM/.git ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git
fi
git -C Megatron-LM fetch --depth 1 origin "$MEGATRON_COMMIT" || true
git -C Megatron-LM checkout "$MEGATRON_COMMIT"

if [ ! -d megatron-performance-lab/.git ]; then
  git clone "$REPO_URL" megatron-performance-lab
fi
git -C megatron-performance-lab fetch origin "$BRANCH" || git -C megatron-performance-lab fetch origin main
git -C megatron-performance-lab checkout "$BRANCH" 2>/dev/null || git -C megatron-performance-lab checkout main

cd /workspace/megatron-performance-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel ninja pybind11 packaging pydantic einops

if [ ! -d /workspace/TransformerEngine/.git ]; then
  git clone https://github.com/NVIDIA/TransformerEngine.git /workspace/TransformerEngine
fi
git -C /workspace/TransformerEngine checkout "$TE_COMMIT"
cd /workspace/TransformerEngine
NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=86 MAX_JOBS=8 \
  pip install --no-build-isolation --no-deps -v .

cd /workspace/megatron-performance-lab
pip check
python - <<'PY'
import torch, importlib.metadata as m
import transformer_engine as te
print("pytorch", torch.__version__)
print("cuda", torch.version.cuda)
print("nccl", m.version("nvidia-nccl-cu12"))
print("te", te.__version__)
PY

echo "SETUP_OK"
