#!/usr/bin/env bash
# Reproducible Phase 7.2 Sequence Parallel A/B on a 2x A40 pod.
set -euo pipefail

ROOT=/workspace/megatron-performance-lab
MEGATRON=/workspace/Megatron-LM
TE=/workspace/TransformerEngine
MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
BRANCH="${PHASE72_BRANCH:-cursor/phase72-sequence-parallel-3b5c}"
POD_ID="${1:?pod id required}"
PRICE="${2:-0.88}"
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
CUDNN_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib

export PYTHONPATH="${MEGATRON}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NVTE_FRAMEWORK=pytorch
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CUDNN_LIB}/libcudnn.so.9"

if [[ ! -d "${ROOT}/.git" ]]; then
  git clone --branch "${BRANCH}" https://github.com/kopernik278/megatron-performance-lab.git "${ROOT}"
fi
cd "${ROOT}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull origin "${BRANCH}"

if [[ ! -d "${MEGATRON}/.git" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON}"
fi
git -C "${MEGATRON}" fetch origin
git -C "${MEGATRON}" checkout "${MEGATRON_COMMIT}"

if [[ ! -d "${TE}/.git" ]]; then
  git clone https://github.com/NVIDIA/TransformerEngine.git "${TE}"
fi
git -C "${TE}" fetch origin
git -C "${TE}" checkout "${TE_COMMIT}"
git -C "${TE}" submodule update --init --recursive

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "${ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
python -m pip install -U pip setuptools wheel ninja "pybind11[global]" "cmake>=3.21,<4"
if ! python -c "import transformer_engine" >/dev/null 2>&1; then
  NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=86 NVTE_WITH_NCCL_EP=0 MAX_JOBS=8 \
    python -m pip install --no-build-isolation --no-deps "${TE}"
fi
python -m pip install pydantic einops importlib-metadata nvdlfw-inspect onnx onnxscript

ln -sfn "${ROOT}" /workspace/megatron-performance-lab
ln -sfn "${MEGATRON}" /workspace/Megatron-LM
ln -sfn "${TE}" /workspace/TransformerEngine

mkdir -p "${ROOT}/results/phase72_work" "${ROOT}/profiles/phase72_work"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
case "${driver}" in
  570.*) ;;
  *)
    echo "Rejecting driver ${driver}; Phase 7 requires 570.x with the cu128 image" >&2
    exit 1
    ;;
esac
if [[ "$(nvidia-smi -L | wc -l)" -ne 2 ]]; then
  echo "Need exactly 2 visible GPUs" >&2
  exit 1
fi

run_tp() {
  local label="$1"
  shift
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/phase7_tp_run.py \
    --tensor-parallel-size 2 \
    --run-label "${label}" \
    "$@"
}

run_tp_nsys() {
  local label="$1"
  local output="$2"
  shift 2
  "${NSYS}" profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="${output}" \
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
      scripts/phase7_tp_run.py \
      --tensor-parallel-size 2 \
      --run-label "${label}" \
      --profile-mode \
      --warmup-iterations 2 \
      --measured-iterations 5 \
      "$@"
}

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/phase7_topology.py \
  --pod-id "${POD_ID}" \
  --price-per-hour-usd "${PRICE}" \
  --output-json results/phase72_work/topology.json

run_tp phase72_A_tp2_sp_false \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json results/phase72_work/A_tp2_sp_false.json

run_tp phase72_B_tp2_sp_true --sequence-parallel \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json results/phase72_work/B_tp2_sp_true.json

run_tp_nsys phase72_A_profile profiles/phase72_work/A_communication \
  --output-json results/phase72_work/A_tp2_sp_false_profile.json

run_tp_nsys phase72_B_profile profiles/phase72_work/B_communication \
  --sequence-parallel \
  --output-json results/phase72_work/B_tp2_sp_true_profile.json

"${NSYS}" export --type sqlite --force-overwrite=true \
  -o profiles/phase72_work/A_communication.sqlite \
  profiles/phase72_work/A_communication.nsys-rep
"${NSYS}" export --type sqlite --force-overwrite=true \
  -o profiles/phase72_work/B_communication.sqlite \
  profiles/phase72_work/B_communication.nsys-rep

GAIN="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
a = json.loads(Path("results/phase72_work/A_tp2_sp_false.json").read_text())
b = json.loads(Path("results/phase72_work/B_tp2_sp_true.json").read_text())
print((b["tokens_per_second"] / a["tokens_per_second"] - 1.0) * 100.0)
PY
)"
echo "PHASE72_FAST_SCREEN_GAIN_PERCENT=${GAIN}"

ANALYZE=(
  "${PYTHON}" scripts/phase7_analyze_sp.py
  --topology results/phase72_work/topology.json
  --variant-a results/phase72_work/A_tp2_sp_false.json
  --variant-b results/phase72_work/B_tp2_sp_true.json
  --variant-a-profile results/phase72_work/A_tp2_sp_false_profile.json
  --variant-b-profile results/phase72_work/B_tp2_sp_true_profile.json
  --sqlite-a profiles/phase72_work/A_communication.sqlite
  --sqlite-b profiles/phase72_work/B_communication.sqlite
  --trace-a profiles/phase72_work/A_communication.nsys-rep
  --trace-b profiles/phase72_work/B_communication.nsys-rep
  --output results/phase7_sequence_parallel.json
)

need_formal="$("${PYTHON}" - <<PY
print(float("${GAIN}") >= 2.0)
PY
)"
if [[ "${need_formal}" == "True" ]]; then
  run_tp phase72_A_formal \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --output-json results/phase72_work/A_tp2_sp_false_formal.json
  run_tp phase72_B_formal --sequence-parallel \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --output-json results/phase72_work/B_tp2_sp_true_formal.json
  ANALYZE+=(--formal-a results/phase72_work/A_tp2_sp_false_formal.json)
  ANALYZE+=(--formal-b results/phase72_work/B_tp2_sp_true_formal.json)
fi

"${ANALYZE[@]}"
echo "PHASE72_POD_RUN_COMPLETE"
