#!/usr/bin/env bash
# Phase 8.1: PP=2 baseline + microbatch sweep on a 2x A40 pod.
# Do not combine TP and PP. Do not disable NCCL P2P. Exactly one pod.
set -euo pipefail

ROOT=/workspace/megatron-performance-lab
MEGATRON=/workspace/Megatron-LM
TE=/workspace/TransformerEngine
MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
BRANCH="${PHASE81_BRANCH:-cursor/phase81-pp2-baseline-3b5c}"
POD_ID="${1:?pod id required}"
PRICE="${2:-0.88}"
PUBLIC_IP="${PHASE81_PUBLIC_IP:-}"
DATA_CENTER="${PHASE81_DATA_CENTER:-}"
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
if [[ ! -x "${NSYS}" ]]; then
  NSYS="$(command -v nsys || true)"
fi
CUDNN_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
ABORT_MARKER=/tmp/phase81_abort_reason.txt
WORK=results/phase81_work
PROF=profiles/phase81_work

export PYTHONPATH="${MEGATRON}:${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=8
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export NVTE_FRAMEWORK=pytorch
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export LD_LIBRARY_PATH="${CUDNN_LIB}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CUDNN_LIB}/libcudnn.so.9"
unset NCCL_P2P_DISABLE || true

if [[ "${NCCL_P2P_DISABLE:-0}" == "1" ]]; then
  echo "NCCL_P2P_DISABLE must not be set for Phase 8.1" >&2
  exit 1
fi

abort() {
  local reason="$1"
  echo "${reason}" | tee "${ABORT_MARKER}"
  echo "PHASE81_ABORT=${reason}"
  exit 2
}

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
python -m pip install pydantic einops importlib-metadata nvdlfw-inspect onnx onnxscript pyyaml

ln -sfn "${ROOT}" /workspace/megatron-performance-lab
ln -sfn "${MEGATRON}" /workspace/Megatron-LM
ln -sfn "${TE}" /workspace/TransformerEngine

mkdir -p "${ROOT}/${WORK}" "${ROOT}/${PROF}"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
case "${driver}" in
  570.*|580.*) ;;
  *)
    abort "Rejecting driver ${driver}; Phase 8.1 requires 570.x or 580.x with the cu128 image"
    ;;
esac
if [[ "$(nvidia-smi -L | wc -l)" -ne 2 ]]; then
  abort "Need exactly 2 visible GPUs"
fi

run_dist() {
  local nproc="$1"
  shift
  if [[ "${RUN_TIMEOUT_SEC:-0}" -gt 0 ]]; then
    timeout "${RUN_TIMEOUT_SEC}" "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" "$@"
  else
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" "$@"
  fi
}

run_pp() {
  local nproc="$1"
  local pp="$2"
  local label="$3"
  shift 3
  run_dist "${nproc}" \
    scripts/phase8_pp_run.py \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size "${pp}" \
    --run-label "${label}" \
    "$@"
}

run_pp_nsys() {
  local nproc="$1"
  local pp="$2"
  local label="$3"
  local output="$4"
  shift 4
  if [[ -z "${NSYS}" ]]; then
    abort "nsys is not installed on this image"
  fi
  local -a cmd=(
    "${NSYS}" profile
    --trace=cuda,nvtx,osrt,cublas,cudnn
    --sample=none
    --cpuctxsw=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --force-overwrite=true
    --output="${output}"
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}"
    scripts/phase8_pp_run.py
    --tensor-parallel-size 1
    --pipeline-parallel-size "${pp}"
    --run-label "${label}"
    --profile-mode
    --warmup-iterations 2
    --measured-iterations 2
  )
  if [[ "${RUN_TIMEOUT_SEC:-0}" -gt 0 ]]; then
    timeout "${RUN_TIMEOUT_SEC}" "${cmd[@]}" "$@"
  else
    "${cmd[@]}" "$@"
  fi
}

set +e
timeout 90 "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/phase7_topology.py \
  --pod-id "${POD_ID}" \
  --price-per-hour-usd "${PRICE}" \
  --output-json "${WORK}/topology.json"
topo_status=$?
set -e
if [[ "${topo_status}" -eq 124 ]]; then
  abort "NCCL P2P hang during topology sanity (timeout 90s)"
fi
if [[ "${topo_status}" -ne 0 ]]; then
  reason="topology probe failed"
  if [[ -f "${WORK}/topology.json" ]]; then
    reason="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("results/phase81_work/topology.json").read_text())
print(payload.get("abort_reason") or "topology probe failed")
PY
)"
  fi
  abort "${reason}"
fi

"${PYTHON}" - "${PUBLIC_IP}" "${DATA_CENTER}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path("results/phase81_work/topology.json").read_text())
if payload.get("abort_reason"):
    raise SystemExit(payload["abort_reason"])
path = payload.get("gpu0_gpu1_path")
if path == "SYS":
    raise SystemExit("cross-NUMA SYS topology")
nvlink = isinstance(path, str) and path.startswith("NV") and str(path)[2:].isdigit()
if path not in {"NODE", "PIX", "PHB", "PXB"} and not nvlink:
    raise SystemExit(f"unsupported GPU0-GPU1 path {path}")
if not payload["p2p_accessibility"]["bidirectional_gpu0_gpu1"]:
    raise SystemExit("CUDA peer access is not bidirectional")
if not payload["nccl_all_reduce_sanity"]["passed"]:
    raise SystemExit("NCCL All-Reduce sanity failed")
payload["public_ip"] = sys.argv[1] or payload.get("public_ip")
payload["data_center"] = sys.argv[2] or payload.get("data_center")
payload["host_public_ip"] = payload["public_ip"]
payload["data_center_id"] = payload["data_center"]
Path("results/phase81_work/topology.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"PHASE81_TOPOLOGY_OK path={path}")
PY

run_pp 1 1 phase81_pp1_ref \
  --num-microbatches 1 --micro-batch-size 8 --global-batch-size 8 \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json "${WORK}/pp1_ref.json"

declare -a MB_COUNTS=(1 2 4 8)
declare -a MB_SIZES=(8 4 2 1)
for i in "${!MB_COUNTS[@]}"; do
  mb="${MB_COUNTS[$i]}"
  size="${MB_SIZES[$i]}"
  label="phase81_pp2_mb${mb}"
  json_path="${WORK}/pp2_mb${mb}.json"
  RUN_TIMEOUT_SEC=900 run_pp 2 2 "${label}" \
    --num-microbatches "${mb}" --micro-batch-size "${size}" --global-batch-size 8 \
    --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
    --output-json "${json_path}"

  profile_label="phase81_pp2_mb${mb}_profile"
  profile_json="${WORK}/pp2_mb${mb}_profile.json"
  nsys_out="${PROF}/pp2_mb${mb}"
  RUN_TIMEOUT_SEC=900 run_pp_nsys 2 2 "${profile_label}" "${nsys_out}" \
    --num-microbatches "${mb}" --micro-batch-size "${size}" --global-batch-size 8 \
    --output-json "${profile_json}"
  "${NSYS}" export --type sqlite --force-overwrite=true \
    -o "${nsys_out}.sqlite" \
    "${nsys_out}.nsys-rep"
done

ANALYZE=(
  "${PYTHON}" scripts/phase8_analyze_pp.py
  --topology "${WORK}/topology.json"
  --pp1 "${WORK}/pp1_ref.json"
  --pod-id "${POD_ID}"
  --price-per-hour-usd "${PRICE}"
  --output results/phase8_pp2_baseline.json
  --markdown docs/experiments/phase8_pp2_baseline.md
)
for mb in 1 2 4 8; do
  ANALYZE+=(--pp2-mb "${WORK}/pp2_mb${mb}.json")
  ANALYZE+=(--sqlite "${PROF}/pp2_mb${mb}.sqlite")
  ANALYZE+=(--trace "${PROF}/pp2_mb${mb}.nsys-rep")
  ANALYZE+=(--label "phase81_pp2_mb${mb}")
done

"${ANALYZE[@]}"
echo "PHASE81_POD_RUN_COMPLETE"
