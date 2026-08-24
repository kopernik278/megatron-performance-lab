#!/usr/bin/env bash
# Reproducible Phase 7.4 TE Userbuffers A/B/C on a 2x A40 pod.
# P2P must work. Do not disable NCCL P2P. Abort on SYS/cross-NUMA or P2P hang.
set -euo pipefail

ROOT=/workspace/megatron-performance-lab
MEGATRON=/workspace/Megatron-LM
TE=/workspace/TransformerEngine
MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
BRANCH="${PHASE74_BRANCH:-cursor/phase74-userbuffers-overlap-3b5c}"
POD_ID="${1:?pod id required}"
PRICE="${2:-0.88}"
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
if [[ ! -x "${NSYS}" ]]; then
  NSYS="$(command -v nsys || true)"
fi
CUDNN_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
ABORT_MARKER=/tmp/phase74_abort_reason.txt

export PYTHONPATH="${MEGATRON}:${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_SKIPMC=1
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
  echo "NCCL_P2P_DISABLE must not be set for Phase 7.4" >&2
  exit 1
fi

write_abort_json() {
  local reason="$1"
  local python_bin="${PYTHON:-python3}"
  "${python_bin}" - "$reason" <<'PY'
import json
import sys
from pathlib import Path

reason = sys.argv[1]
topology = {}
path = Path("results/phase74_work/topology.json")
if path.exists():
    try:
        topology = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        topology = {"unparsed": path.read_text(encoding="utf-8")[:4000]}
Path("results").mkdir(parents=True, exist_ok=True)
Path("results/phase7_tp_userbuffers_overlap.json").write_text(
    json.dumps(
        {
            "status": "aborted",
            "experiment": "Phase 7.4 TE Userbuffers TP communication overlap",
            "iteration_mode": "FAST ITERATION MODE",
            "abort_reason": reason,
            "topology": topology,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

abort() {
  local reason="$1"
  echo "${reason}" | tee "${ABORT_MARKER}"
  echo "PHASE74_ABORT=${reason}"
  write_abort_json "${reason}" || true
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

mkdir -p "${ROOT}/results/phase74_work" "${ROOT}/profiles/phase74_work"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
case "${driver}" in
  570.*) ;;
  *)
    abort "Rejecting driver ${driver}; Phase 7 requires 570.x with the cu128 image"
    ;;
esac
if [[ "$(nvidia-smi -L | wc -l)" -ne 2 ]]; then
  abort "Need exactly 2 visible GPUs"
fi

run_dist() {
  if [[ "${RUN_TIMEOUT_SEC:-0}" -gt 0 ]]; then
    timeout "${RUN_TIMEOUT_SEC}" "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 "$@"
  else
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 "$@"
  fi
}

run_tp() {
  local label="$1"
  shift
  run_dist \
    scripts/phase7_tp_run.py \
    --tensor-parallel-size 2 \
    --run-label "${label}" \
    "$@"
}

run_tp_nsys() {
  local label="$1"
  local output="$2"
  shift 2
  if [[ -z "${NSYS}" ]]; then
    abort "nsys is not installed on this image"
  fi
  if [[ "${RUN_TIMEOUT_SEC:-0}" -gt 0 ]]; then
    timeout "${RUN_TIMEOUT_SEC}" "${NSYS}" profile \
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
  else
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
  fi
}

set +e
timeout 90 "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/phase7_topology.py \
  --pod-id "${POD_ID}" \
  --price-per-hour-usd "${PRICE}" \
  --output-json results/phase74_work/topology.json
topo_status=$?
set -e
if [[ "${topo_status}" -eq 124 ]]; then
  abort "NCCL P2P hang during topology sanity (timeout 90s)"
fi
if [[ "${topo_status}" -ne 0 ]]; then
  reason="topology probe failed"
  if [[ -f results/phase74_work/topology.json ]]; then
    reason="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("results/phase74_work/topology.json").read_text())
print(payload.get("abort_reason") or "topology probe failed")
PY
)"
  fi
  abort "${reason}"
fi

"${PYTHON}" - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("results/phase74_work/topology.json").read_text())
if payload.get("abort_reason"):
    raise SystemExit(payload["abort_reason"])
path = payload.get("gpu0_gpu1_path")
if path == "SYS":
    raise SystemExit("cross-NUMA SYS topology")
if path not in {"NODE", "PIX", "PHB", "PXB"}:
    raise SystemExit(f"unsupported GPU0-GPU1 path {path}")
if not payload["p2p_accessibility"]["bidirectional_gpu0_gpu1"]:
    raise SystemExit("CUDA peer access is not bidirectional")
if not payload["nccl_all_reduce_sanity"]["passed"]:
    raise SystemExit("NCCL All-Reduce sanity failed")
print(f"PHASE74_TOPOLOGY_OK path={path}")
PY

run_tp phase74_A_tp2_baseline \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json results/phase74_work/A_tp2_baseline.json

run_tp phase74_B_te_linear_sp \
  --sequence-parallel --te-linear \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json results/phase74_work/B_te_linear_sp.json

set +e
RUN_TIMEOUT_SEC=1800 run_tp phase74_C_userbuffers \
  --sequence-parallel --te-linear --tp-comm-overlap \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json results/phase74_work/C_userbuffers.json
c_status=$?
set -e
if [[ "${c_status}" -eq 124 ]]; then
  abort "Userbuffers initialize/run hung (timeout 1800s)"
fi
if [[ "${c_status}" -ne 0 ]]; then
  abort "Variant C Userbuffers run failed"
fi

run_tp_nsys phase74_A_profile profiles/phase74_work/A_communication \
  --output-json results/phase74_work/A_tp2_baseline_profile.json

run_tp_nsys phase74_B_profile profiles/phase74_work/B_communication \
  --sequence-parallel --te-linear \
  --output-json results/phase74_work/B_te_linear_sp_profile.json

set +e
RUN_TIMEOUT_SEC=1800 run_tp_nsys phase74_C_profile profiles/phase74_work/C_communication \
  --sequence-parallel --te-linear --tp-comm-overlap \
  --output-json results/phase74_work/C_userbuffers_profile.json
c_prof_status=$?
set -e
if [[ "${c_prof_status}" -ne 0 ]]; then
  abort "Variant C Nsight profile failed"
fi

"${NSYS}" export --type sqlite --force-overwrite=true \
  -o profiles/phase74_work/A_communication.sqlite \
  profiles/phase74_work/A_communication.nsys-rep
"${NSYS}" export --type sqlite --force-overwrite=true \
  -o profiles/phase74_work/B_communication.sqlite \
  profiles/phase74_work/B_communication.nsys-rep
"${NSYS}" export --type sqlite --force-overwrite=true \
  -o profiles/phase74_work/C_communication.sqlite \
  profiles/phase74_work/C_communication.nsys-rep

GAIN="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
b = json.loads(Path("results/phase74_work/B_te_linear_sp.json").read_text())
c = json.loads(Path("results/phase74_work/C_userbuffers.json").read_text())
print((c["tokens_per_second"] / b["tokens_per_second"] - 1.0) * 100.0)
PY
)"
echo "PHASE74_B_TO_C_GAIN_PERCENT=${GAIN}"

ANALYZE=(
  "${PYTHON}" scripts/phase7_analyze_ub.py
  --topology results/phase74_work/topology.json
  --variant-a results/phase74_work/A_tp2_baseline.json
  --variant-b results/phase74_work/B_te_linear_sp.json
  --variant-c results/phase74_work/C_userbuffers.json
  --variant-a-profile results/phase74_work/A_tp2_baseline_profile.json
  --variant-b-profile results/phase74_work/B_te_linear_sp_profile.json
  --variant-c-profile results/phase74_work/C_userbuffers_profile.json
  --sqlite-a profiles/phase74_work/A_communication.sqlite
  --sqlite-b profiles/phase74_work/B_communication.sqlite
  --sqlite-c profiles/phase74_work/C_communication.sqlite
  --trace-a profiles/phase74_work/A_communication.nsys-rep
  --trace-b profiles/phase74_work/B_communication.nsys-rep
  --trace-c profiles/phase74_work/C_communication.nsys-rep
  --output results/phase7_tp_userbuffers_overlap.json
)

need_formal="$("${PYTHON}" - <<PY
print(float("${GAIN}") >= 3.0)
PY
)"
if [[ "${need_formal}" == "True" ]]; then
  run_tp phase74_B_formal \
    --sequence-parallel --te-linear \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --output-json results/phase74_work/B_te_linear_sp_formal.json
  run_tp phase74_C_formal \
    --sequence-parallel --te-linear --tp-comm-overlap \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --output-json results/phase74_work/C_userbuffers_formal.json
  ANALYZE+=(--formal-b results/phase74_work/B_te_linear_sp_formal.json)
  ANALYZE+=(--formal-c results/phase74_work/C_userbuffers_formal.json)
fi

"${ANALYZE[@]}"
echo "PHASE74_POD_RUN_COMPLETE"
