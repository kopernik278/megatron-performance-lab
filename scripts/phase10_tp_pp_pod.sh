#!/usr/bin/env bash
# Phase 10.1: TP=2 + PP=2 hybrid baseline on one 4-GPU pod (DP=1).
# Default GPU: auto-detect via nvidia-smi; set PHASE101_GPU_TYPE to override.
set -euo pipefail

ROOT=/workspace/megatron-performance-lab
MEGATRON=/workspace/Megatron-LM
TE=/workspace/TransformerEngine
MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
BRANCH="${PHASE101_BRANCH:-cursor/phase101-tp2-pp2-hybrid-3b5c}"
POD_ID="${1:?pod id required}"
PRICE="${2:-1.36}"
PHASE101_GPU_TYPE="${PHASE101_GPU_TYPE:-}"
PUBLIC_IP="${PHASE101_PUBLIC_IP:-}"
DATA_CENTER="${PHASE101_DATA_CENTER:-}"
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
if [[ ! -x "${NSYS}" ]]; then
  NSYS="$(command -v nsys || true)"
fi
CUDNN_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
ABORT_MARKER=/tmp/phase101_abort_reason.txt
RUN_DONE_MARKER=/tmp/phase101_run_done
WORK=results/phase101_work
PROF=profiles/phase101_work

if [[ -f "${RUN_DONE_MARKER}" ]]; then
  if python3 - "${ROOT}" <<'PY' 2>/dev/null
import json
import sys
from pathlib import Path
path = Path(sys.argv[1]) / "results/phase10_tp2_pp2_hybrid_baseline.json"
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "success" else 1)
PY
  then
    echo "PHASE101_ALREADY_RAN"
    exit 0
  fi
  rm -f "${RUN_DONE_MARKER}"
fi

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
# Defer libcudnn LD_PRELOAD until after CUDA preflight; it can break torch init on some hosts.
CUDNN_LD_PRELOAD="${CUDNN_LIB}/libcudnn.so.9"
unset LD_PRELOAD || true
unset NCCL_P2P_DISABLE || true
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PHASE101_GPU_TYPE

if [[ "${NCCL_P2P_DISABLE:-0}" == "1" ]]; then
  echo "NCCL_P2P_DISABLE must not be set for Phase 10.1" >&2
  exit 1
fi

abort() {
  local reason="$1"
  echo "${reason}" | tee "${ABORT_MARKER}"
  echo "PHASE101_ABORT=${reason}"
  "${PYTHON:-python3}" - "$reason" "${POD_ID}" "${PRICE}" <<'PY' || true
import json, sys
from pathlib import Path
reason, pod_id, price = sys.argv[1], sys.argv[2], float(sys.argv[3])
topology = {}
path = Path("results/phase101_work/topology.json")
if path.exists():
    try:
        topology = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        topology = {"unparsed": path.read_text(encoding="utf-8")[:4000]}
Path("results").mkdir(parents=True, exist_ok=True)
Path("docs/experiments").mkdir(parents=True, exist_ok=True)
Path("results/phase10_tp2_pp2_hybrid_baseline.json").write_text(
    json.dumps(
        {
            "status": "aborted",
            "experiment": "Phase 10.1 TP=2 PP=2 hybrid baseline",
            "abort_reason": reason,
            "infrastructure": {
                "pod_id": pod_id,
                "price_per_hour_usd": price,
                "pod_status": "to_be_deleted",
            },
            "topology": topology,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
Path("docs/experiments/phase10_tp2_pp2_hybrid_baseline.md").write_text(
    f"# Phase 10.1 aborted\n\n{reason}\n",
    encoding="utf-8",
)
PY
  exit 0
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

ln -sfn "${ROOT}" /workspace/megatron-performance-lab
ln -sfn "${MEGATRON}" /workspace/Megatron-LM
ln -sfn "${TE}" /workspace/TransformerEngine

mkdir -p "${ROOT}/${WORK}" "${ROOT}/${PROF}"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"

read -r GPU_PROFILE_JSON < <("${PYTHON}" - <<'PY'
import json
import os
from phase10_gpu_profile import resolve_profile
profile = resolve_profile(os.environ.get("PHASE101_GPU_TYPE"))
print(json.dumps({
    "nvte_cuda_archs": profile.nvte_cuda_archs,
    "gpu_type_id": profile.gpu_type_id,
    "display_name": profile.display_name,
}))
PY
)
NVTE_CUDA_ARCHS="$(echo "${GPU_PROFILE_JSON}" | "${PYTHON}" -c "import json,sys; print(json.load(sys.stdin)['nvte_cuda_archs'])")"
GPU_TYPE_ID="$(echo "${GPU_PROFILE_JSON}" | "${PYTHON}" -c "import json,sys; print(json.load(sys.stdin)['gpu_type_id'])")"
GPU_DISPLAY="$(echo "${GPU_PROFILE_JSON}" | "${PYTHON}" -c "import json,sys; print(json.load(sys.stdin)['display_name'])")"
echo "PHASE101_GPU_PROFILE=${GPU_TYPE_ID} (${GPU_DISPLAY}) NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}"

TE_ARCH_MARKER="${ROOT}/.venv/.te_cuda_archs"
need_te_install=0
if ! python -c "import transformer_engine" >/dev/null 2>&1; then
  need_te_install=1
elif [[ ! -f "${TE_ARCH_MARKER}" ]] || [[ "$(cat "${TE_ARCH_MARKER}")" != "${NVTE_CUDA_ARCHS}" ]]; then
  need_te_install=1
fi
if [[ "${need_te_install}" -eq 1 ]]; then
  python -m pip uninstall -y transformer-engine transformer_engine 2>/dev/null || true
  NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" NVTE_WITH_NCCL_EP=0 MAX_JOBS=8 \
    python -m pip install --no-build-isolation --no-deps "${TE}"
  echo "${NVTE_CUDA_ARCHS}" > "${TE_ARCH_MARKER}"
fi
python -m pip install pydantic einops importlib-metadata nvdlfw-inspect onnx onnxscript pyyaml

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
case "${driver}" in
  570.*|580.*) ;;
  *)
    abort "Rejecting driver ${driver}; Phase 10.1 requires 570.x or 580.x"
    ;;
esac
gpu_count="$(nvidia-smi -L | wc -l)"
if [[ "${gpu_count}" -ne 4 ]]; then
  abort "Need exactly 4 visible GPUs, got ${gpu_count}"
fi

wait_for_cuda_ready() {
  local max_wait="${1:-120}"
  local elapsed=0
  while (( elapsed < max_wait )); do
    if env -u LD_PRELOAD "${PYTHON}" - <<'PY' 2>/dev/null
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
torch.zeros(1, device="cuda:0")
assert torch.cuda.can_device_access_peer(0, 1)
PY
    then
      echo "PHASE101_CUDA_READY elapsed=${elapsed}s"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  abort "CUDA not ready after ${max_wait}s (nvidia-smi ok but torch CUDA init failed)"
}

wait_for_cuda_ready 120

run_dist() {
  local nproc="$1"
  shift
  if [[ "${RUN_TIMEOUT_SEC:-0}" -gt 0 ]]; then
    timeout "${RUN_TIMEOUT_SEC}" "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" "$@"
  else
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" "$@"
  fi
}

run_hybrid() {
  local nproc="$1"
  local tp="$2"
  local pp="$3"
  local label="$4"
  shift 4
  run_dist "${nproc}" \
    scripts/phase10_tp_pp_run.py \
    --tensor-parallel-size "${tp}" \
    --pipeline-parallel-size "${pp}" \
    --run-label "${label}" \
    "$@"
}

run_hybrid_nsys() {
  local nproc="$1"
  local tp="$2"
  local pp="$3"
  local label="$4"
  local output="$5"
  shift 5
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
    scripts/phase10_tp_pp_run.py
    --tensor-parallel-size "${tp}"
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
"${PYTHON}" scripts/phase10_preflight.py \
  --pod-id "${POD_ID}" \
  --price-per-hour-usd "${PRICE}" \
  --output-json "${WORK}/preflight.json" \
  --allow-sys-topology
preflight_status=$?
set -e
if [[ "${preflight_status}" -ne 0 ]]; then
  reason="preflight failed"
  if [[ -f "${WORK}/preflight.json" ]]; then
    reason="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("results/phase101_work/preflight.json").read_text())
print(payload.get("abort_reason") or "preflight failed")
PY
)"
  fi
  abort "${reason}"
fi

export LD_PRELOAD="${CUDNN_LD_PRELOAD}"

set +e
timeout 180 "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/phase10_topology.py \
  --pod-id "${POD_ID}" \
  --price-per-hour-usd "${PRICE}" \
  --output-json "${WORK}/topology.json" \
  --preflight-json "${WORK}/preflight.json"
topo_status=$?
set -e
if [[ "${topo_status}" -eq 124 ]]; then
  abort "NCCL hang during 4-GPU topology sanity (timeout 180s)"
fi
if [[ "${topo_status}" -ne 0 ]]; then
  reason="topology probe failed"
  if [[ -f "${WORK}/topology.json" ]]; then
    reason="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("results/phase101_work/topology.json").read_text())
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
payload = json.loads(Path("results/phase101_work/topology.json").read_text())
if payload.get("abort_reason"):
    raise SystemExit(payload["abort_reason"])
if not payload["p2p_accessibility"]["fully_bidirectional"]:
    raise SystemExit("CUDA peer access is not fully bidirectional")
if not payload["nccl_all_reduce_sanity"]["passed"]:
    raise SystemExit("NCCL All-Reduce sanity failed")
payload["public_ip"] = sys.argv[1] or payload.get("public_ip")
payload["data_center"] = sys.argv[2] or payload.get("data_center")
Path("results/phase101_work/topology.json").write_text(json.dumps(payload, indent=2) + "\n")
print("PHASE101_TOPOLOGY_OK")
PY

echo "PHASE101_CORRECTNESS_SMOKE_START"
RUN_TIMEOUT_SEC=300 run_hybrid 4 2 2 phase101_hybrid_smoke \
  --num-microbatches 4 --micro-batch-size 2 --global-batch-size 8 \
  --smoke-iterations 3 --warmup-iterations 1 --measured-iterations 1 \
  --output-json "${WORK}/hybrid_smoke.json"
echo "PHASE101_CORRECTNESS_SMOKE_OK"

echo "PHASE101_REFERENCE_R1_START"
RUN_TIMEOUT_SEC=600 run_hybrid 1 1 1 phase101_r1_tp1_pp1 \
  --num-microbatches 1 --micro-batch-size 8 --global-batch-size 8 \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json "${WORK}/r1_tp1_pp1.json"

echo "PHASE101_REFERENCE_R2_START"
RUN_TIMEOUT_SEC=600 run_hybrid 2 2 1 phase101_r2_tp2_pp1 \
  --num-microbatches 1 --micro-batch-size 8 --global-batch-size 8 \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json "${WORK}/r2_tp2_pp1.json"

declare -a MB_COUNTS=(2 4 8)
declare -a MB_SIZES=(4 2 1)
for i in "${!MB_COUNTS[@]}"; do
  mb="${MB_COUNTS[$i]}"
  size="${MB_SIZES[$i]}"
  label="phase101_hybrid_mb${mb}"
  json_path="${WORK}/hybrid_mb${mb}.json"
  echo "PHASE101_SWEEP_M${mb}_START"
  RUN_TIMEOUT_SEC=900 run_hybrid 4 2 2 "${label}" \
    --num-microbatches "${mb}" --micro-batch-size "${size}" --global-batch-size 8 \
    --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
    --output-json "${json_path}"
done

read -r BEST_MB BEST_SIZE BEST_JSON BEST_TPS < <("${PYTHON}" - <<'PY'
import json
from pathlib import Path
best = None
for mb, size in ((2, 4), (4, 2), (8, 1)):
    path = Path(f"results/phase101_work/hybrid_mb{mb}.json")
    payload = json.loads(path.read_text())
    tps = payload["tokens_per_second"]
    if best is None or tps > best[3]:
        best = (mb, size, str(path), tps)
print(*best)
PY
)
echo "PHASE101_BEST_MB=${BEST_MB} BEST_TPS=${BEST_TPS}"

echo "PHASE101_PROFILE_BEST_START"
profile_json="${WORK}/hybrid_mb${BEST_MB}_profile.json"
nsys_out="${PROF}/hybrid_mb${BEST_MB}"
RUN_TIMEOUT_SEC=900 run_hybrid_nsys 4 2 2 "phase101_hybrid_mb${BEST_MB}_profile" "${nsys_out}" \
  --num-microbatches "${BEST_MB}" --micro-batch-size "${BEST_SIZE}" --global-batch-size 8 \
  --output-json "${profile_json}"
"${NSYS}" export --type sqlite --force-overwrite=true \
  -o "${nsys_out}.sqlite" \
  "${nsys_out}.nsys-rep"

echo "PHASE101_REFERENCE_R3_START"
RUN_TIMEOUT_SEC=900 run_hybrid 2 1 2 phase101_r3_tp1_pp2 \
  --num-microbatches "${BEST_MB}" --micro-batch-size "${BEST_SIZE}" --global-batch-size 8 \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json "${WORK}/r3_tp1_pp2.json"

ANALYZE=(
  "${PYTHON}" scripts/phase10_analyze_tp_pp.py
  --topology "${WORK}/topology.json"
  --best-run "${BEST_JSON}"
  --sqlite "${nsys_out}.sqlite"
  --trace "${nsys_out}.nsys-rep"
  --pod-id "${POD_ID}"
  --price-per-hour-usd "${PRICE}"
  --output results/phase10_tp2_pp2_hybrid_baseline.json
  --markdown docs/experiments/phase10_tp2_pp2_hybrid_baseline.md
)
for mb in 2 4 8; do
  ANALYZE+=(--microbatch-sweep "${WORK}/hybrid_mb${mb}.json")
done
ANALYZE+=(--reference "${WORK}/r1_tp1_pp1.json" --reference-label R1)
ANALYZE+=(--reference "${WORK}/r2_tp2_pp1.json" --reference-label R2)
ANALYZE+=(--reference "${WORK}/r3_tp1_pp2.json" --reference-label R3)
ANALYZE+=(--reference "${BEST_JSON}" --reference-label R4)
"${ANALYZE[@]}"

tar czf /tmp/phase101_artifacts.tgz results/phase10_tp2_pp2_hybrid_baseline.json docs/experiments/phase10_tp2_pp2_hybrid_baseline.md "${WORK}" 2>/dev/null || true
if [[ -f /tmp/phase101_artifacts.tgz ]]; then
  echo "PHASE101_ARTIFACT_B64_BEGIN"
  base64 -w0 /tmp/phase101_artifacts.tgz
  echo
  echo "PHASE101_ARTIFACT_B64_END"
fi

touch "${RUN_DONE_MARKER}"
echo "PHASE101_POD_RUN_COMPLETE"
