#!/usr/bin/env bash
# Phase 7.4b: B vs C1 AG-only Userbuffers overlap on a 2x A40 pod.
# Do not enable Reduce-Scatter overlap. Do not disable NCCL P2P.
# Abort C1 within 120s on hang. Exactly one pod.
set -euo pipefail

ROOT=/workspace/megatron-performance-lab
MEGATRON=/workspace/Megatron-LM
TE=/workspace/TransformerEngine
MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
BRANCH="${PHASE74B_BRANCH:-cursor/phase74b-ag-overlap-3b5c}"
POD_ID="${1:?pod id required}"
PRICE="${2:-0.88}"
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
if [[ ! -x "${NSYS}" ]]; then
  NSYS="$(command -v nsys || true)"
fi
CUDNN_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
ABORT_MARKER=/tmp/phase74b_abort_reason.txt
HANG_KERNEL=userbuffers_fp16_sum_inplace_gpu_rr_rs_oop
C1_TIMEOUT_SEC=120
WORK=results/phase74b_work
PROF=profiles/phase74b_work

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
  echo "NCCL_P2P_DISABLE must not be set for Phase 7.4b" >&2
  exit 1
fi

C1_OVERLAP_FLAGS=(
  --tp-comm-overlap
  --tp-comm-overlap-ag
  --no-tp-comm-overlap-rs
  --no-tp-comm-bulk-dgrad
  --no-tp-comm-bulk-wgrad
  --no-tp-comm-overlap-rs-dgrad
)
C2_OVERLAP_FLAGS=(
  --tp-comm-overlap
  --tp-comm-overlap-ag
  --no-tp-comm-overlap-rs
  --tp-comm-bulk-dgrad
  --no-tp-comm-bulk-wgrad
  --no-tp-comm-overlap-rs-dgrad
)

write_abort_json() {
  local reason="$1"
  local python_bin="${PYTHON:-python3}"
  "${python_bin}" - "$reason" <<'PY'
import json
import sys
from pathlib import Path

reason = sys.argv[1]
topology = {}
path = Path("results/phase74b_work/topology.json")
if path.exists():
    try:
        topology = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        topology = {"unparsed": path.read_text(encoding="utf-8")[:4000]}
variant_b = None
b_path = Path("results/phase74b_work/B_te_linear_sp.json")
if b_path.exists():
    try:
        variant_b = json.loads(b_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        variant_b = None
Path("results").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "scripts")
from phase7_analyze_partial import abort_payload, write
write(Path("results/phase7_tp_partial_comm_overlap.json"), abort_payload(topology, reason, variant_b))
PY
}

abort() {
  local reason="$1"
  echo "${reason}" | tee "${ABORT_MARKER}"
  echo "PHASE74B_ABORT=${reason}"
  write_abort_json "${reason}" || true
  exit 2
}

contains_hang_kernel() {
  local log="$1"
  [[ -f "${log}" ]] && grep -F "${HANG_KERNEL}" "${log}" >/dev/null 2>&1
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
    abort "Rejecting driver ${driver}; Phase 7.4b requires 570.x or 580.x with the cu128 image"
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
payload = json.loads(Path("results/phase74b_work/topology.json").read_text())
print(payload.get("abort_reason") or "topology probe failed")
PY
)"
  fi
  abort "${reason}"
fi

"${PYTHON}" - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("results/phase74b_work/topology.json").read_text())
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
print(f"PHASE74B_TOPOLOGY_OK path={path}")
PY

run_tp phase74b_B_te_linear_sp \
  --sequence-parallel --te-linear \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json "${WORK}/B_te_linear_sp.json"

run_tp_nsys phase74b_B_profile "${PROF}/B_communication" \
  --sequence-parallel --te-linear \
  --output-json "${WORK}/B_te_linear_sp_profile.json"

set +e
RUN_TIMEOUT_SEC="${C1_TIMEOUT_SEC}" run_tp phase74b_C1_ag_only \
  --sequence-parallel --te-linear \
  "${C1_OVERLAP_FLAGS[@]}" \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --output-json "${WORK}/C1_ag_only.json" \
  > "${WORK}/C1_ag_only.log" 2>&1
c1_status=$?
set -e
if contains_hang_kernel "${WORK}/C1_ag_only.log"; then
  abort "C1 launched ${HANG_KERNEL}; aborting without retry"
fi
if [[ "${c1_status}" -eq 124 ]]; then
  abort "C1 AG-only Userbuffers hung (timeout ${C1_TIMEOUT_SEC}s)"
fi
if [[ "${c1_status}" -ne 0 ]]; then
  abort "Variant C1 AG-only Userbuffers run failed"
fi

set +e
RUN_TIMEOUT_SEC="${C1_TIMEOUT_SEC}" run_tp_nsys phase74b_C1_profile "${PROF}/C1_communication" \
  --sequence-parallel --te-linear \
  "${C1_OVERLAP_FLAGS[@]}" \
  --output-json "${WORK}/C1_ag_only_profile.json" \
  > "${WORK}/C1_profile.log" 2>&1
c1_prof_status=$?
set -e
if contains_hang_kernel "${WORK}/C1_profile.log"; then
  abort "C1 profile launched ${HANG_KERNEL}; aborting without retry"
fi
if [[ "${c1_prof_status}" -eq 124 ]]; then
  abort "C1 Nsight profile hung (timeout ${C1_TIMEOUT_SEC}s)"
fi
if [[ "${c1_prof_status}" -ne 0 ]]; then
  abort "Variant C1 Nsight profile failed"
fi

"${NSYS}" export --type sqlite --force-overwrite=true \
  -o "${PROF}/B_communication.sqlite" \
  "${PROF}/B_communication.nsys-rep"
"${NSYS}" export --type sqlite --force-overwrite=true \
  -o "${PROF}/C1_communication.sqlite" \
  "${PROF}/C1_communication.nsys-rep"

ANALYZE=(
  "${PYTHON}" scripts/phase7_analyze_partial.py
  --topology "${WORK}/topology.json"
  --variant-b "${WORK}/B_te_linear_sp.json"
  --variant-c1 "${WORK}/C1_ag_only.json"
  --variant-b-profile "${WORK}/B_te_linear_sp_profile.json"
  --variant-c1-profile "${WORK}/C1_ag_only_profile.json"
  --sqlite-b "${PROF}/B_communication.sqlite"
  --sqlite-c1 "${PROF}/C1_communication.sqlite"
  --trace-b "${PROF}/B_communication.nsys-rep"
  --trace-c1 "${PROF}/C1_communication.nsys-rep"
  --output results/phase7_tp_partial_comm_overlap.json
)

read -r GAIN OVERLAP < <("${PYTHON}" - <<'PY'
import json
from pathlib import Path
from phase7_analyze_tp import analyze_trace
from phase7_analyze_partial import ag_overlap_snapshot

b = json.loads(Path("results/phase74b_work/B_te_linear_sp.json").read_text())
c1 = json.loads(Path("results/phase74b_work/C1_ag_only.json").read_text())
gain = (c1["tokens_per_second"] / b["tokens_per_second"] - 1.0) * 100.0
profile = json.loads(Path("results/phase74b_work/C1_ag_only_profile.json").read_text())
comm = analyze_trace(
    Path("profiles/phase74b_work/C1_communication.sqlite"),
    Path("profiles/phase74b_work/C1_communication.nsys-rep"),
    profile,
)
ag = ag_overlap_snapshot(comm)
overlap = max(
    ag["average_ag_gemm_overlap_percent"],
    sum(item["communication_overlap_percent"] for item in comm["per_device_overlap"].values())
    / max(len(comm["per_device_overlap"]), 1),
)
print(f"{gain:.6f} {overlap:.6f}")
PY
)
echo "PHASE74B_B_TO_C1_GAIN_PERCENT=${GAIN}"
echo "PHASE74B_C1_OVERLAP_PERCENT=${OVERLAP}"

C1_WORKED=1
if awk "BEGIN {exit !(${OVERLAP} > 0)}"; then
  echo "PHASE74B_C1_HAS_OVERLAP=1"
else
  echo "PHASE74B_C1_HAS_OVERLAP=0"
  C1_WORKED=0
fi

if [[ "${C1_WORKED}" -eq 1 ]]; then
  set +e
  RUN_TIMEOUT_SEC="${C1_TIMEOUT_SEC}" run_tp phase74b_C2_bulk_dgrad \
    --sequence-parallel --te-linear \
    "${C2_OVERLAP_FLAGS[@]}" \
    --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
    --output-json "${WORK}/C2_bulk_dgrad.json" \
    > "${WORK}/C2_bulk_dgrad.log" 2>&1
  c2_status=$?
  set -e
  if contains_hang_kernel "${WORK}/C2_bulk_dgrad.log"; then
    echo "PHASE74B_C2_SKIP=launched ${HANG_KERNEL}"
  elif [[ "${c2_status}" -eq 124 ]]; then
    echo "PHASE74B_C2_SKIP=timeout ${C1_TIMEOUT_SEC}s"
  elif [[ "${c2_status}" -ne 0 ]]; then
    echo "PHASE74B_C2_SKIP=run failed"
  else
    set +e
    RUN_TIMEOUT_SEC="${C1_TIMEOUT_SEC}" run_tp_nsys phase74b_C2_profile "${PROF}/C2_communication" \
      --sequence-parallel --te-linear \
      "${C2_OVERLAP_FLAGS[@]}" \
      --output-json "${WORK}/C2_bulk_dgrad_profile.json" \
      > "${WORK}/C2_profile.log" 2>&1
    c2_prof_status=$?
    set -e
    if contains_hang_kernel "${WORK}/C2_profile.log"; then
      echo "PHASE74B_C2_SKIP=profile launched ${HANG_KERNEL}"
    elif [[ "${c2_prof_status}" -eq 0 ]]; then
      "${NSYS}" export --type sqlite --force-overwrite=true \
        -o "${PROF}/C2_communication.sqlite" \
        "${PROF}/C2_communication.nsys-rep"
      ANALYZE+=(
        --variant-c2 "${WORK}/C2_bulk_dgrad.json"
        --variant-c2-profile "${WORK}/C2_bulk_dgrad_profile.json"
        --sqlite-c2 "${PROF}/C2_communication.sqlite"
        --trace-c2 "${PROF}/C2_communication.nsys-rep"
      )
      echo "PHASE74B_C2_RAN=1"
    else
      echo "PHASE74B_C2_SKIP=profile failed"
    fi
  fi
fi

if awk "BEGIN {exit !(${OVERLAP} > 0 && ${GAIN} >= 2.0)}"; then
  run_tp phase74b_B_formal \
    --sequence-parallel --te-linear \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --output-json "${WORK}/B_te_linear_sp_formal.json"
  set +e
  RUN_TIMEOUT_SEC=1800 run_tp phase74b_C1_formal \
    --sequence-parallel --te-linear \
    "${C1_OVERLAP_FLAGS[@]}" \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --output-json "${WORK}/C1_ag_only_formal.json" \
    > "${WORK}/C1_formal.log" 2>&1
  c1_formal_status=$?
  set -e
  if contains_hang_kernel "${WORK}/C1_formal.log"; then
    echo "PHASE74B_FORMAL_SKIP=launched ${HANG_KERNEL}"
  elif [[ "${c1_formal_status}" -eq 0 ]]; then
    ANALYZE+=(--formal-b "${WORK}/B_te_linear_sp_formal.json")
    ANALYZE+=(--formal-c1 "${WORK}/C1_ag_only_formal.json")
  else
    echo "PHASE74B_FORMAL_SKIP=C1 formal failed"
  fi
else
  echo "PHASE74B_FORMAL_SKIP=overlap or 2% gain gate not met"
fi

"${ANALYZE[@]}"
echo "PHASE74B_POD_RUN_COMPLETE"
