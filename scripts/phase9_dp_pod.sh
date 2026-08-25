#!/usr/bin/env bash
# Phase 9.1: DP=2 gradient All-Reduce overlap A/B on one 2x A40 pod.
# Do not combine TP/PP. Do not disable NCCL P2P. Do not enable distributed optimizer.
set -euo pipefail

ROOT=/workspace/megatron-performance-lab
MEGATRON=/workspace/Megatron-LM
TE=/workspace/TransformerEngine
MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
BRANCH="${PHASE91_BRANCH:-cursor/phase91-dp2-grad-overlap-3b5c}"
POD_ID="${1:?pod id required}"
PRICE="${2:-0.88}"
PUBLIC_IP="${PHASE91_PUBLIC_IP:-}"
DATA_CENTER="${PHASE91_DATA_CENTER:-}"
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
if [[ ! -x "${NSYS}" ]]; then
  NSYS="$(command -v nsys || true)"
fi
CUDNN_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
ABORT_MARKER=/tmp/phase91_abort_reason.txt
WORK=results/phase91_work
PROF=profiles/phase91_work

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
  echo "NCCL_P2P_DISABLE must not be set for Phase 9.1" >&2
  exit 1
fi

write_abort() {
  local reason="$1"
  local python_bin="${PYTHON:-python3}"
  "${python_bin}" - "$reason" "${POD_ID}" "${PRICE}" <<'PY' || true
import json, sys
from pathlib import Path
reason, pod_id, price = sys.argv[1], sys.argv[2], float(sys.argv[3])
topology = {}
path = Path("results/phase91_work/topology.json")
if path.exists():
    try:
        topology = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        topology = {"unparsed": path.read_text(encoding="utf-8")[:4000]}
Path("results").mkdir(parents=True, exist_ok=True)
Path("docs/experiments").mkdir(parents=True, exist_ok=True)
Path("results/phase9_dp2_grad_overlap.json").write_text(
    json.dumps(
        {
            "status": "aborted",
            "experiment": "Phase 9.1 DP=2 gradient-communication overlap",
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
Path("docs/experiments/phase9_dp2_grad_overlap.md").write_text(
    f"# Phase 9.1 aborted\n\n{reason}\n",
    encoding="utf-8",
)
PY
}

abort() {
  local reason="$1"
  echo "${reason}" | tee "${ABORT_MARKER}"
  echo "PHASE91_ABORT=${reason}"
  write_abort "${reason}" || true
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

mkdir -p "${ROOT}/${WORK}" "${ROOT}/${PROF}" docs/experiments results profiles
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
case "${driver}" in
  570.*|580.*) ;;
  *)
    abort "Rejecting driver ${driver}; Phase 9.1 requires 570.x or 580.x with the cu128 image"
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

run_nsys() {
  local nproc="$1"
  local label="$2"
  local output="$3"
  shift 3
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
    scripts/phase9_dp_run.py
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
payload = json.loads(Path("results/phase91_work/topology.json").read_text())
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
payload = json.loads(Path("results/phase91_work/topology.json").read_text())
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
Path("results/phase91_work/topology.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"PHASE91_TOPOLOGY_OK path={path}")
PY

echo "PHASE91_DP1_REF_START"
RUN_TIMEOUT_SEC=900 run_dist 1 \
  scripts/phase9_dp_run.py \
  --data-parallel-size 1 \
  --no-overlap-grad-reduce \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --run-label phase91_dp1_ref \
  --output-json "${WORK}/dp1_ref.json"

echo "PHASE91_CORRECTNESS_SMOKE_START"
set +e
RUN_TIMEOUT_SEC=180 run_dist 2 \
  scripts/phase9_dp_run.py \
  --data-parallel-size 2 \
  --overlap-grad-reduce \
  --run-label phase91_b_smoke \
  --smoke-only \
  --output-json "${WORK}/b_smoke.json"
smoke_status=$?
set -e
if [[ "${smoke_status}" -eq 124 ]]; then
  abort "variant B smoke hung (timeout 180s)"
fi
if [[ "${smoke_status}" -ne 0 ]]; then
  abort "variant B correctness smoke failed"
fi
echo "PHASE91_CORRECTNESS_SMOKE_OK"

echo "PHASE91_VARIANT_A_START"
RUN_TIMEOUT_SEC=900 run_dist 2 \
  scripts/phase9_dp_run.py \
  --data-parallel-size 2 \
  --no-overlap-grad-reduce \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --run-label phase91_a_overlap_off \
  --output-json "${WORK}/variant_a.json"

RUN_TIMEOUT_SEC=900 run_nsys 2 phase91_a_profile "${PROF}/variant_a" \
  --data-parallel-size 2 \
  --no-overlap-grad-reduce \
  --output-json "${WORK}/variant_a_profile.json"
"${NSYS}" export --type sqlite --force-overwrite=true \
  -o "${PROF}/variant_a.sqlite" \
  "${PROF}/variant_a.nsys-rep"

echo "PHASE91_VARIANT_B_START"
RUN_TIMEOUT_SEC=900 run_dist 2 \
  scripts/phase9_dp_run.py \
  --data-parallel-size 2 \
  --overlap-grad-reduce \
  --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
  --run-label phase91_b_overlap_on \
  --output-json "${WORK}/variant_b.json"

RUN_TIMEOUT_SEC=900 run_nsys 2 phase91_b_profile "${PROF}/variant_b" \
  --data-parallel-size 2 \
  --overlap-grad-reduce \
  --output-json "${WORK}/variant_b_profile.json"
"${NSYS}" export --type sqlite --force-overwrite=true \
  -o "${PROF}/variant_b.sqlite" \
  "${PROF}/variant_b.nsys-rep"

FORMAL_ARGS=()
gain="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
a = json.loads(Path("results/phase91_work/variant_a.json").read_text())
b = json.loads(Path("results/phase91_work/variant_b.json").read_text())
print((b["tokens_per_second"] - a["tokens_per_second"]) / a["tokens_per_second"] * 100.0)
PY
)"
echo "PHASE91_FAST_GAIN_PERCENT=${gain}"
need_formal="$("${PYTHON}" -c "print('yes' if float('${gain}') >= 2.0 else 'no')")"
if [[ "${need_formal}" == "yes" ]]; then
  echo "PHASE91_FORMAL_A_START"
  RUN_TIMEOUT_SEC=2400 run_dist 2 \
    scripts/phase9_dp_run.py \
    --data-parallel-size 2 \
    --no-overlap-grad-reduce \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --run-label phase91_a_formal \
    --output-json "${WORK}/variant_a_formal.json"
  echo "PHASE91_FORMAL_B_START"
  RUN_TIMEOUT_SEC=2400 run_dist 2 \
    scripts/phase9_dp_run.py \
    --data-parallel-size 2 \
    --overlap-grad-reduce \
    --smoke-iterations 3 --warmup-iterations 20 --measured-iterations 100 \
    --run-label phase91_b_formal \
    --output-json "${WORK}/variant_b_formal.json"
  FORMAL_ARGS=(--formal-a "${WORK}/variant_a_formal.json" --formal-b "${WORK}/variant_b_formal.json")
fi

"${PYTHON}" scripts/phase9_analyze_dp.py \
  --topology "${WORK}/topology.json" \
  --dp1 "${WORK}/dp1_ref.json" \
  --variant-a "${WORK}/variant_a.json" \
  --variant-b "${WORK}/variant_b.json" \
  --sqlite-a "${PROF}/variant_a.sqlite" \
  --sqlite-b "${PROF}/variant_b.sqlite" \
  --trace-a "${PROF}/variant_a.nsys-rep" \
  --trace-b "${PROF}/variant_b.nsys-rep" \
  --pod-id "${POD_ID}" \
  --price-per-hour-usd "${PRICE}" \
  --output results/phase9_dp2_grad_overlap.json \
  --markdown docs/experiments/phase9_dp2_grad_overlap.md \
  "${FORMAL_ARGS[@]+"${FORMAL_ARGS[@]}"}"

echo "PHASE91_POD_RUN_COMPLETE"
