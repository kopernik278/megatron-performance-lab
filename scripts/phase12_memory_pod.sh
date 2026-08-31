#!/usr/bin/env bash
# Phase 12: Training memory and capacity A/B/C/D on one 2x A40 pod (DP=2).
# No TP/PP/SP/VPP/Userbuffers/CUDA Graph. Do not disable NCCL P2P.
set -euo pipefail

ROOT=/workspace/megatron-performance-lab
MEGATRON=/workspace/Megatron-LM
TE=/workspace/TransformerEngine
MEGATRON_COMMIT=09fde85ea25fb67e9b32019089fae163a3233bd3
TE_COMMIT=4329ff84bfbdaa778a33cba02a15fb0807c64689
BRANCH="${PHASE12_BRANCH:-cursor/phase12-memory-capacity-3b5c}"
POD_ID="${1:?pod id required}"
PRICE="${2:-0.88}"
PUBLIC_IP="${PHASE12_PUBLIC_IP:-}"
DATA_CENTER="${PHASE12_DATA_CENTER:-}"
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
if [[ ! -x "${NSYS}" ]]; then
  NSYS="$(command -v nsys || true)"
fi
CUDNN_LIB=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
ABORT_MARKER=/tmp/phase12_abort_reason.txt
RUN_DONE_MARKER=/tmp/phase12_run_done
WORK=results/phase12_work
PROF=profiles/phase12_work

if [[ -f "${RUN_DONE_MARKER}" ]]; then
  if [[ -f "${ROOT}/results/phase12_memory_capacity.json" ]]; then
    status="$("${ROOT}/.venv/bin/python" -c "import json;print(json.load(open('${ROOT}/results/phase12_memory_capacity.json')).get('status',''))" 2>/dev/null || true)"
    if [[ "${status}" == "success" ]]; then
      echo "PHASE12_ALREADY_RAN"
      exit 0
    fi
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
export LD_PRELOAD="${CUDNN_LIB}/libcudnn.so.9"
unset NCCL_P2P_DISABLE || true

if [[ "${NCCL_P2P_DISABLE:-0}" == "1" ]]; then
  echo "NCCL_P2P_DISABLE must not be set for Phase 12" >&2
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
path = Path("results/phase12_work/topology.json")
if path.exists():
    try:
        topology = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        topology = {"unparsed": path.read_text(encoding="utf-8")[:4000]}
Path("results").mkdir(parents=True, exist_ok=True)
Path("docs/experiments").mkdir(parents=True, exist_ok=True)
Path("results/phase12_memory_capacity.json").write_text(
    json.dumps(
        {
            "status": "aborted",
            "experiment": "Phase 12 training memory and capacity",
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
Path("docs/experiments/phase12_memory_capacity.md").write_text(
    f"# Phase 12 aborted\n\n{reason}\n",
    encoding="utf-8",
)
PY
}

abort() {
  local reason="$1"
  echo "${reason}" | tee "${ABORT_MARKER}"
  echo "PHASE12_ABORT=${reason}"
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
    abort "Rejecting driver ${driver}; Phase 12 requires 570.x or 580.x with the cu128 image"
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

variant_flags() {
  case "$1" in
    A) echo --no-use-distributed-optimizer --overlap-grad-reduce --no-overlap-param-gather --no-activation-checkpointing ;;
    B) echo --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather --no-activation-checkpointing ;;
    C) echo --no-use-distributed-optimizer --overlap-grad-reduce --no-overlap-param-gather --activation-checkpointing ;;
    D) echo --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather --activation-checkpointing ;;
    *) return 1 ;;
  esac
}

run_nsys_variant() {
  local variant="$1"
  local label="$2"
  local output="$3"
  shift 3
  if [[ -z "${NSYS}" ]]; then
    echo "PHASE12_NSYS_SKIP nsys missing"
    return 0
  fi
  local -a flags
  # shellcheck disable=SC2207
  flags=($(variant_flags "${variant}"))
  local -a cmd=(
    "${NSYS}" profile
    --trace=cuda,nvtx,osrt,cublas,cudnn
    --sample=none
    --cpuctxsw=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --force-overwrite=true
    --output="${output}"
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2
    scripts/phase12_memory_run.py
    --variant "${variant}"
    --run-label "${label}"
    --profile-mode
    --mode benchmark
    --warmup-iterations 2
    --measured-iterations 2
    --micro-batch-size 8
    --sequence-length 2048
    "${flags[@]}"
  )
  if [[ "${RUN_TIMEOUT_SEC:-0}" -gt 0 ]]; then
    timeout "${RUN_TIMEOUT_SEC}" "${cmd[@]}" "$@" || true
  else
    "${cmd[@]}" "$@" || true
  fi
}

set +e
timeout 90 "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/phase7_topology.py \
  --allow-sys-topology \
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
payload = json.loads(Path("results/phase12_work/topology.json").read_text())
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
payload = json.loads(Path("results/phase12_work/topology.json").read_text())
if payload.get("abort_reason"):
    raise SystemExit(payload["abort_reason"])
path = payload.get("gpu0_gpu1_path")
nccl_ok = payload.get("nccl_all_reduce_sanity", {}).get("passed")
if path == "SYS" and not nccl_ok:
    raise SystemExit("cross-NUMA SYS topology with failed NCCL sanity")
nvlink = isinstance(path, str) and path.startswith("NV") and str(path)[2:].isdigit()
allowed = {"NODE", "PIX", "PHB", "PXB"}
if path == "SYS" and nccl_ok:
    allowed = allowed | {"SYS"}
if path not in allowed and not nvlink:
    raise SystemExit(f"unsupported GPU0-GPU1 path {path}")
if not payload["p2p_accessibility"]["bidirectional_gpu0_gpu1"]:
    raise SystemExit("CUDA peer access is not bidirectional")
if not payload["nccl_all_reduce_sanity"]["passed"]:
    raise SystemExit("NCCL All-Reduce sanity failed")
payload["public_ip"] = sys.argv[1] or payload.get("public_ip")
payload["data_center"] = sys.argv[2] or payload.get("data_center")
payload["host_public_ip"] = payload["public_ip"]
payload["data_center_id"] = payload["data_center"]
Path("results/phase12_work/topology.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"PHASE12_TOPOLOGY_OK path={path}")
PY

echo "PHASE12_CORRECTNESS_SMOKE_START"
for V in A B C D; do
  # shellcheck disable=SC2207
  FLAGS=($(variant_flags "${V}"))
  set +e
  RUN_TIMEOUT_SEC=240 run_dist 2 \
    scripts/phase12_memory_run.py \
    --variant "${V}" \
    --run-label "phase12_${V}_smoke" \
    --mode smoke \
    --micro-batch-size 8 \
    --sequence-length 2048 \
    --output-json "${WORK}/${V}_smoke.json" \
    "${FLAGS[@]}"
  st=$?
  set -e
  if [[ "${st}" -eq 124 ]]; then
    abort "correctness smoke hung on variant ${V}"
  fi
  if [[ "${st}" -ne 0 ]]; then
    abort "correctness smoke failed on variant ${V} (rc=${st})"
  fi
done
echo "PHASE12_CORRECTNESS_SMOKE_OK"

echo "PHASE12_FIXED_WORKLOAD_START"
for V in A B C D; do
  echo "PHASE12_VARIANT_${V}_FIXED_START"
  # shellcheck disable=SC2207
  FLAGS=($(variant_flags "${V}"))
  RUN_TIMEOUT_SEC=900 run_dist 2 \
    scripts/phase12_memory_run.py \
    --variant "${V}" \
    --run-label "phase12_${V}_fixed_mb8" \
    --mode benchmark \
    --smoke-iterations 3 --warmup-iterations 5 --measured-iterations 20 \
    --micro-batch-size 8 \
    --sequence-length 2048 \
    --output-json "${WORK}/variant_${V}_fixed.json" \
    "${FLAGS[@]}"
done

# Short nsys on A vs C for recompute evidence (best-effort).
echo "PHASE12_NSYS_A_C_START"
RUN_TIMEOUT_SEC=600 run_nsys_variant A phase12_a_profile "${PROF}/variant_a" \
  --output-json "${WORK}/variant_A_profile.json" || true
RUN_TIMEOUT_SEC=600 run_nsys_variant C phase12_c_profile "${PROF}/variant_c" \
  --output-json "${WORK}/variant_C_profile.json" || true
if [[ -n "${NSYS}" && -f "${PROF}/variant_a.nsys-rep" ]]; then
  "${NSYS}" export --type sqlite --force-overwrite=true \
    -o "${PROF}/variant_a.sqlite" "${PROF}/variant_a.nsys-rep" || true
fi
if [[ -n "${NSYS}" && -f "${PROF}/variant_c.nsys-rep" ]]; then
  "${NSYS}" export --type sqlite --force-overwrite=true \
    -o "${PROF}/variant_c.sqlite" "${PROF}/variant_c.nsys-rep" || true
fi

echo "PHASE12_CAPACITY_SEARCH_START"
for V in A B C D; do
  echo "PHASE12_CAPACITY_${V}_START"
  "${PYTHON}" scripts/phase12_capacity_search.py \
    --python "${PYTHON}" \
    --variant "${V}" \
    --run-script scripts/phase12_memory_run.py \
    --work-dir "${WORK}" \
    --sequence-length 2048 \
    --start-mb 8 \
    --max-mb-cap 64 \
    --timeout-sec 300 \
    --output-json "${WORK}/capacity_${V}.json"
done

echo "PHASE12_CAPACITY_BENCH_START"
for V in A B C D; do
  max_mb="$("${PYTHON}" -c "import json;print(json.load(open('${WORK}/capacity_${V}.json'))['max_micro_batch_size'])")"
  if [[ "${max_mb}" -lt 1 ]]; then
    abort "capacity search returned max_mb=0 for ${V}"
  fi
  # shellcheck disable=SC2207
  FLAGS=($(variant_flags "${V}"))
  echo "PHASE12_CAPACITY_BENCH_${V}_MB${max_mb}"
  RUN_TIMEOUT_SEC=900 run_dist 2 \
    scripts/phase12_memory_run.py \
    --variant "${V}" \
    --run-label "phase12_${V}_cap_mb${max_mb}" \
    --mode benchmark \
    --smoke-iterations 2 --warmup-iterations 3 --measured-iterations 10 \
    --micro-batch-size "${max_mb}" \
    --sequence-length 2048 \
    --output-json "${WORK}/capacity_bench_${V}.json" \
    "${FLAGS[@]}"
done

SQLITE_ARGS=()
if [[ -f "${PROF}/variant_a.sqlite" ]]; then
  SQLITE_ARGS+=(--sqlite-a "${PROF}/variant_a.sqlite")
fi
if [[ -f "${PROF}/variant_c.sqlite" ]]; then
  SQLITE_ARGS+=(--sqlite-c "${PROF}/variant_c.sqlite")
fi

echo "PHASE12_ANALYZE_START"
"${PYTHON}" scripts/phase12_analyze_memory.py \
  --topology "${WORK}/topology.json" \
  --variant-a "${WORK}/variant_A_fixed.json" \
  --variant-b "${WORK}/variant_B_fixed.json" \
  --variant-c "${WORK}/variant_C_fixed.json" \
  --variant-d "${WORK}/variant_D_fixed.json" \
  --capacity-a "${WORK}/capacity_A.json" \
  --capacity-b "${WORK}/capacity_B.json" \
  --capacity-c "${WORK}/capacity_C.json" \
  --capacity-d "${WORK}/capacity_D.json" \
  --capacity-bench-a "${WORK}/capacity_bench_A.json" \
  --capacity-bench-b "${WORK}/capacity_bench_B.json" \
  --capacity-bench-c "${WORK}/capacity_bench_C.json" \
  --capacity-bench-d "${WORK}/capacity_bench_D.json" \
  --pod-id "${POD_ID}" \
  --price-per-hour-usd "${PRICE}" \
  --output results/phase12_memory_capacity.json \
  --markdown docs/experiments/phase12_memory_capacity.md \
  "${SQLITE_ARGS[@]}"

# Encode compact artifacts for Cloud Agent pullback.
tar czf /tmp/phase12_artifacts.tgz \
  results/phase12_memory_capacity.json \
  docs/experiments/phase12_memory_capacity.md \
  "${WORK}" 2>/dev/null || true
if [[ -f /tmp/phase12_artifacts.tgz ]]; then
  echo "PHASE12_ARTIFACT_B64_BEGIN"
  base64 -w0 /tmp/phase12_artifacts.tgz
  echo
  echo "PHASE12_ARTIFACT_B64_END"
fi

touch "${RUN_DONE_MARKER}"
echo "PHASE12_POD_RUN_COMPLETE"
