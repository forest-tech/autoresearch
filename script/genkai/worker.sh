#!/bin/bash

set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
    echo "usage: $0 WORKER_ID GPU_ID WORKTREE OUT_DIR IMAGE REPO_ROOT [PRIMARY_METRIC] [OBJECTIVE_DIRECTION]" >&2
    exit 2
fi

WORKER_ID="$1"
GPU_ID="$2"
WORKTREE="$3"
OUT_DIR="$4"
IMAGE="$5"
REPO_ROOT="$6"
PRIMARY_METRIC="${7:-${PRIMARY_METRIC:-val_bpb}}"
OBJECTIVE_DIRECTION="${8:-${OBJECTIVE_DIRECTION:-min}}"

TRAIN_LOG="${OUT_DIR}/train.log"
CONFIG_FILE="${OUT_DIR}/config.json"
RESULT_FILE="${OUT_DIR}/worker_result.json"
EXPERIMENT_TOOL="${REPO_ROOT}/experiment_utils.py"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/torchinductor_cache" "${OUT_DIR}/triton_cache"

[[ -f "${WORKTREE}/train.py" ]] || { echo "[ERROR] train.py not found: ${WORKTREE}/train.py" >&2; exit 1; }
[[ -f "${WORKTREE}/prepare.py" ]] || { echo "[ERROR] prepare.py not found: ${WORKTREE}/prepare.py" >&2; exit 1; }
[[ -f "${EXPERIMENT_TOOL}" ]] || { echo "[ERROR] experiment_utils.py not found" >&2; exit 1; }

experiment_tool() {
    singularity exec \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --pwd "${WORKTREE}" \
        "${IMAGE}" \
        python "${EXPERIMENT_TOOL}" "$@"
}

echo "[WORKER ${WORKER_ID}] snapshot config"
experiment_tool snapshot-config \
    --train "${WORKTREE}/train.py" \
    --prepare "${WORKTREE}/prepare.py" \
    --output "${CONFIG_FILE}"

echo "[WORKER ${WORKER_ID}] start training on GPU ${GPU_ID}"
TRAIN_EXIT=0
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
SINGULARITYENV_CUDA_VISIBLE_DEVICES="${GPU_ID}" \
SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="${OUT_DIR}/torchinductor_cache" \
SINGULARITYENV_TRITON_CACHE_DIR="${OUT_DIR}/triton_cache" \
timeout 600 singularity exec \
    --nv \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${WORKTREE}" \
    "${IMAGE}" \
    bash -lc "uv run train.py" \
    >"${TRAIN_LOG}" 2>&1 || TRAIN_EXIT=$?

experiment_tool write-run-result \
    --log "${TRAIN_LOG}" \
    --output "${RESULT_FILE}" \
    --train-exit "${TRAIN_EXIT}" \
    --worker "${WORKER_ID}" \
    --gpu "${GPU_ID}" \
    --primary-metric "${PRIMARY_METRIC}" \
    --direction "${OBJECTIVE_DIRECTION}"

OBJECTIVE_VALUE=$(experiment_tool value "${RESULT_FILE}" --metric "${PRIMARY_METRIC}")
if [[ ${TRAIN_EXIT} -eq 0 && -n "${OBJECTIVE_VALUE}" ]]; then
    RUN_STATUS="ok"
else
    RUN_STATUS="crash"
fi

echo "[WORKER ${WORKER_ID}] status=${RUN_STATUS} ${PRIMARY_METRIC}=${OBJECTIVE_VALUE:-NA}"

# A training crash is an experiment result, not a worker infrastructure failure.
exit 0
