#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:10:00
#PJM -L jobenv=singularity
#PJM -j

set -e

module load singularity-ce

# =========================
# 設定
# =========================
# 保存先の設定: `EXP_NAME=my-experiment bash ...` で上書きできます。
EXP_NAME="${EXP_NAME:-unnamed}"
# 未指定時は最新の実験ディレクトリを使用します。DATE を指定するとその日時を使用します。
DATE="${DATE:-}"
if [[ ! "${EXP_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "[ERROR] invalid EXP_NAME: ${EXP_NAME}" >&2
    exit 1
fi
EXPERIMENT_ROOT="${HOME}/experiments/${EXP_NAME}"
REQUESTED_RUN="${DATE}"

if [[ ! -d "${EXPERIMENT_ROOT}" ]]; then
    echo "[ERROR] experiment root directory not found: ${EXPERIMENT_ROOT}" >&2
    exit 1
fi

if [[ -n "${REQUESTED_RUN}" ]]; then
    if [[ "${REQUESTED_RUN}" = /* ]]; then
        RUN_ROOT="${REQUESTED_RUN}"
    else
        RUN_ROOT="${EXPERIMENT_ROOT}/${REQUESTED_RUN}"
    fi
    if [[ ! -d "${RUN_ROOT}" ]]; then
        echo "[ERROR] experiment directory not found: ${RUN_ROOT}" >&2
        exit 1
    fi
else
    RUN_ROOT=""
    while IFS= read -r -d '' candidate; do
        candidate_name="${candidate##*/}"
        if [[ "${candidate_name}" =~ ^[0-9]{8}_[0-9]{6}$ ]] &&
           [[ -z "${RUN_ROOT}" || "${candidate_name}" > "${RUN_ROOT##*/}" ]]; then
            RUN_ROOT="${candidate}"
        fi
    done < <(find "${EXPERIMENT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0)

    if [[ -z "${RUN_ROOT}" ]]; then
        echo "[ERROR] no experiment directories matching YYYYMMDD_HHMMSS found in: ${EXPERIMENT_ROOT}" >&2
        exit 1
    fi
fi

RESULT_FILE="${RUN_ROOT}/results.jsonl"
OUTPUT_FILE="${RUN_ROOT}/progress.png"

IMAGE="${HOME}/nlp-singularity/nlp-singularity.sif"
WORKDIR="${HOME}/projects/autoresearch"
METRIC="${METRIC:-val_bpb}"

# 絶対パスに変換
if [[ "${RESULT_FILE}" != /* ]]; then
    RESULT_FILE="${WORKDIR}/${RESULT_FILE}"
fi

if [ ! -f "${RESULT_FILE}" ]; then
    echo "Error: results.jsonl not found:"
    echo "  ${RESULT_FILE}"
    exit 1
fi

echo "========================================"
echo "Plot experiment results"
echo "========================================"
echo "Results : ${RESULT_FILE}"
echo

# =========================
# プロット
# =========================

singularity exec \
    --bind "${WORKDIR}:${WORKDIR}" --bind "${RUN_ROOT}:${RUN_ROOT}" \
    --pwd "${WORKDIR}" \
    "${IMAGE}" \
    bash -lc "
        uv run plot_results_parallel.py '${RESULT_FILE}' --metric '${METRIC}' -o '${OUTPUT_FILE}'
    "

echo
echo "Done."
echo "Output:"
echo "  ${OUTPUT_FILE}"
