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
# 保存先の設定: このスクリプト内で実験名を編集してください。
EXP_NAME="unnamed"  # 実験名を指定しない場合の名前
# プロットする実験の日時を入力してください (例: 20260914_123456)。
DATE=""
if [[ ! "${EXP_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "[ERROR] invalid EXP_NAME: ${EXP_NAME}" >&2
    exit 1
fi
if [[ ! "${DATE}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    echo "[ERROR] set DATE in this script to YYYYMMDD_HHMMSS" >&2
    exit 1
fi
RUN_ROOT="${HOME}/experiments/autoresearch/${EXP_NAME}/${DATE}"
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
