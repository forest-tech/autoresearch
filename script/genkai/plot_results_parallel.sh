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
EXP_DIR="20260909_201352"
RESULT_FILE="./results/${EXP_DIR}/results.jsonl"
METRIC=val_loss
OUTPUT_FILE="./progress_parallel_${EXP_DIR}.png"

IMAGE=/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif
WORKDIR=/home/pj24001974/ku50001532/projects/autoresearch
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
    --bind "${WORKDIR}:${WORKDIR}" \
    --pwd "${WORKDIR}" \
    "${IMAGE}" \
    bash -lc "
        uv run plot_results_parallel.py '${RESULT_FILE}' --metric '${METRIC}' -o '${OUTPUT_FILE}'
    "

echo
echo "Done."
echo "Output:"
echo "  ${OUTPUT_FILE}"
