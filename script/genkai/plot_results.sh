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

IMAGE=/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif
WORKDIR=/home/pj24001974/ku50001532/projects/autoresearch

# =========================
# 引数チェック
# =========================

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  pjsub $0 <results.jsonl>"
    echo
    echo "Example:"
    echo "  pjsub $0 results/20260828_192954/results.jsonl"
    exit 1
fi

RESULT_FILE="$1"

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
        uv run python script/plot_results.py '${RESULT_FILE}'
    "

echo
echo "Done."
echo "Output:"
echo "  $(dirname "${RESULT_FILE}")/progress.png"
