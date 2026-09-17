#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:10:00
#PJM -L jobenv=singularity
#PJM -j

set -euo pipefail

# worker別PNGと、全workerを重ねた all_workers.png を生成します。
# bash scripts/genkai/plot_results_workers.sh
# RESULT_FILE / OUTPUT_DIR / METRIC / DIRECTION / ROWS_PER_PAGE で設定を上書きできます。
RESULT_FILE="${RESULT_FILE:-${HOME}/experiments/autoresearch/parallel-strategy-diversity/20260916_224138/results.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${RESULT_FILE}")/worker_plots}"
WORKDIR="${WORKDIR:-${HOME}/projects/autoresearch}"
IMAGE="${IMAGE:-${HOME}/nlp-singularity/nlp-singularity.sif}"
METRIC="${METRIC:-val_bpb}"
ROWS_PER_PAGE="${ROWS_PER_PAGE:-15}"

if [[ ! -f "${RESULT_FILE}" ]]; then
    echo "[ERROR] results.jsonl not found: ${RESULT_FILE}" >&2
    exit 1
fi

WORKDIR="$(cd "${WORKDIR}" && pwd -P)"
INPUT_DIR="$(cd "$(dirname "${RESULT_FILE}")" && pwd -P)"
RESULT_FILE="${INPUT_DIR}/$(basename "${RESULT_FILE}")"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd -P)"
ARGS=("${RESULT_FILE}" --metric "${METRIC}" --rows-per-page "${ROWS_PER_PAGE}" -o "${OUTPUT_DIR}")
if [[ -n "${DIRECTION:-}" ]]; then
    ARGS+=(--direction "${DIRECTION}")
fi

module load singularity-ce
singularity exec \
    --bind "${WORKDIR}:${WORKDIR}" \
    --bind "${INPUT_DIR}:${INPUT_DIR}" \
    --bind "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    --pwd "${WORKDIR}" "${IMAGE}" \
    bash -lc 'uv run --project "$1" "$1/plot_results_workers.py" "${@:2}"' \
    plot-results-workers "${WORKDIR}" "${ARGS[@]}"

echo "Output: ${OUTPUT_DIR}"
