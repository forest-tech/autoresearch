#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=0:10:00
#PJM -L jobenv=singularity
#PJM -j

set -euo pipefail

# 使い方:
# 1. 下の RESULT_FILES に比較したい results.jsonl の絶対パスを記入する。
# 2. RESULT_LABELS にグラフの凡例名を同じ順番で記入する。
# 3. bash scripts/genkai/plot_results_comparison.sh を実行する。
RESULT_FILES=(
    "${HOME}/experiments/autoresearch/history-1/20260911_174435/results.jsonl"
    "${HOME}/experiments/autoresearch/history-10/20260911_175929/results.jsonl"
    "${HOME}/experiments/autoresearch/history-all/20260914_233436/results.jsonl"
)
RESULT_LABELS=(
    "history-1"
    "history-10"
    "history-all(baseline)"
)

WORKDIR="${WORKDIR:-${HOME}/projects/autoresearch}"
IMAGE="${IMAGE:-${HOME}/nlp-singularity/nlp-singularity.sif}"
METRIC="${METRIC:-val_bpb}"
# Comparisons are separate from individual experiment outputs.
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/experiments/autoresearch/comparisons}"
SAFE_METRIC="${METRIC//[^a-zA-Z0-9._-]/_}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/parallel_comparison_${SAFE_METRIC}_$(date +%Y%m%d_%H%M%S).png}"

if (( ${#RESULT_FILES[@]} < 2 )); then
    echo "RESULT_FILES に2つ以上の results.jsonl を指定してください。" >&2
    exit 1
fi

for result_file in "${RESULT_FILES[@]}"; do
    if [[ -z "${result_file}" || ! -f "${result_file}" ]]; then
        echo "RESULT_FILES の空欄またはファイルが存在しないパスを修正してください: ${result_file:-<empty>}" >&2
        exit 1
    fi
done

if (( ${#RESULT_LABELS[@]} != ${#RESULT_FILES[@]} )); then
    echo "RESULT_LABELS は RESULT_FILES と同じ個数にしてください。" >&2
    exit 1
fi

module load singularity-ce

# Bind input directories so results outside the project are also accessible.
mkdir -p "$(dirname "${OUTPUT_FILE}")"
OUTPUT_PARENT="$(cd "$(dirname "${OUTPUT_FILE}")" && pwd)"
OUTPUT_FILE="${OUTPUT_PARENT}/$(basename "${OUTPUT_FILE}")"
BIND_ARGS=(--bind "${WORKDIR}:${WORKDIR}" --bind "${PWD}:${PWD}" --bind "${OUTPUT_PARENT}:${OUTPUT_PARENT}")
for argument in "${RESULT_FILES[@]}"; do
    if [[ -f "${argument}" ]]; then
        input_parent="$(cd "$(dirname "${argument}")" && pwd)"
        BIND_ARGS+=(--bind "${input_parent}:${input_parent}")
    fi
done
singularity exec \
    "${BIND_ARGS[@]}" \
    --pwd "${PWD}" \
    "${IMAGE}" \
    uv run --project "${WORKDIR}" "${WORKDIR}/plot_results_comparison.py" \
    --metric "${METRIC}" \
    --labels "${RESULT_LABELS[@]}" \
    -o "${OUTPUT_FILE}" \
    "${RESULT_FILES[@]}"

echo "Output: ${OUTPUT_FILE}"
