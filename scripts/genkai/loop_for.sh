#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=00:29:00
#PJM -L jobenv=singularity
#PJM -j

set -e

module load singularity-ce

# =========================
# 設定
# =========================

IMAGE="${HOME}/nlp-singularity/nlp-singularity.sif"
WORKDIR="${HOME}/projects/autoresearch"

NUM_ITERATIONS=3

# 保存先の設定: `EXP_NAME=my-experiment bash ...` で上書きできます。
EXP_NAME="${EXP_NAME:-unnamed}"
DATE=$(date +%Y%m%d_%H%M%S)
if [[ ! "${EXP_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "[ERROR] invalid EXP_NAME: ${EXP_NAME}" >&2
    exit 1
fi
if [[ ! "${DATE}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    echo "[ERROR] set DATE in this script to YYYYMMDD_HHMMSS" >&2
    exit 1
fi
RUN_ROOT="${HOME}/experiments/autoresearch/${EXP_NAME}/${DATE}"
mkdir -p "${RUN_ROOT}"
RESULT_FILE="${RUN_ROOT}/result.jsonl"
TRAIN_LOG="${RUN_ROOT}/train.log"


# =========================
# 実験ループ
# =========================

for i in $(seq 1 "${NUM_ITERATIONS}"); do
    echo "========================================"
    echo "Iteration ${i}/${NUM_ITERATIONS}"
    echo "========================================"

    # -------------------------
    # 1. train
    # -------------------------
    echo "[TRAIN] Starting training..."

    singularity exec \
        --nv \
        --bind "${WORKDIR}:${WORKDIR}" --bind "${RUN_ROOT}:${RUN_ROOT}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        bash -lc "uv run train.py" \
        2>&1 | tee "${TRAIN_LOG}"

    echo "[TRAIN] Training finished."

    # -------------------------
    # 2. Codex
    #    - train.log を解析
    #    - result.jsonl に結果を追記
    #    - 次のパラメータを調整
    # -------------------------
    echo "[CODEX] Starting analysis and parameter adjustment..."

    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" --bind "${RUN_ROOT}:${RUN_ROOT}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        bash -lc "
            codex exec --skip-git-repo-check '
現在は実験 iteration ${i} が終了した直後です。

まず ${TRAIN_LOG} を確認し、
今回の学習結果と評価指標を読み取ってください。

次に、今回使用した主要な学習パラメータもコードや設定ファイルから確認し、
${RESULT_FILE} に今回の実験結果を1行追記してください。

${RESULT_FILE} が存在しない場合は新しく作成してください。
既に存在する場合は、これまでの結果を絶対に削除・上書きせず、
末尾に今回の結果だけを追記してください。

${RESULT_FILE} には少なくとも以下の情報が分かるようにしてください。

- iteration
- 今回使用した主要な学習パラメータ
- 評価指標
- 評価結果

今回の iteration は ${i} です。

その後、${RESULT_FILE} に記録されている
これまでの実験結果を比較してください。

過去の結果を参考に、
次の uv run train.py の実行で性能が改善する可能性が高くなるように
学習パラメータを調整してください。

条件:
- ${TRAIN_LOG} から今回の結果を正しく読み取ること
- ${RESULT_FILE} に今回の結果を必ず記録すること
- ${RESULT_FILE} の過去の記録は削除・上書きしないこと
- 過去に試したパラメータ設定を可能な限り繰り返さないこと
- 次のループで uv run train.py をそのまま実行できる状態にすること
- 学習コードの大幅な変更は避け、パラメータ調整を中心に行うこと
'"

    echo "[CODEX] Finished."

    # result.jsonl をログにも表示
    if [ -f "${RESULT_FILE}" ]; then
        echo "[RESULT] Current result.jsonl:"
        cat "${RESULT_FILE}"
    else
        echo "[WARNING] result.jsonl was not created."
        exit 1
    fi

done

echo "========================================"
echo "All experiments finished."
echo "========================================"
