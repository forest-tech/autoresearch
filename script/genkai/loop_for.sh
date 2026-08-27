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

IMAGE=/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif
WORKDIR=/home/pj24001974/ku50001532/projects/autoresearch

NUM_ITERATIONS=3

RESULT_FILE="${WORKDIR}/result.jsonl"
TRAIN_LOG="${WORKDIR}/train.log"


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
        --bind "${WORKDIR}:${WORKDIR}" \
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
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        bash -lc "
            codex exec --skip-git-repo-check '
現在は実験 iteration ${i} が終了した直後です。

まず train.log を確認し、
今回の学習結果と評価指標を読み取ってください。

次に、今回使用した主要な学習パラメータもコードや設定ファイルから確認し、
result.jsonl に今回の実験結果を1行追記してください。

result.jsonl が存在しない場合は新しく作成してください。
既に存在する場合は、これまでの結果を絶対に削除・上書きせず、
末尾に今回の結果だけを追記してください。

result.jsonl には少なくとも以下の情報が分かるようにしてください。

- iteration
- 今回使用した主要な学習パラメータ
- 評価指標
- 評価結果

今回の iteration は ${i} です。

その後、result.jsonl に記録されている
これまでの実験結果を比較してください。

過去の結果を参考に、
次の uv run train.py の実行で性能が改善する可能性が高くなるように
学習パラメータを調整してください。

条件:
- train.log から今回の結果を正しく読み取ること
- result.jsonl に今回の結果を必ず記録すること
- result.jsonl の過去の記録は削除・上書きしないこと
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