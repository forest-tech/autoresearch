```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Git repository information
# ============================================================

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "[ERROR] not inside a Git repository." >&2
    exit 1
}

# origin URL からプロジェクト名を取得
# e.g.
#   git@github.com:forest-tech/autoresearch.git -> autoresearch
#   https://github.com/forest-tech/autoresearch.git -> autoresearch
REMOTE_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"

if [[ -n "$REMOTE_URL" ]]; then
    PROJECT_NAME="$(basename "$REMOTE_URL" .git)"
else
    # origin がない場合はリポジトリルート名を使用
    PROJECT_NAME="$(basename "$REPO_ROOT")"
fi

# ============================================================
# Branch / experiment name
# ============================================================

BRANCH="$(git -C "$REPO_ROOT" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"

if [[ -z "$BRANCH" ]]; then
    # detached HEAD
    SHORT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    EXP_NAME="detached-${SHORT_SHA}"

elif [[ "$BRANCH" == "${PROJECT_NAME}/"* ]]; then
    # e.g.
    # autoresearch/history-1 -> history-1
    # myproject/exp-a         -> exp-a
    EXP_NAME="${BRANCH#"${PROJECT_NAME}/"}"

else
    # master, main, feature/foo など
    # '/' はディレクトリ階層にならないよう '-' に変換
    EXP_NAME="${BRANCH//\//-}"
fi

# ============================================================
# Experiment directory
# ============================================================

DATE="$(date +%Y%m%d_%H%M%S)"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${HOME}/experiments}"
RESULT_DIR="${EXPERIMENT_ROOT}/${PROJECT_NAME}/${EXP_NAME}/${DATE}"

mkdir -p "$RESULT_DIR"

# ============================================================
# Job configuration
# ============================================================

JOB_SCRIPT="${JOB_SCRIPT:-${REPO_ROOT}/scripts/genkai/parallel_loop_for_codex.sh}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] job script not found: $JOB_SCRIPT" >&2
    exit 1
fi

JOB_NAME="$EXP_NAME"
OUT_FILE="${RESULT_DIR}/job.out"

# ジョブ側へ同じ値を渡す
export PROJECT_NAME
export EXP_NAME
export DATE
export RESULT_DIR

# ============================================================
# Show configuration
# ============================================================

echo "[INFO] project    : $PROJECT_NAME"
echo "[INFO] repository : $REPO_ROOT"
echo "[INFO] branch     : ${BRANCH:-detached HEAD}"
echo "[INFO] exp_name   : $EXP_NAME"
echo "[INFO] date       : $DATE"
echo "[INFO] result_dir : $RESULT_DIR"
echo "[INFO] job_name   : $JOB_NAME"
echo "[INFO] job_script : $JOB_SCRIPT"
echo

# ============================================================
# Submit
# ============================================================

SUBMIT_OUTPUT="$(
    pjsub \
        -X \
        -N "$JOB_NAME" \
        -o "$OUT_FILE" \
        -j \
        "$JOB_SCRIPT"
)"

echo "$SUBMIT_OUTPUT"

# ============================================================
# Extract Job ID
# ============================================================

JOB_ID="$(
    printf '%s\n' "$SUBMIT_OUTPUT" |
        sed -n 's/.*Job \([0-9][0-9]*\).*/\1/p' |
        tail -n 1
)"

# ============================================================
# Save metadata
# ============================================================

cat > "${RESULT_DIR}/job.info" <<EOF
JOB_ID=${JOB_ID}
JOB_NAME=${JOB_NAME}
PROJECT_NAME=${PROJECT_NAME}
BRANCH=${BRANCH:-DETACHED_HEAD}
EXP_NAME=${EXP_NAME}
DATE=${DATE}
REPO_ROOT=${REPO_ROOT}
RESULT_DIR=${RESULT_DIR}
JOB_SCRIPT=${JOB_SCRIPT}
EOF

echo
echo "[INFO] submitted"
echo "[INFO] job_id   : ${JOB_ID:-unknown}"
echo "[INFO] job.out  : $OUT_FILE"
echo "[INFO] job.info : ${RESULT_DIR}/job.info"
```

