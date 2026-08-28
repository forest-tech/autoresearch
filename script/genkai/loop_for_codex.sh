#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -L jobenv=singularity
#PJM -j

set -euo pipefail

module load singularity-ce

IMAGE=/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif
WORKDIR=/home/pj24001974/ku50001532/projects/autoresearch

NUM_ITERATIONS=10

DATE=$(date +%Y%m%d_%H%M%S)

RUN_ROOT="${WORKDIR}/results/${DATE}"
RESULT_FILE="${RUN_ROOT}/results.jsonl"

RESULT_REL="${RESULT_FILE#${WORKDIR}/}"
RUN_ROOT_REL="${RUN_ROOT#${WORKDIR}/}"

cd "${WORKDIR}"

# ============================================================
# helper
# ============================================================

# results.jsonl に1実験を安全に追記する
append_result() {
    local iteration="$1"
    local commit="$2"
    local val_bpb="$3"
    local memory_gb="$4"
    local status="$5"
    local description="$6"
    local log="$7"
    local patch="$8"

    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        python - \
        "${RESULT_FILE}" \
        "${iteration}" \
        "${commit}" \
        "${val_bpb}" \
        "${memory_gb}" \
        "${status}" \
        "${description}" \
        "${log}" \
        "${patch}" <<'PY'
import json
import sys

(
    result_file,
    iteration,
    commit,
    val_bpb,
    memory_gb,
    status,
    description,
    log,
    patch,
) = sys.argv[1:]

record = {
    "iteration": int(iteration),
    "commit": commit if commit else None,
    "val_bpb": float(val_bpb) if val_bpb else None,
    "memory_gb": float(memory_gb) if memory_gb else None,
    "status": status,
    "description": description,
    "log": log,
    "patch": patch,
}

with open(result_file, "a", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False)
    f.write("\n")
PY
}


# results.jsonl から最後のiterationを取得
get_last_iteration() {
    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        "${IMAGE}" \
        python - "${RESULT_FILE}" <<'PY'
import json
import os
import sys

path = sys.argv[1]

last = 0

if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            last = max(last, record["iteration"])

print(last)
PY
}


# 現在のbest val_bpbを取得
get_best_val() {
    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        "${IMAGE}" \
        python - "${RESULT_FILE}" <<'PY'
import json
import os
import sys

path = sys.argv[1]

best = None

if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("status") != "keep":
                continue

            value = record.get("val_bpb")

            if value is None:
                continue

            if best is None or value < best:
                best = value

if best is not None:
    print(best)
PY
}

snapshot_config() {
    local output_file="$1"

    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        python - \
        "${WORKDIR}/train.py" \
        "${WORKDIR}/prepare.py" \
        "${output_file}" <<'PY'
import ast
import hashlib
import json
import operator
import sys

train_path, prepare_path, output_path = sys.argv[1:]


# ------------------------------------------------------------
# simple Python expression evaluator
# ------------------------------------------------------------

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate(node, env):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Tuple):
        return tuple(evaluate(x, env) for x in node.elts)

    if isinstance(node, ast.List):
        return [evaluate(x, env) for x in node.elts]

    if isinstance(node, ast.Dict):
        return {
            evaluate(k, env): evaluate(v, env)
            for k, v in zip(node.keys, node.values)
        }

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(node.id)

    if isinstance(node, ast.BinOp):
        op = BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(type(node.op).__name__)

        return op(
            evaluate(node.left, env),
            evaluate(node.right, env),
        )

    if isinstance(node, ast.UnaryOp):
        op = UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(type(node.op).__name__)

        return op(evaluate(node.operand, env))

    raise ValueError(type(node).__name__)


# ------------------------------------------------------------
# Extract "# Hyperparameters" block from train.py
# ------------------------------------------------------------

with open(train_path, encoding="utf-8") as f:
    train_source = f.read()

lines = train_source.splitlines()

start_line = None
end_line = None

for i, line in enumerate(lines, 1):
    if "Hyperparameters (edit these directly" in line:
        start_line = i

    if start_line is not None and i > start_line:
        if "Setup: tokenizer, model, optimizer" in line:
            end_line = i
            break

if start_line is None or end_line is None:
    raise SystemExit(
        "[ERROR] Hyperparameter section not found in train.py"
    )

tree = ast.parse(train_source)

hyperparameters = {}
env = {}

for node in tree.body:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        continue

    if not (start_line < node.lineno < end_line):
        continue

    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            continue

        target = node.targets[0]
        value_node = node.value

    else:
        target = node.target
        value_node = node.value

    if not isinstance(target, ast.Name):
        continue

    name = target.id

    if not name.isupper():
        continue

    try:
        value = evaluate(value_node, env)
    except Exception:
        # Codexが複雑な式を追加した場合も、
        # 少なくとも式自体は記録する
        value = {
            "expression": ast.unparse(value_node)
        }

    env[name] = value
    hyperparameters[name] = value


# ------------------------------------------------------------
# Fixed experimental configuration from prepare.py
# ------------------------------------------------------------

with open(prepare_path, encoding="utf-8") as f:
    prepare_source = f.read()

prepare_tree = ast.parse(prepare_source)
prepare_env = {}
fixed_config = {}

wanted = {
    "MAX_SEQ_LEN",
    "TIME_BUDGET",
}

for node in prepare_tree.body:
    if not isinstance(node, ast.Assign):
        continue

    if len(node.targets) != 1:
        continue

    target = node.targets[0]

    if not isinstance(target, ast.Name):
        continue

    try:
        value = evaluate(node.value, prepare_env)
    except Exception:
        continue

    prepare_env[target.id] = value

    if target.id in wanted:
        fixed_config[target.id] = value


# ------------------------------------------------------------
# Code fingerprint
# ------------------------------------------------------------

train_sha256 = hashlib.sha256(
    train_source.encode("utf-8")
).hexdigest()


record = {
    "hyperparameters": hyperparameters,
    "fixed_config": fixed_config,
    "train_sha256": train_sha256,
}


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        record,
        f,
        ensure_ascii=False,
        indent=2,
    )
    f.write("\n")
PY
}

# ============================================================
# preflight
# ============================================================

[[ -d .git ]] || {
    echo "[ERROR] git repositoryではありません"
    exit 1
}

[[ -f train.py ]] || {
    echo "[ERROR] train.py がありません"
    exit 1
}

git diff --quiet && git diff --cached --quiet || {
    echo "[ERROR] 未commitの変更があります"
    exit 1
}

BRANCH=$(git branch --show-current)

case "${BRANCH}" in
    autoresearch/*)
        ;;
    *)
        echo "[ERROR] autoresearch/* ブランチで実行してください"
        echo "current branch: ${BRANCH}"
        exit 1
        ;;
esac

mkdir -p "${RUN_ROOT}"

touch "${RESULT_FILE}"

# JSONLとして壊れていないか確認
singularity exec \
    --bind "${WORKDIR}:${WORKDIR}" \
    "${IMAGE}" \
    python - "${RESULT_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        if not line.strip():
            continue

        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"[ERROR] invalid JSONL at line {lineno}: {e}"
            )
PY

LAST_ITER=$(get_last_iteration)
BEST_VAL=$(get_best_val)

# ============================================================
# experiment loop
# ============================================================

for n in $(seq 1 "${NUM_ITERATIONS}"); do

    ITER=$((LAST_ITER + n))
    TAG=$(printf '%03d' "${ITER}")

    ITER_DIR="${RUN_ROOT}/iter_${TAG}"

    TRAIN_LOG="${ITER_DIR}/train.log"
    PATCH_FILE="${ITER_DIR}/change.patch"
    CONFIG_FILE="${ITER_DIR}/config.json"

    PROMPT_FILE="${ITER_DIR}/codex_prompt.txt"
    CODEX_MESSAGE="${ITER_DIR}/codex_message.txt"
    CODEX_STDOUT="${ITER_DIR}/codex.stdout"
    CODEX_STDERR="${ITER_DIR}/codex.stderr"

    mkdir -p "${ITER_DIR}"

    echo
    echo "========================================"
    echo "Iteration ${ITER}"
    echo "========================================"

    IS_BASELINE=0
    DESCRIPTION="baseline"

    # ========================================================
    # Candidate generation
    # ========================================================

    if [[ -z "${BEST_VAL}" ]]; then

        IS_BASELINE=1
        : > "${PATCH_FILE}"

        echo "[BASELINE] current train.py"

    else

        BASE_COMMIT=$(git rev-parse --short=7 HEAD)

        cat > "${PROMPT_FILE}" <<EOF
You are preparing exactly one candidate experiment for an autonomous
LLM training loop on HPC environment.

Current iteration: ${ITER}
Current accepted commit: ${BASE_COMMIT}

Primary metric:
val_bpb (lower is better)

Current experiment run directory:
${RUN_ROOT_REL}

Current experiment history:
${RESULT_REL}

The shell script owns:

- training execution
- ${RESULT_REL}
- keep/discard decisions
- git commits
- experiment artifact management

Your task is ONLY to choose the next experiment and edit train.py.

Instructions:

- Read README.md, prepare.py, train.py, and ${RESULT_REL}.
- Inspect artifacts from the current run under ${RUN_ROOT_REL}/ when useful.
- Use previous keep/discard results and recorded hyperparameters to avoid repeating unsuccessful experiments.
- Compare the current accepted configuration with previous candidate configurations when deciding the next experiment.
- Choose one change that has a plausible chance to lower val_bpb.
- Prefer one-factor-at-a-time and small/local changes initially.
- Avoid repeating previous experiments.
- Edit ONLY train.py.
- Do NOT run uv run train.py.
- Do NOT edit prepare.py.
- Do NOT edit program.md.
- Do NOT edit ${RESULT_REL}.
- Do NOT edit files under results/.
- Do NOT edit scripts.
- Do NOT modify .git.
- Do NOT run git commit/reset/checkout/restore.
- Leave train.py runnable as-is.

Final response:
output only a short one-line description of the experiment.
Do not use tabs.
EOF

        echo "[CODEX] generating candidate"

        if ! singularity exec \
            --bind "${WORKDIR}:${WORKDIR}" \
            --pwd "${WORKDIR}" \
            "${IMAGE}" \
            bash -lc \
            "codex exec \
                --sandbox danger-full-access \
                --skip-git-repo-check \
                -o '${CODEX_MESSAGE}' \
                - < '${PROMPT_FILE}'" \
            >"${CODEX_STDOUT}" \
            2>"${CODEX_STDERR}"
        then
            echo "[ERROR] Codex failed"
            git restore --worktree -- .
            exit 1
        fi

        CHANGED=$(git diff --name-only)

        if [[ "${CHANGED}" != "train.py" ]]; then
            echo "[ERROR] Codex modified unexpected files:"
            printf '%s\n' "${CHANGED}"

            git restore --worktree -- .
            exit 1
        fi

        git diff -- train.py > "${PATCH_FILE}"

        DESCRIPTION=$(
            tr '\t\r\n' '   ' < "${CODEX_MESSAGE}" \
            | tr -s ' ' \
            | sed 's/^ //; s/ $//'
        )

        [[ -n "${DESCRIPTION}" ]] \
            || DESCRIPTION="candidate change"

    fi

    # ========================================================
    # Snapshot experiment configuration
    # ========================================================

    echo "[CONFIG] snapshotting experiment configuration"

    snapshot_config "${CONFIG_FILE}"

    # ========================================================
    # Training
    # ========================================================

    echo "[TRAIN] starting"

    TRAIN_EXIT=0

    timeout 600 singularity exec \
        --nv \
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        bash -lc "uv run train.py" \
        >"${TRAIN_LOG}" 2>&1 \
        || TRAIN_EXIT=$?

    # ========================================================
    # Result extraction
    # ========================================================

    VAL_BPB=$(
        awk '/^val_bpb:/ {v=$2} END {print v}' \
        "${TRAIN_LOG}"
    )

    PEAK_VRAM_MB=$(
        awk '/^peak_vram_mb:/ {v=$2} END {print v}' \
        "${TRAIN_LOG}"
    )

    # ========================================================
    # Crash
    # ========================================================

    if [[ ${TRAIN_EXIT} -ne 0 || -z "${VAL_BPB}" ]]; then

        echo "[TRAIN] failed (exit=${TRAIN_EXIT})"

        append_result \
            "${ITER}" \
            "" \
            "" \
            "" \
            "crash" \
            "${DESCRIPTION}" \
            "results/iter_${TAG}/train.log" \
            "results/iter_${TAG}/change.patch"

        if [[ ${IS_BASELINE} -eq 0 ]]; then
            git restore --worktree -- train.py
            continue
        fi

        echo "[ERROR] baseline failed"
        exit 1
    fi

    MEMORY_GB=$(
        awk -v mb="${PEAK_VRAM_MB:-0}" \
            'BEGIN {printf "%.1f", mb/1024}'
    )

    # ========================================================
    # keep / discard
    # ========================================================

    if [[ ${IS_BASELINE} -eq 1 ]]; then

        STATUS="keep"
        COMMIT=$(git rev-parse --short=7 HEAD)
        BEST_VAL="${VAL_BPB}"

    elif awk \
        -v v="${VAL_BPB}" \
        -v b="${BEST_VAL}" \
        'BEGIN {exit !(v < b)}'
    then

        STATUS="keep"

        git add train.py
        git commit \
            -m "experiment: iter ${ITER} val_bpb=${VAL_BPB}" \
            >/dev/null

        COMMIT=$(git rev-parse --short=7 HEAD)
        BEST_VAL="${VAL_BPB}"

    else

        STATUS="discard"
        COMMIT=""

        git restore --worktree -- train.py

    fi

    # ========================================================
    # results.jsonl
    # ========================================================

    append_result \
        "${ITER}" \
        "${COMMIT}" \
        "${VAL_BPB}" \
        "${MEMORY_GB}" \
        "${STATUS}" \
        "${DESCRIPTION}" \
        "results/iter_${TAG}/train.log" \
        "results/iter_${TAG}/change.patch"

    echo "[RESULT] val_bpb=${VAL_BPB}"
    echo "[RESULT] memory=${MEMORY_GB} GB"
    echo "[RESULT] status=${STATUS}"
    echo "[BEST]   val_bpb=${BEST_VAL}"

done

echo
echo "========================================"
echo "Finished ${NUM_ITERATIONS} experiments"
echo "Results: ${RESULT_FILE}"
echo "========================================"