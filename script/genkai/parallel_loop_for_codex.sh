#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:29:00
#PJM -L jobenv=singularity
#PJM -j

set -euo pipefail

module load singularity-ce

# ============================================================
# Configuration
# ============================================================
IMAGE=/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif
WORKDIR=/home/pj24001974/ku50001532/projects/autoresearch
WORKER_SCRIPT="${WORKDIR}/script/genkai/worker.sh"

NUM_ROUNDS=10
NUM_WORKERS=4

DATE=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${WORKDIR}/results/${DATE}"
RESULT_FILE="${RUN_ROOT}/results.jsonl"
WORKTREE_ROOT="${WORKDIR}/worktrees/${DATE}"

cd "${WORKDIR}"

# ============================================================
# Helper: cleanup worktrees
# ============================================================
declare -a ACTIVE_WORKTREES=()

cleanup_worktrees() {
    local wt
    for wt in "${ACTIVE_WORKTREES[@]:-}"; do
        if [[ -n "${wt}" && -e "${wt}/.git" ]]; then
            git worktree remove --force "${wt}" >/dev/null 2>&1 || true
        fi
    done
    git worktree prune >/dev/null 2>&1 || true
}

trap cleanup_worktrees EXIT

register_worktree() {
    ACTIVE_WORKTREES+=("$1")
}

remove_worktree() {
    local wt="$1"
    git worktree remove --force "${wt}" >/dev/null
}

create_worktree() {
    local wt="$1"
    local commit="$2"

    mkdir -p "$(dirname "${wt}")"
    git worktree add --detach "${wt}" "${commit}" >/dev/null
    register_worktree "${wt}"
}

# ============================================================
# Helper: results.jsonl
# Only the coordinator writes this file.
# ============================================================
append_result() {
    local iteration="$1"
    local round="$2"
    local worker="$3"
    local base_commit="$4"
    local commit="$5"
    local status="$6"
    local description="$7"
    local worker_result="$8"
    local config_file="$9"
    local log_rel="${10}"
    local patch_rel="${11}"
    local config_rel="${12}"

    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        python - \
        "${RESULT_FILE}" \
        "${iteration}" \
        "${round}" \
        "${worker}" \
        "${base_commit}" \
        "${commit}" \
        "${status}" \
        "${description}" \
        "${worker_result}" \
        "${config_file}" \
        "${log_rel}" \
        "${patch_rel}" \
        "${config_rel}" <<'PY'
import json
import sys

(
    result_file,
    iteration,
    round_id,
    worker,
    base_commit,
    commit,
    status,
    description,
    worker_result_file,
    config_file,
    log_rel,
    patch_rel,
    config_rel,
) = sys.argv[1:]

with open(worker_result_file, encoding="utf-8") as f:
    run = json.load(f)

with open(config_file, encoding="utf-8") as f:
    config = json.load(f)

worker_id = int(worker)

record = {
    "iteration": int(iteration),
    "round": int(round_id),
    "worker": worker_id if worker_id >= 0 else None,
    "gpu": run.get("gpu"),
    "base_commit": base_commit or None,
    "commit": commit or None,
    "val_bpb": run.get("val_bpb"),
    "memory_gb": run.get("memory_gb"),
    "train_exit": run.get("train_exit"),
    "status": status,
    "description": description,
    "hyperparameters": config.get("hyperparameters", {}),
    "fixed_config": config.get("fixed_config", {}),
    "train_sha256": config.get("train_sha256"),
    "log": log_rel,
    "patch": patch_rel,
    "config": config_rel,
}

with open(result_file, "a", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False)
    f.write("\n")
PY
}

validate_jsonl() {
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
            raise SystemExit(f"[ERROR] invalid JSONL at line {lineno}: {e}")
PY
}

get_result_val() {
    local result_file="$1"
    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        "${IMAGE}" \
        python - "${result_file}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
v = d.get("val_bpb")
if v is not None:
    print(v)
PY
}

select_winner() {
    local best_val="$1"
    shift

    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        "${IMAGE}" \
        python - "${best_val}" "$@" <<'PY'
import json
import sys

best = float(sys.argv[1])
winner = None
winner_val = best

for path in sys.argv[2:]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    if d.get("run_status") != "ok":
        continue

    value = d.get("val_bpb")
    if value is None:
        continue

    if value < winner_val:
        winner_val = value
        winner = d["worker"]

if winner is not None:
    print(winner)
PY
}

# ============================================================
# Helper: Codex candidate generation
# ============================================================
generate_candidate() {
    local round="$1"
    local worker="$2"
    local base_commit="$3"
    local best_val="$4"
    local wt="$5"
    local worker_dir="$6"
    local candidates_file="$7"

    local prompt_file="${worker_dir}/codex_prompt.txt"
    local codex_message="${worker_dir}/codex_message.txt"
    local codex_stdout="${worker_dir}/codex.stdout"
    local codex_stderr="${worker_dir}/codex.stderr"
    local patch_file="${worker_dir}/change.patch"
    local description_file="${worker_dir}/description.txt"

    mkdir -p "${worker_dir}"

    cat > "${prompt_file}" <<EOF_PROMPT
You are preparing exactly one candidate experiment for an autonomous
LLM training loop on an HPC environment.

Current round: ${round}
Current worker: ${worker}
Current accepted commit: ${base_commit}
Current best val_bpb: ${best_val}

Primary metric:
val_bpb (lower is better)

Experiment history (read only):
${RESULT_FILE}

Candidates already proposed in this round (read only):
${candidates_file}

Artifacts from this run may be inspected under (read only):
${RUN_ROOT}

The coordinator owns:
- training execution
- results.jsonl
- keep/discard decisions
- git commits
- worktree management
- experiment artifact management

Your task is ONLY to choose one new candidate experiment and edit train.py
in the current worktree.

Instructions:
- Read README.md, prepare.py, train.py, and ${RESULT_FILE}.
- Read ${candidates_file} before choosing the candidate.
- Choose a candidate different from candidates already proposed in this round.
- Use previous keep/discard results and recorded hyperparameters to avoid repeating unsuccessful experiments.
- Compare the current accepted configuration with previous candidate configurations.
- Choose one change that has a plausible chance to lower val_bpb.
- Prefer one-factor-at-a-time and small/local changes initially.
- Edit ONLY train.py in the current worktree.
- Do NOT run uv run train.py.
- Do NOT edit prepare.py.
- Do NOT edit program.md.
- Do NOT edit ${RESULT_FILE}.
- Do NOT edit anything under ${RUN_ROOT}.
- Do NOT edit scripts.
- Do NOT modify .git.
- Do NOT run git commit/reset/checkout/restore.
- Leave train.py runnable as-is.

Final response:
output only a short one-line description of the experiment.
Do not use tabs.
EOF_PROMPT

    echo "[CODEX] round=${round} worker=${worker} generating candidate"

    if ! singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${wt}" \
        "${IMAGE}" \
        bash -lc \
        "codex exec \
            --sandbox danger-full-access \
            --skip-git-repo-check \
            -o '${codex_message}' \
            - < '${prompt_file}'" \
        >"${codex_stdout}" \
        2>"${codex_stderr}"
    then
        echo "[ERROR] Codex failed: round=${round} worker=${worker}" >&2
        return 1
    fi

    # Only train.py may be modified. Catch tracked, staged, and untracked files.
    local changed_tracked
    local changed_staged
    local untracked
    local all_changed

    changed_tracked=$(git -C "${wt}" diff --name-only)
    changed_staged=$(git -C "${wt}" diff --cached --name-only)
    untracked=$(git -C "${wt}" ls-files --others --exclude-standard)

    all_changed=$(
        printf '%s\n%s\n%s\n' \
            "${changed_tracked}" \
            "${changed_staged}" \
            "${untracked}" \
        | sed '/^$/d' \
        | sort -u
    )

    if [[ "${all_changed}" != "train.py" ]]; then
        echo "[ERROR] Codex must modify only train.py." >&2
        echo "[ERROR] changed files:" >&2
        printf '%s\n' "${all_changed}" >&2
        return 1
    fi

    if git -C "${wt}" diff --quiet -- train.py; then
        echo "[ERROR] Codex did not leave an unstaged train.py diff." >&2
        return 1
    fi

    git -C "${wt}" diff --binary -- train.py > "${patch_file}"

    local description
    description=$(
        tr '\t\r\n' '   ' < "${codex_message}" \
        | tr -s ' ' \
        | sed 's/^ //; s/ $//'
    )

    [[ -n "${description}" ]] || description="candidate change"

    printf '%s\n' "${description}" > "${description_file}"
    printf 'worker %s: %s\n' "${worker}" "${description}" >> "${candidates_file}"
}

# ============================================================
# Preflight
# ============================================================
[[ -d .git ]] || {
    echo "[ERROR] git repositoryではありません"
    exit 1
}

[[ -f train.py ]] || {
    echo "[ERROR] train.py がありません"
    exit 1
}

[[ -f "${WORKER_SCRIPT}" ]] || {
    echo "[ERROR] worker script not found: ${WORKER_SCRIPT}"
    exit 1
}

# Tracked changes in the accepted worktree are not allowed.
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

mkdir -p "${RUN_ROOT}" "${WORKTREE_ROOT}"
touch "${RESULT_FILE}"
validate_jsonl

GPU_COUNT=$(
    singularity exec --nv "${IMAGE}" nvidia-smi -L \
    | grep -c '^GPU ' || true
)

if [[ "${GPU_COUNT}" -lt "${NUM_WORKERS}" ]]; then
    echo "[ERROR] ${NUM_WORKERS} GPUs required, but ${GPU_COUNT} visible" >&2
    exit 1
fi

echo "[SETUP] branch=${BRANCH}"
echo "[SETUP] GPUs=${GPU_COUNT}"
echo "[SETUP] results=${RESULT_FILE}"

# ============================================================
# Baseline: evaluate accepted HEAD once on GPU 0
# ============================================================
ITERATION=1
BASELINE_DIR="${RUN_ROOT}/baseline"
BASELINE_WT="${WORKTREE_ROOT}/baseline"
BASELINE_PATCH="${BASELINE_DIR}/change.patch"
BASELINE_RESULT="${BASELINE_DIR}/worker_result.json"
BASELINE_CONFIG="${BASELINE_DIR}/config.json"
BASE_COMMIT_FULL=$(git rev-parse HEAD)
BASE_COMMIT_SHORT=$(git rev-parse --short=7 HEAD)

mkdir -p "${BASELINE_DIR}"
: > "${BASELINE_PATCH}"

create_worktree "${BASELINE_WT}" "${BASE_COMMIT_FULL}"

bash "${WORKER_SCRIPT}" \
    -1 0 \
    "${BASELINE_WT}" \
    "${BASELINE_DIR}" \
    "${IMAGE}" \
    "${WORKDIR}" \
    >"${BASELINE_DIR}/worker.stdout" \
    2>"${BASELINE_DIR}/worker.stderr"

BEST_VAL=$(get_result_val "${BASELINE_RESULT}")

if [[ -z "${BEST_VAL}" ]]; then
    echo "[ERROR] baseline failed" >&2
    exit 1
fi

append_result \
    "${ITERATION}" \
    0 \
    -1 \
    "${BASE_COMMIT_SHORT}" \
    "${BASE_COMMIT_SHORT}" \
    "keep" \
    "baseline" \
    "${BASELINE_RESULT}" \
    "${BASELINE_CONFIG}" \
    "results/${DATE}/baseline/train.log" \
    "results/${DATE}/baseline/change.patch" \
    "results/${DATE}/baseline/config.json"

remove_worktree "${BASELINE_WT}"

echo "[BASELINE] val_bpb=${BEST_VAL} commit=${BASE_COMMIT_SHORT}"

# ============================================================
# Parallel best-of-4 rounds
# ============================================================
for ROUND in $(seq 1 "${NUM_ROUNDS}"); do
    TAG=$(printf '%03d' "${ROUND}")
    ROUND_DIR="${RUN_ROOT}/round_${TAG}"
    CANDIDATES_FILE="${ROUND_DIR}/candidates.txt"

    mkdir -p "${ROUND_DIR}"
    : > "${CANDIDATES_FILE}"

    BASE_COMMIT_FULL=$(git rev-parse HEAD)
    BASE_COMMIT_SHORT=$(git rev-parse --short=7 HEAD)

    echo
    echo "========================================"
    echo "Round ${ROUND}/${NUM_ROUNDS}"
    echo "base=${BASE_COMMIT_SHORT} best=${BEST_VAL}"
    echo "========================================"

    declare -a WT_DIRS=()
    declare -a WORKER_DIRS=()
    declare -a RESULT_FILES=()
    declare -a PATCH_FILES=()
    declare -a CONFIG_FILES=()
    declare -a DESCRIPTIONS=()

    # --------------------------------------------------------
    # 1. Generate four distinct candidates sequentially
    # --------------------------------------------------------
    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        WT_DIR="${WORKTREE_ROOT}/round_${TAG}/worker_${WORKER}"
        WORKER_DIR="${ROUND_DIR}/worker_${WORKER}"

        create_worktree "${WT_DIR}" "${BASE_COMMIT_FULL}"

        generate_candidate \
            "${ROUND}" \
            "${WORKER}" \
            "${BASE_COMMIT_SHORT}" \
            "${BEST_VAL}" \
            "${WT_DIR}" \
            "${WORKER_DIR}" \
            "${CANDIDATES_FILE}"

        WT_DIRS[${WORKER}]="${WT_DIR}"
        WORKER_DIRS[${WORKER}]="${WORKER_DIR}"
        RESULT_FILES[${WORKER}]="${WORKER_DIR}/worker_result.json"
        PATCH_FILES[${WORKER}]="${WORKER_DIR}/change.patch"
        CONFIG_FILES[${WORKER}]="${WORKER_DIR}/config.json"
        DESCRIPTIONS[${WORKER}]=$(cat "${WORKER_DIR}/description.txt")
    done

    # --------------------------------------------------------
    # 2. Run one candidate per GPU in parallel
    # --------------------------------------------------------
    declare -a PIDS=()

    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        echo "[LAUNCH] worker=${WORKER} gpu=${WORKER}"

        bash "${WORKER_SCRIPT}" \
            "${WORKER}" \
            "${WORKER}" \
            "${WT_DIRS[${WORKER}]}" \
            "${WORKER_DIRS[${WORKER}]}" \
            "${IMAGE}" \
            "${WORKDIR}" \
            >"${WORKER_DIRS[${WORKER}]}/worker.stdout" \
            2>"${WORKER_DIRS[${WORKER}]}/worker.stderr" &

        PIDS[${WORKER}]=$!
    done

    WORKER_INFRA_FAILED=0

    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        if ! wait "${PIDS[${WORKER}]}"; then
            echo "[ERROR] worker infrastructure failure: ${WORKER}" >&2
            WORKER_INFRA_FAILED=1
        fi
    done

    if [[ ${WORKER_INFRA_FAILED} -ne 0 ]]; then
        echo "[ERROR] aborting round because worker.sh itself failed" >&2
        exit 1
    fi

    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        [[ -f "${RESULT_FILES[${WORKER}]}" ]] || {
            echo "[ERROR] missing worker result: ${RESULT_FILES[${WORKER}]}" >&2
            exit 1
        }
    done

    # --------------------------------------------------------
    # 3. Select only the best candidate that beats current best
    # --------------------------------------------------------
    WINNER=$(
        select_winner \
            "${BEST_VAL}" \
            "${RESULT_FILES[@]}"
    )

    WINNER_COMMIT=""

    if [[ -n "${WINNER}" ]]; then
        # The accepted worktree must still be exactly the round base.
        CURRENT_HEAD=$(git rev-parse HEAD)
        if [[ "${CURRENT_HEAD}" != "${BASE_COMMIT_FULL}" ]]; then
            echo "[ERROR] accepted HEAD changed during round" >&2
            exit 1
        fi

        git apply --check "${PATCH_FILES[${WINNER}]}"
        git apply "${PATCH_FILES[${WINNER}]}"
        git add train.py

        WINNER_VAL=$(get_result_val "${RESULT_FILES[${WINNER}]}")

        git commit \
            -m "experiment: round ${ROUND} worker ${WINNER} val_bpb=${WINNER_VAL}" \
            >/dev/null

        WINNER_COMMIT=$(git rev-parse --short=7 HEAD)
        BEST_VAL="${WINNER_VAL}"

        echo "[KEEP] worker=${WINNER} val_bpb=${WINNER_VAL} commit=${WINNER_COMMIT}"
    else
        echo "[DISCARD] no candidate improved val_bpb=${BEST_VAL}"
    fi

    # --------------------------------------------------------
    # 4. Coordinator appends all four results sequentially
    # --------------------------------------------------------
    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        ITERATION=$((ITERATION + 1))

        RUN_STATUS=$(
            singularity exec \
                --bind "${WORKDIR}:${WORKDIR}" \
                "${IMAGE}" \
                python - "${RESULT_FILES[${WORKER}]}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("run_status", "crash"))
PY
        )

        STATUS="discard"
        COMMIT=""

        if [[ "${RUN_STATUS}" != "ok" ]]; then
            STATUS="crash"
        elif [[ -n "${WINNER}" && "${WORKER}" -eq "${WINNER}" ]]; then
            STATUS="keep"
            COMMIT="${WINNER_COMMIT}"
        fi

        append_result \
            "${ITERATION}" \
            "${ROUND}" \
            "${WORKER}" \
            "${BASE_COMMIT_SHORT}" \
            "${COMMIT}" \
            "${STATUS}" \
            "${DESCRIPTIONS[${WORKER}]}" \
            "${RESULT_FILES[${WORKER}]}" \
            "${CONFIG_FILES[${WORKER}]}" \
            "results/${DATE}/round_${TAG}/worker_${WORKER}/train.log" \
            "results/${DATE}/round_${TAG}/worker_${WORKER}/change.patch" \
            "results/${DATE}/round_${TAG}/worker_${WORKER}/config.json"
    done

    validate_jsonl

    # --------------------------------------------------------
    # 5. Remove detached worktrees after artifacts are saved
    # --------------------------------------------------------
    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        remove_worktree "${WT_DIRS[${WORKER}]}"
    done

    git worktree prune >/dev/null

    echo "[BEST] val_bpb=${BEST_VAL}"
done

echo
echo "========================================"
echo "Finished ${NUM_ROUNDS} parallel rounds"
echo "Best val_bpb: ${BEST_VAL}"
echo "Results: ${RESULT_FILE}"
echo "========================================"
