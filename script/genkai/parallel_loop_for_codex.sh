#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -L jobenv=singularity
#PJM -j

set -euo pipefail

module load singularity-ce

# =========================
# custom
# =========================
WORKDIR="/home/pj24001974/ku50001532/projects/autoresearch-history-10"
HISTORY_MODE="recent"
HISTORY_LIMIT="10"
# -------------------------
IMAGE="${IMAGE:-/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif}"
WORKDIR="${WORKDIR:-/home/pj24001974/ku50001532/projects/autoresearch}"
WORKER_SCRIPT="${WORKDIR}/script/genkai/worker.sh"
NUM_ROUNDS="${NUM_ROUNDS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"

CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-medium}"

PRIMARY_METRIC="${PRIMARY_METRIC:-val_bpb}"
OBJECTIVE_DIRECTION="${OBJECTIVE_DIRECTION:-min}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-prompts/candidate_default.txt}"
HISTORY_MODE="${HISTORY_MODE:-all}"
HISTORY_LIMIT="${HISTORY_LIMIT:-10}"

if [[ "${PROMPT_TEMPLATE}" != /* ]]; then
    PROMPT_TEMPLATE="${WORKDIR}/${PROMPT_TEMPLATE}"
fi

DATE=${DATE:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT="${WORKDIR}/results/${DATE}"
RESULT_FILE="${RUN_ROOT}/results.jsonl"
RUN_CONFIG="${RUN_ROOT}/run_config.json"
WORKTREE_ROOT="${WORKDIR}/worktrees/${DATE}"
EXPERIMENT_TOOL="${WORKDIR}/experiment_utils.py"

cd "${WORKDIR}"

experiment_tool() {
    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        bash -lc "uv run '${EXPERIMENT_TOOL}' '$@'"
}

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

create_worktree() {
    local wt="$1" commit="$2"
    mkdir -p "$(dirname "${wt}")"
    git worktree add --detach "${wt}" "${commit}" >/dev/null
    ACTIVE_WORKTREES+=("${wt}")
}

remove_worktree() {
    git worktree remove --force "$1" >/dev/null
}

append_result() {
    local iteration="$1" round="$2" worker="$3" base_commit="$4" commit="$5"
    local status="$6" description="$7" worker_result="$8" config_file="$9"
    local artifact_prefix="${10}" prompt_metadata="${11}" candidate_generated="${12}"
    local -a args=(
        append-result --results "${RESULT_FILE}"
        --run-result "${worker_result}" --config "${config_file}"
        --prompt-metadata "${prompt_metadata}" --iteration "${iteration}"
        --round "${round}" --base-commit "${base_commit}"
        --status "${status}" --description "${description}"
        --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}"
        --log "${artifact_prefix}/train.log"
        --patch "${artifact_prefix}/change.patch"
        --config-artifact "${artifact_prefix}/config.json"
    )
    [[ "${worker}" -ge 0 ]] && args+=(--worker "${worker}")
    [[ -n "${commit}" ]] && args+=(--commit "${commit}")
    if [[ "${candidate_generated}" -eq 1 ]]; then
        args+=(--prompt "${artifact_prefix}/codex_prompt.txt" --history "${artifact_prefix}/history_context.jsonl")
    fi
    experiment_tool "${args[@]}"
}

generate_candidate() {
    local round="$1" worker="$2" base_commit="$3" best_val="$4"
    local wt="$5" worker_dir="$6" candidates_file="$7"
    local prompt_file="${worker_dir}/codex_prompt.txt"
    local prompt_metadata="${worker_dir}/prompt_metadata.json"
    local history_file="${worker_dir}/history_context.jsonl"
    local codex_message="${worker_dir}/codex_message.txt"
    local codex_stdout="${worker_dir}/codex.stdout"
    local codex_stderr="${worker_dir}/codex.stderr"
    local patch_file="${worker_dir}/change.patch"
    local description_file="${worker_dir}/description.txt"

    mkdir -p "${worker_dir}"
    experiment_tool render-prompt \
        --template "${PROMPT_TEMPLATE}" --results "${RESULT_FILE}" \
        --output "${prompt_file}" --history-output "${history_file}" \
        --metadata-output "${prompt_metadata}" \
        --history-mode "${HISTORY_MODE}" --history-limit "${HISTORY_LIMIT}" \
        --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}" \
        --iteration "${round}" --worker "${worker}" --base-commit "${base_commit}" \
        --current-best "${best_val}" --candidates-file "${candidates_file}" \
        --repo-root "${WORKDIR}"

    echo "[CODEX] round=${round} worker=${worker} generating candidate"
    if ! singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" --pwd "${wt}" "${IMAGE}" \
        bash -lc \
        "codex exec \
            --model '${CODEX_MODEL}' \
            -c 'model_reasoning_effort=${CODEX_REASONING_EFFORT}' \
            --sandbox danger-full-access \
            --skip-git-repo-check \
            -o '${codex_message}' \
            - < '${prompt_file}'" \
        >"${codex_stdout}" 2>"${codex_stderr}"
    then
        echo "[ERROR] Codex failed: round=${round} worker=${worker}" >&2
        return 1
    fi

    local changed_tracked changed_staged untracked all_changed
    changed_tracked=$(git -C "${wt}" diff --name-only)
    changed_staged=$(git -C "${wt}" diff --cached --name-only)
    untracked=$(git -C "${wt}" ls-files --others --exclude-standard)
    all_changed=$(printf '%s\n%s\n%s\n' "${changed_tracked}" "${changed_staged}" "${untracked}" | sed '/^$/d' | sort -u)
    if [[ "${all_changed}" != "train.py" ]] || git -C "${wt}" diff --quiet -- train.py; then
        echo "[ERROR] Codex must leave only an unstaged train.py change; got: ${all_changed}" >&2
        return 1
    fi

    git -C "${wt}" diff --binary -- train.py > "${patch_file}"
    local description
    description=$(tr '\t\r\n' '   ' < "${codex_message}" | tr -s ' ' | sed 's/^ //; s/ $//')
    [[ -n "${description}" ]] || description="candidate change"
    printf '%s\n' "${description}" > "${description_file}"
    printf 'worker %s: %s\n' "${worker}" "${description}" >> "${candidates_file}"
}

[[ -e .git ]] || { echo "[ERROR] not a Git repository" >&2; exit 1; }
[[ -f train.py ]] || { echo "[ERROR] train.py not found" >&2; exit 1; }
[[ -f "${WORKER_SCRIPT}" ]] || { echo "[ERROR] worker script not found: ${WORKER_SCRIPT}" >&2; exit 1; }
[[ -f "${EXPERIMENT_TOOL}" ]] || { echo "[ERROR] experiment_utils.py not found" >&2; exit 1; }

experiment_tool validate-config \
    --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}" \
    --history-mode "${HISTORY_MODE}" --history-limit "${HISTORY_LIMIT}" \
    --template "${PROMPT_TEMPLATE}"

git diff --quiet && git diff --cached --quiet || {
    echo "[ERROR] tracked changes must be committed before a run" >&2
    exit 1
}

BRANCH=$(git branch --show-current)
case "${BRANCH}" in
    autoresearch/*) ;;
    *) echo "[ERROR] run on an autoresearch/* branch (current: ${BRANCH})" >&2; exit 1 ;;
esac

mkdir -p "${RUN_ROOT}" "${WORKTREE_ROOT}"
touch "${RESULT_FILE}"
experiment_tool validate-jsonl "${RESULT_FILE}"
experiment_tool write-run-config \
    --output "${RUN_CONFIG}" --template "${PROMPT_TEMPLATE}" \
    --history-mode "${HISTORY_MODE}" --history-limit "${HISTORY_LIMIT}" \
    --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}" \
    --repo-root "${WORKDIR}"

GPU_COUNT=$(singularity exec --nv "${IMAGE}" nvidia-smi -L | grep -c '^GPU ' || true)
if [[ "${GPU_COUNT}" -lt "${NUM_WORKERS}" ]]; then
    echo "[ERROR] ${NUM_WORKERS} GPUs required, but ${GPU_COUNT} visible" >&2
    exit 1
fi

echo "[SETUP] objective=${PRIMARY_METRIC}/${OBJECTIVE_DIRECTION}"
echo "[SETUP] prompt=${PROMPT_TEMPLATE#${WORKDIR}/} history=${HISTORY_MODE} limit=${HISTORY_LIMIT}"
echo "[SETUP] GPUs=${GPU_COUNT} results=${RESULT_FILE}"

# Evaluate accepted HEAD once for this run's baseline.
ITERATION=1
BASELINE_DIR="${RUN_ROOT}/baseline"
BASELINE_WT="${WORKTREE_ROOT}/baseline"
BASELINE_RESULT="${BASELINE_DIR}/worker_result.json"
BASELINE_CONFIG="${BASELINE_DIR}/config.json"
BASE_COMMIT_FULL=$(git rev-parse HEAD)
BASE_COMMIT_SHORT=$(git rev-parse --short=7 HEAD)
mkdir -p "${BASELINE_DIR}"
: > "${BASELINE_DIR}/change.patch"
create_worktree "${BASELINE_WT}" "${BASE_COMMIT_FULL}"

bash "${WORKER_SCRIPT}" -1 0 "${BASELINE_WT}" "${BASELINE_DIR}" "${IMAGE}" "${WORKDIR}" \
    "${PRIMARY_METRIC}" "${OBJECTIVE_DIRECTION}" \
    >"${BASELINE_DIR}/worker.stdout" 2>"${BASELINE_DIR}/worker.stderr"

BEST_VAL=$(experiment_tool value "${BASELINE_RESULT}" --metric "${PRIMARY_METRIC}")
BASELINE_STATUS=$(experiment_tool run-status "${BASELINE_RESULT}")
if [[ "${BASELINE_STATUS}" != "ok" || -z "${BEST_VAL}" ]]; then
    echo "[ERROR] baseline failed or did not emit ${PRIMARY_METRIC}" >&2
    exit 1
fi

append_result "${ITERATION}" 0 -1 "${BASE_COMMIT_SHORT}" "${BASE_COMMIT_SHORT}" \
    keep baseline "${BASELINE_RESULT}" "${BASELINE_CONFIG}" \
    "results/${DATE}/baseline" "${RUN_CONFIG}" 0
remove_worktree "${BASELINE_WT}"
echo "[BASELINE] ${PRIMARY_METRIC}=${BEST_VAL} commit=${BASE_COMMIT_SHORT}"

for ROUND in $(seq 1 "${NUM_ROUNDS}"); do
    TAG=$(printf '%03d' "${ROUND}")
    ROUND_DIR="${RUN_ROOT}/round_${TAG}"
    CANDIDATES_FILE="${ROUND_DIR}/candidates.txt"
    mkdir -p "${ROUND_DIR}"
    : > "${CANDIDATES_FILE}"

    BASE_COMMIT_FULL=$(git rev-parse HEAD)
    BASE_COMMIT_SHORT=$(git rev-parse --short=7 HEAD)
    echo
    echo "Round ${ROUND}/${NUM_ROUNDS}: base=${BASE_COMMIT_SHORT} best=${BEST_VAL}"

    declare -a WT_DIRS=() WORKER_DIRS=() RESULT_FILES=() PATCH_FILES=()
    declare -a CONFIG_FILES=() PROMPT_METADATA_FILES=() DESCRIPTIONS=()

    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        WT_DIR="${WORKTREE_ROOT}/round_${TAG}/worker_${WORKER}"
        WORKER_DIR="${ROUND_DIR}/worker_${WORKER}"
        create_worktree "${WT_DIR}" "${BASE_COMMIT_FULL}"
        generate_candidate "${ROUND}" "${WORKER}" "${BASE_COMMIT_SHORT}" "${BEST_VAL}" \
            "${WT_DIR}" "${WORKER_DIR}" "${CANDIDATES_FILE}"
        WT_DIRS[${WORKER}]="${WT_DIR}"
        WORKER_DIRS[${WORKER}]="${WORKER_DIR}"
        RESULT_FILES[${WORKER}]="${WORKER_DIR}/worker_result.json"
        PATCH_FILES[${WORKER}]="${WORKER_DIR}/change.patch"
        CONFIG_FILES[${WORKER}]="${WORKER_DIR}/config.json"
        PROMPT_METADATA_FILES[${WORKER}]="${WORKER_DIR}/prompt_metadata.json"
        DESCRIPTIONS[${WORKER}]=$(<"${WORKER_DIR}/description.txt")
    done

    declare -a PIDS=()
    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        echo "[LAUNCH] worker=${WORKER} gpu=${WORKER}"
        bash "${WORKER_SCRIPT}" "${WORKER}" "${WORKER}" \
            "${WT_DIRS[${WORKER}]}" "${WORKER_DIRS[${WORKER}]}" "${IMAGE}" "${WORKDIR}" \
            "${PRIMARY_METRIC}" "${OBJECTIVE_DIRECTION}" \
            >"${WORKER_DIRS[${WORKER}]}/worker.stdout" \
            2>"${WORKER_DIRS[${WORKER}]}/worker.stderr" &
        PIDS[${WORKER}]=$!
    done

    WORKER_INFRA_FAILED=0
    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        wait "${PIDS[${WORKER}]}" || WORKER_INFRA_FAILED=1
    done
    [[ ${WORKER_INFRA_FAILED} -eq 0 ]] || { echo "[ERROR] worker infrastructure failure" >&2; exit 1; }

    WINNER=$(experiment_tool select-winner --best "${BEST_VAL}" \
        --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}" \
        "${RESULT_FILES[@]}")
    WINNER_COMMIT=""

    if [[ -n "${WINNER}" ]]; then
        [[ "$(git rev-parse HEAD)" == "${BASE_COMMIT_FULL}" ]] || { echo "[ERROR] accepted HEAD changed" >&2; exit 1; }
        git apply --check "${PATCH_FILES[${WINNER}]}"
        git apply "${PATCH_FILES[${WINNER}]}"
        git add train.py
        WINNER_VAL=$(experiment_tool value "${RESULT_FILES[${WINNER}]}" --metric "${PRIMARY_METRIC}")
        git commit -m "experiment: round ${ROUND} worker ${WINNER} ${PRIMARY_METRIC}=${WINNER_VAL}" >/dev/null
        WINNER_COMMIT=$(git rev-parse --short=7 HEAD)
        BEST_VAL="${WINNER_VAL}"
        echo "[KEEP] worker=${WINNER} ${PRIMARY_METRIC}=${WINNER_VAL} commit=${WINNER_COMMIT}"
    else
        echo "[DISCARD] no candidate improved ${PRIMARY_METRIC}=${BEST_VAL}"
    fi

    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        ITERATION=$((ITERATION + 1))
        VALUE=$(experiment_tool value "${RESULT_FILES[${WORKER}]}" --metric "${PRIMARY_METRIC}")
        RUN_STATUS=$(experiment_tool run-status "${RESULT_FILES[${WORKER}]}")
        STATUS="discard"
        COMMIT=""
        if [[ "${RUN_STATUS}" != "ok" || -z "${VALUE}" ]]; then
            STATUS="crash"
        elif [[ -n "${WINNER}" && "${WORKER}" -eq "${WINNER}" ]]; then
            STATUS="keep"
            COMMIT="${WINNER_COMMIT}"
        fi
        append_result "${ITERATION}" "${ROUND}" "${WORKER}" "${BASE_COMMIT_SHORT}" "${COMMIT}" \
            "${STATUS}" "${DESCRIPTIONS[${WORKER}]}" "${RESULT_FILES[${WORKER}]}" \
            "${CONFIG_FILES[${WORKER}]}" \
            "results/${DATE}/round_${TAG}/worker_${WORKER}" \
            "${PROMPT_METADATA_FILES[${WORKER}]}" 1
    done

    experiment_tool validate-jsonl "${RESULT_FILE}"
    for WORKER in $(seq 0 $((NUM_WORKERS - 1))); do
        remove_worktree "${WT_DIRS[${WORKER}]}"
    done
    git worktree prune >/dev/null
    echo "[BEST] ${PRIMARY_METRIC}=${BEST_VAL} (${OBJECTIVE_DIRECTION})"
done

echo
echo "Finished ${NUM_ROUNDS} parallel rounds"
echo "Best ${PRIMARY_METRIC}: ${BEST_VAL} (${OBJECTIVE_DIRECTION})"
echo "Results: ${RESULT_FILE}"
