#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -L jobenv=singularity
#PJM -j

set -euo pipefail

module load singularity-ce

# Infrastructure, objective, and prompt settings are independent.
IMAGE="${IMAGE:-/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif}"
WORKDIR="${WORKDIR:-/home/pj24001974/ku50001532/projects/autoresearch}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
PRIMARY_METRIC="${PRIMARY_METRIC:-val_bpb}"
OBJECTIVE_DIRECTION="${OBJECTIVE_DIRECTION:-min}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-prompts/candidate_default.txt}"
HISTORY_MODE="${HISTORY_MODE:-all}"
HISTORY_LIMIT="${HISTORY_LIMIT:-10}"

if [[ "${PROMPT_TEMPLATE}" != /* ]]; then
    PROMPT_TEMPLATE="${WORKDIR}/${PROMPT_TEMPLATE}"
fi

DATE=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${WORKDIR}/results/${DATE}"
RESULT_FILE="${RUN_ROOT}/results.jsonl"
RUN_CONFIG="${RUN_ROOT}/run_config.json"
EXPERIMENT_TOOL="${WORKDIR}/experiment_utils.py"

cd "${WORKDIR}"

experiment_tool() {
    singularity exec \
        --bind "${WORKDIR}:${WORKDIR}" \
        --pwd "${WORKDIR}" \
        "${IMAGE}" \
        bash -lc 'uv run "$@"' _ "${EXPERIMENT_TOOL}" "$@"
}

append_result() {
    local iteration="$1" commit="$2" status="$3" description="$4"
    local run_result="$5" config_file="$6" prompt_metadata="$7" tag="$8"
    local base_commit="$9"
    local -a args=(
        append-result --results "${RESULT_FILE}"
        --run-result "${run_result}" --config "${config_file}"
        --prompt-metadata "${prompt_metadata}" --iteration "${iteration}"
        --status "${status}" --description "${description}"
        --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}"
        --log "results/${DATE}/iter_${tag}/train.log"
        --patch "results/${DATE}/iter_${tag}/change.patch"
        --config-artifact "results/${DATE}/iter_${tag}/config.json"
    )
    [[ -n "${commit}" ]] && args+=(--commit "${commit}")
    [[ -n "${base_commit}" ]] && args+=(--base-commit "${base_commit}")
    if [[ "${prompt_metadata}" != "${RUN_CONFIG}" ]]; then
        args+=(
            --prompt "results/${DATE}/iter_${tag}/codex_prompt.txt"
            --history "results/${DATE}/iter_${tag}/history_context.jsonl"
        )
    fi
    experiment_tool "${args[@]}"
}

[[ -d .git ]] || { echo "[ERROR] not a Git repository" >&2; exit 1; }
[[ -f train.py ]] || { echo "[ERROR] train.py not found" >&2; exit 1; }
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

mkdir -p "${RUN_ROOT}"
touch "${RESULT_FILE}"
experiment_tool validate-jsonl "${RESULT_FILE}"
experiment_tool write-run-config \
    --output "${RUN_CONFIG}" --template "${PROMPT_TEMPLATE}" \
    --history-mode "${HISTORY_MODE}" --history-limit "${HISTORY_LIMIT}" \
    --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}" \
    --repo-root "${WORKDIR}"

LAST_ITER=$(experiment_tool last-iteration "${RESULT_FILE}")
BEST_VAL=$(experiment_tool best "${RESULT_FILE}" \
    --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}")

echo "[SETUP] objective=${PRIMARY_METRIC}/${OBJECTIVE_DIRECTION}"
echo "[SETUP] prompt=${PROMPT_TEMPLATE#${WORKDIR}/} history=${HISTORY_MODE} limit=${HISTORY_LIMIT}"
echo "[SETUP] results=${RESULT_FILE}"

for n in $(seq 1 "${NUM_ITERATIONS}"); do
    ITER=$((LAST_ITER + n))
    TAG=$(printf '%03d' "${ITER}")
    ITER_DIR="${RUN_ROOT}/iter_${TAG}"
    TRAIN_LOG="${ITER_DIR}/train.log"
    PATCH_FILE="${ITER_DIR}/change.patch"
    CONFIG_FILE="${ITER_DIR}/config.json"
    RUN_RESULT="${ITER_DIR}/run_result.json"
    PROMPT_FILE="${ITER_DIR}/codex_prompt.txt"
    PROMPT_METADATA="${ITER_DIR}/prompt_metadata.json"
    HISTORY_FILE="${ITER_DIR}/history_context.jsonl"
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
    BASE_COMMIT=$(git rev-parse --short=7 HEAD)
    RECORD_PROMPT_METADATA="${RUN_CONFIG}"

    if [[ -z "${BEST_VAL}" ]]; then
        IS_BASELINE=1
        : > "${PATCH_FILE}"
        echo "[BASELINE] current train.py"
    else
        PREEXISTING_UNTRACKED="${ITER_DIR}/preexisting_untracked.txt"
        git ls-files --others --exclude-standard | sort -u > "${PREEXISTING_UNTRACKED}"
        experiment_tool render-prompt \
            --template "${PROMPT_TEMPLATE}" --results "${RESULT_FILE}" \
            --output "${PROMPT_FILE}" --history-output "${HISTORY_FILE}" \
            --metadata-output "${PROMPT_METADATA}" \
            --history-mode "${HISTORY_MODE}" --history-limit "${HISTORY_LIMIT}" \
            --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}" \
            --iteration "${ITER}" --base-commit "${BASE_COMMIT}" \
            --current-best "${BEST_VAL}" --repo-root "${WORKDIR}"
        RECORD_PROMPT_METADATA="${PROMPT_METADATA}"

        echo "[CODEX] generating candidate"
        if ! singularity exec \
            --bind "${WORKDIR}:${WORKDIR}" --pwd "${WORKDIR}" "${IMAGE}" \
            bash -lc \
            "codex exec --sandbox danger-full-access --skip-git-repo-check -o '${CODEX_MESSAGE}' - < '${PROMPT_FILE}'" \
            >"${CODEX_STDOUT}" 2>"${CODEX_STDERR}"
        then
            echo "[ERROR] Codex failed" >&2
            git restore --worktree -- train.py
            exit 1
        fi

        NEW_UNTRACKED=$(comm -13 "${PREEXISTING_UNTRACKED}" <(git ls-files --others --exclude-standard | sort -u))
        CHANGED=$(
            printf '%s\n%s\n%s\n' \
                "$(git diff --name-only)" \
                "$(git diff --cached --name-only)" \
                "${NEW_UNTRACKED}" \
            | sed '/^$/d' | sort -u
        )
        if [[ "${CHANGED}" != "train.py" ]] || git diff --quiet -- train.py; then
            echo "[ERROR] Codex must leave only an unstaged train.py change" >&2
            printf '%s\n' "${CHANGED}" >&2
            git restore --staged --worktree -- .
            exit 1
        fi

        git diff --binary -- train.py > "${PATCH_FILE}"
        DESCRIPTION=$(tr '\t\r\n' '   ' < "${CODEX_MESSAGE}" | tr -s ' ' | sed 's/^ //; s/ $//')
        [[ -n "${DESCRIPTION}" ]] || DESCRIPTION="candidate change"
    fi

    experiment_tool snapshot-config \
        --train "${WORKDIR}/train.py" --prepare "${WORKDIR}/prepare.py" \
        --output "${CONFIG_FILE}"

    echo "[TRAIN] starting"
    TRAIN_EXIT=0
    timeout 600 singularity exec \
        --nv --bind "${WORKDIR}:${WORKDIR}" --pwd "${WORKDIR}" "${IMAGE}" \
        bash -lc "uv run train.py" >"${TRAIN_LOG}" 2>&1 || TRAIN_EXIT=$?

    experiment_tool write-run-result \
        --log "${TRAIN_LOG}" --output "${RUN_RESULT}" --train-exit "${TRAIN_EXIT}" \
        --primary-metric "${PRIMARY_METRIC}" --direction "${OBJECTIVE_DIRECTION}"
    OBJECTIVE_VALUE=$(experiment_tool value "${RUN_RESULT}" --metric "${PRIMARY_METRIC}")

    if [[ ${TRAIN_EXIT} -ne 0 || -z "${OBJECTIVE_VALUE}" ]]; then
        echo "[TRAIN] failed or did not emit ${PRIMARY_METRIC} (exit=${TRAIN_EXIT})"
        append_result "${ITER}" "" "crash" "${DESCRIPTION}" "${RUN_RESULT}" \
            "${CONFIG_FILE}" "${RECORD_PROMPT_METADATA}" "${TAG}" "${BASE_COMMIT}"
        if [[ ${IS_BASELINE} -eq 0 ]]; then
            git restore --worktree -- train.py
            continue
        fi
        echo "[ERROR] baseline failed" >&2
        exit 1
    fi

    if [[ ${IS_BASELINE} -eq 1 ]]; then
        STATUS="keep"
        COMMIT="${BASE_COMMIT}"
        BEST_VAL="${OBJECTIVE_VALUE}"
    elif experiment_tool compare "${OBJECTIVE_VALUE}" "${BEST_VAL}" --direction "${OBJECTIVE_DIRECTION}"; then
        STATUS="keep"
        git add train.py
        git commit -m "experiment: iter ${ITER} ${PRIMARY_METRIC}=${OBJECTIVE_VALUE}" >/dev/null
        COMMIT=$(git rev-parse --short=7 HEAD)
        BEST_VAL="${OBJECTIVE_VALUE}"
    else
        STATUS="discard"
        COMMIT=""
        git restore --worktree -- train.py
    fi

    append_result "${ITER}" "${COMMIT}" "${STATUS}" "${DESCRIPTION}" "${RUN_RESULT}" \
        "${CONFIG_FILE}" "${RECORD_PROMPT_METADATA}" "${TAG}" "${BASE_COMMIT}"
    echo "[RESULT] ${PRIMARY_METRIC}=${OBJECTIVE_VALUE} status=${STATUS}"
    echo "[BEST]   ${PRIMARY_METRIC}=${BEST_VAL} (${OBJECTIVE_DIRECTION})"
done

echo
echo "Finished ${NUM_ITERATIONS} experiments"
echo "Best ${PRIMARY_METRIC}: ${BEST_VAL} (${OBJECTIVE_DIRECTION})"
echo "Results: ${RESULT_FILE}"
