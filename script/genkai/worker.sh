#!/bin/bash

set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 WORKER_ID GPU_ID WORKTREE OUT_DIR IMAGE REPO_ROOT" >&2
    exit 2
fi

WORKER_ID="$1"
GPU_ID="$2"
WORKTREE="$3"
OUT_DIR="$4"
IMAGE="$5"
REPO_ROOT="$6"

TRAIN_LOG="${OUT_DIR}/train.log"
CONFIG_FILE="${OUT_DIR}/config.json"
RESULT_FILE="${OUT_DIR}/worker_result.json"

mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/torchinductor_cache" "${OUT_DIR}/triton_cache"

[[ -f "${WORKTREE}/train.py" ]] || {
    echo "[ERROR] train.py not found: ${WORKTREE}/train.py" >&2
    exit 1
}

[[ -f "${WORKTREE}/prepare.py" ]] || {
    echo "[ERROR] prepare.py not found: ${WORKTREE}/prepare.py" >&2
    exit 1
}

# ============================================================
# Snapshot experiment configuration
# ============================================================
snapshot_config() {
    local train_path="$1"
    local prepare_path="$2"
    local output_file="$3"

    singularity exec \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --pwd "${WORKTREE}" \
        "${IMAGE}" \
        python - \
        "${train_path}" \
        "${prepare_path}" \
        "${output_file}" <<'PY'
import ast
import hashlib
import json
import operator
import sys

train_path, prepare_path, output_path = sys.argv[1:]

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
        return op(evaluate(node.left, env), evaluate(node.right, env))
    if isinstance(node, ast.UnaryOp):
        op = UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(type(node.op).__name__)
        return op(evaluate(node.operand, env))
    raise ValueError(type(node).__name__)


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
    raise SystemExit("[ERROR] Hyperparameter section not found in train.py")

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
        value = {"expression": ast.unparse(value_node)}

    env[name] = value
    hyperparameters[name] = value

with open(prepare_path, encoding="utf-8") as f:
    prepare_source = f.read()

prepare_tree = ast.parse(prepare_source)
prepare_env = {}
fixed_config = {}
wanted = {"MAX_SEQ_LEN", "TIME_BUDGET"}

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

train_sha256 = hashlib.sha256(train_source.encode("utf-8")).hexdigest()

record = {
    "hyperparameters": hyperparameters,
    "fixed_config": fixed_config,
    "train_sha256": train_sha256,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

# ============================================================
# Snapshot
# ============================================================
echo "[WORKER ${WORKER_ID}] snapshot config"
snapshot_config \
    "${WORKTREE}/train.py" \
    "${WORKTREE}/prepare.py" \
    "${CONFIG_FILE}"

# ============================================================
# Train on exactly one GPU
# ============================================================
echo "[WORKER ${WORKER_ID}] start training on GPU ${GPU_ID}"

TRAIN_EXIT=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
SINGULARITYENV_CUDA_VISIBLE_DEVICES="${GPU_ID}" \
SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="${OUT_DIR}/torchinductor_cache" \
SINGULARITYENV_TRITON_CACHE_DIR="${OUT_DIR}/triton_cache" \
timeout 600 singularity exec \
    --nv \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${WORKTREE}" \
    "${IMAGE}" \
    bash -lc "uv run train.py" \
    >"${TRAIN_LOG}" 2>&1 \
    || TRAIN_EXIT=$?

VAL_BPB=$(
    awk '/^val_bpb:/ {v=$2} END {print v}' "${TRAIN_LOG}"
)

PEAK_VRAM_MB=$(
    awk '/^peak_vram_mb:/ {v=$2} END {print v}' "${TRAIN_LOG}"
)

MEMORY_GB=""
if [[ -n "${PEAK_VRAM_MB}" ]]; then
    MEMORY_GB=$(
        awk -v mb="${PEAK_VRAM_MB}" \
            'BEGIN {printf "%.1f", mb/1024}'
    )
fi

if [[ ${TRAIN_EXIT} -eq 0 && -n "${VAL_BPB}" ]]; then
    RUN_STATUS="ok"
else
    RUN_STATUS="crash"
fi

# ============================================================
# Write worker-local result atomically
# ============================================================
TMP_RESULT="${RESULT_FILE}.tmp"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${WORKTREE}" \
    "${IMAGE}" \
    python - \
    "${TMP_RESULT}" \
    "${WORKER_ID}" \
    "${GPU_ID}" \
    "${RUN_STATUS}" \
    "${TRAIN_EXIT}" \
    "${VAL_BPB}" \
    "${MEMORY_GB}" <<'PY'
import json
import sys

(
    output_file,
    worker_id,
    gpu_id,
    run_status,
    train_exit,
    val_bpb,
    memory_gb,
) = sys.argv[1:]

record = {
    "worker": int(worker_id),
    "gpu": int(gpu_id),
    "run_status": run_status,
    "train_exit": int(train_exit),
    "val_bpb": float(val_bpb) if val_bpb else None,
    "memory_gb": float(memory_gb) if memory_gb else None,
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

mv "${TMP_RESULT}" "${RESULT_FILE}"

echo "[WORKER ${WORKER_ID}] status=${RUN_STATUS} val_bpb=${VAL_BPB:-NA}"

# A training crash is an experiment result, not a worker infrastructure failure.
exit 0
