"""Small, dependency-free helpers for autoresearch experiment orchestration.

The shell scripts deliberately retain ownership of process execution and Git.
This module centralizes the data protocol shared by sequential and parallel runs:
objective comparison, training-summary parsing, JSONL records, and prompt context.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import os
from pathlib import Path
from string import Template
from typing import Any, Iterable


DEFAULT_PRIMARY_METRIC = "val_bpb"
DEFAULT_OBJECTIVE_DIRECTION = "min"
DEFAULT_HISTORY_MODE = "all"

METRIC_DESCRIPTIONS = {
    "val_bpb": (
        "Validation bits per UTF-8 byte. Token cross-entropies are summed only "
        "for non-special targets, divided by their UTF-8 byte count, and "
        "converted from nats to bits. This preserves the original BPB metric."
    ),
    "val_loss": (
        "Mean validation next-token cross-entropy in nats per target token, "
        "computed directly from model losses over the fixed validation pass "
        "and including BOS/special-token targets."
    ),
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSONL at line {line_number}: expected object")
            records.append(value)
    return records


def metric_value(record: dict[str, Any], metric: str) -> float | None:
    """Read a metric without reinterpreting historical fields.

    New records use ``metrics``. The explicit objective pair is also understood.
    The only legacy alias is top-level ``val_bpb``, matching the historical schema.
    """
    metrics = record.get("metrics")
    value: Any = metrics.get(metric) if isinstance(metrics, dict) else None
    if value is None and record.get("primary_metric") == metric:
        value = record.get("objective_value")
    if value is None and metric == "val_bpb":
        value = record.get("val_bpb")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def is_better(candidate: float, best: float, direction: str) -> bool:
    validate_direction(direction)
    return candidate < best if direction == "min" else candidate > best


def best_value(
    records: Iterable[dict[str, Any]], metric: str, direction: str
) -> float | None:
    validate_direction(direction)
    values = [
        value
        for record in records
        if record.get("status") == "keep"
        if (value := metric_value(record, metric)) is not None
    ]
    if not values:
        return None
    return min(values) if direction == "min" else max(values)


def select_history(
    records: list[dict[str, Any]], mode: str, limit: int | None
) -> list[dict[str, Any]]:
    validate_history(mode, limit)
    if mode == "all":
        return records
    assert limit is not None
    return records[-limit:]


def history_record_for_prompt(record: dict[str, Any]) -> dict[str, Any]:
    """Remove artifact paths that could lead around the selected history policy."""
    excluded = {"artifacts", "log", "patch", "config"}
    return {key: value for key, value in record.items() if key not in excluded}


def validate_direction(direction: str) -> None:
    if direction not in {"min", "max"}:
        raise ValueError("objective direction must be 'min' or 'max'")


def validate_history(mode: str, limit: int | None) -> None:
    if mode not in {"all", "recent"}:
        raise ValueError("history mode must be 'all' or 'recent'")
    if mode == "recent" and (limit is None or limit < 1):
        raise ValueError("recent history mode requires a positive history limit")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is not None:
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def render_prompt(
    *,
    template_path: str | Path,
    results_path: str | Path,
    history_mode: str,
    history_limit: int | None,
    primary_metric: str,
    objective_direction: str,
    iteration: int,
    base_commit: str,
    current_best: float,
    worker: int | None = None,
    candidates: str = "",
    repo_root: str | Path | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Render one candidate prompt and return prompt, selected JSONL, metadata."""
    validate_direction(objective_direction)
    records = load_jsonl(results_path)
    selected = select_history(records, history_mode, history_limit)
    history_text = "".join(
        json.dumps(history_record_for_prompt(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in selected
    )
    history_for_prompt = history_text.rstrip() or "(no completed experiment records)"

    template_path = Path(template_path)
    template_text = template_path.read_text(encoding="utf-8")
    preference = "lower is better" if objective_direction == "min" else "higher is better"
    description = METRIC_DESCRIPTIONS.get(
        primary_metric,
        "Custom evaluation metric emitted by the configured training/evaluation procedure.",
    )
    worker_text = "none (sequential run)" if worker is None else str(worker)
    candidates_text = candidates.strip() or "(no other candidates proposed in this round)"
    substitutions = {
        "iteration": str(iteration),
        "worker": worker_text,
        "base_commit": base_commit,
        "current_best": format(current_best, ".12g"),
        "primary_metric": primary_metric,
        "objective_direction": objective_direction,
        "objective_preference": preference,
        "metric_description": description,
        "history_mode": history_mode,
        "history_limit": str(history_limit) if history_mode == "recent" else "not applicable",
        "history_count": str(len(selected)),
        "history_records": history_for_prompt,
        "candidates": candidates_text,
    }
    try:
        prompt = Template(template_text).substitute(substitutions)
    except KeyError as error:
        raise ValueError(f"unknown prompt template variable: {error.args[0]}") from error

    root = Path(repo_root) if repo_root is not None else None
    metadata = {
        "template": _display_path(template_path, root),
        "template_sha256": sha256_text(template_text),
        "rendered_prompt_sha256": sha256_text(prompt),
        "history_mode": history_mode,
        "history_limit": history_limit if history_mode == "recent" else None,
        "history_records_supplied": len(selected),
        "primary_metric": primary_metric,
        "objective_direction": objective_direction,
        "candidate_generated": True,
    }
    return prompt, history_text, metadata


def make_run_config(
    template_path: str | Path,
    history_mode: str,
    history_limit: int | None,
    primary_metric: str,
    direction: str,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    validate_direction(direction)
    validate_history(history_mode, history_limit)
    path = Path(template_path)
    text = path.read_text(encoding="utf-8")
    root = Path(repo_root) if repo_root is not None else None
    return {
        "template": _display_path(path, root),
        "template_sha256": sha256_text(text),
        "rendered_prompt_sha256": None,
        "history_mode": history_mode,
        "history_limit": history_limit if history_mode == "recent" else None,
        "history_records_supplied": 0,
        "primary_metric": primary_metric,
        "objective_direction": direction,
        "candidate_generated": False,
    }


def parse_evaluation_metrics(log_text: str) -> dict[str, float]:
    """Parse the last ``eval_metrics`` JSON object from a training log.

    Individual val_bpb/val_loss lines remain a compatibility fallback for old logs.
    """
    metrics: dict[str, float] = {}
    for line in log_text.splitlines():
        if line.startswith("eval_metrics:"):
            try:
                candidate = json.loads(line.split(":", 1)[1].strip())
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            parsed: dict[str, float] = {}
            for name, value in candidate.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    continue
                parsed[str(name)] = numeric
            metrics = parsed
    if metrics:
        return metrics
    for line in log_text.splitlines():
        for name in ("val_bpb", "val_loss"):
            if line.startswith(f"{name}:"):
                try:
                    value = float(line.split(":", 1)[1].strip().split()[0])
                    if math.isfinite(value):
                        metrics[name] = value
                except (ValueError, IndexError):
                    pass
    return metrics


def make_run_result(
    *,
    log_text: str,
    primary_metric: str,
    direction: str,
    train_exit: int,
    worker: int | None = None,
    gpu: int | None = None,
) -> dict[str, Any]:
    validate_direction(direction)
    metrics = parse_evaluation_metrics(log_text)
    objective = metrics.get(primary_metric)
    peak_vram_mb = None
    for line in log_text.splitlines():
        if line.startswith("peak_vram_mb:"):
            try:
                peak_vram_mb = float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return {
        "worker": worker,
        "gpu": gpu,
        "run_status": "ok" if train_exit == 0 and objective is not None else "crash",
        "train_exit": train_exit,
        "primary_metric": primary_metric,
        "objective_direction": direction,
        "objective_value": objective,
        "metrics": metrics,
        "memory_gb": round(peak_vram_mb / 1024, 1) if peak_vram_mb is not None else None,
    }


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_ast(item, env) for item in node.elts)
    if isinstance(node, ast.List):
        return [_evaluate_ast(item, env) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _evaluate_ast(key, env): _evaluate_ast(value, env)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _evaluate_ast(node.left, env), _evaluate_ast(node.right, env)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate_ast(node.operand, env))
    raise ValueError(type(node).__name__)


def snapshot_config(train_path: str | Path, prepare_path: str | Path) -> dict[str, Any]:
    train_source = Path(train_path).read_text(encoding="utf-8")
    lines = train_source.splitlines()
    start_line = next(
        (i for i, line in enumerate(lines, 1) if "Hyperparameters (edit these directly" in line),
        None,
    )
    end_line = next(
        (
            i
            for i, line in enumerate(lines, 1)
            if start_line is not None
            and i > start_line
            and "Setup: tokenizer, model, optimizer" in line
        ),
        None,
    )
    if start_line is None or end_line is None:
        raise ValueError("hyperparameter section not found in train.py")

    hyperparameters: dict[str, Any] = {}
    env: dict[str, Any] = {}
    for node in ast.parse(train_source).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not (start_line < node.lineno < end_line):
            continue
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target, value_node = node.targets[0], node.value
        else:
            target, value_node = node.target, node.value
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        try:
            value = _evaluate_ast(value_node, env)
        except Exception:
            value = {"expression": ast.unparse(value_node)}
        env[target.id] = value
        hyperparameters[target.id] = value

    prepare_source = Path(prepare_path).read_text(encoding="utf-8")
    prepare_env: dict[str, Any] = {}
    fixed_config: dict[str, Any] = {}
    for node in ast.parse(prepare_source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = _evaluate_ast(node.value, prepare_env)
        except Exception:
            continue
        prepare_env[target.id] = value
        if target.id in {"MAX_SEQ_LEN", "TIME_BUDGET", "EVAL_TOKENS"}:
            fixed_config[target.id] = value
    return {
        "hyperparameters": hyperparameters,
        "fixed_config": fixed_config,
        "train_sha256": sha256_text(train_source),
    }


def make_result_record(
    *,
    iteration: int,
    status: str,
    description: str,
    primary_metric: str,
    direction: str,
    run: dict[str, Any],
    config: dict[str, Any],
    prompt: dict[str, Any],
    commit: str | None = None,
    base_commit: str | None = None,
    round_id: int | None = None,
    worker: int | None = None,
    artifacts: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    validate_direction(direction)
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    objective = metric_value({"metrics": metrics}, primary_metric)
    artifacts = {key: value for key, value in (artifacts or {}).items() if value}
    record = {
        "iteration": iteration,
        "primary_metric": primary_metric,
        "objective_direction": direction,
        "objective_value": objective,
        "metrics": metrics,
        # Explicit compatibility alias for existing BPB readers; never used for other metrics.
        "val_bpb": metrics.get("val_bpb"),
        "status": status,
        "description": description,
        "commit": commit,
        "base_commit": base_commit,
        "round": round_id,
        "worker": worker,
        "gpu": run.get("gpu"),
        "memory_gb": run.get("memory_gb"),
        "train_exit": run.get("train_exit"),
        "hyperparameters": config.get("hyperparameters", {}),
        "fixed_config": config.get("fixed_config", {}),
        "train_sha256": config.get("train_sha256"),
        "prompt": prompt,
        "artifacts": artifacts,
    }
    # Keep familiar flat artifact fields for older ad-hoc readers.
    record.update({key: value for key, value in artifacts.items() if key in {"log", "patch", "config"}})
    return record


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _add_objective_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC)
    parser.add_argument("--direction", default=DEFAULT_OBJECTIVE_DIRECTION)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config")
    _add_objective_arguments(validate)
    validate.add_argument("--history-mode", default=DEFAULT_HISTORY_MODE)
    validate.add_argument("--history-limit", type=int)
    validate.add_argument("--template", type=Path, required=True)

    validate_jsonl = commands.add_parser("validate-jsonl")
    validate_jsonl.add_argument("path", type=Path)

    last = commands.add_parser("last-iteration")
    last.add_argument("path", type=Path)

    best = commands.add_parser("best")
    best.add_argument("path", type=Path)
    _add_objective_arguments(best)

    compare = commands.add_parser("compare")
    compare.add_argument("candidate", type=float)
    compare.add_argument("best", type=float)
    compare.add_argument("--direction", default=DEFAULT_OBJECTIVE_DIRECTION)

    value = commands.add_parser("value")
    value.add_argument("path", type=Path)
    value.add_argument("--metric", default=DEFAULT_PRIMARY_METRIC)

    run_status = commands.add_parser("run-status")
    run_status.add_argument("path", type=Path)

    winner = commands.add_parser("select-winner")
    winner.add_argument("paths", nargs="+", type=Path)
    winner.add_argument("--best", required=True, type=float)
    _add_objective_arguments(winner)

    prompt = commands.add_parser("render-prompt")
    prompt.add_argument("--template", type=Path, required=True)
    prompt.add_argument("--results", type=Path, required=True)
    prompt.add_argument("--output", type=Path, required=True)
    prompt.add_argument("--history-output", type=Path, required=True)
    prompt.add_argument("--metadata-output", type=Path, required=True)
    prompt.add_argument("--history-mode", default=DEFAULT_HISTORY_MODE)
    prompt.add_argument("--history-limit", type=int)
    _add_objective_arguments(prompt)
    prompt.add_argument("--iteration", type=int, required=True)
    prompt.add_argument("--base-commit", required=True)
    prompt.add_argument("--current-best", type=float, required=True)
    prompt.add_argument("--worker", type=int)
    prompt.add_argument("--candidates-file", type=Path)
    prompt.add_argument("--repo-root", type=Path)

    run_config = commands.add_parser("write-run-config")
    run_config.add_argument("--output", type=Path, required=True)
    run_config.add_argument("--template", type=Path, required=True)
    run_config.add_argument("--history-mode", default=DEFAULT_HISTORY_MODE)
    run_config.add_argument("--history-limit", type=int)
    _add_objective_arguments(run_config)
    run_config.add_argument("--repo-root", type=Path)

    snapshot = commands.add_parser("snapshot-config")
    snapshot.add_argument("--train", type=Path, required=True)
    snapshot.add_argument("--prepare", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)

    run_result = commands.add_parser("write-run-result")
    run_result.add_argument("--log", type=Path, required=True)
    run_result.add_argument("--output", type=Path, required=True)
    run_result.add_argument("--train-exit", type=int, required=True)
    _add_objective_arguments(run_result)
    run_result.add_argument("--worker", type=int)
    run_result.add_argument("--gpu", type=int)

    append = commands.add_parser("append-result")
    append.add_argument("--results", type=Path, required=True)
    append.add_argument("--run-result", type=Path, required=True)
    append.add_argument("--config", type=Path, required=True)
    append.add_argument("--prompt-metadata", type=Path, required=True)
    append.add_argument("--iteration", type=int, required=True)
    append.add_argument("--round", dest="round_id", type=int)
    append.add_argument("--worker", type=int)
    append.add_argument("--base-commit")
    append.add_argument("--commit")
    append.add_argument("--status", required=True)
    append.add_argument("--description", required=True)
    _add_objective_arguments(append)
    for name in ("log", "patch", "config-artifact", "prompt", "history"):
        append.add_argument(f"--{name}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        validate_direction(args.direction)
        validate_history(args.history_mode, args.history_limit)
        args.template.read_text(encoding="utf-8")
    elif args.command == "validate-jsonl":
        load_jsonl(args.path)
    elif args.command == "last-iteration":
        print(max((int(record.get("iteration", 0)) for record in load_jsonl(args.path)), default=0))
    elif args.command == "best":
        value = best_value(load_jsonl(args.path), args.primary_metric, args.direction)
        if value is not None:
            print(format(value, ".12g"))
    elif args.command == "compare":
        return 0 if is_better(args.candidate, args.best, args.direction) else 1
    elif args.command == "value":
        value = metric_value(_read_json(args.path), args.metric)
        if value is not None:
            print(format(value, ".12g"))
    elif args.command == "run-status":
        print(_read_json(args.path).get("run_status", "crash"))
    elif args.command == "select-winner":
        winner: int | None = None
        winner_value = args.best
        for path in args.paths:
            record = _read_json(path)
            value = metric_value(record, args.primary_metric)
            if record.get("run_status") == "ok" and value is not None and is_better(value, winner_value, args.direction):
                winner = int(record["worker"])
                winner_value = value
        if winner is not None:
            print(winner)
    elif args.command == "render-prompt":
        candidates = args.candidates_file.read_text(encoding="utf-8") if args.candidates_file else ""
        rendered, history, metadata = render_prompt(
            template_path=args.template,
            results_path=args.results,
            history_mode=args.history_mode,
            history_limit=args.history_limit,
            primary_metric=args.primary_metric,
            objective_direction=args.direction,
            iteration=args.iteration,
            base_commit=args.base_commit,
            current_best=args.current_best,
            worker=args.worker,
            candidates=candidates,
            repo_root=args.repo_root,
        )
        args.output.write_text(rendered, encoding="utf-8")
        args.history_output.write_text(history, encoding="utf-8")
        _write_json(args.metadata_output, metadata)
    elif args.command == "write-run-config":
        _write_json(
            args.output,
            make_run_config(
                args.template,
                args.history_mode,
                args.history_limit,
                args.primary_metric,
                args.direction,
                args.repo_root,
            ),
        )
    elif args.command == "snapshot-config":
        _write_json(args.output, snapshot_config(args.train, args.prepare))
    elif args.command == "write-run-result":
        _write_json(
            args.output,
            make_run_result(
                log_text=args.log.read_text(encoding="utf-8", errors="replace"),
                primary_metric=args.primary_metric,
                direction=args.direction,
                train_exit=args.train_exit,
                worker=args.worker,
                gpu=args.gpu,
            ),
        )
    elif args.command == "append-result":
        artifacts = {
            "log": args.log,
            "patch": args.patch,
            "config": args.config_artifact,
            "prompt": args.prompt,
            "history": args.history,
        }
        record = make_result_record(
            iteration=args.iteration,
            round_id=args.round_id,
            worker=args.worker,
            base_commit=args.base_commit,
            commit=args.commit,
            status=args.status,
            description=args.description,
            primary_metric=args.primary_metric,
            direction=args.direction,
            run=_read_json(args.run_result),
            config=_read_json(args.config),
            prompt=_read_json(args.prompt_metadata),
            artifacts=artifacts,
        )
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with args.results.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"[ERROR] {error}") from error
