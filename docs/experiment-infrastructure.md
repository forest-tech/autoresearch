# Experiment infrastructure

The autonomous HPC loops treat three groups of variables independently:

- Training configuration is the current contents and hyperparameters of `train.py`.
- Objective configuration is `PRIMARY_METRIC` plus `OBJECTIVE_DIRECTION` (`min` or `max`).
- Prompt configuration is `PROMPT_TEMPLATE`, `HISTORY_MODE`, and `HISTORY_LIMIT`.

The sequential and parallel Codex loops use the same dependency-free helper, `experiment_utils.py`, for objective comparisons, training-output parsing, JSONL records, and prompt rendering. The shell remains responsible for training processes, keep/discard decisions, Git, and artifacts. Candidate Codex runs remain restricted to editing `train.py` and never run training themselves.

## Configurations

Defaults are `PRIMARY_METRIC=val_bpb`, `OBJECTIVE_DIRECTION=min`, `PROMPT_TEMPLATE=prompts/candidate_default.txt`, and `HISTORY_MODE=all`.

```bash
# 1. Default BPB optimization with full history
PRIMARY_METRIC=val_bpb OBJECTIVE_DIRECTION=min HISTORY_MODE=all \
  bash script/genkai/loop_for_codex.sh

# 2. Validation-loss optimization with full history
PRIMARY_METRIC=val_loss OBJECTIVE_DIRECTION=min HISTORY_MODE=all \
  bash script/genkai/loop_for_codex.sh

# 3. BPB optimization with recent-10 history
PRIMARY_METRIC=val_bpb OBJECTIVE_DIRECTION=min \
  HISTORY_MODE=recent HISTORY_LIMIT=10 \
  bash script/genkai/loop_for_codex.sh

# 4. BPB optimization with the included exploratory prompt
PRIMARY_METRIC=val_bpb OBJECTIVE_DIRECTION=min HISTORY_MODE=all \
  PROMPT_TEMPLATE=prompts/candidate_exploratory.txt \
  bash script/genkai/loop_for_codex.sh
```

Use `script/genkai/parallel_loop_for_codex.sh` in the same commands for parallel best-of-N rounds. Existing `IMAGE`, `WORKDIR`, `NUM_ITERATIONS`, `NUM_ROUNDS`, and `NUM_WORKERS` values can also be overridden through the environment.

## Evaluation metrics

`prepare.evaluate_metrics` computes both current validation metrics in the same fixed validation pass:

- `val_bpb`: the original validation bits per UTF-8 byte. Per-token cross-entropy is summed for non-special targets, divided by target byte counts, then converted from nats to bits. Its mathematical definition and evaluation-token budget are unchanged.
- `val_loss`: mean validation next-token cross-entropy in nats per target token, directly accumulated from unreduced model losses. It includes special-token targets and is tokenizer-dependent; it is not derived or approximated from BPB.

`OBJECTIVE_DIRECTION=min` keeps strictly smaller values; `max` keeps strictly larger values. A missing configured objective, nonnumeric value, nonzero training exit, or non-finite value makes the run a crash. Both metrics are still recorded regardless of which is primary.

Training writes an `eval_metrics: { ... }` JSON summary line. This is the evaluator-to-orchestrator extension point. To add a real downstream metric such as `task_accuracy`:

1. Implement the actual fixed evaluation procedure (including any benchmark data/setup) in the evaluation layer and return `task_accuracy` in the evaluation metric mapping. A separate pass is appropriate when the task cannot share language-model validation batches.
2. Ensure `train.py` includes it in the `eval_metrics` JSON object. The generic parser will record it without changes.
3. Optionally add its human-readable definition to `METRIC_DESCRIPTIONS` in `experiment_utils.py`.
4. Run with `PRIMARY_METRIC=task_accuracy OBJECTIVE_DIRECTION=max`.

No downstream score is fabricated by the current implementation.

## Result records and compatibility

New `results.jsonl` lines contain the selected objective and all evaluation metrics, for example:

```json
{
  "iteration": 2,
  "primary_metric": "val_loss",
  "objective_direction": "min",
  "objective_value": 2.34,
  "metrics": {"val_bpb": 1.0021, "val_loss": 2.34},
  "status": "keep",
  "prompt": {
    "template": "prompts/candidate_default.txt",
    "template_sha256": "...",
    "rendered_prompt_sha256": "...",
    "history_mode": "recent",
    "history_limit": 10,
    "history_records_supplied": 7,
    "primary_metric": "val_loss",
    "objective_direction": "min"
  }
}
```

Records also retain applicable iteration/round/worker/GPU, commits, memory, exit status, description, hyperparameters, code hash, and artifact paths. A top-level `val_bpb` compatibility alias is written when BPB was measured. Readers prefer `metrics`; they fall back only to the historical top-level `val_bpb` field for old records. Old directories are not modified, and an old `val_bpb` is never interpreted as another metric.

Each run has `run_config.json`. Each generated candidate saves `codex_prompt.txt`, `prompt_metadata.json`, and `history_context.jsonl`. These make the prompt condition and exact rendered input reproducible.

Plotting defaults to BPB and understands old and new records. Select another metric with, for example:

```bash
uv run python plot_results.py results/RUN/results.jsonl --metric val_loss
uv run python plot_results_parallel.py results/RUN/results.jsonl --metric val_loss
```

## History and prompt templates

`HISTORY_MODE=all` embeds every completed record from the current run. `HISTORY_MODE=recent` embeds at most the final positive `HISTORY_LIMIT` records; if fewer exist, it embeds all available records. The rendered prompt and history artifact contain only that selected slice. Artifact paths are removed from history records so that a recent-history prompt does not point to older prompt/history artifacts. Templates tell the candidate to treat the supplied inline slice as authoritative and do not expose or instruct it to read the complete `results.jsonl` path.

To add a prompt strategy, copy a file under `prompts/`, edit its instructions, and set `PROMPT_TEMPLATE` to that path. Templates use Python `string.Template` variables:

`iteration`, `worker`, `base_commit`, `current_best`, `primary_metric`, `objective_direction`, `objective_preference`, `metric_description`, `history_mode`, `history_limit`, `history_count`, `history_records`, and `candidates`.

Keep the ownership and safety constraints from `candidate_default.txt` in variants unless changing one of those constraints is itself the explicit research condition.
