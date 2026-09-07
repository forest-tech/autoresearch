You are working on the experiment infrastructure of this autoresearch repository.

Repository:
https://github.com/forest-tech/autoresearch

Your task is to refactor the repository so that I can systematically run two kinds of research experiments:

1. Experiments comparing different optimization/evaluation metrics.
2. Experiments comparing different Codex candidate-generation prompts and history/context policies.

This is an implementation task, not a design-only task.

Inspect the current repository, design the smallest clean architecture that satisfies the requirements below, implement it completely, and validate it with lightweight tests.

Do not ask me for confirmation before editing files, creating files, or running lightweight validation commands.

Make reasonable implementation decisions yourself and continue until the requested refactoring and validation are complete.

Do not start expensive GPU training unless it is genuinely necessary.
Prefer static checks, lightweight tests, and synthetic/mock data.

# 1. Inspect the current implementation first

Before modifying anything, inspect the repository carefully.

At minimum, read:

* README.md
* program.md
* prepare.py
* train.py
* script/genkai/loop_for_codex.sh
* plot_results.py
* plot_results_parallel.py, if relevant
* existing results.jsonl files or examples, if available

Also inspect any other files required to understand the experiment workflow.

Identify every place where:

* `val_bpb` is hard-coded as the optimization objective
* lower-is-better behavior is assumed
* evaluation results are parsed from training output
* results are written to `results.jsonl`
* keep/discard decisions are made
* the candidate-generation prompt is hard-coded
* the full experiment history is exposed to Codex
* prompt artifacts and experiment metadata are stored

The restrictions inside the existing autonomous candidate-generation prompt, such as "Edit ONLY train.py", apply to candidate-generation Codex runs, not to this infrastructure-refactoring task.

For this task, you may modify any repository files necessary to implement the infrastructure cleanly.

# 2. Goal A: make the optimization objective configurable

The current autonomous loop is strongly coupled to `val_bpb`.

Refactor it so that the metric used for experiment selection can be changed through configuration without modifying the core experiment loop.

Separate these concepts clearly:

1. metrics measured during evaluation
2. the primary objective metric used for keep/discard
3. whether the objective is minimized or maximized

The configuration should support something conceptually equivalent to:

PRIMARY_METRIC=val_bpb
OBJECTIVE_DIRECTION=min

and:

PRIMARY_METRIC=val_loss
OBJECTIVE_DIRECTION=min

and, in the future:

PRIMARY_METRIC=task_accuracy
OBJECTIVE_DIRECTION=max

You may choose a different configuration mechanism if it is simpler and cleaner for this repository.

Requirements:

* `val_bpb` must remain the default objective.
* Existing BPB experiments must continue to work.
* Support at least `val_bpb` and a proper validation loss end-to-end.
* Do not derive or approximate validation loss from BPB.
* Compute a mathematically appropriate validation loss directly.
* When practical, compute BPB and validation loss in the same evaluation pass to avoid unnecessary evaluation overhead.
* Preserve the existing mathematical definition and comparability of `val_bpb`.
* Clearly document what the new validation-loss metric represents.

The selected objective must control all objective-specific behavior, including:

* result extraction
* baseline initialization
* current best value
* keep/discard comparison
* missing-objective/crash detection
* progress output
* final result output
* commit messages where appropriate
* metric description shown to the candidate-generating Codex

The comparison implementation must correctly handle both:

* `min`
* `max`

Avoid scattering metric-specific branches throughout the shell script.

Prefer a generic objective abstraction over repeated logic such as:

if metric == "val_bpb"
...
elif metric == "val_loss"
...

# 3. Multi-metric results.jsonl

Currently the experiment records primarily expose `val_bpb`.

Change the result representation so that multiple evaluation metrics can be recorded for each experiment.

Prefer a schema conceptually similar to:

{
"iteration": 1,
"primary_metric": "val_bpb",
"objective_direction": "min",
"objective_value": 1.0021,
"metrics": {
"val_bpb": 1.0021,
"val_loss": 2.34
},
"status": "keep",
...
}

The exact schema may differ if you find a cleaner design.

Requirements:

* Keep existing useful metadata such as iteration, round, worker, GPU, commit, status, description, hyperparameters, artifacts, memory information, etc. when applicable.
* Store enough information to determine exactly which objective was optimized.
* Do not modify or migrate old experiment directories.
* Preserve backward compatibility with old records containing a top-level `val_bpb` where reasonably practical.
* New readers should ideally understand both old and new records.
* Do not silently reinterpret historical metrics.

If plotting utilities assume `val_bpb`, refactor them where appropriate so that the plotted metric can be selected while preserving `val_bpb` as the default.

# 4. Extensibility to task-performance metrics

A future objective should be able to use a downstream task-performance metric such as accuracy.

Do NOT invent, simulate, or fake task performance.

Inspect the repository and existing dependencies and determine the cleanest extension point for metrics requiring a different evaluation procedure.

If adding a real downstream benchmark now would require substantial dependencies, downloads, GPU time, or unrelated infrastructure, do not add it merely to satisfy an example.

Instead:

* create a clean evaluator/metric extension point
* ensure maximize-type objectives are supported
* document how a future task-performance evaluator would plug into the system

The current implementation must concretely support at least:

* val_bpb / min
* val_loss / min

The architecture must also be capable of supporting something like:

* task_accuracy / max

without redesigning the experiment loop.

# 5. Goal B: make the candidate-generation prompt configurable

Currently the candidate-generation prompt is embedded directly in the experiment orchestration script.

Refactor it so that prompt experiments can be performed without editing the main autonomous loop.

Separate:

* experiment orchestration
* candidate-generation prompt template
* history/context selection policy
* runtime values inserted into the prompt

Prefer a simple, transparent architecture suitable for a small research repository.

For example, this may involve:

* one or more prompt template files
* prompt-related configuration
* a small prompt-building helper

Do not introduce an unnecessary framework.

# 6. History/context experiments

At minimum, support these two history modes:

1. full history
2. most recent N experiment records

The configuration should support something conceptually similar to:

HISTORY_MODE=all

and:

HISTORY_MODE=recent
HISTORY_LIMIT=10

The exact names and mechanism are up to you.

The current/full-history behavior should remain the default unless there is a strong reason otherwise.

CRITICAL EXPERIMENTAL REQUIREMENT:

When recent-history mode is selected, the candidate-generating Codex must not be able to bypass the condition simply because the prompt tells it to read the complete `results.jsonl`.

For example, if the experiment condition is `recent-10`, Codex should actually receive only the intended recent-history context for the experiment.

Do not accidentally leak the complete history through:

* prompt instructions
* alternate generated files
* explicit paths in the prompt
* another automatically included history representation

The purpose of the experiment is to control the information supplied through the candidate-generation prompt.

Do not over-engineer filesystem isolation beyond what is necessary for this experiment, but make the intended context policy real rather than cosmetic.

# 7. Prompt templates

Make the candidate-generation prompt itself easy to replace.

I want to be able to compare prompt strategies such as:

* current/default prompt
* more exploratory instructions
* more conservative instructions
* different descriptions of previous failures
* different instructions about one-factor-at-a-time changes

The orchestration loop should not have to be rewritten to perform these comparisons.

Use a template file or similarly transparent representation.

Preserve the important behavioral constraints of the current candidate-generation workflow by default:

* candidate Codex proposes exactly one experiment
* candidate Codex edits only `train.py`
* candidate Codex does not run the expensive training itself
* shell/orchestration code owns training execution
* shell/orchestration code owns keep/discard decisions
* shell/orchestration code owns Git operations
* shell/orchestration code owns experiment result recording
* candidate Codex returns a concise description of its proposed change

# 8. Prompt reproducibility

Every autonomous experiment run must leave enough metadata to reconstruct the prompt condition that produced it.

Record appropriate prompt metadata in `results.jsonl`, run-level metadata, or iteration artifacts.

At minimum, preserve:

* prompt template identifier or path
* prompt template version or content hash
* history mode
* history limit when applicable
* number of history records actually supplied
* primary metric
* objective direction

Continue saving the fully rendered prompt for each candidate iteration as an artifact.

This is important because I eventually want to compare experiment conditions such as:

val_bpb + full history + prompt A

versus:

val_bpb + recent-10 + prompt A

versus:

val_loss + recent-10 + prompt B

without ambiguity about the condition used.

# 9. Experimental variables must be independent

After the refactoring, treat these as independent experimental variables:

A. Training configuration
Examples:

* learning rate
* model architecture
* optimizer
* attention configuration

B. Objective configuration
Examples:

* val_bpb / min
* val_loss / min
* future task_accuracy / max

C. Prompt configuration
Examples:

* default prompt
* alternative prompt template
* full history
* recent-10 history

Changing B or C must not require rewriting the core autonomous experiment loop.

Do not couple the selected objective unnecessarily to the selected prompt policy.

# 10. Preserve current behavior by default

The default configuration should remain as close as practical to the existing experiment behavior.

Default behavior should be:

* primary metric: val_bpb
* objective direction: min
* current/default candidate-generation instructions
* full experiment history
* shell script owns training and evaluation orchestration
* shell script owns keep/discard
* shell script owns Git commits
* shell script owns results and artifacts
* candidate Codex modifies only train.py

Do not change the scientific meaning of the current BPB baseline as a side effect of the refactoring.

Do not break the existing HPC/Singularity workflow.

Use the generic term "HPC environment" in documentation rather than relying on a machine-specific name when possible.

# 11. Keep the implementation easy to inspect

This is research code.

Prioritize:

* reproducibility
* explicit configuration
* minimal hidden behavior
* simple files
* understandable experiment metadata
* easy comparison between experimental conditions

Avoid:

* unnecessary classes
* elaborate configuration frameworks
* excessive abstraction
* large new dependencies
* unrelated cleanup

Refactor only as much as needed to separate the experimental variables cleanly.

# 12. Validation

After implementation, perform lightweight validation.

At minimum validate:

* shell syntax with `bash -n` for modified shell scripts
* Python syntax/imports where practical
* objective comparison for `min`
* objective comparison for `max`
* parsing old JSONL records containing top-level `val_bpb`
* writing/reading new multi-metric JSONL records
* default `val_bpb` behavior
* `val_loss` objective selection
* full-history prompt construction
* recent-N prompt construction
* recent-N with fewer than N existing records
* rendered prompt metadata
* prompt template selection

Use synthetic/mock result records and lightweight commands instead of running expensive training.

If a useful check cannot run because the local environment lacks GPU/runtime dependencies, report that clearly rather than attempting an expensive workaround.

# 13. Documentation

Add concise documentation showing exactly how to run at least these configurations:

1. Default BPB optimization with full history

2. Validation-loss optimization with full history

3. BPB optimization with only the most recent 10 experiments supplied as candidate-generation history

4. BPB optimization using an alternative prompt template, if the template interface supports this directly

Show the exact configuration values or commands.

Also briefly document:

* meaning of the supported metrics
* min/max objective direction
* new results.jsonl fields
* how to add a future metric
* how to add a future prompt template

# 14. Implementation workflow

Follow this workflow autonomously:

1. Inspect the repository.
2. Understand the existing experiment lifecycle.
3. Identify objective-specific and prompt-specific coupling.
4. Design a minimal refactoring.
5. Implement it.
6. Run lightweight validation.
7. Fix issues found by validation.
8. Inspect the final diff for accidental unrelated changes.
9. Report the completed implementation.

Do not stop after producing a plan.

Do not ask me whether you should proceed.

Do not ask for approval before modifying repository files or running normal lightweight development commands.

Do not wait for additional instructions when a reasonable implementation decision can be made from the repository.

Do not make Git commits or push to the remote unless that is already strictly required by a validation path. Leave the final changes visible in the working tree so I can inspect them.

# 15. Final response

When the implementation is finished, give a concise but complete report containing:

1. Files changed or added
2. Architecture chosen
3. Metric/objective configuration interface
4. Prompt/history configuration interface
5. New results.jsonl representation
6. Backward-compatibility behavior
7. Validation performed and results
8. Exact examples for:

   * val_bpb + full history
   * val_loss + full history
   * val_bpb + recent-10
9. How a future task-performance metric should be added
10. Any remaining limitations or decisions I should know about

Do not merely tell me what could be implemented. Complete the implementation first.

