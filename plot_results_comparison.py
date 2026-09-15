"""Compare parallel runs: uv run plot_results_comparison.py run1/results.jsonl run2/results.jsonl."""

import argparse
import math
import re
from pathlib import Path

from experiment_utils import is_better, load_jsonl, metric_value
from plot_results import infer_direction


def round_points(records, metric):
    points = []
    for record in records:
        round_id = record.get("round")
        value = metric_value(record, metric)
        if (record.get("status") != "crash" and value is not None
                and isinstance(round_id, (int, float))
                and not isinstance(round_id, bool) and math.isfinite(round_id)):
            points.append((round_id, value, record.get("status") == "keep"))
    return sorted(points, key=lambda point: point[0])


def running_best(points, direction):
    rounds, values = [], []
    best = None
    for round_id, value, _ in points:
        if best is None or is_better(value, best, direction):
            best = value
        if rounds and rounds[-1] == round_id:
            values[-1] = best
        else:
            rounds.append(round_id)
            values.append(best)
    return rounds, values


def default_labels(paths):
    # Include enough parent directories to distinguish experiments with the same date.
    for depth in range(1, max(len(path.parts) for path in paths) + 1):
        labels = ["/".join(path.parent.parts[-depth:]) for path in paths]
        if len(set(labels)) == len(labels):
            return labels
    return [str(path) for path in paths]


def main():
    parser = argparse.ArgumentParser(description="Compare running best metrics across parallel runs.")
    parser.add_argument("results", nargs="+", type=Path, help="Two or more results.jsonl paths")
    parser.add_argument("--labels", nargs="+", help="Legend names, in input order")
    parser.add_argument("--metric", default="val_bpb")
    parser.add_argument("--direction", choices=("min", "max"))
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument("--best-only", action="store_true", help="Hide individual experiment points")
    parser.add_argument("-o", "--output", type=Path, help="Default: results/comparisons/parallel_comparison_<metric>.png")
    args = parser.parse_args()
    paths = [path.expanduser().resolve() for path in args.results]
    if len(paths) < 2:
        parser.error("provide at least two results.jsonl files")
    if len(set(paths)) != len(paths):
        parser.error("input files must be distinct")
    if args.labels is not None and len(args.labels) != len(paths):
        parser.error("--labels must contain one name per input file")

    runs = []
    for path in paths:
        if not path.is_file():
            parser.error(f"file not found: {path}")
        try:
            records = load_jsonl(path)
        except ValueError as error:
            parser.error(f"{path}: {error}")
        points = round_points(records, args.metric)
        if not points:
            parser.error(f"{path}: no valid round records for {args.metric}")
        runs.append((records, points))
    try:
        direction = infer_direction([record for records, _ in runs for record in records], args.metric, args.direction)
    except ValueError as error:
        parser.error(str(error))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    labels = args.labels or default_labels(paths)
    fig, ax = plt.subplots(figsize=(14, 8))
    for index, ((_, points), label) in enumerate(zip(runs, labels)):
        color = plt.get_cmap("tab20")(index % 20)
        rounds, values = running_best(points, direction)
        ax.step(rounds, values, where="post", color=color, linewidth=2,
                marker="o", markersize=3, label=label, zorder=3)
        if not args.best_only:
            for kept in (False, True):
                selected = [(x, y) for x, y, accepted in points if accepted == kept]
                if selected:
                    ax.scatter(*zip(*selected), color=color, s=40 if kept else 18,
                               alpha=0.9 if kept else 0.2,
                               edgecolors="black" if kept else "none",
                               linewidths=0.5, zorder=4 if kept else 2)
    arrow = "lower is better" if direction == "min" else "higher is better"
    ax.set_title(args.title or f"Parallel Run Comparison: {args.metric} ({len(runs)} runs)")
    ax.set_xlabel("Round")
    ax.set_ylabel(f"{args.metric} ({arrow})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)
    handles, legend_labels = ax.get_legend_handles_labels()
    if not args.best_only:
        handles.extend([
            Line2D([], [], color="gray", linewidth=2, label="Running best"),
            Line2D([], [], color="gray", marker="o", linestyle="none", markeredgecolor="black", label="Kept"),
            Line2D([], [], color="gray", marker="o", linestyle="none", alpha=0.3, label="Discarded / other"),
        ])
        legend_labels.extend(["Running best", "Kept", "Discarded / other"])
    ax.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    safe_metric = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.metric)
    output = args.output or Path("results/comparisons") / f"parallel_comparison_{safe_metric}.png"
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to: {output.resolve()}")


if __name__ == "__main__":
    main()
