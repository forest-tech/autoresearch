import argparse
import textwrap
from pathlib import Path

from experiment_utils import is_better, load_jsonl, metric_value


def parse_args():
    parser = argparse.ArgumentParser(description="Plot metric progress from results.jsonl.")
    parser.add_argument("results_file", type=Path, help="Path to results.jsonl")
    parser.add_argument("--metric", default="val_bpb", help="Metric to plot (default: val_bpb)")
    parser.add_argument("--direction", choices=("min", "max"), help="Override objective direction")
    parser.add_argument("-o", "--output", type=Path, help="Default: progress.png beside results")
    return parser.parse_args()


def infer_direction(records, metric, explicit=None):
    if explicit:
        return explicit
    directions = {
        record.get("objective_direction")
        for record in records
        if record.get("primary_metric") == metric
        and record.get("objective_direction") in {"min", "max"}
    }
    if len(directions) > 1:
        raise ValueError(f"mixed objective directions for {metric}; pass --direction")
    return next(iter(directions), "min")


def main():
    import matplotlib.pyplot as plt

    args = parse_args()
    if not args.results_file.exists():
        raise FileNotFoundError(f"Not found: {args.results_file}")

    records = load_jsonl(args.results_file)
    points = [
        (record, metric_value(record, args.metric))
        for record in records
        if record.get("status") != "crash"
        and metric_value(record, args.metric) is not None
    ]
    if not points:
        raise RuntimeError(f"No records with metric {args.metric!r}")
    direction = infer_direction(records, args.metric, args.direction)
    output = args.output or args.results_file.parent / "progress.png"
    kept = [(record, value) for record, value in points if record.get("status") == "keep"]
    discarded = [(record, value) for record, value in points if record.get("status") == "discard"]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        [record["iteration"] for record, _ in points],
        [value for _, value in points],
        marker="o", linewidth=1.5, alpha=0.5, label="All experiments",
    )
    if kept:
        ax.scatter(
            [record["iteration"] for record, _ in kept], [value for _, value in kept],
            marker="o", s=70, label="keep", zorder=3,
        )
    if discarded:
        ax.scatter(
            [record["iteration"] for record, _ in discarded], [value for _, value in discarded],
            marker="x", s=70, label="discard", zorder=3,
        )

    accepted_x, accepted_y = [], []
    current = None
    for record, value in points:
        if current is None or record.get("status") == "keep":
            current = value
        accepted_x.append(record["iteration"])
        accepted_y.append(current)
    ax.step(accepted_x, accepted_y, where="post", linestyle="--", linewidth=2, alpha=0.8, label="Accepted")

    previous = None
    annotation_index = 0
    for record, value in kept:
        if previous is None:
            previous = value
            continue
        description = "\n".join(textwrap.wrap(record.get("description", ""), width=35))
        delta = value - previous
        text = f"{description}\nΔ = {delta:+.6f}".strip()
        ax.annotate(
            text, xy=(record["iteration"], value),
            xytext=(15, 35 if annotation_index % 2 == 0 else -65),
            textcoords="offset points", fontsize=8,
            arrowprops={"arrowstyle": "->", "alpha": 0.6},
            bbox={"boxstyle": "round,pad=0.3", "alpha": 0.8},
        )
        previous = value
        annotation_index += 1

    best = points[0][1]
    for _, value in points[1:]:
        if is_better(value, best, direction):
            best = value
    ax.axhline(best, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(points[-1][0]["iteration"] + 0.1, best, f"best = {best:.6f}", va="center", fontsize=9)
    arrow = "↓" if direction == "min" else "↑"
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"{args.metric} {arrow}")
    ax.set_title(f"Autoresearch Experiment Progress: {args.metric}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.margins(x=0.05, y=0.15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
