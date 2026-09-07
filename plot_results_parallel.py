import argparse
import textwrap
from pathlib import Path

from experiment_utils import is_better, load_jsonl, metric_value
from plot_results import infer_direction


def shorten(description: str, width: int = 42) -> str:
    return textwrap.shorten(" ".join(description.split()), width=width, placeholder="...")


def plot_results(records, output_path: Path, metric="val_bpb", direction=None):
    import matplotlib.pyplot as plt

    records = [record for record in records if "round" in record and record.get("round") is not None]
    records.sort(key=lambda record: (record["round"], record.get("worker") is not None, record.get("worker") or -1))
    points = [
        (record, metric_value(record, metric))
        for record in records
        if record.get("status") != "crash"
        and metric_value(record, metric) is not None
    ]
    if not points:
        raise ValueError(f"No records with valid {metric} were found.")
    direction = infer_direction(records, metric, direction)
    kept = [(record, value) for record, value in points if record.get("status") == "keep"]
    discarded = [(record, value) for record, value in points if record.get("status") != "keep"]
    rounds = sorted({record["round"] for record, _ in points})

    running_x, running_y = [], []
    best = None
    for round_id in rounds:
        values = [value for record, value in points if record["round"] == round_id]
        for value in values:
            if best is None or is_better(value, best, direction):
                best = value
        running_x.append(round_id)
        running_y.append(best)

    fig, ax = plt.subplots(figsize=(18, 8))
    if discarded:
        ax.scatter(
            [record["round"] for record, _ in discarded], [value for _, value in discarded],
            s=18, alpha=0.22, edgecolors="none", label="Discarded", zorder=2,
        )
    if kept:
        ax.scatter(
            [record["round"] for record, _ in kept], [value for _, value in kept],
            s=55, edgecolors="black", linewidths=0.6, label="Kept", zorder=4,
        )
    ax.step(running_x, running_y, where="post", linewidth=2, alpha=0.75, label="Running best", zorder=3)
    for record, value in kept:
        description = shorten(record.get("description", ""))
        if description:
            ax.annotate(
                description, xy=(record["round"], value), xytext=(6, 7),
                textcoords="offset points", fontsize=9, rotation=28,
                ha="left", va="bottom", alpha=0.9, annotation_clip=True,
            )

    arrow = "lower is better" if direction == "min" else "higher is better"
    improvements = max(0, len(kept) - 1)
    ax.set_title(f"Autoresearch Progress: {len(records)} Experiments, {improvements} Kept Improvements", fontsize=16)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel(f"{metric} ({arrow})", fontsize=13)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks(rounds)
    values = [value for _, value in points]
    margin = max((max(values) - min(values)) * 0.04, 0.0005)
    ax.set_ylim(min(values) - margin, max(values) + margin * 3)
    ax.margins(x=0.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot parallel autoresearch progress.")
    parser.add_argument("results", type=Path, help="Path to results.jsonl")
    parser.add_argument("--metric", default="val_bpb", help="Metric to plot (default: val_bpb)")
    parser.add_argument("--direction", choices=("min", "max"), help="Override objective direction")
    parser.add_argument("-o", "--output", type=Path, help="Output image path")
    args = parser.parse_args()
    output = args.output or args.results.parent / "progress.png"
    plot_results(load_jsonl(args.results), output, args.metric, args.direction)


if __name__ == "__main__":
    main()
