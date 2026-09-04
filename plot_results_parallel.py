import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {e}"
                ) from e

            if "round" not in record:
                continue

            records.append(record)

    if not records:
        raise ValueError("No experiment records were found.")

    return sorted(
        records,
        key=lambda r: (
            r["round"],
            r.get("worker") is not None,
            r.get("worker") if r.get("worker") is not None else -1,
        ),
    )


def shorten_description(description: str, width: int = 42) -> str:
    if not description:
        return ""

    description = " ".join(description.split())

    return textwrap.shorten(
        description,
        width=width,
        placeholder="...",
    )


def plot_results(records, output_path: Path):
    valid_records = [
        r for r in records
        if r.get("val_bpb") is not None
    ]

    if not valid_records:
        raise ValueError("No records with valid val_bpb were found.")

    kept_records = [
        r for r in valid_records
        if r.get("status") == "keep"
    ]

    discarded_records = [
        r for r in valid_records
        if r.get("status") != "keep"
    ]

    # ---------------------------------------------
    # Running best by round
    # ---------------------------------------------
    rounds = sorted({
        r["round"]
        for r in valid_records
    })

    running_best_x = []
    running_best_y = []

    best = float("inf")

    for round_id in rounds:
        round_records = [
            r for r in valid_records
            if r["round"] == round_id
        ]

        round_values = [
            r["val_bpb"]
            for r in round_records
        ]

        if round_values:
            best = min(best, min(round_values))

        running_best_x.append(round_id)
        running_best_y.append(best)

    # ---------------------------------------------
    # Figure
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 8))

    # Discarded
    if discarded_records:
        ax.scatter(
            [r["round"] for r in discarded_records],
            [r["val_bpb"] for r in discarded_records],
            s=18,
            alpha=0.22,
            edgecolors="none",
            label="Discarded",
            zorder=2,
        )

    # Kept
    if kept_records:
        ax.scatter(
            [r["round"] for r in kept_records],
            [r["val_bpb"] for r in kept_records],
            s=55,
            edgecolors="black",
            linewidths=0.6,
            label="Kept",
            zorder=4,
        )

    # Running best
    ax.step(
        running_best_x,
        running_best_y,
        where="post",
        linewidth=2.0,
        alpha=0.75,
        label="Running best",
        zorder=3,
    )

    # ---------------------------------------------
    # Annotations
    # ---------------------------------------------
    for r in kept_records:
        x = r["round"]
        y = r["val_bpb"]

        description = shorten_description(
            r.get("description", "")
        )

        if not description:
            continue

        ax.annotate(
            description,
            xy=(x, y),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=9,
            rotation=28,
            ha="left",
            va="bottom",
            alpha=0.9,
            annotation_clip=True,
        )

    # ---------------------------------------------
    # Labels / title
    # ---------------------------------------------
    num_experiments = len(records)

    num_kept = sum(
        1
        for r in records
        if r.get("status") == "keep"
    )

    num_improvements = max(0, num_kept - 1)

    ax.set_title(
        f"Autoresearch Progress: "
        f"{num_experiments} Experiments, "
        f"{num_improvements} Kept Improvements",
        fontsize=16,
    )

    ax.set_xlabel(
        "Round",
        fontsize=13,
    )

    ax.set_ylabel(
        "Validation BPB (lower is better)",
        fontsize=13,
    )

    # ---------------------------------------------
    # Appearance
    # ---------------------------------------------
    ax.grid(
        True,
        alpha=0.18,
        linewidth=0.8,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
    )

    # round を整数目盛りにする
    ax.set_xticks(rounds)

    values = [
        r["val_bpb"]
        for r in valid_records
    ]

    ymin = min(values)
    ymax = max(values)

    margin = max(
        (ymax - ymin) * 0.04,
        0.0005,
    )

    ax.set_ylim(
        ymin - margin,
        ymax + margin * 3,
    )

    ax.margins(x=0.02)

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot autoresearch experiment progress."
    )

    parser.add_argument(
        "results",
        type=Path,
        help="Path to results.jsonl",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path",
    )

    args = parser.parse_args()

    if args.output is None:
        output_path = (
            args.results.parent
            / "progress.png"
        )
    else:
        output_path = args.output

    records = load_results(
        args.results
    )

    plot_results(
        records,
        output_path,
    )


if __name__ == "__main__":
    main()