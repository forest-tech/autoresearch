import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot val_bpb progress from results.jsonl."
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to results.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: progress.png next to results.jsonl",
    )
    return parser.parse_args()


def load_results(path: Path):
    results = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            results.append(json.loads(line))

    return results


def main():
    args = parse_args()

    results_file = args.results_file

    if not results_file.exists():
        raise FileNotFoundError(f"Not found: {results_file}")

    output_file = (
        args.output
        if args.output is not None
        else results_file.parent / "progress.png"
    )

    results = load_results(results_file)

    if not results:
        raise RuntimeError(f"No results found in {results_file}")

    iterations = [r["iteration"] for r in results]
    bpb = [r["val_bpb"] for r in results]

    keep_results = [r for r in results if r["status"] == "keep"]
    discard_results = [r for r in results if r["status"] == "discard"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # 全実験
    ax.plot(
        iterations,
        bpb,
        marker="o",
        linewidth=1.5,
        alpha=0.5,
        label="All experiments",
    )

    # keep
    ax.scatter(
        [r["iteration"] for r in keep_results],
        [r["val_bpb"] for r in keep_results],
        marker="o",
        s=70,
        label="keep",
        zorder=3,
    )

    # discard
    ax.scatter(
        [r["iteration"] for r in discard_results],
        [r["val_bpb"] for r in discard_results],
        marker="x",
        s=70,
        label="discard",
        zorder=3,
    )

    # accepted BPB
    accepted_iterations = []
    accepted_bpb = []

    current_best = None

    for r in results:
        if current_best is None:
            current_best = r["val_bpb"]

        if r["status"] == "keep":
            current_best = r["val_bpb"]

        accepted_iterations.append(r["iteration"])
        accepted_bpb.append(current_best)

    ax.step(
        accepted_iterations,
        accepted_bpb,
        where="post",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Accepted BPB",
    )

    # keepされた改善点に注釈
    annotation_index = 0

    for r in keep_results:
        if r["iteration"] == results[0]["iteration"]:
            continue

        description = r.get("description", "")
        description = "\n".join(
            textwrap.wrap(description, width=35)
        )

        previous_keeps = [
            x
            for x in keep_results
            if x["iteration"] < r["iteration"]
        ]

        improvement = ""

        if previous_keeps:
            previous_bpb = previous_keeps[-1]["val_bpb"]
            delta = r["val_bpb"] - previous_bpb
            improvement = f"\nΔBPB = {delta:+.6f}"

        annotation_text = description + improvement

        xytext = (
            15,
            35 if annotation_index % 2 == 0 else -65,
        )

        ax.annotate(
            annotation_text,
            xy=(r["iteration"], r["val_bpb"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8,
            arrowprops={
                "arrowstyle": "->",
                "alpha": 0.6,
            },
            bbox={
                "boxstyle": "round,pad=0.3",
                "alpha": 0.8,
            },
        )

        annotation_index += 1

    # best
    best = min(results, key=lambda x: x["val_bpb"])

    ax.axhline(
        best["val_bpb"],
        linestyle=":",
        linewidth=1,
        alpha=0.6,
    )

    ax.text(
        iterations[-1] + 0.1,
        best["val_bpb"],
        f'best = {best["val_bpb"]:.6f}',
        va="center",
        fontsize=9,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation BPB ↓")
    ax.set_title("Autoresearch Experiment Progress")

    ax.set_xticks(iterations)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend()
    ax.margins(x=0.05, y=0.15)

    fig.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Saved: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
