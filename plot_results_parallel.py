import argparse
import json
import textwrap
from collections import defaultdict
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

            if "round" not in record or "val_bpb" not in record:
                continue

            # val_bpb が null の場合は 10 とする
            if record["val_bpb"] is None:
                record["val_bpb"] = 10

            records.append(record)

    if not records:
        raise ValueError(
            "No records containing both 'round' and 'val_bpb' were found."
        )

    return records


def plot_results(records, output_path: Path):
    records = sorted(
        records,
        key=lambda r: (
            r["round"],
            r.get("worker") is not None,
            r.get("worker") if r.get("worker") is not None else -1,
            r.get("iteration", 0),
        ),
    )

    by_round = defaultdict(list)
    for record in records:
        by_round[record["round"]].append(record)

    fig, ax = plt.subplots(figsize=(11, 6))

    # 全実験候補
    ax.scatter(
        [r["round"] for r in records],
        [r["val_bpb"] for r in records],
        s=55,
        alpha=0.75,
        label="Candidates",
        zorder=2,
    )

    # keep された結果をつないで、採用系列を見やすくする
    kept = [r for r in records if r.get("status") == "keep"]
    kept.sort(key=lambda r: (r["round"], r.get("iteration", 0)))

    if kept:
        ax.plot(
            [r["round"] for r in kept],
            [r["val_bpb"] for r in kept],
            marker="o",
            linewidth=2,
            label="Accepted",
            zorder=3,
        )

    # 各 round は同じ base から並列探索される想定。
    # その round が始まる前の accepted BPB より低い候補だけ注釈する。
    incumbent_bpb = None

    for round_id in sorted(by_round):
        round_records = by_round[round_id]

        if incumbent_bpb is not None:
            improved = [
                r for r in round_records
                if r["val_bpb"] < incumbent_bpb
            ]

            for i, record in enumerate(improved):
                description = record.get("description", "")
                if not description:
                    continue

                wrapped = "\n".join(
                    textwrap.wrap(description, width=42)
                )

                ax.annotate(
                    wrapped,
                    xy=(record["round"], record["val_bpb"]),
                    xytext=(12, 18 + i * 38),
                    textcoords="offset points",
                    fontsize=9,
                    arrowprops={"arrowstyle": "->"},
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "alpha": 0.85,
                    },
                    zorder=4,
                )

        # 次 round の比較基準は、この round で keep された結果。
        round_kept = [
            r for r in round_records
            if r.get("status") == "keep"
        ]

        if round_kept:
            incumbent_bpb = min(r["val_bpb"] for r in round_kept)
        elif incumbent_bpb is None:
            # baseline に status=keep が無い場合の保険
            incumbent_bpb = min(r["val_bpb"] for r in round_records)

    ax.set_xlabel("Round")
    ax.set_ylabel("BPB (↓)")
    ax.set_title("BPB by Round")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(sorted(by_round.keys()))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot val_bpb by round from results.jsonl."
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
        help="Output image path (default: <results_dir>/bpb_by_round.png)",
    )
    args = parser.parse_args()

    output_path = (
        args.output
        if args.output is not None
        else args.results.parent / "bpb_by_round.png"
    )

    records = load_results(args.results)
    plot_results(records, output_path)


if __name__ == "__main__":
    main()
