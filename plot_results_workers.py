"""Plot combined and per-worker progress with a paginated change/result ledger."""

import argparse
import math
import textwrap
from collections import defaultdict
from pathlib import Path

from experiment_utils import is_better, load_jsonl, metric_value
from plot_results import infer_direction


DEFAULT_RESULTS = Path(
    "~/experiments/autoresearch/parallel-strategy-diversity/20260916_224138/results.jsonl"
).expanduser()
COLORS = {"keep": "#238b45", "discard": "#d97706", "crash": "#cb181d"}


def plot_all_workers(plt, workers, output_dir, metric, direction):
    """Overlay every worker's metric history in one round-based plot."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    worker_ids = sorted(workers)
    color_map = plt.get_cmap("tab20", max(len(worker_ids), 1))
    fig, ax = plt.subplots(figsize=(18, 9), layout="constrained")
    plotted_workers = []

    for color_index, worker in enumerate(worker_ids):
        trials = workers[worker]
        points = [
            (record.get("round"), metric_value(record, metric), str(record.get("status", "unknown")))
            for record in trials
            if record.get("round") is not None and record.get("status") != "crash"
        ]
        points = [(round_id, value, status) for round_id, value, status in points if value is not None]
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        color = color_map(color_index)
        ax.plot(
            [point[0] for point in points], [point[1] for point in points],
            color=color, linewidth=1.5, alpha=0.65, zorder=1,
        )
        for status, marker in (("keep", "o"), ("discard", "x")):
            status_points = [point for point in points if point[2] == status]
            if status_points:
                ax.scatter(
                    [point[0] for point in status_points], [point[1] for point in status_points],
                    color=color, marker=marker, s=55 if status == "keep" else 42,
                    linewidths=1.4, zorder=3,
                )
        other_points = [point for point in points if point[2] not in {"keep", "discard"}]
        if other_points:
            ax.scatter(
                [point[0] for point in other_points], [point[1] for point in other_points],
                color=color, marker="s", s=38, zorder=3,
            )
        plotted_workers.append((worker, color))

    if not plotted_workers:
        raise ValueError(f"No worker records with valid round and {metric} were found")
    worker_handles = [
        Line2D([0], [0], color=color, marker="o", linewidth=1.5, label=f"Worker {worker}")
        for worker, color in plotted_workers
    ]
    status_handles = [
        Line2D([0], [0], color="0.25", marker="o", linestyle="none", label="keep"),
        Line2D([0], [0], color="0.25", marker="x", linestyle="none", label="discard"),
        Line2D([0], [0], color="0.25", marker="s", linestyle="none", label="other"),
    ]
    worker_legend = ax.legend(handles=worker_handles, title="Workers", loc="upper left",
                              bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.add_artist(worker_legend)
    ax.legend(handles=status_handles, title="Status", loc="lower left",
              bbox_to_anchor=(1.01, 0), fontsize=9)
    arrow = "lower is better" if direction == "min" else "higher is better"
    ax.set_xlabel("Round")
    ax.set_ylabel(f"{metric} ({arrow})")
    ax.set_title(f"All Workers: {metric} by Round")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    output = output_dir / "all_workers.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return output


def plot_results(records, output_dir, metric="val_bpb", direction=None, rows_per_page=15):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if rows_per_page < 1:
        raise ValueError("rows_per_page must be positive")
    direction = infer_direction(records, metric, direction)
    workers = defaultdict(list)
    for record in records:
        worker = record.get("worker")
        if worker is None:
            continue  # Shared baseline is not a worker trial.
        if isinstance(worker, bool) or not isinstance(worker, int) or worker < 0:
            raise ValueError(f"Invalid worker ID: {worker!r}")
        workers[worker].append(record)
    if not workers:
        raise ValueError("No records with worker IDs were found")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for trials in workers.values():
        trials.sort(key=lambda r: (
            r.get("round") if r.get("round") is not None else math.inf,
            r.get("iteration") if r.get("iteration") is not None else math.inf,
        ))
    outputs = [plot_all_workers(plt, workers, output_dir, metric, direction)]
    # Disable math parsing so descriptions containing '$' remain literal text.
    with plt.rc_context({"text.parse_math": False}):
        for worker, trials in sorted(workers.items()):
            values = [metric_value(r, metric) if r.get("status") != "crash" else None for r in trials]
            best, running_best = None, []
            for value in values:
                if value is not None and (best is None or is_better(value, best, direction)):
                    best = value
                running_best.append(best if best is not None else math.nan)
            pages = math.ceil(len(trials) / rows_per_page)
            for page, start in enumerate(range(0, len(trials), rows_per_page), 1):
                end = min(start + rows_per_page, len(trials))
                rows = []
                for index in range(start, end):
                    record = trials[index]
                    description = "\n".join(textwrap.wrap(
                        str(record.get("description") or "(no description)"), width=100
                    ))
                    rows.append([
                        str(index + 1), str(record.get("round", "—")),
                        str(record.get("iteration", "—")), str(record.get("status", "unknown")),
                        f"{values[index]:.6f}" if values[index] is not None else "N/A",
                        description,
                    ])
                heights = [max(0.38, 0.17 * (row[-1].count("\n") + 1) + 0.16) for row in rows]
                table_height = sum(heights) + 0.4
                fig, (ax, ledger) = plt.subplots(
                    2, 1, figsize=(18, 5 + table_height),
                    gridspec_kw={"height_ratios": [4, table_height]}, layout="constrained",
                )
                x = list(range(1, len(trials) + 1))
                ax.plot(x, [v if v is not None else math.nan for v in values],
                        color="0.65", linewidth=1, zorder=1)
                for status in sorted({str(r.get("status", "unknown")) for r in trials}):
                    indices = [i for i, r in enumerate(trials)
                               if str(r.get("status", "unknown")) == status and values[i] is not None]
                    if indices:
                        ax.scatter([i + 1 for i in indices], [values[i] for i in indices],
                                   color=COLORS.get(status, "#636363"),
                                   marker="o" if status == "keep" else "x", label=status, zorder=3)
                missing = [i + 1 for i, value in enumerate(values) if value is None]
                if missing:
                    ax.scatter(missing, [0.03] * len(missing), transform=ax.get_xaxis_transform(),
                               color=COLORS["crash"], marker="v", label="crash / missing metric")
                if best is not None:
                    ax.step(x, running_best, where="post", color="#2171b5",
                            label="Worker best observed (including discard)")
                ax.axvspan(start + 0.5, end + 0.5, color="#3182bd", alpha=0.08,
                           label="Trials listed below")
                ax.set_xlim(0.5, len(trials) + 0.5)
                ax.set_xlabel("Trial within worker (ordered by round, iteration)")
                from matplotlib.ticker import MaxNLocator

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                arrow = "lower is better" if direction == "min" else "higher is better"
                ax.set_ylabel(f"{metric} ({arrow})")
                ax.set_title(f"Worker {worker} | {len(trials)} trials | page {page}/{pages}")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=9)
                ledger.axis("off")
                table = ledger.table(
                    cellText=rows, colLabels=["Trial", "Round", "Iteration", "Status", metric, "Change description"],
                    colWidths=[0.045, 0.045, 0.055, 0.065, 0.08, 0.71],
                    cellLoc="left", loc="center", bbox=[0, 0, 1, 1],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                for (row, col), cell in table.get_celld().items():
                    cell.set_height((0.4 if row == 0 else heights[row - 1]) / table_height)
                    cell.set_edgecolor("#dddddd")
                    if row == 0:
                        cell.set_facecolor("#e8edf2")
                        cell.get_text().set_weight("bold")
                    else:
                        cell.set_facecolor("#f7f7f7" if row % 2 else "white")
                        if col == 3:
                            cell.get_text().set_color(COLORS.get(rows[row - 1][3], "#636363"))
                output = output_dir / f"worker_{worker}_page_{page:03d}.png"
                fig.savefig(output, dpi=150)
                plt.close(fig)
                outputs.append(output)
                print(f"Saved: {output}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metric", default="val_bpb")
    parser.add_argument("--direction", choices=("min", "max"))
    parser.add_argument("--rows-per-page", type=int, default=15)
    parser.add_argument("-o", "--output-dir", type=Path)
    args = parser.parse_args()
    results = args.results.expanduser()
    if not results.is_file():
        parser.error(f"Results file not found: {results}")
    if args.rows_per_page < 1:
        parser.error("--rows-per-page must be positive")
    output = args.output_dir.expanduser() if args.output_dir else results.parent / "worker_plots"
    plot_results(load_jsonl(results), output, args.metric, args.direction, args.rows_per_page)


if __name__ == "__main__":
    main()
