"""Plot multi-dataset graph-kernel experiment results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from graph_kernel_svm.scripts.plot_results import plt


def plot_all_results(
    csv_path: str | Path = "outputs/all_datasets_kernel_comparison.csv",
    figures_dir: str | Path = "outputs/figures",
) -> list[Path]:
    """Create best-method and grouped kernel comparison figures."""

    rows = _read_rows(csv_path)
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_rows(rows)

    best_path = output_dir / "all_datasets_best_macro_f1.png"
    _plot_best_macro_f1(grouped, best_path)
    comparison_path = output_dir / "all_datasets_kernel_comparison.png"
    _plot_kernel_comparison(grouped, comparison_path)
    return [best_path, comparison_path]


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Multi-dataset results CSV must contain at least one row.")
    return rows


def _group_rows(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return dict(grouped)


def _plot_best_macro_f1(
    grouped: dict[str, list[dict[str, str]]],
    output_path: Path,
) -> None:
    datasets = list(grouped)
    best_rows = [
        max(rows, key=lambda row: float(row["mean_macro_f1"]))
        for rows in grouped.values()
    ]
    values = [float(row["mean_macro_f1"]) for row in best_rows]
    labels = [row["setting"] for row in best_rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(datasets, values, color="#2f6f8f", edgecolor="#173b4d")
    axis.set_title("Best Mean Macro F1 by Dataset")
    axis.set_ylabel("Mean macro F1")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.bar_label(
        bars,
        labels=[f"{value:.3f}\n{label}" for value, label in zip(values, labels, strict=True)],
        padding=4,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_kernel_comparison(
    grouped: dict[str, list[dict[str, str]]],
    output_path: Path,
) -> None:
    datasets = list(grouped)
    settings = [row["setting"] for row in next(iter(grouped.values()))]
    positions = np.arange(len(datasets))
    width = 0.8 / len(settings)
    figure, axis = plt.subplots(figsize=(12, 6))
    for index, setting in enumerate(settings):
        values = [
            float(next(row["mean_macro_f1"] for row in grouped[dataset] if row["setting"] == setting))
            for dataset in datasets
        ]
        offsets = positions - 0.4 + width / 2 + index * width
        axis.bar(offsets, values, width=width, label=setting)
    axis.set_title("Kernel Mean Macro F1 Across Datasets")
    axis.set_ylabel("Mean macro F1")
    axis.set_xticks(positions, labels=datasets)
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncols=4, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/all_datasets_kernel_comparison.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output_path in plot_all_results(args.input, args.figures_dir):
        print(output_path)


if __name__ == "__main__":
    main()
