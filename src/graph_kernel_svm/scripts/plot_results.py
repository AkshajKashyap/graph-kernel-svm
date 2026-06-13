"""Plot graph-kernel experiment results from CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

matplotlib_config_dir = Path(tempfile.gettempdir()) / "graph-kernel-svm-matplotlib"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_results(
    csv_path: str | Path = "outputs/mutag_kernel_comparison.csv",
    figures_dir: str | Path = "outputs/figures",
) -> list[Path]:
    """Create accuracy, macro F1, and kernel runtime comparison charts."""

    rows = _read_results(csv_path)
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = [row["setting"] for row in rows]
    outputs = [
        _plot_metric(
            settings,
            values=[float(row["mean_accuracy"]) for row in rows],
            errors=[float(row["std_accuracy"]) for row in rows],
            title="MUTAG Accuracy by Kernel",
            ylabel="Mean accuracy",
            output_path=output_dir / "mutag_accuracy_comparison.png",
            ylim=(0.0, 1.05),
        ),
        _plot_metric(
            settings,
            values=[float(row["mean_macro_f1"]) for row in rows],
            errors=[float(row["std_macro_f1"]) for row in rows],
            title="MUTAG Macro F1 by Kernel",
            ylabel="Mean macro F1",
            output_path=output_dir / "mutag_macro_f1_comparison.png",
            ylim=(0.0, 1.05),
        ),
        _plot_metric(
            settings,
            values=[float(row["kernel_time_seconds"]) for row in rows],
            errors=[0.0] * len(rows),
            title="MUTAG Kernel Computation Time",
            ylabel="Seconds",
            output_path=output_dir / "mutag_kernel_time_comparison.png",
        ),
    ]
    return outputs


def _read_results(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path)
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Experiment results CSV must contain at least one row.")
    required_columns = {
        "setting",
        "mean_accuracy",
        "std_accuracy",
        "mean_macro_f1",
        "std_macro_f1",
        "kernel_time_seconds",
    }
    missing = required_columns.difference(rows[0])
    if missing:
        raise ValueError(f"Experiment results CSV is missing columns: {sorted(missing)}")
    return rows


def _plot_metric(
    settings: list[str],
    values: list[float],
    errors: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
) -> Path:
    positions = np.arange(len(settings))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=4,
        color="#2f6f8f",
        edgecolor="#173b4d",
        linewidth=0.7,
    )
    axis.set_title(title, pad=12)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, labels=settings, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/mutag_kernel_comparison.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output_path in plot_results(args.input, args.figures_dir):
        print(output_path)


if __name__ == "__main__":
    main()
