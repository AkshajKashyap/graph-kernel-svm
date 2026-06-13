"""Compare graph kernels across repeated stratified train/test splits."""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from graph_kernel_svm.data import DatasetSummary, summarize_dataset
from graph_kernel_svm.graphs import GraphExample
from graph_kernel_svm.scripts.train_baseline import _build_kernel, _load_dataset


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Aggregated metrics for one kernel setting."""

    kernel: str
    wl_iterations: int | None
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float

    @property
    def setting(self) -> str:
        return self.kernel if self.wl_iterations is None else f"wl_{self.wl_iterations}"


def run_kernel_experiments(
    examples: list[GraphExample],
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
    normalize: bool = False,
) -> list[ExperimentResult]:
    """Evaluate stats and WL kernels on shared repeated stratified splits."""

    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    labels = np.array([example.label for example in examples])
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed,
    )
    splits = list(splitter.split(np.zeros(len(examples)), labels))
    settings = [("stats", None), *(("wl", iteration) for iteration in range(6))]

    results = []
    for kernel_name, wl_iterations in settings:
        kernel_matrix = _build_kernel(
            examples,
            kernel=kernel_name,
            wl_iterations=wl_iterations or 0,
            normalize=normalize,
        )
        accuracies = []
        macro_f1_scores = []
        for train_indices, test_indices in splits:
            train_kernel = kernel_matrix[np.ix_(train_indices, train_indices)]
            test_kernel = kernel_matrix[np.ix_(test_indices, train_indices)]
            classifier = SVC(kernel="precomputed")
            classifier.fit(train_kernel, labels[train_indices])
            predictions = classifier.predict(test_kernel)
            accuracies.append(accuracy_score(labels[test_indices], predictions))
            macro_f1_scores.append(
                f1_score(labels[test_indices], predictions, average="macro", zero_division=0)
            )

        results.append(
            ExperimentResult(
                kernel=kernel_name,
                wl_iterations=wl_iterations,
                mean_accuracy=float(np.mean(accuracies)),
                std_accuracy=float(np.std(accuracies)),
                mean_macro_f1=float(np.mean(macro_f1_scores)),
                std_macro_f1=float(np.std(macro_f1_scores)),
            )
        )
    return results


def write_results_csv(results: list[ExperimentResult], output_path: str | Path) -> Path:
    """Write aggregated experiment results as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "setting",
                "kernel",
                "wl_iterations",
                "mean_accuracy",
                "std_accuracy",
                "mean_macro_f1",
                "std_macro_f1",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "setting": result.setting,
                    "kernel": result.kernel,
                    "wl_iterations": (
                        "" if result.wl_iterations is None else result.wl_iterations
                    ),
                    "mean_accuracy": f"{result.mean_accuracy:.6f}",
                    "std_accuracy": f"{result.std_accuracy:.6f}",
                    "mean_macro_f1": f"{result.mean_macro_f1:.6f}",
                    "std_macro_f1": f"{result.std_macro_f1:.6f}",
                }
            )
    return path


def write_markdown_report(
    results: list[ExperimentResult],
    summary: DatasetSummary,
    dataset: str,
    command: str,
    output_path: str | Path,
) -> Path:
    """Write a readable Markdown experiment report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# MUTAG Kernel Comparison",
        "",
        "## Dataset Summary",
        "",
        f"- Dataset: `{dataset}`",
        f"- Graphs: {summary.num_graphs}",
        f"- Class balance: `{summary.class_balance}`",
        f"- Average nodes: {summary.avg_nodes:.2f}",
        f"- Average edges: {summary.avg_edges:.2f}",
        "",
        "## Reproduction",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Results",
        "",
        "| Setting | Mean accuracy | Std accuracy | Mean macro F1 | Std macro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    rows.extend(
        "| "
        f"{result.setting} | {result.mean_accuracy:.4f} | {result.std_accuracy:.4f} | "
        f"{result.mean_macro_f1:.4f} | {result.std_macro_f1:.4f} |"
        for result in results
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = _load_dataset(args.dataset, args.data_root)
    summary = summarize_dataset(examples)
    results = run_kernel_experiments(
        examples,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        normalize=args.normalize,
    )
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "graph_kernel_svm.scripts.run_experiments",
            *sys.argv[1:],
        ]
    )
    csv_path = write_results_csv(results, "outputs/mutag_kernel_comparison.csv")
    report_path = write_markdown_report(
        results,
        summary=summary,
        dataset=args.dataset,
        command=command,
        output_path="reports/mutag_kernel_comparison.md",
    )

    for result in results:
        print(
            f"{result.setting}: accuracy={result.mean_accuracy:.3f}+/-{result.std_accuracy:.3f} "
            f"macro_f1={result.mean_macro_f1:.3f}+/-{result.std_macro_f1:.3f}"
        )
    print(f"csv={csv_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
