"""Compare graph kernels across repeated stratified train/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from graph_kernel_svm.data import DatasetSummary, summarize_dataset
from graph_kernel_svm.graphs import GraphExample
from graph_kernel_svm.models import DEFAULT_C_VALUES, select_best_c
from graph_kernel_svm.scripts.train_baseline import _build_kernel, _load_dataset
from graph_kernel_svm.utils import KernelDiagnostics, diagnose_kernel_matrix, get_kernel_matrix


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Aggregated metrics for one kernel setting."""

    kernel: str
    wl_iterations: int | None
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    kernel_time_seconds: float
    cache_hit: bool
    best_c: float
    c_distribution: dict[float, int]
    per_class_f1: dict[int, float]
    confusion_matrix: list[list[int]]
    kernel_diagnostics: KernelDiagnostics

    @property
    def setting(self) -> str:
        return self.kernel if self.wl_iterations is None else f"wl_{self.wl_iterations}"


def run_kernel_experiments(
    examples: list[GraphExample],
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
    normalize: bool = False,
    dataset_name: str = "synthetic",
    use_cache: bool = False,
    force_recompute: bool = False,
    cache_dir: str | Path = "outputs/cache",
    c_values: Sequence[float] = DEFAULT_C_VALUES,
) -> list[ExperimentResult]:
    """Evaluate stats, shortest-path, and WL kernels on shared splits."""

    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    labels = np.array([example.label for example in examples])
    classes = np.unique(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed,
    )
    splits = list(splitter.split(np.zeros(len(examples)), labels))
    settings = [
        ("stats", None),
        ("shortest_path", None),
        *(("wl", iteration) for iteration in range(6)),
    ]

    results = []
    for kernel_name, wl_iterations in settings:
        cached_kernel = get_kernel_matrix(
            examples=examples,
            dataset_name=dataset_name,
            kernel_name=kernel_name,
            normalize=normalize,
            wl_iterations=wl_iterations,
            use_cache=use_cache,
            force_recompute=force_recompute,
            cache_dir=cache_dir,
            compute=lambda kernel_name=kernel_name, wl_iterations=wl_iterations: _build_kernel(
                examples,
                kernel=kernel_name,
                wl_iterations=wl_iterations or 0,
                normalize=normalize,
            ),
        )
        kernel_matrix = cached_kernel.matrix
        accuracies = []
        macro_f1_scores = []
        selected_c_values = []
        per_class_f1_scores = []
        aggregate_confusion = np.zeros((len(classes), len(classes)), dtype=int)
        for split_index, (train_indices, test_indices) in enumerate(splits):
            best_c = select_best_c(
                kernel_matrix,
                labels,
                train_indices,
                c_values=c_values,
                seed=seed + split_index,
            )
            train_kernel = kernel_matrix[np.ix_(train_indices, train_indices)]
            test_kernel = kernel_matrix[np.ix_(test_indices, train_indices)]
            classifier = SVC(kernel="precomputed", C=best_c)
            classifier.fit(train_kernel, labels[train_indices])
            predictions = classifier.predict(test_kernel)
            selected_c_values.append(best_c)
            accuracies.append(accuracy_score(labels[test_indices], predictions))
            macro_f1_scores.append(
                f1_score(labels[test_indices], predictions, average="macro", zero_division=0)
            )
            per_class_f1_scores.append(
                f1_score(
                    labels[test_indices],
                    predictions,
                    labels=classes,
                    average=None,
                    zero_division=0,
                )
            )
            aggregate_confusion += confusion_matrix(
                labels[test_indices],
                predictions,
                labels=classes,
            )

        results.append(
            ExperimentResult(
                kernel=kernel_name,
                wl_iterations=wl_iterations,
                mean_accuracy=float(np.mean(accuracies)),
                std_accuracy=float(np.std(accuracies)),
                mean_macro_f1=float(np.mean(macro_f1_scores)),
                std_macro_f1=float(np.std(macro_f1_scores)),
                kernel_time_seconds=cached_kernel.elapsed_seconds,
                cache_hit=cached_kernel.cache_hit,
                best_c=_most_common_c(selected_c_values, c_values),
                c_distribution=dict(sorted(Counter(selected_c_values).items())),
                per_class_f1={
                    int(label): float(score)
                    for label, score in zip(
                        classes,
                        np.mean(per_class_f1_scores, axis=0),
                        strict=True,
                    )
                },
                confusion_matrix=aggregate_confusion.tolist(),
                kernel_diagnostics=diagnose_kernel_matrix(kernel_matrix),
            )
        )
    return results


def _most_common_c(
    selected_values: list[float],
    c_values: Sequence[float],
) -> float:
    counts = Counter(selected_values)
    return max((float(value) for value in c_values), key=lambda value: counts[value])


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
                "kernel_time_seconds",
                "best_c",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "setting": result.setting,
                    "kernel": result.kernel,
                    "wl_iterations": ("" if result.wl_iterations is None else result.wl_iterations),
                    "mean_accuracy": f"{result.mean_accuracy:.6f}",
                    "std_accuracy": f"{result.std_accuracy:.6f}",
                    "mean_macro_f1": f"{result.mean_macro_f1:.6f}",
                    "std_macro_f1": f"{result.std_macro_f1:.6f}",
                    "kernel_time_seconds": f"{result.kernel_time_seconds:.6f}",
                    "best_c": f"{result.best_c:g}",
                }
            )
    return path


def write_markdown_report(
    results: list[ExperimentResult],
    summary: DatasetSummary,
    dataset: str,
    command: str,
    output_path: str | Path,
    c_values: Sequence[float] = DEFAULT_C_VALUES,
    n_splits: int | None = None,
    test_size: float | None = None,
    seed: int | None = None,
    normalize: bool | None = None,
    use_cache: bool | None = None,
    force_recompute: bool | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write a readable Markdown experiment report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_result = max(results, key=lambda result: result.mean_macro_f1)
    fastest_result = min(results, key=lambda result: result.kernel_time_seconds)
    cache_hits = sum(result.cache_hit for result in results)
    interpretation = _build_interpretation(results, best_result)
    rows = [
        f"# {dataset} Kernel Comparison",
        "",
        "## Reproducibility Metadata",
        "",
        f"- Timestamp: `{timestamp or datetime.now(timezone.utc).isoformat()}`",
        f"- Dataset: `{dataset}`",
        f"- Splits: `{n_splits if n_splits is not None else 'not recorded'}`",
        f"- Test size: `{test_size if test_size is not None else 'not recorded'}`",
        f"- Random seed: `{seed if seed is not None else 'not recorded'}`",
        f"- Normalize kernels: `{normalize if normalize is not None else 'not recorded'}`",
        f"- C grid: `{[float(value) for value in c_values]}`",
        f"- Use cache: `{use_cache if use_cache is not None else 'not recorded'}`",
        f"- Force recompute: "
        f"`{force_recompute if force_recompute is not None else 'not recorded'}`",
        "",
        "## Dataset Summary",
        "",
        f"- Dataset: `{dataset}`",
        f"- Graphs: {summary.num_graphs}",
        f"- Class balance: `{summary.class_balance}`",
        f"- Average nodes: {summary.avg_nodes:.2f}",
        f"- Average edges: {summary.avg_edges:.2f}",
        "",
        "## Methods Compared",
        "",
        "- Graph-stat baseline using compact structural counts.",
        "- Shortest-path kernel using labeled endpoint and distance features.",
        "- Weisfeiler-Lehman subtree kernel with 0 through 5 refinement iterations.",
        '- Every classifier is an `SVC(kernel="precomputed")` evaluated on shared splits.',
        f"- C values searched inside each outer training split: "
        f"`{[float(value) for value in c_values]}`.",
        "",
        "## Results",
        "",
        "| Setting | Mean accuracy | Std accuracy | Mean macro F1 | Std macro F1 | "
        "Kernel time (s) | Most common C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows.extend(
        "| "
        f"{result.setting} | {result.mean_accuracy:.4f} | {result.std_accuracy:.4f} | "
        f"{result.mean_macro_f1:.4f} | {result.std_macro_f1:.4f} | "
        f"{result.kernel_time_seconds:.6f} | `{result.best_c:g}` |"
        for result in results
    )
    rows.extend(
        [
            "",
            "## Kernel Matrix Diagnostics",
            "",
            "| Setting | Shape | Symmetry error | Diagonal range | Min eigenvalue | Warning |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    rows.extend(_kernel_diagnostic_row(result) for result in results)
    rows.extend(
        [
            "",
            "## Best Method",
            "",
            f"`{best_result.setting}` achieved the highest mean macro F1 "
            f"({best_result.mean_macro_f1:.4f}) with mean accuracy "
            f"{best_result.mean_accuracy:.4f}. Its most common selected C was "
            f"`{best_result.best_c:g}`.",
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "## Timing and Cache Notes",
            "",
            f"- Fastest recorded kernel step: `{fastest_result.setting}` "
            f"at {fastest_result.kernel_time_seconds:.6f} seconds.",
            f"- Cache hits: {cache_hits} of {len(results)} settings.",
            "- Kernel time measures matrix retrieval or computation, including cache I/O.",
            "",
            "## Reproduction",
            "",
            "```bash",
            command,
            "```",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _kernel_diagnostic_row(result: ExperimentResult) -> str:
    diagnostics = result.kernel_diagnostics
    min_eigenvalue = (
        "not computed"
        if diagnostics.approximate_min_eigenvalue is None
        else f"{diagnostics.approximate_min_eigenvalue:.3e}"
    )
    return (
        f"| {result.setting} | {diagnostics.shape[0]} x {diagnostics.shape[1]} | "
        f"{diagnostics.symmetry_error:.3e} | {diagnostics.min_diagonal:.3e} to "
        f"{diagnostics.max_diagonal:.3e} | {min_eigenvalue} | "
        f"`{diagnostics.condition_warning or 'none'}` |"
    )


def write_diagnostics_report(
    results: list[ExperimentResult],
    dataset: str,
    output_path: str | Path,
    *,
    command: str,
    n_splits: int,
    test_size: float,
    seed: int,
    normalize: bool,
    c_values: Sequence[float],
    use_cache: bool,
    force_recompute: bool,
    timestamp: str,
) -> Path:
    """Write detailed classification, tuning, cache, and kernel diagnostics."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_result = max(results, key=lambda result: result.mean_macro_f1)
    lines = [
        f"# {dataset} Experiment Diagnostics",
        "",
        "## Reproducibility Metadata",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Dataset: `{dataset}`",
        f"- Splits: `{n_splits}`",
        f"- Test size: `{test_size}`",
        f"- Random seed: `{seed}`",
        f"- Normalize kernels: `{normalize}`",
        f"- C grid: `{[float(value) for value in c_values]}`",
        f"- Use cache: `{use_cache}`",
        f"- Force recompute: `{force_recompute}`",
        "",
        "## Method Diagnostics",
        "",
        "| Setting | Per-class F1 | Selected C distribution | Cache | Kernel warning |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {result.setting} | `{result.per_class_f1}` | "
        f"`{result.c_distribution}` | `{'hit' if result.cache_hit else 'miss'}` | "
        f"`{result.kernel_diagnostics.condition_warning or 'none'}` |"
        for result in results
    )
    lines.extend(
        [
            "",
            f"## Best Method Confusion Matrix: {best_result.setting}",
            "",
            _format_confusion_matrix(best_result),
            "",
            "## Reproduction",
            "",
            "```bash",
            command,
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _format_confusion_matrix(result: ExperimentResult) -> str:
    labels = list(result.per_class_f1)
    header = "| Actual / Predicted | " + " | ".join(map(str, labels)) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in labels) + " |"
    rows = [
        f"| {label} | " + " | ".join(map(str, values)) + " |"
        for label, values in zip(labels, result.confusion_matrix, strict=True)
    ]
    return "\n".join([header, divider, *rows])


def _build_interpretation(
    results: list[ExperimentResult],
    best_result: ExperimentResult,
) -> str:
    runner_up = sorted(results, key=lambda result: result.mean_macro_f1, reverse=True)[1]
    margin = best_result.mean_macro_f1 - runner_up.mean_macro_f1
    if margin <= 0.001:
        comparison = (
            f"The leading methods are effectively tied: `{best_result.setting}` and "
            f"`{runner_up.setting}` differ by only {margin:.4f} mean macro F1."
        )
    else:
        comparison = (
            f"`{best_result.setting}` leads `{runner_up.setting}` by {margin:.4f} mean macro F1."
        )

    wl_results = [result for result in results if result.kernel == "wl"]
    best_wl = max(wl_results, key=lambda result: result.mean_macro_f1)
    return (
        f"{comparison} The strongest WL depth is `{best_wl.setting}` "
        f"with mean macro F1 {best_wl.mean_macro_f1:.4f}. "
        "Differences should be read alongside split variability and kernel runtime."
    )


def write_experiment_config(
    output_path: str | Path,
    *,
    dataset: str,
    n_splits: int,
    test_size: float,
    seed: int,
    normalize: bool,
    use_cache: bool,
    force_recompute: bool,
    c_values: Sequence[float] = DEFAULT_C_VALUES,
    timestamp: str | None = None,
) -> Path:
    """Save experiment configuration as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": dataset,
        "n_splits": n_splits,
        "test_size": test_size,
        "seed": seed,
        "normalize": normalize,
        "use_cache": use_cache,
        "force_recompute": force_recompute,
        "c_values": [float(value) for value in c_values],
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_C_VALUES),
    )
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
        dataset_name=args.dataset,
        use_cache=args.use_cache,
        force_recompute=args.force_recompute,
        c_values=args.c_values,
    )
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "graph_kernel_svm.scripts.run_experiments",
            *sys.argv[1:],
        ]
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    dataset_slug = args.dataset.lower()
    csv_path = write_results_csv(
        results,
        f"outputs/{dataset_slug}_kernel_comparison.csv",
    )
    config_path = write_experiment_config(
        f"outputs/{dataset_slug}_kernel_comparison_config.json",
        dataset=args.dataset,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        normalize=args.normalize,
        use_cache=args.use_cache,
        force_recompute=args.force_recompute,
        c_values=args.c_values,
        timestamp=timestamp,
    )
    report_path = write_markdown_report(
        results,
        summary=summary,
        dataset=args.dataset,
        command=command,
        output_path=f"reports/{dataset_slug}_kernel_comparison.md",
        c_values=args.c_values,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        normalize=args.normalize,
        use_cache=args.use_cache,
        force_recompute=args.force_recompute,
        timestamp=timestamp,
    )
    diagnostics_path = write_diagnostics_report(
        results,
        args.dataset,
        f"reports/{dataset_slug}_diagnostics.md",
        command=command,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        normalize=args.normalize,
        c_values=args.c_values,
        use_cache=args.use_cache,
        force_recompute=args.force_recompute,
        timestamp=timestamp,
    )

    for result in results:
        print(
            f"{result.setting}: accuracy={result.mean_accuracy:.3f}+/-{result.std_accuracy:.3f} "
            f"macro_f1={result.mean_macro_f1:.3f}+/-{result.std_macro_f1:.3f} "
            f"kernel_time_seconds={result.kernel_time_seconds:.6f} "
            f"cache_hit={result.cache_hit} best_c={result.best_c:g}"
        )
    print(f"csv={csv_path}")
    print(f"config={config_path}")
    print(f"report={report_path}")
    print(f"diagnostics={diagnostics_path}")


if __name__ == "__main__":
    main()
