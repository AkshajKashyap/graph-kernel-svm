"""Run graph-kernel comparisons across multiple TU datasets."""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from graph_kernel_svm.data import SUPPORTED_TU_DATASETS, DatasetSummary, summarize_dataset
from graph_kernel_svm.models import DEFAULT_C_VALUES
from graph_kernel_svm.scripts.run_experiments import (
    ExperimentResult,
    _format_confusion_matrix,
    run_kernel_experiments,
)
from graph_kernel_svm.scripts.train_baseline import _load_dataset


def run_all_experiments(
    datasets: list[str],
    *,
    data_root: str | Path = "data/raw",
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
    normalize: bool = False,
    use_cache: bool = False,
    force_recompute: bool = False,
    cache_dir: str | Path = "outputs/cache",
    c_values: Sequence[float] = DEFAULT_C_VALUES,
) -> tuple[dict[str, list[ExperimentResult]], dict[str, DatasetSummary]]:
    """Run the existing kernel comparison for each requested dataset."""

    results_by_dataset = {}
    summaries = {}
    for dataset in datasets:
        dataset_name = dataset.upper()
        examples = _load_dataset(dataset_name, Path(data_root))
        summaries[dataset_name] = summarize_dataset(examples)
        results_by_dataset[dataset_name] = run_kernel_experiments(
            examples,
            n_splits=n_splits,
            test_size=test_size,
            seed=seed,
            normalize=normalize,
            dataset_name=dataset_name,
            use_cache=use_cache,
            force_recompute=force_recompute,
            cache_dir=cache_dir,
            c_values=c_values,
        )
    return results_by_dataset, summaries


def write_all_results_csv(
    results_by_dataset: dict[str, list[ExperimentResult]],
    output_path: str | Path,
) -> Path:
    """Write all dataset results in a single long-form CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "dataset",
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
        for dataset, results in results_by_dataset.items():
            for result in results:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "setting": result.setting,
                        "kernel": result.kernel,
                        "wl_iterations": (
                            "" if result.wl_iterations is None else result.wl_iterations
                        ),
                        "mean_accuracy": f"{result.mean_accuracy:.6f}",
                        "std_accuracy": f"{result.std_accuracy:.6f}",
                        "mean_macro_f1": f"{result.mean_macro_f1:.6f}",
                        "std_macro_f1": f"{result.std_macro_f1:.6f}",
                        "kernel_time_seconds": f"{result.kernel_time_seconds:.6f}",
                        "best_c": f"{result.best_c:g}",
                    }
                )
    return path


def write_all_results_report(
    results_by_dataset: dict[str, list[ExperimentResult]],
    summaries: dict[str, DatasetSummary],
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
    """Write per-dataset result tables and a best-method summary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_results = {
        dataset: max(results, key=lambda result: result.mean_macro_f1)
        for dataset, results in results_by_dataset.items()
    }
    lines = [
        "# Multi-Dataset Graph Kernel Comparison",
        "",
        "## Reproducibility Metadata",
        "",
        f"- Timestamp: `{timestamp or datetime.now(timezone.utc).isoformat()}`",
        f"- Datasets: `{list(results_by_dataset)}`",
        f"- Splits: `{n_splits if n_splits is not None else 'not recorded'}`",
        f"- Test size: `{test_size if test_size is not None else 'not recorded'}`",
        f"- Random seed: `{seed if seed is not None else 'not recorded'}`",
        f"- Normalize kernels: `{normalize if normalize is not None else 'not recorded'}`",
        f"- C grid: `{[float(value) for value in c_values]}`",
        f"- Use cache: `{use_cache if use_cache is not None else 'not recorded'}`",
        f"- Force recompute: "
        f"`{force_recompute if force_recompute is not None else 'not recorded'}`",
        "",
        f"C values searched within each outer training split: "
        f"`{[float(value) for value in c_values]}`.",
        "",
        "## Best Method by Dataset",
        "",
        "| Dataset | Best method | Most common C | Mean macro F1 | Mean accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {dataset} | {result.setting} | `{result.best_c:g}` | "
        f"{result.mean_macro_f1:.4f} | "
        f"{result.mean_accuracy:.4f} |"
        for dataset, result in best_results.items()
    )

    for dataset, results in results_by_dataset.items():
        summary = summaries[dataset]
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                f"Graphs: {summary.num_graphs}; class balance: `{summary.class_balance}`; "
                f"average nodes: {summary.avg_nodes:.2f}; "
                f"average edges: {summary.avg_edges:.2f}.",
                "",
                "| Setting | Mean accuracy | Std accuracy | Mean macro F1 | "
                "Std macro F1 | Kernel time (s) | Most common C |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {result.setting} | {result.mean_accuracy:.4f} | "
            f"{result.std_accuracy:.4f} | {result.mean_macro_f1:.4f} | "
            f"{result.std_macro_f1:.4f} | {result.kernel_time_seconds:.6f} | "
            f"`{result.best_c:g}` |"
            for result in results
        )

    lines.extend(
        [
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


def write_all_diagnostics_report(
    results_by_dataset: dict[str, list[ExperimentResult]],
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
    """Write detailed diagnostics across all evaluated datasets."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multi-Dataset Experiment Diagnostics",
        "",
        "## Reproducibility Metadata",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Datasets: `{list(results_by_dataset)}`",
        f"- Splits: `{n_splits}`",
        f"- Test size: `{test_size}`",
        f"- Random seed: `{seed}`",
        f"- Normalize kernels: `{normalize}`",
        f"- C grid: `{[float(value) for value in c_values]}`",
        f"- Use cache: `{use_cache}`",
        f"- Force recompute: `{force_recompute}`",
    ]
    for dataset, results in results_by_dataset.items():
        best_result = max(results, key=lambda result: result.mean_macro_f1)
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                "| Setting | Per-class F1 | C distribution | Cache | Symmetry error | "
                "Min eigenvalue | Warning |",
                "| --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        lines.extend(_all_diagnostics_row(result) for result in results)
        lines.extend(
            [
                "",
                f"### Best Method Confusion Matrix: {best_result.setting}",
                "",
                _format_confusion_matrix(best_result),
            ]
        )
    lines.extend(
        [
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


def _all_diagnostics_row(result: ExperimentResult) -> str:
    diagnostics = result.kernel_diagnostics
    min_eigenvalue = (
        "not computed"
        if diagnostics.approximate_min_eigenvalue is None
        else f"{diagnostics.approximate_min_eigenvalue:.3e}"
    )
    return (
        f"| {result.setting} | `{result.per_class_f1}` | `{result.c_distribution}` | "
        f"`{'hit' if result.cache_hit else 'miss'}` | {diagnostics.symmetry_error:.3e} | "
        f"{min_eigenvalue} | `{diagnostics.condition_warning or 'none'}` |"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_TU_DATASETS,
        default=list(SUPPORTED_TU_DATASETS),
    )
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
    results_by_dataset, summaries = run_all_experiments(
        args.datasets,
        data_root=args.data_root,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        normalize=args.normalize,
        use_cache=args.use_cache,
        force_recompute=args.force_recompute,
        c_values=args.c_values,
    )
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "graph_kernel_svm.scripts.run_all_experiments",
            *sys.argv[1:],
        ]
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    csv_path = write_all_results_csv(
        results_by_dataset,
        "outputs/all_datasets_kernel_comparison.csv",
    )
    report_path = write_all_results_report(
        results_by_dataset,
        summaries,
        command,
        "reports/all_datasets_kernel_comparison.md",
        c_values=args.c_values,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        normalize=args.normalize,
        use_cache=args.use_cache,
        force_recompute=args.force_recompute,
        timestamp=timestamp,
    )
    diagnostics_path = write_all_diagnostics_report(
        results_by_dataset,
        "reports/all_datasets_diagnostics.md",
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
    print(f"csv={csv_path}")
    print(f"report={report_path}")
    print(f"diagnostics={diagnostics_path}")


if __name__ == "__main__":
    main()
