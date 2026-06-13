"""Run graph-kernel comparisons across multiple TU datasets."""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path

from graph_kernel_svm.data import SUPPORTED_TU_DATASETS, DatasetSummary, summarize_dataset
from graph_kernel_svm.scripts.run_experiments import ExperimentResult, run_kernel_experiments
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
                    }
                )
    return path


def write_all_results_report(
    results_by_dataset: dict[str, list[ExperimentResult]],
    summaries: dict[str, DatasetSummary],
    command: str,
    output_path: str | Path,
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
        "## Best Method by Dataset",
        "",
        "| Dataset | Best method | Mean macro F1 | Mean accuracy |",
        "| --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {dataset} | {result.setting} | {result.mean_macro_f1:.4f} | "
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
                "Std macro F1 | Kernel time (s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {result.setting} | {result.mean_accuracy:.4f} | "
            f"{result.std_accuracy:.4f} | {result.mean_macro_f1:.4f} | "
            f"{result.std_macro_f1:.4f} | {result.kernel_time_seconds:.6f} |"
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
    )
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "graph_kernel_svm.scripts.run_all_experiments",
            *sys.argv[1:],
        ]
    )
    csv_path = write_all_results_csv(
        results_by_dataset,
        "outputs/all_datasets_kernel_comparison.csv",
    )
    report_path = write_all_results_report(
        results_by_dataset,
        summaries,
        command,
        "reports/all_datasets_kernel_comparison.md",
    )
    print(f"csv={csv_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
