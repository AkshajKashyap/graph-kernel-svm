import csv
from pathlib import Path

from graph_kernel_svm.scripts.plot_all_results import plot_all_results
from graph_kernel_svm.scripts.run_all_experiments import (
    run_all_experiments,
    write_all_diagnostics_report,
    write_all_results_csv,
    write_all_results_report,
)
from graph_kernel_svm.scripts.train_baseline import _load_dataset


def _write_tiny_dataset(root: Path, dataset: str) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir()
    graph_indicators = []
    graph_labels = []
    node_labels = []
    edges = []
    node_id = 1
    for graph_index in range(6):
        graph_id = graph_index + 1
        graph_indicators.extend([graph_id, graph_id, graph_id])
        graph_labels.append("-1" if graph_index < 3 else "1")
        node_labels.extend(["1", "2", "1"])
        edges.extend([(node_id, node_id + 1), (node_id + 1, node_id + 2)])
        node_id += 3

    files = {
        f"{dataset}_A.txt": "\n".join(f"{left}, {right}" for left, right in edges) + "\n",
        f"{dataset}_graph_indicator.txt": "\n".join(map(str, graph_indicators)) + "\n",
        f"{dataset}_graph_labels.txt": "\n".join(graph_labels) + "\n",
        f"{dataset}_node_labels.txt": "\n".join(node_labels) + "\n",
    }
    for filename, contents in files.items():
        (dataset_dir / filename).write_text(contents, encoding="utf-8")


def test_run_all_experiments_with_fake_tu_datasets(tmp_path: Path) -> None:
    _write_tiny_dataset(tmp_path, "MUTAG")
    _write_tiny_dataset(tmp_path, "PTC_MR")

    results, summaries = run_all_experiments(
        ["MUTAG", "PTC_MR"],
        data_root=tmp_path,
        n_splits=1,
        test_size=0.34,
        seed=3,
    )

    assert set(results) == {"MUTAG", "PTC_MR"}
    assert all(len(dataset_results) == 8 for dataset_results in results.values())
    assert summaries["MUTAG"].num_graphs == 6


def test_multi_dataset_csv_report_and_plots(tmp_path: Path) -> None:
    _write_tiny_dataset(tmp_path, "MUTAG")
    _write_tiny_dataset(tmp_path, "PROTEINS")
    results, summaries = run_all_experiments(
        ["MUTAG", "PROTEINS"],
        data_root=tmp_path,
        n_splits=1,
        test_size=0.34,
    )

    csv_path = write_all_results_csv(results, tmp_path / "outputs" / "all.csv")
    report_path = write_all_results_report(
        results,
        summaries,
        "python -m graph_kernel_svm.scripts.run_all_experiments",
        tmp_path / "reports" / "all.md",
        n_splits=1,
        test_size=0.34,
        seed=42,
        normalize=False,
        use_cache=False,
        force_recompute=False,
        timestamp="2026-06-13T12:00:00+00:00",
    )
    diagnostics_path = write_all_diagnostics_report(
        results,
        tmp_path / "reports" / "all_diagnostics.md",
        command="python -m graph_kernel_svm.scripts.run_all_experiments",
        n_splits=1,
        test_size=0.34,
        seed=42,
        normalize=False,
        c_values=[0.1, 1.0, 10.0],
        use_cache=False,
        force_recompute=False,
        timestamp="2026-06-13T12:00:00+00:00",
    )
    figures = plot_all_results(csv_path, tmp_path / "figures")

    with csv_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    report = report_path.read_text(encoding="utf-8")
    diagnostics = diagnostics_path.read_text(encoding="utf-8")

    assert len(rows) == 16
    assert {row["dataset"] for row in rows} == {"MUTAG", "PROTEINS"}
    assert "best_c" in rows[0]
    assert "## Best Method by Dataset" in report
    assert "Most common C" in report
    assert "C values searched" in report
    assert "## MUTAG" in report
    assert "## PROTEINS" in report
    assert "run_all_experiments" in report
    assert "## Reproducibility Metadata" in report
    assert "Datasets: `['MUTAG', 'PROTEINS']`" in report
    assert "Best Method Confusion Matrix" in diagnostics
    assert "C distribution" in diagnostics
    assert [path.name for path in figures] == [
        "all_datasets_best_macro_f1.png",
        "all_datasets_kernel_comparison.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)


def test_missing_dataset_error_suggests_download_command(tmp_path: Path) -> None:
    try:
        _load_dataset("PTC_MR", tmp_path)
    except FileNotFoundError as error:
        message = str(error)
        assert "download_tu_dataset" in message
        assert "--dataset PTC_MR" in message
    else:
        raise AssertionError("Expected missing dataset files to raise FileNotFoundError.")
