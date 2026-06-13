import csv
import json
from pathlib import Path

from graph_kernel_svm.data import load_synthetic_graph_classification, summarize_dataset
from graph_kernel_svm.scripts.run_experiments import (
    run_kernel_experiments,
    write_experiment_config,
    write_markdown_report,
    write_results_csv,
)


def test_experiment_aggregation_covers_all_kernel_settings() -> None:
    examples = load_synthetic_graph_classification()

    results = run_kernel_experiments(examples, n_splits=2, test_size=0.4, seed=7)

    assert [result.setting for result in results] == [
        "stats",
        "shortest_path",
        "wl_0",
        "wl_1",
        "wl_2",
        "wl_3",
        "wl_4",
        "wl_5",
    ]
    assert all(0.0 <= result.mean_accuracy <= 1.0 for result in results)
    assert all(0.0 <= result.mean_macro_f1 <= 1.0 for result in results)
    assert all(result.std_accuracy >= 0.0 for result in results)
    assert all(result.std_macro_f1 >= 0.0 for result in results)
    assert all(result.kernel_time_seconds >= 0.0 for result in results)


def test_experiment_aggregation_is_deterministic() -> None:
    examples = load_synthetic_graph_classification()

    first = run_kernel_experiments(examples, n_splits=2, test_size=0.4, seed=11)
    second = run_kernel_experiments(examples, n_splits=2, test_size=0.4, seed=11)

    assert [
        (
            result.setting,
            result.mean_accuracy,
            result.std_accuracy,
            result.mean_macro_f1,
            result.std_macro_f1,
        )
        for result in first
    ] == [
        (
            result.setting,
            result.mean_accuracy,
            result.std_accuracy,
            result.mean_macro_f1,
            result.std_macro_f1,
        )
        for result in second
    ]


def test_experiment_outputs_csv_and_markdown(tmp_path: Path) -> None:
    examples = load_synthetic_graph_classification()
    results = run_kernel_experiments(examples, n_splits=1, test_size=0.4)

    csv_path = write_results_csv(results, tmp_path / "outputs" / "results.csv")
    report_path = write_markdown_report(
        results,
        summary=summarize_dataset(examples),
        dataset="synthetic",
        command="python -m graph_kernel_svm.scripts.run_experiments --dataset synthetic",
        output_path=tmp_path / "reports" / "report.md",
    )

    with csv_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    report = report_path.read_text(encoding="utf-8")

    assert len(rows) == 8
    assert rows[0]["setting"] == "stats"
    assert rows[1]["setting"] == "shortest_path"
    assert "kernel_time_seconds" in rows[0]
    assert "# MUTAG Kernel Comparison" in report
    assert "## Dataset Summary" in report
    assert "## Reproduction" in report
    assert "## Best Method" in report
    assert "highest mean macro F1" in report
    assert "python -m graph_kernel_svm.scripts.run_experiments" in report
    assert "Kernel time (s)" in report
    assert "wl_5" in report


def test_experiment_config_is_saved(tmp_path: Path) -> None:
    config_path = write_experiment_config(
        tmp_path / "config.json",
        dataset="synthetic",
        n_splits=3,
        test_size=0.4,
        seed=7,
        normalize=True,
        use_cache=True,
        force_recompute=False,
        timestamp="2026-06-13T12:00:00+00:00",
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config == {
        "dataset": "synthetic",
        "n_splits": 3,
        "test_size": 0.4,
        "seed": 7,
        "normalize": True,
        "use_cache": True,
        "force_recompute": False,
        "timestamp": "2026-06-13T12:00:00+00:00",
    }
