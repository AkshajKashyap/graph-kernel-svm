import csv
from pathlib import Path

from graph_kernel_svm.scripts.plot_results import plot_results


def test_plot_results_creates_expected_figures(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "setting",
                "mean_accuracy",
                "std_accuracy",
                "mean_macro_f1",
                "std_macro_f1",
                "kernel_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "setting": "stats",
                    "mean_accuracy": "0.70",
                    "std_accuracy": "0.05",
                    "mean_macro_f1": "0.68",
                    "std_macro_f1": "0.04",
                    "kernel_time_seconds": "0.01",
                },
                {
                    "setting": "wl_2",
                    "mean_accuracy": "0.80",
                    "std_accuracy": "0.03",
                    "mean_macro_f1": "0.79",
                    "std_macro_f1": "0.02",
                    "kernel_time_seconds": "0.04",
                },
            ]
        )

    outputs = plot_results(csv_path, tmp_path / "figures")

    assert [path.name for path in outputs] == [
        "mutag_accuracy_comparison.png",
        "mutag_macro_f1_comparison.png",
        "mutag_kernel_time_comparison.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
