from graph_kernel_svm.scripts.train_baseline import train_baseline


def test_train_baseline_runs_with_precomputed_kernel() -> None:
    accuracy = train_baseline()

    assert 0.0 <= accuracy <= 1.0
