from graph_kernel_svm.scripts.train_baseline import build_parser, train_baseline


def test_train_baseline_runs_with_precomputed_kernel() -> None:
    accuracy = train_baseline()

    assert 0.0 <= accuracy <= 1.0


def test_train_baseline_runs_with_wl_kernel() -> None:
    accuracy = train_baseline(kernel="wl")

    assert 0.0 <= accuracy <= 1.0


def test_train_parser_accepts_dataset_and_wl_kernel() -> None:
    args = build_parser().parse_args(["--dataset", "MUTAG", "--kernel", "wl", "--normalize"])

    assert args.dataset == "MUTAG"
    assert args.kernel == "wl"
    assert args.normalize is True
