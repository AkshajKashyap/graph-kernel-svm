"""Train an SVM with a baseline graph kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from graph_kernel_svm.data import (
    load_synthetic_graph_classification,
    load_tu_dataset,
    summarize_dataset,
)
from graph_kernel_svm.graphs import GraphExample
from graph_kernel_svm.kernels import graph_stat_kernel, weisfeiler_lehman_subtree_kernel


def train_baseline(
    dataset: str = "synthetic",
    data_root: str | Path = "data/raw",
    kernel: str = "stats",
    random_state: int = 42,
    wl_iterations: int = 3,
    normalize: bool = False,
) -> float:
    """Train and evaluate SVC on a graph-classification dataset."""

    examples = _load_dataset(dataset, Path(data_root))
    labels = np.array([example.label for example in examples])
    indices = np.arange(len(examples))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.33,
        random_state=random_state,
        stratify=labels,
    )

    full_kernel = _build_kernel(
        examples,
        kernel=kernel,
        wl_iterations=wl_iterations,
        normalize=normalize,
    )
    train_kernel = full_kernel[np.ix_(train_idx, train_idx)]
    test_kernel = full_kernel[np.ix_(test_idx, train_idx)]

    classifier = SVC(kernel="precomputed")
    classifier.fit(train_kernel, labels[train_idx])
    predictions = classifier.predict(test_kernel)
    return float(accuracy_score(labels[test_idx], predictions))


def _load_dataset(dataset: str, data_root: Path) -> list[GraphExample]:
    if dataset.lower() == "synthetic":
        return load_synthetic_graph_classification()
    return load_tu_dataset(data_root / dataset)


def _build_kernel(
    examples: list[GraphExample],
    kernel: str,
    wl_iterations: int,
    normalize: bool,
) -> np.ndarray:
    if kernel == "stats":
        return graph_stat_kernel(examples)
    if kernel == "wl":
        return weisfeiler_lehman_subtree_kernel(
            examples,
            num_iterations=wl_iterations,
            normalize=normalize,
        )
    raise ValueError(f"Unsupported kernel: {kernel}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--kernel", choices=["stats", "wl"], default="stats")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--wl-iterations", type=int, default=3)
    parser.add_argument("--normalize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = _load_dataset(args.dataset, args.data_root)
    summary = summarize_dataset(examples)
    print(
        f"dataset={args.dataset} graphs={summary.num_graphs} "
        f"class_balance={summary.class_balance} "
        f"avg_nodes={summary.avg_nodes:.2f} avg_edges={summary.avg_edges:.2f}"
    )
    accuracy = train_baseline(
        dataset=args.dataset,
        data_root=args.data_root,
        kernel=args.kernel,
        random_state=args.random_state,
        wl_iterations=args.wl_iterations,
        normalize=args.normalize,
    )
    print(f"accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
