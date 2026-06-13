"""Train an SVM with a baseline graph kernel."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from graph_kernel_svm.data import load_synthetic_graph_classification
from graph_kernel_svm.kernels import graph_stat_kernel, weisfeiler_lehman_subtree_kernel


def train_baseline(
    kernel: str = "stats",
    random_state: int = 42,
    wl_iterations: int = 3,
    normalize: bool = False,
) -> float:
    """Train and evaluate SVC on the synthetic dataset."""

    examples = load_synthetic_graph_classification()
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


def _build_kernel(
    examples: list,
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
    parser.add_argument("--kernel", choices=["stats", "wl"], default="stats")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--wl-iterations", type=int, default=3)
    parser.add_argument("--normalize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    accuracy = train_baseline(
        kernel=args.kernel,
        random_state=args.random_state,
        wl_iterations=args.wl_iterations,
        normalize=args.normalize,
    )
    print(f"accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
