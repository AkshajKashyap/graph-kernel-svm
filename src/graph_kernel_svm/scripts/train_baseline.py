"""Train an SVM with the baseline graph-stat kernel."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from graph_kernel_svm.data import load_synthetic_graph_classification
from graph_kernel_svm.kernels import graph_stat_kernel


def train_baseline(random_state: int = 42) -> float:
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
    train_examples = [examples[index] for index in train_idx]
    test_examples = [examples[index] for index in test_idx]

    train_kernel = graph_stat_kernel(train_examples)
    test_kernel = graph_stat_kernel(test_examples, train_examples)

    classifier = SVC(kernel="precomputed")
    classifier.fit(train_kernel, labels[train_idx])
    predictions = classifier.predict(test_kernel)
    return float(accuracy_score(labels[test_idx], predictions))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    accuracy = train_baseline(random_state=args.random_state)
    print(f"accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
