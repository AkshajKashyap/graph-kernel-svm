"""SVM helpers for precomputed graph kernels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

DEFAULT_C_VALUES = (0.1, 1.0, 10.0)


def select_best_c(
    kernel_matrix: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    c_values: Sequence[float] = DEFAULT_C_VALUES,
    seed: int = 42,
) -> float:
    """Tune SVC C using only the supplied training indices."""

    candidates = tuple(float(value) for value in c_values)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("c_values must contain at least one positive value.")

    train_indices = np.asarray(train_indices, dtype=int)
    train_labels = np.asarray(labels)[train_indices]
    train_kernel = kernel_matrix[np.ix_(train_indices, train_indices)]
    _, class_counts = np.unique(train_labels, return_counts=True)
    inner_splits = min(3, int(class_counts.min()))
    if inner_splits < 2:
        return candidates[0]

    splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    mean_scores = []
    for c_value in candidates:
        fold_scores = []
        for inner_train, inner_validation in splitter.split(train_kernel, train_labels):
            inner_train_kernel = train_kernel[np.ix_(inner_train, inner_train)]
            inner_validation_kernel = train_kernel[
                np.ix_(inner_validation, inner_train)
            ]
            classifier = SVC(kernel="precomputed", C=c_value)
            classifier.fit(inner_train_kernel, train_labels[inner_train])
            predictions = classifier.predict(inner_validation_kernel)
            fold_scores.append(
                f1_score(
                    train_labels[inner_validation],
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )
        mean_scores.append(float(np.mean(fold_scores)))

    best_index = max(range(len(candidates)), key=lambda index: mean_scores[index])
    return candidates[best_index]
