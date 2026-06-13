import numpy as np
from sklearn.svm import SVC

from graph_kernel_svm.models import select_best_c


def _precomputed_kernel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.8],
            [-1.0, -1.2],
            [-0.5, -0.6],
            [0.5, 0.7],
            [1.0, 1.1],
            [1.5, 0.9],
            [2.0, 1.3],
            [10.0, -10.0],
            [-10.0, 10.0],
        ]
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    train_indices = np.arange(8)
    return features @ features.T, labels, train_indices


def test_c_tuning_only_uses_training_indices() -> None:
    kernel, labels, train_indices = _precomputed_kernel()
    altered_kernel = kernel.copy()
    altered_kernel[8:, :] = 1_000_000.0
    altered_kernel[:, 8:] = -1_000_000.0
    altered_labels = labels.copy()
    altered_labels[8:] = 99

    original = select_best_c(kernel, labels, train_indices, [0.1, 1.0, 10.0], seed=3)
    altered = select_best_c(
        altered_kernel,
        altered_labels,
        train_indices,
        [0.1, 1.0, 10.0],
        seed=3,
    )

    assert original == altered


def test_c_tuning_returns_value_from_grid_and_supports_precomputed_svc() -> None:
    kernel, labels, train_indices = _precomputed_kernel()
    c_values = [0.01, 0.1, 1.0]

    best_c = select_best_c(kernel, labels, train_indices, c_values, seed=5)
    train_kernel = kernel[np.ix_(train_indices, train_indices)]
    classifier = SVC(kernel="precomputed", C=best_c)
    classifier.fit(train_kernel, labels[train_indices])

    assert best_c in c_values
    assert classifier.predict(train_kernel).shape == (len(train_indices),)
