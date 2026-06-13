import numpy as np

from graph_kernel_svm.data import load_synthetic_graph_classification
from graph_kernel_svm.kernels import graph_stat_feature_matrix, graph_stat_kernel


def test_graph_stat_feature_matrix_has_one_row_per_example() -> None:
    dataset = load_synthetic_graph_classification()

    features = graph_stat_feature_matrix(dataset)

    assert features.shape == (len(dataset), 6)
    assert np.all(features[:, 0] > 0)


def test_graph_stat_kernel_is_symmetric_and_square_by_default() -> None:
    dataset = load_synthetic_graph_classification()

    kernel = graph_stat_kernel(dataset)

    assert kernel.shape == (len(dataset), len(dataset))
    assert np.allclose(kernel, kernel.T)
    assert np.all(np.diag(kernel) > 0)


def test_graph_stat_kernel_supports_cross_kernel_shape() -> None:
    dataset = load_synthetic_graph_classification()

    kernel = graph_stat_kernel(dataset[:2], dataset[2:])

    assert kernel.shape == (2, len(dataset) - 2)
