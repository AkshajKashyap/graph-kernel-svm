import networkx as nx
import numpy as np

from graph_kernel_svm.graphs import GraphExample
from graph_kernel_svm.kernels import (
    weisfeiler_lehman_feature_counts,
    weisfeiler_lehman_subtree_kernel,
)


def _labeled_graph(labels: list[str], edges: list[tuple[int, int]]) -> GraphExample:
    graph = nx.Graph()
    for node, label in enumerate(labels):
        graph.add_node(node, node_label=label)
    graph.add_edges_from(edges)
    return GraphExample(graph=graph, label=0)


def test_wl_kernel_matrix_shape() -> None:
    examples = [
        _labeled_graph(["a", "a"], [(0, 1)]),
        _labeled_graph(["a", "b"], [(0, 1)]),
        _labeled_graph(["b", "b"], [(0, 1)]),
    ]

    kernel = weisfeiler_lehman_subtree_kernel(examples)

    assert kernel.shape == (3, 3)


def test_wl_kernel_is_symmetric() -> None:
    examples = [
        _labeled_graph(["a", "a", "a"], [(0, 1), (1, 2)]),
        _labeled_graph(["a", "a", "a"], [(0, 1), (0, 2)]),
    ]

    kernel = weisfeiler_lehman_subtree_kernel(examples)

    assert np.allclose(kernel, kernel.T)


def test_wl_kernel_output_is_deterministic() -> None:
    examples = [
        _labeled_graph(["a", "b", "a"], [(0, 1), (1, 2)]),
        _labeled_graph(["b", "a", "b"], [(0, 1), (0, 2)]),
    ]

    first = weisfeiler_lehman_subtree_kernel(examples, num_iterations=2)
    second = weisfeiler_lehman_subtree_kernel(examples, num_iterations=2)

    assert np.array_equal(first, second)


def test_wl_normalized_kernel_has_unit_diagonal() -> None:
    examples = [
        _labeled_graph(["a", "a"], [(0, 1)]),
        _labeled_graph(["a", "b", "a"], [(0, 1), (1, 2)]),
    ]

    kernel = weisfeiler_lehman_subtree_kernel(examples, normalize=True)

    assert np.allclose(np.diag(kernel), np.ones(len(examples)))


def test_wl_zero_iterations_only_uses_original_node_labels() -> None:
    examples = [
        _labeled_graph(["a", "a"], [(0, 1)]),
        _labeled_graph(["a", "a"], []),
        _labeled_graph(["a", "b"], [(0, 1)]),
    ]

    kernel = weisfeiler_lehman_subtree_kernel(examples, num_iterations=0)
    counts = weisfeiler_lehman_feature_counts(examples, num_iterations=0)

    assert counts[0] == counts[1]
    assert kernel[0, 1] == kernel[0, 0]
    assert kernel[0, 2] == 2
