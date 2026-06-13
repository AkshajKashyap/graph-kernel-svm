import networkx as nx
import numpy as np

from graph_kernel_svm.graphs import GraphExample
from graph_kernel_svm.kernels import shortest_path_feature_counts, shortest_path_kernel


def _labeled_graph(labels: list[str], edges: list[tuple[int, int]]) -> GraphExample:
    graph = nx.Graph()
    for node, label in enumerate(labels):
        graph.add_node(node, node_label=label)
    graph.add_edges_from(edges)
    return GraphExample(graph=graph, label=0)


def test_shortest_path_kernel_shape() -> None:
    examples = [
        _labeled_graph(["a", "a"], [(0, 1)]),
        _labeled_graph(["a", "b"], [(0, 1)]),
        _labeled_graph(["b", "b"], [(0, 1)]),
    ]

    kernel = shortest_path_kernel(examples)

    assert kernel.shape == (3, 3)


def test_shortest_path_kernel_is_symmetric() -> None:
    examples = [
        _labeled_graph(["a", "b", "a"], [(0, 1), (1, 2)]),
        _labeled_graph(["a", "a", "b"], [(0, 1), (0, 2)]),
    ]

    kernel = shortest_path_kernel(examples)

    assert np.allclose(kernel, kernel.T)


def test_shortest_path_kernel_is_deterministic() -> None:
    examples = [
        _labeled_graph(["a", "b", "c"], [(0, 1), (1, 2)]),
        _labeled_graph(["c", "b", "a"], [(0, 1), (1, 2)]),
    ]

    first = shortest_path_kernel(examples)
    second = shortest_path_kernel(examples)

    assert np.array_equal(first, second)


def test_shortest_path_normalized_kernel_has_unit_diagonal() -> None:
    examples = [
        _labeled_graph(["a", "b"], [(0, 1)]),
        _labeled_graph(["a", "b", "a"], [(0, 1), (1, 2)]),
    ]

    kernel = shortest_path_kernel(examples, normalize=True)

    assert np.allclose(np.diag(kernel), np.ones(len(examples)))


def test_shortest_path_ignores_unreachable_node_pairs() -> None:
    connected = _labeled_graph(["a", "b"], [(0, 1)])
    disconnected = _labeled_graph(["a", "b", "c"], [(0, 1)])

    counts = shortest_path_feature_counts([connected, disconnected])
    kernel = shortest_path_kernel([connected, disconnected])

    assert counts[0] == counts[1]
    assert sum(counts[1].values()) == 1
    assert kernel[0, 1] == kernel[0, 0]
