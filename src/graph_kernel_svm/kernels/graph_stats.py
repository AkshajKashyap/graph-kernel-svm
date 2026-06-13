"""A simple baseline kernel over graph-level statistics."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np

from graph_kernel_svm.graphs import GraphExample


def graph_stat_features(example: GraphExample) -> np.ndarray:
    """Compute compact graph statistics for a single example."""

    graph = example.graph
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    average_degree = float(degrees.mean()) if degrees.size else 0.0
    density = nx.density(graph)
    triangles = sum(nx.triangles(graph).values()) / 3
    components = nx.number_connected_components(graph)

    return np.array(
        [node_count, edge_count, average_degree, density, triangles, components],
        dtype=float,
    )


def graph_stat_feature_matrix(examples: Sequence[GraphExample]) -> np.ndarray:
    """Return graph-stat feature rows for examples."""

    if not examples:
        raise ValueError("At least one graph example is required.")
    return np.vstack([graph_stat_features(example) for example in examples])


def graph_stat_kernel(
    left: Sequence[GraphExample],
    right: Sequence[GraphExample] | None = None,
) -> np.ndarray:
    """Compute a linear kernel over baseline graph-stat features."""

    right = left if right is None else right
    left_features = graph_stat_feature_matrix(left)
    right_features = graph_stat_feature_matrix(right)
    return left_features @ right_features.T
