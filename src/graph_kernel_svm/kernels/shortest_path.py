"""Shortest-path graph kernel."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Hashable, Sequence

import numpy as np

from graph_kernel_svm.graphs import GraphExample

ShortestPathFeature = tuple[str, str, int]
FeatureCounts = Counter[ShortestPathFeature]
Adjacency = dict[Hashable, list[Hashable]]


def shortest_path_kernel(
    examples: Sequence[GraphExample],
    normalize: bool = False,
) -> np.ndarray:
    """Compute a square shortest-path kernel matrix."""

    feature_counts = shortest_path_feature_counts(examples)
    kernel = _dot_product_kernel(feature_counts)
    if normalize:
        return _normalize_kernel(kernel)
    return kernel


def shortest_path_feature_counts(
    examples: Sequence[GraphExample],
) -> list[FeatureCounts]:
    """Count labeled shortest-path features for each graph."""

    if not examples:
        raise ValueError("At least one graph example is required.")
    return [_graph_feature_counts(example) for example in examples]


def _graph_feature_counts(example: GraphExample) -> FeatureCounts:
    adjacency = _build_adjacency(example)
    node_labels = {
        node: str(attributes.get("node_label", "0"))
        for node, attributes in example.graph.nodes(data=True)
    }
    node_order = {node: index for index, node in enumerate(example.graph.nodes)}
    counts: FeatureCounts = Counter()

    for source in example.graph.nodes:
        distances = _bfs_distances(source, adjacency)
        for target, distance in distances.items():
            if node_order[source] >= node_order[target]:
                continue
            left_label, right_label = sorted((node_labels[source], node_labels[target]))
            counts[(left_label, right_label, distance)] += 1
    return counts


def _build_adjacency(example: GraphExample) -> Adjacency:
    adjacency: Adjacency = {node: [] for node in example.graph.nodes}
    for source, target in example.graph.edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    return adjacency


def _bfs_distances(source: Hashable, adjacency: Adjacency) -> dict[Hashable, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    return distances


def _dot_product_kernel(feature_counts: Sequence[FeatureCounts]) -> np.ndarray:
    kernel = np.zeros((len(feature_counts), len(feature_counts)), dtype=float)
    for row, left_counts in enumerate(feature_counts):
        for column in range(row, len(feature_counts)):
            value = sum(
                count * feature_counts[column].get(feature, 0)
                for feature, count in left_counts.items()
            )
            kernel[row, column] = value
            kernel[column, row] = value
    return kernel


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    diagonal = np.diag(kernel)
    scale = np.sqrt(np.outer(diagonal, diagonal))
    return np.divide(kernel, scale, out=np.zeros_like(kernel), where=scale > 0)
