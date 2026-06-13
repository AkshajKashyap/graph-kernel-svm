"""Weisfeiler-Lehman subtree kernel."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np

from graph_kernel_svm.graphs import GraphExample

FeatureCounts = Counter[str]
NodeLabels = dict[Hashable, str]
Adjacency = dict[Hashable, list[Hashable]]


def weisfeiler_lehman_subtree_kernel(
    examples: Sequence[GraphExample],
    num_iterations: int = 3,
    normalize: bool = False,
) -> np.ndarray:
    """Compute a square Weisfeiler-Lehman subtree kernel matrix."""

    feature_counts = weisfeiler_lehman_feature_counts(
        examples,
        num_iterations=num_iterations,
    )
    kernel = _dot_product_kernel(feature_counts)
    if normalize:
        return _normalize_kernel(kernel)
    return kernel


def weisfeiler_lehman_feature_counts(
    examples: Sequence[GraphExample],
    num_iterations: int = 3,
) -> list[FeatureCounts]:
    """Return WL feature count vectors for each graph."""

    if not examples:
        raise ValueError("At least one graph example is required.")
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative.")

    adjacencies = [_build_adjacency(example) for example in examples]
    current_labels = [_initial_node_labels(example) for example in examples]
    feature_counts = [Counter() for _ in examples]

    _add_iteration_counts(feature_counts, current_labels, iteration=0)

    for iteration in range(1, num_iterations + 1):
        signatures_by_graph = [
            _node_signatures(labels, adjacency)
            for labels, adjacency in zip(current_labels, adjacencies, strict=True)
        ]
        label_map = _compress_signatures(signatures_by_graph)
        current_labels = [
            {node: label_map[signature] for node, signature in signatures.items()}
            for signatures in signatures_by_graph
        ]
        _add_iteration_counts(feature_counts, current_labels, iteration=iteration)

    return feature_counts


def _build_adjacency(example: GraphExample) -> Adjacency:
    adjacency: Adjacency = {node: [] for node in example.graph.nodes}
    for source, target in example.graph.edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    return adjacency


def _initial_node_labels(example: GraphExample) -> NodeLabels:
    return {
        node: str(attributes.get("node_label", "0"))
        for node, attributes in example.graph.nodes(data=True)
    }


def _node_signatures(labels: NodeLabels, adjacency: Adjacency) -> dict[Hashable, tuple[str, tuple[str, ...]]]:
    return {
        node: (labels[node], tuple(sorted(labels[neighbor] for neighbor in neighbors)))
        for node, neighbors in adjacency.items()
    }


def _compress_signatures(
    signatures_by_graph: Sequence[dict[Hashable, tuple[str, tuple[str, ...]]]],
) -> dict[tuple[str, tuple[str, ...]], str]:
    unique_signatures = sorted({signature for signatures in signatures_by_graph for signature in signatures.values()})
    return {signature: str(index) for index, signature in enumerate(unique_signatures)}


def _add_iteration_counts(
    feature_counts: Sequence[FeatureCounts],
    labels_by_graph: Sequence[NodeLabels],
    iteration: int,
) -> None:
    for counts, labels in zip(feature_counts, labels_by_graph, strict=True):
        counts.update(f"{iteration}:{label}" for label in labels.values())


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
