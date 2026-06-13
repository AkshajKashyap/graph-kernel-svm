"""Synthetic graph datasets for smoke tests and examples."""

from __future__ import annotations

import networkx as nx

from graph_kernel_svm.graphs import GraphExample


def load_synthetic_graph_classification() -> list[GraphExample]:
    """Return a tiny deterministic binary graph-classification dataset.

    Class 0 contains sparse path-like graphs. Class 1 contains cycle-like graphs
    with higher average degree. The dataset is deliberately small so tests and
    examples run quickly.
    """

    specs = [
        *[(f"path_{nodes}", 0, nx.path_graph(nodes)) for nodes in range(4, 9)],
        *[(f"cycle_{nodes}", 1, nx.cycle_graph(nodes)) for nodes in range(4, 9)],
    ]
    return [
        GraphExample(graph=graph, label=label, graph_id=graph_id)
        for graph_id, label, graph in specs
    ]
