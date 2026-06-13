"""Core graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass(frozen=True, slots=True)
class GraphExample:
    """A labeled graph sample used by kernels and training code."""

    graph: nx.Graph
    label: int
    graph_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.graph.number_of_nodes() == 0:
            raise ValueError("GraphExample graph must contain at least one node.")
        if not isinstance(self.label, int):
            raise TypeError("GraphExample label must be an int.")
