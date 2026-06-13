"""Load graph classification datasets in the TU Dortmund text format."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from graph_kernel_svm.graphs import GraphExample

SUPPORTED_TU_DATASETS = ("MUTAG", "PTC_MR", "PROTEINS")


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Basic graph-classification dataset statistics."""

    num_graphs: int
    class_balance: dict[int, int]
    avg_nodes: float
    avg_edges: float


def load_tu_dataset(dataset_dir: str | Path) -> list[GraphExample]:
    """Load a local TU Dortmund dataset directory."""

    directory = Path(dataset_dir)
    dataset_name = directory.name
    paths = {
        "edges": directory / f"{dataset_name}_A.txt",
        "graph_indicator": directory / f"{dataset_name}_graph_indicator.txt",
        "graph_labels": directory / f"{dataset_name}_graph_labels.txt",
        "node_labels": directory / f"{dataset_name}_node_labels.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TU dataset files: {', '.join(missing)}")

    graph_indicators = _read_int_lines(paths["graph_indicator"])
    raw_graph_labels = _read_string_lines(paths["graph_labels"])
    node_labels = _read_string_lines(paths["node_labels"])

    if len(graph_indicators) != len(node_labels):
        raise ValueError("Graph indicator and node label files must have equal lengths.")
    if not graph_indicators:
        raise ValueError("TU dataset must contain at least one node.")

    num_graphs = max(graph_indicators)
    if min(graph_indicators) < 1 or set(graph_indicators) != set(range(1, num_graphs + 1)):
        raise ValueError("Graph indicators must be contiguous and one-based.")
    if len(raw_graph_labels) != num_graphs:
        raise ValueError("Graph label count must match the number of graphs.")

    graphs = [nx.Graph() for _ in range(num_graphs)]
    for node_id, (graph_id, node_label) in enumerate(
        zip(graph_indicators, node_labels, strict=True),
        start=1,
    ):
        graphs[graph_id - 1].add_node(node_id, node_label=node_label)

    for source, target in _read_edges(paths["edges"]):
        if source < 1 or source > len(graph_indicators):
            raise ValueError(f"Edge references unknown node {source}.")
        if target < 1 or target > len(graph_indicators):
            raise ValueError(f"Edge references unknown node {target}.")
        source_graph = graph_indicators[source - 1]
        target_graph = graph_indicators[target - 1]
        if source_graph != target_graph:
            raise ValueError(f"Edge ({source}, {target}) connects different graphs.")
        graphs[source_graph - 1].add_edge(source, target)

    label_mapping = {
        raw_label: mapped_label
        for mapped_label, raw_label in enumerate(sorted(set(raw_graph_labels)))
    }
    return [
        GraphExample(
            graph=graph,
            label=label_mapping[raw_label],
            graph_id=f"{dataset_name}_{graph_index}",
            metadata={"dataset": dataset_name, "original_label": raw_label},
        )
        for graph_index, (graph, raw_label) in enumerate(
            zip(graphs, raw_graph_labels, strict=True),
            start=1,
        )
    ]


def summarize_dataset(examples: list[GraphExample]) -> DatasetSummary:
    """Compute a concise summary for a graph-classification dataset."""

    if not examples:
        raise ValueError("At least one graph example is required.")
    return DatasetSummary(
        num_graphs=len(examples),
        class_balance=dict(sorted(Counter(example.label for example in examples).items())),
        avg_nodes=sum(example.graph.number_of_nodes() for example in examples) / len(examples),
        avg_edges=sum(example.graph.number_of_edges() for example in examples) / len(examples),
    )


def _read_string_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_int_lines(path: Path) -> list[int]:
    try:
        return [int(value) for value in _read_string_lines(path)]
    except ValueError as error:
        raise ValueError(f"Expected integer values in {path}.") from error


def _read_edges(path: Path) -> list[tuple[int, int]]:
    edges = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid edge at {path}:{line_number}.")
        try:
            edges.append((int(parts[0]), int(parts[1])))
        except ValueError as error:
            raise ValueError(f"Invalid edge at {path}:{line_number}.") from error
    return edges
