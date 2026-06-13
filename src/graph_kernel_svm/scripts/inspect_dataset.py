"""Inspect a graph-classification dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from graph_kernel_svm.graphs import GraphExample
from graph_kernel_svm.scripts.train_baseline import _load_dataset


def format_dataset_inspection(
    examples: list[GraphExample],
    dataset: str,
) -> str:
    """Return a compact diagnostic summary for a graph dataset."""

    if not examples:
        raise ValueError("At least one graph example is required.")

    node_counts = [example.graph.number_of_nodes() for example in examples]
    edge_counts = [example.graph.number_of_edges() for example in examples]
    class_balance = dict(sorted(Counter(example.label for example in examples).items()))
    node_labels = sorted(
        {
            str(attributes.get("node_label", "0"))
            for example in examples
            for _, attributes in example.graph.nodes(data=True)
        }
    )
    lines = [
        f"dataset={dataset}",
        f"graphs={len(examples)}",
        f"class_balance={class_balance}",
        (
            f"nodes avg={sum(node_counts) / len(node_counts):.2f} "
            f"min={min(node_counts)} max={max(node_counts)}"
        ),
        (
            f"edges avg={sum(edge_counts) / len(edge_counts):.2f} "
            f"min={min(edge_counts)} max={max(edge_counts)}"
        ),
        f"unique_node_labels={node_labels}",
        "examples:",
    ]
    lines.extend(
        "  "
        f"id={example.graph_id or index} label={example.label} "
        f"nodes={example.graph.number_of_nodes()} edges={example.graph.number_of_edges()}"
        for index, example in enumerate(examples[:3], start=1)
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = _load_dataset(args.dataset, args.data_root)
    print(format_dataset_inspection(examples, args.dataset))


if __name__ == "__main__":
    main()
