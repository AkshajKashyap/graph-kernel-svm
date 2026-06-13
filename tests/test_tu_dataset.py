from pathlib import Path

from graph_kernel_svm.data import load_tu_dataset, summarize_dataset


def _write_tiny_tu_dataset(root: Path) -> Path:
    dataset_dir = root / "MUTAG"
    dataset_dir.mkdir()
    files = {
        "MUTAG_A.txt": "1, 2\n2, 3\n4, 5\n",
        "MUTAG_graph_indicator.txt": "1\n1\n1\n2\n2\n",
        "MUTAG_graph_labels.txt": "-1\n1\n",
        "MUTAG_node_labels.txt": "6\n7\n6\n8\n9\n",
    }
    for filename, contents in files.items():
        (dataset_dir / filename).write_text(contents, encoding="utf-8")
    return dataset_dir


def test_load_tu_dataset_parses_graphs_labels_and_edges(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_tu_dataset(tmp_path)

    examples = load_tu_dataset(dataset_dir)

    assert len(examples) == 2
    assert [example.label for example in examples] == [0, 1]
    assert [example.graph.number_of_nodes() for example in examples] == [3, 2]
    assert [example.graph.number_of_edges() for example in examples] == [2, 1]
    assert examples[0].graph.nodes[1]["node_label"] == "6"
    assert examples[0].metadata["original_label"] == "-1"


def test_tu_dataset_summary(tmp_path: Path) -> None:
    examples = load_tu_dataset(_write_tiny_tu_dataset(tmp_path))

    summary = summarize_dataset(examples)

    assert summary.num_graphs == 2
    assert summary.class_balance == {0: 1, 1: 1}
    assert summary.avg_nodes == 2.5
    assert summary.avg_edges == 1.5


def test_load_tu_dataset_rejects_cross_graph_edges(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_tu_dataset(tmp_path)
    (dataset_dir / "MUTAG_A.txt").write_text("1, 4\n", encoding="utf-8")

    try:
        load_tu_dataset(dataset_dir)
    except ValueError as error:
        assert "connects different graphs" in str(error)
    else:
        raise AssertionError("Expected a cross-graph edge to be rejected.")
