from graph_kernel_svm.data import load_synthetic_graph_classification
from graph_kernel_svm.graphs import GraphExample


def test_synthetic_loader_returns_labeled_graph_examples() -> None:
    dataset = load_synthetic_graph_classification()

    assert len(dataset) == 10
    assert all(isinstance(example, GraphExample) for example in dataset)
    assert {example.label for example in dataset} == {0, 1}
    assert all(example.graph.number_of_nodes() > 0 for example in dataset)
