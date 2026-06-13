from graph_kernel_svm.data import load_synthetic_graph_classification
from graph_kernel_svm.scripts.inspect_dataset import format_dataset_inspection


def test_inspect_dataset_formats_synthetic_data_without_crashing() -> None:
    output = format_dataset_inspection(
        load_synthetic_graph_classification(),
        dataset="synthetic",
    )

    assert "graphs=10" in output
    assert "class_balance={0: 5, 1: 5}" in output
    assert "unique_node_labels=['0']" in output
    assert output.count("  id=") == 3
