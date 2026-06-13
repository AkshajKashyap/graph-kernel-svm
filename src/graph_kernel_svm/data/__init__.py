"""Dataset loaders."""

from graph_kernel_svm.data.synthetic import load_synthetic_graph_classification
from graph_kernel_svm.data.tu_dataset import (
    SUPPORTED_TU_DATASETS,
    DatasetSummary,
    load_tu_dataset,
    summarize_dataset,
)

__all__ = [
    "SUPPORTED_TU_DATASETS",
    "DatasetSummary",
    "load_synthetic_graph_classification",
    "load_tu_dataset",
    "summarize_dataset",
]
