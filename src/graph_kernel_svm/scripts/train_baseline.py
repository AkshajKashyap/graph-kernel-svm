"""Train an SVM with a baseline graph kernel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from graph_kernel_svm.data import (
    SUPPORTED_TU_DATASETS,
    load_synthetic_graph_classification,
    load_tu_dataset,
    summarize_dataset,
)
from graph_kernel_svm.graphs import GraphExample
from graph_kernel_svm.kernels import (
    graph_stat_kernel,
    shortest_path_kernel,
    weisfeiler_lehman_subtree_kernel,
)
from graph_kernel_svm.models import DEFAULT_C_VALUES, select_best_c
from graph_kernel_svm.utils import get_kernel_matrix


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Metrics and kernel timing for one train/test split."""

    accuracy: float
    kernel_time_seconds: float
    cache_hit: bool
    best_c: float


def train_baseline(
    dataset: str = "synthetic",
    data_root: str | Path = "data/raw",
    kernel: str = "stats",
    random_state: int = 42,
    wl_iterations: int = 3,
    normalize: bool = False,
    use_cache: bool = False,
    force_recompute: bool = False,
    cache_dir: str | Path = "outputs/cache",
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
) -> float:
    """Train and evaluate SVC on a graph-classification dataset."""

    examples = _load_dataset(dataset, Path(data_root))
    result = _train_on_examples(
        examples,
        dataset=dataset,
        kernel=kernel,
        random_state=random_state,
        wl_iterations=wl_iterations,
        normalize=normalize,
        use_cache=use_cache,
        force_recompute=force_recompute,
        cache_dir=cache_dir,
        c_values=c_values,
    )
    return result.accuracy


def _train_on_examples(
    examples: list[GraphExample],
    *,
    dataset: str,
    kernel: str,
    random_state: int,
    wl_iterations: int,
    normalize: bool,
    use_cache: bool,
    force_recompute: bool,
    cache_dir: str | Path,
    c_values: tuple[float, ...],
) -> TrainingResult:
    labels = np.array([example.label for example in examples])
    indices = np.arange(len(examples))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.33,
        random_state=random_state,
        stratify=labels,
    )

    cached_kernel = get_kernel_matrix(
        examples=examples,
        dataset_name=dataset,
        kernel_name=kernel,
        normalize=normalize,
        wl_iterations=wl_iterations if kernel == "wl" else None,
        use_cache=use_cache,
        force_recompute=force_recompute,
        cache_dir=cache_dir,
        compute=lambda: _build_kernel(
            examples,
            kernel=kernel,
            wl_iterations=wl_iterations,
            normalize=normalize,
        ),
    )
    full_kernel = cached_kernel.matrix
    best_c = select_best_c(
        full_kernel,
        labels,
        train_idx,
        c_values=c_values,
        seed=random_state,
    )
    train_kernel = full_kernel[np.ix_(train_idx, train_idx)]
    test_kernel = full_kernel[np.ix_(test_idx, train_idx)]

    classifier = SVC(kernel="precomputed", C=best_c)
    classifier.fit(train_kernel, labels[train_idx])
    predictions = classifier.predict(test_kernel)
    return TrainingResult(
        accuracy=float(accuracy_score(labels[test_idx], predictions)),
        kernel_time_seconds=cached_kernel.elapsed_seconds,
        cache_hit=cached_kernel.cache_hit,
        best_c=best_c,
    )


def _load_dataset(dataset: str, data_root: Path) -> list[GraphExample]:
    if dataset.lower() == "synthetic":
        return load_synthetic_graph_classification()
    dataset_name = dataset.upper()
    if dataset_name not in SUPPORTED_TU_DATASETS:
        supported = ", ".join(("synthetic", *SUPPORTED_TU_DATASETS))
        raise ValueError(f"Unsupported dataset {dataset!r}. Choose one of: {supported}.")
    dataset_dir = data_root / dataset_name
    try:
        return load_tu_dataset(dataset_dir)
    except FileNotFoundError as error:
        command = (
            ".venv/bin/python -m graph_kernel_svm.scripts.download_tu_dataset "
            f"--dataset {dataset_name}"
        )
        raise FileNotFoundError(
            f"{dataset_name} is missing required raw files under {dataset_dir}. "
            f"Download it with: {command}"
        ) from error


def _build_kernel(
    examples: list[GraphExample],
    kernel: str,
    wl_iterations: int,
    normalize: bool,
) -> np.ndarray:
    if kernel == "stats":
        kernel_matrix = graph_stat_kernel(examples)
        return _normalize_kernel(kernel_matrix) if normalize else kernel_matrix
    if kernel == "shortest_path":
        return shortest_path_kernel(examples, normalize=normalize)
    if kernel == "wl":
        return weisfeiler_lehman_subtree_kernel(
            examples,
            num_iterations=wl_iterations,
            normalize=normalize,
        )
    raise ValueError(f"Unsupported kernel: {kernel}")


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    diagonal = np.diag(kernel)
    scale = np.sqrt(np.outer(diagonal, diagonal))
    return np.divide(kernel, scale, out=np.zeros_like(kernel), where=scale > 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--kernel",
        choices=["stats", "wl", "shortest_path"],
        default="stats",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--wl-iterations", type=int, default=3)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_C_VALUES),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = _load_dataset(args.dataset, args.data_root)
    summary = summarize_dataset(examples)
    print(
        f"dataset={args.dataset} graphs={summary.num_graphs} "
        f"class_balance={summary.class_balance} "
        f"avg_nodes={summary.avg_nodes:.2f} avg_edges={summary.avg_edges:.2f}"
    )
    result = _train_on_examples(
        examples,
        dataset=args.dataset,
        kernel=args.kernel,
        random_state=args.random_state,
        wl_iterations=args.wl_iterations,
        normalize=args.normalize,
        use_cache=args.use_cache,
        force_recompute=args.force_recompute,
        cache_dir="outputs/cache",
        c_values=tuple(args.c_values),
    )
    print(
        f"accuracy={result.accuracy:.3f} "
        f"kernel_time_seconds={result.kernel_time_seconds:.6f} "
        f"cache_hit={result.cache_hit} best_c={result.best_c:g}"
    )


if __name__ == "__main__":
    main()
