"""Kernel matrix caching utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from graph_kernel_svm.graphs import GraphExample


@dataclass(frozen=True, slots=True)
class KernelCacheResult:
    """A kernel matrix together with cache and timing metadata."""

    matrix: np.ndarray
    elapsed_seconds: float
    cache_hit: bool
    cache_path: Path | None


def dataset_fingerprint(examples: Sequence[GraphExample]) -> str:
    """Return a stable fingerprint for the graph data relevant to kernels."""

    graph_records = []
    for example in examples:
        node_label_counts = Counter(
            str(attributes.get("node_label", "0"))
            for _, attributes in example.graph.nodes(data=True)
        )
        graph_records.append(
            {
                "label": example.label,
                "nodes": example.graph.number_of_nodes(),
                "edges": example.graph.number_of_edges(),
                "node_labels": sorted(node_label_counts.items()),
            }
        )
    payload = {"graph_count": len(examples), "graphs": graph_records}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def kernel_cache_key(
    dataset_name: str,
    kernel_name: str,
    normalize: bool,
    examples: Sequence[GraphExample],
    wl_iterations: int | None = None,
) -> str:
    """Build a deterministic cache key for a kernel setting and dataset."""

    setting = {
        "dataset": dataset_name,
        "kernel": kernel_name,
        "normalize": normalize,
        "wl_iterations": wl_iterations if kernel_name == "wl" else None,
        "dataset_fingerprint": dataset_fingerprint(examples),
    }
    encoded = json.dumps(setting, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    dataset_slug = "".join(
        character.lower() if character.isalnum() else "_" for character in dataset_name
    ).strip("_")
    kernel_slug = kernel_name
    if kernel_name == "wl":
        kernel_slug = f"wl_{wl_iterations}"
    normalization_slug = "normalized" if normalize else "raw"
    return f"{dataset_slug}_{kernel_slug}_{normalization_slug}_{digest}"


def save_kernel_matrix(matrix: np.ndarray, cache_path: str | Path) -> Path:
    """Save a kernel matrix to a NumPy binary file."""

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix, allow_pickle=False)
    return path


def load_kernel_matrix(cache_path: str | Path) -> np.ndarray:
    """Load a cached kernel matrix."""

    return np.load(Path(cache_path), allow_pickle=False)


def get_kernel_matrix(
    *,
    examples: Sequence[GraphExample],
    dataset_name: str,
    kernel_name: str,
    normalize: bool,
    compute: Callable[[], np.ndarray],
    wl_iterations: int | None = None,
    use_cache: bool = False,
    force_recompute: bool = False,
    cache_dir: str | Path = "outputs/cache",
) -> KernelCacheResult:
    """Load a matching matrix from cache or compute and optionally save it."""

    cache_path = Path(cache_dir) / (
        kernel_cache_key(
            dataset_name=dataset_name,
            kernel_name=kernel_name,
            normalize=normalize,
            examples=examples,
            wl_iterations=wl_iterations,
        )
        + ".npy"
    )
    start = perf_counter()
    if use_cache and cache_path.is_file() and not force_recompute:
        matrix = load_kernel_matrix(cache_path)
        return KernelCacheResult(
            matrix=matrix,
            elapsed_seconds=perf_counter() - start,
            cache_hit=True,
            cache_path=cache_path,
        )

    matrix = compute()
    if use_cache:
        save_kernel_matrix(matrix, cache_path)
    return KernelCacheResult(
        matrix=matrix,
        elapsed_seconds=perf_counter() - start,
        cache_hit=False,
        cache_path=cache_path if use_cache else None,
    )
