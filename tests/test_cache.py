from pathlib import Path

import numpy as np

from graph_kernel_svm.data import load_synthetic_graph_classification
from graph_kernel_svm.utils import (
    get_kernel_matrix,
    kernel_cache_key,
    load_kernel_matrix,
    save_kernel_matrix,
)


def test_cache_save_load_roundtrip(tmp_path: Path) -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    cache_path = tmp_path / "kernel.npy"

    save_kernel_matrix(matrix, cache_path)
    loaded = load_kernel_matrix(cache_path)

    assert np.array_equal(loaded, matrix)


def test_force_recompute_bypasses_existing_cache(tmp_path: Path) -> None:
    examples = load_synthetic_graph_classification()
    calls = 0

    def compute() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.full((len(examples), len(examples)), calls, dtype=float)

    first = get_kernel_matrix(
        examples=examples,
        dataset_name="synthetic",
        kernel_name="stats",
        normalize=False,
        compute=compute,
        use_cache=True,
        cache_dir=tmp_path,
    )
    cached = get_kernel_matrix(
        examples=examples,
        dataset_name="synthetic",
        kernel_name="stats",
        normalize=False,
        compute=compute,
        use_cache=True,
        cache_dir=tmp_path,
    )
    forced = get_kernel_matrix(
        examples=examples,
        dataset_name="synthetic",
        kernel_name="stats",
        normalize=False,
        compute=compute,
        use_cache=True,
        force_recompute=True,
        cache_dir=tmp_path,
    )

    assert calls == 2
    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert forced.cache_hit is False
    assert np.all(cached.matrix == 1.0)
    assert np.all(forced.matrix == 2.0)


def test_different_kernel_settings_have_different_cache_keys() -> None:
    examples = load_synthetic_graph_classification()

    stats_key = kernel_cache_key("synthetic", "stats", False, examples)
    normalized_key = kernel_cache_key("synthetic", "stats", True, examples)
    wl_one_key = kernel_cache_key("synthetic", "wl", False, examples, wl_iterations=1)
    wl_two_key = kernel_cache_key("synthetic", "wl", False, examples, wl_iterations=2)

    assert len({stats_key, normalized_key, wl_one_key, wl_two_key}) == 4
