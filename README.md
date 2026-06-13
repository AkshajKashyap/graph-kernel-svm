# graph-kernel-svm

From-scratch graph kernels for graph classification with reproducible MUTAG experiments.

## What this project does

This project parses TU Dortmund graph datasets, computes graph kernel matrices,
and trains scikit-learn SVM classifiers with `kernel="precomputed"`. It includes
repeatable evaluation, kernel caching, timing, Markdown reports, and publication-ready
comparison plots.

Implemented kernels:

- **Graph-stat baseline:** compares compact structural statistics such as graph size,
  density, degree, components, and triangles.
- **Weisfeiler-Lehman subtree kernel:** iteratively refines node labels from labeled
  neighborhoods and compares the resulting label-frequency features.
- **Shortest-path kernel:** compares labeled endpoint pairs grouped by shortest-path
  distance.

A precomputed SVM keeps kernel construction separate from classification, allowing the
same train/test procedure to evaluate every custom kernel matrix.

## Current results on MUTAG

The full experiment compares the graph-stat baseline, shortest-path kernel, and WL
depths 0 through 5 over repeated stratified splits. Generated metrics are saved in
`outputs/mutag_kernel_comparison.csv`, the interpretation is in
`reports/mutag_kernel_comparison.md`, and figures are written to `outputs/figures/`.

## Commands

Download MUTAG:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.download_mutag
```

Run tests:

```bash
.venv/bin/python -m pytest
```

Inspect the dataset:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.inspect_dataset \
  --dataset MUTAG --data-root data/raw
```

Run one baseline:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.train_baseline \
  --dataset MUTAG --kernel wl --normalize
```

Run the full experiment:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.run_experiments \
  --dataset MUTAG --data-root data/raw --n-splits 10 \
  --test-size 0.25 --seed 42 --normalize --use-cache
```

Plot results:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.plot_results
```

MUTAG is loaded from `data/raw/MUTAG`. Kernel matrices are cached under
`outputs/cache/`; add `--force-recompute` to refresh matching entries.
