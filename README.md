# graph-kernel-svm

From-scratch graph kernels for graph classification with reproducible TU dataset experiments.

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
same train/test procedure to evaluate every custom kernel matrix. SVM `C` is tuned
inside each training split using stratified inner cross-validation, so outer test
examples never influence model selection.

## Current Results

Experiments support MUTAG, PTC_MR, and PROTEINS. The runner compares the graph-stat
baseline, shortest-path kernel, and WL depths 0 through 5 over repeated stratified
splits. Results vary by dataset because graph size, labels, and structural patterns
differ, so evaluating multiple datasets gives a more useful picture than one benchmark.

## Commands

Download each dataset:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.download_tu_dataset --dataset MUTAG
.venv/bin/python -m graph_kernel_svm.scripts.download_tu_dataset --dataset PTC_MR
.venv/bin/python -m graph_kernel_svm.scripts.download_tu_dataset --dataset PROTEINS
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
  --test-size 0.25 --seed 42 --normalize --use-cache \
  --c-values 0.1 1.0 10.0
```

Run all supported datasets:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.run_all_experiments \
  --datasets MUTAG PTC_MR PROTEINS --data-root data/raw \
  --n-splits 10 --test-size 0.25 --seed 42 --normalize --use-cache \
  --c-values 0.1 1.0 10.0
```

Plot results:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.plot_results
.venv/bin/python -m graph_kernel_svm.scripts.plot_all_results
```

Raw datasets are stored under `data/raw/{DATASET_NAME}`. Combined metrics are written
to `outputs/all_datasets_kernel_comparison.csv`, with the report at
`reports/all_datasets_kernel_comparison.md`. Kernel matrices are cached under
`outputs/cache/`; add `--force-recompute` to refresh matching entries.
