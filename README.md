# graph-kernel-svm

A script-first graph classification project implementing graph-stat,
Weisfeiler-Lehman subtree, and shortest-path kernels from scratch. It loads local TU
Dortmund datasets, trains `SVC(kernel="precomputed")` models with leakage-safe inner
`C` tuning, and produces cached, timed, reproducible experiment reports and figures.

## Why This Project Matters

Graph kernels provide an interpretable way to compare structured objects without
learning a neural representation. This repository makes the full experimental path
visible: parsing raw graph files, constructing kernel matrices, checking numerical
health, tuning only on training data, measuring split variability, and recording the
configuration needed to reproduce a result.

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
make test
make download-data
make all-experiments
make plots
```

## Commands

```bash
make install
make test
make lint
make format-check
make download-data
make inspect
make experiment
make all-experiments
make plots
make clean-outputs
```

Direct commands remain available:

```bash
.venv/bin/python -m graph_kernel_svm.scripts.train_baseline \
  --dataset MUTAG --kernel wl --normalize \
  --c-values 0.1 1.0 10.0
```

## Current Results

Experiments support MUTAG, PTC_MR, and PROTEINS. Results vary by dataset and random
split; the project does not claim state-of-the-art performance. Generated artifacts:

- Per-dataset comparisons: `reports/{dataset}_kernel_comparison.md`
- Per-dataset diagnostics: `reports/{dataset}_diagnostics.md`
- Multi-dataset comparison: `reports/all_datasets_kernel_comparison.md`
- Multi-dataset diagnostics: `reports/all_datasets_diagnostics.md`
- Figures: `outputs/figures/`
- Technical method details: [reports/method_notes.md](reports/method_notes.md)

## Repository Structure

```text
src/graph_kernel_svm/
  data/       TU and synthetic dataset loaders
  graphs/     graph example data structures
  kernels/    graph-stat, WL, and shortest-path kernels
  models/     leakage-safe precomputed-SVM tuning
  scripts/    download, inspect, train, experiment, and plot commands
  utils/      caching and kernel diagnostics
tests/        offline unit and integration tests
reports/      method notes and generated Markdown reports
```

## Methods

- **Graph-stat baseline:** a linear kernel over compact global graph statistics.
- **WL subtree kernel:** counts deterministic neighborhood-refinement labels over
  multiple iterations.
- **Shortest-path kernel:** counts labeled endpoint pairs by shortest-path distance.
- **Precomputed SVM:** consumes custom kernel slices while keeping classifier logic
  consistent across methods.

See [method notes](reports/method_notes.md) for implementation and evaluation details.

## Reproducibility

Reports record timestamps, datasets, split count, test size, random seed,
normalization, C grid, cache settings, and the exact command. Diagnostics include
per-class F1, confusion matrices, selected-C distributions, cache hits, and kernel
matrix numerical checks. CI runs Ruff and pytest on Python 3.11 without downloading
datasets.

## Limitations

- Kernels prioritize readable implementations over large-scale performance.
- Continuous node attributes and edge labels are not currently used.
- Reported estimates depend on small benchmark datasets and repeated random splits.
- The experiment suite covers classical graph kernels only.

## Next Steps

- Profile and vectorize kernel construction for larger datasets.
- Add support for continuous node attributes and edge labels.
- Expand statistical comparison across repeated runs without changing model families.
