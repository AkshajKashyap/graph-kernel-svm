# graph-kernel-svm

Minimal graph-kernel SVM experiments using precomputed kernels.

```bash
.venv/bin/python -m graph_kernel_svm.scripts.download_mutag
.venv/bin/python -m graph_kernel_svm.scripts.train_baseline \
  --dataset synthetic --kernel wl
.venv/bin/python -m graph_kernel_svm.scripts.train_baseline \
  --dataset MUTAG --kernel wl --normalize
.venv/bin/python -m graph_kernel_svm.scripts.run_experiments \
  --dataset MUTAG --data-root data/raw --n-splits 10 \
  --test-size 0.25 --seed 42 --normalize
```

MUTAG is loaded from `data/raw/MUTAG` using its raw TU Dortmund text files.
Experiment results are written to `outputs/` and `reports/`.
