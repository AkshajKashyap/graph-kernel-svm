PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install test lint format-check download-data inspect experiment all-experiments plots clean-outputs

install:
	$(PIP) install -e ".[dev]"

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

download-data:
	$(PYTHON) -m graph_kernel_svm.scripts.download_tu_dataset --dataset MUTAG
	$(PYTHON) -m graph_kernel_svm.scripts.download_tu_dataset --dataset PTC_MR
	$(PYTHON) -m graph_kernel_svm.scripts.download_tu_dataset --dataset PROTEINS

inspect:
	$(PYTHON) -m graph_kernel_svm.scripts.inspect_dataset --dataset MUTAG --data-root data/raw

experiment:
	$(PYTHON) -m graph_kernel_svm.scripts.run_experiments --dataset MUTAG --data-root data/raw --n-splits 10 --test-size 0.25 --seed 42 --normalize --use-cache --c-values 0.1 1.0 10.0

all-experiments:
	$(PYTHON) -m graph_kernel_svm.scripts.run_all_experiments --datasets MUTAG PTC_MR PROTEINS --data-root data/raw --n-splits 10 --test-size 0.25 --seed 42 --normalize --use-cache --c-values 0.1 1.0 10.0

plots:
	$(PYTHON) -m graph_kernel_svm.scripts.plot_results
	$(PYTHON) -m graph_kernel_svm.scripts.plot_all_results

clean-outputs:
	rm -rf outputs
	rm -f reports/*_kernel_comparison.md reports/*_diagnostics.md
