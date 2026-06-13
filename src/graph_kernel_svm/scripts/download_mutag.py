"""Backward-compatible MUTAG download command."""

from __future__ import annotations

from pathlib import Path

from graph_kernel_svm.scripts.download_tu_dataset import download_tu_dataset


def download_mutag(data_root: str | Path = "data/raw") -> Path:
    """Download MUTAG into data_root/MUTAG and return that directory."""

    return download_tu_dataset("MUTAG", data_root)


def main() -> None:
    print(download_mutag())


if __name__ == "__main__":
    main()
