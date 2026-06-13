"""Download and extract supported TU Dortmund graph datasets."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from graph_kernel_svm.data import SUPPORTED_TU_DATASETS

TU_DATASET_BASE_URL = "https://www.chrsmrrs.com/graphkerneldatasets"


def download_tu_dataset(
    dataset: str,
    data_root: str | Path = "data/raw",
) -> Path:
    """Download a supported TU dataset into data_root/dataset."""

    dataset_name = dataset.upper()
    if dataset_name not in SUPPORTED_TU_DATASETS:
        supported = ", ".join(SUPPORTED_TU_DATASETS)
        raise ValueError(f"Unsupported dataset {dataset!r}. Choose one of: {supported}.")

    root = Path(data_root)
    destination = root / dataset_name
    required_edge_file = destination / f"{dataset_name}_A.txt"
    if required_edge_file.is_file():
        return destination

    root.mkdir(parents=True, exist_ok=True)
    archive_url = f"{TU_DATASET_BASE_URL}/{dataset_name}.zip"
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / f"{dataset_name}.zip"
        urllib.request.urlretrieve(archive_url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            _validate_archive_paths(archive)
            archive.extractall(temporary_path)

        extracted = temporary_path / dataset_name
        if not extracted.is_dir():
            raise RuntimeError(
                f"Downloaded archive does not contain a {dataset_name} directory."
            )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(extracted, destination)

    return destination


def _validate_archive_paths(archive: zipfile.ZipFile) -> None:
    for member in archive.infolist():
        if member.filename.startswith("/") or ".." in Path(member.filename).parts:
            raise RuntimeError(f"Unsafe path in TU dataset archive: {member.filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_TU_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = download_tu_dataset(args.dataset, args.data_root)
    print(destination)


if __name__ == "__main__":
    main()
