"""Download and extract the MUTAG dataset."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MUTAG_URL = "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip"


def download_mutag(data_root: str | Path = "data/raw") -> Path:
    """Download MUTAG into data_root/MUTAG and return that directory."""

    root = Path(data_root)
    destination = root / "MUTAG"
    if (destination / "MUTAG_A.txt").is_file():
        return destination

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "MUTAG.zip"
        urllib.request.urlretrieve(MUTAG_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            _validate_archive_paths(archive)
            archive.extractall(temporary_directory)

        extracted = Path(temporary_directory) / "MUTAG"
        if not extracted.is_dir():
            raise RuntimeError("Downloaded archive does not contain a MUTAG directory.")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(extracted, destination)

    return destination


def _validate_archive_paths(archive: zipfile.ZipFile) -> None:
    for member in archive.infolist():
        if member.filename.startswith("/") or ".." in Path(member.filename).parts:
            raise RuntimeError(f"Unsafe path in MUTAG archive: {member.filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = download_mutag(args.data_root)
    print(destination)


if __name__ == "__main__":
    main()
