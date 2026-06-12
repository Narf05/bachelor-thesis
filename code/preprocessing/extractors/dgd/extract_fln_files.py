"""
Copy DGD .fln files from per-download folders into one fln_files directory.

Default input:
    ~/Downloads/DGD_PF

Default output:
    ~/Downloads/DGD_PF/fln_files

Usage:
    python code/preprocessing/extractors/dgd/extract_fln_files.py
    python code/preprocessing/extractors/dgd/extract_fln_files.py --dry-run
    python code/preprocessing/extractors/dgd/extract_fln_files.py --source path/to/DGD_PF --output path/to/fln_files
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path.home() / "Downloads" / "DGD_ZW"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "fln_files"


def _unique_destination(output_dir: Path, filename: str, folder_name: str) -> Path:
    destination = output_dir / filename
    if not destination.exists():
        return destination

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    destination = output_dir / f"{folder_name}{suffix}"
    if not destination.exists():
        return destination

    counter = 2
    while True:
        destination = output_dir / f"{folder_name}_{counter}{suffix}"
        if not destination.exists():
            return destination
        counter += 1


def collect_fln_files(source_dir: Path, output_dir: Path, dry_run: bool = False) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a folder: {source_dir}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for folder in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        if folder.resolve() == output_dir.resolve():
            continue

        fln_files = sorted(folder.glob("*.fln"))
        if not fln_files:
            print(f"Skipping {folder.name}: no .fln file found")
            continue

        if len(fln_files) > 1:
            print(f"Warning: {folder.name} contains {len(fln_files)} .fln files")

        for fln_file in fln_files:
            destination = _unique_destination(output_dir, fln_file.name, folder.name)
            print(f"{'Would copy' if dry_run else 'Copying'} {fln_file} -> {destination}")
            if not dry_run:
                shutil.copy2(fln_file, destination)
            copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect .fln files from DGD_PF subfolders into one fln_files folder."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder containing DGD download subfolders. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Folder where .fln files will be copied. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be copied without creating or copying anything.",
    )
    args = parser.parse_args()

    count = collect_fln_files(args.source, args.output, dry_run=args.dry_run)
    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {count} .fln file(s).")


if __name__ == "__main__":
    main()
