#!/usr/bin/env python3
"""CLI script to validate an existing RUTAM memory instruction dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rutam.data.dataset_validator import DatasetValidator
from rutam.data.memory_dataset import RUTAMMemoryDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RUTAM memory training dataset.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/memory_training",
        help="Directory containing dataset splits.",
    )
    args = parser.parse_args()

    dir_path = Path(args.path)
    if not dir_path.is_dir():
        print(f"Error: dataset directory not found: {dir_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating RUTAM memory dataset at: {dir_path}")

    # Load splits
    train_records = RUTAMMemoryDatasetBuilder.load_split(dir_path / "train.jsonl")
    val_records = RUTAMMemoryDatasetBuilder.load_split(dir_path / "validation.jsonl")
    test_records = RUTAMMemoryDatasetBuilder.load_split(dir_path / "test.jsonl")

    validator = DatasetValidator()
    is_valid = validator.validate_splits(train_records, val_records, test_records)

    # Check manifest
    manifest_file = dir_path / "manifest.json"
    manifest_ok = manifest_file.is_file()

    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Manifest present: {manifest_ok}")

    if not is_valid:
        print("Validation errors detected:", file=sys.stderr)
        for err in validator.errors:
            print(f" - {err}", file=sys.stderr)
        sys.exit(1)

    print("ALL DATASET CHECKS PASSED: ZERO LEAKAGE, VALID SCHEMAS, CONSISTENT TRANSITIONS.")


if __name__ == "__main__":
    main()
