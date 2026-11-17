"""Tests verifying cryptographic integrity and schema compliance of canonical generated dataset."""

import hashlib
import json
from pathlib import Path

from rutam.data.dataset_validator import DatasetValidator
from rutam.data.memory_dataset import RUTAMMemoryDatasetBuilder


def test_canonical_dataset_integrity():
    dataset_dir = Path("data/memory_training")
    manifest_path = dataset_dir / "manifest.json"
    assert manifest_path.is_file(), "Canonical dataset not found: run build_memory_dataset.py first"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_count"] == 1000
    assert manifest["validation_status"] == "VALID"

    # Verify SHA-256 hashes
    for filename, expected_hash in manifest["file_hashes"].items():
        file_path = dataset_dir / filename
        assert file_path.is_file(), f"Missing dataset file: {filename}"
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {filename}"

    # Load and validate splits
    train_records = RUTAMMemoryDatasetBuilder.load_split(dataset_dir / "train.jsonl")
    val_records = RUTAMMemoryDatasetBuilder.load_split(dataset_dir / "validation.jsonl")
    test_records = RUTAMMemoryDatasetBuilder.load_split(dataset_dir / "test.jsonl")

    validator = DatasetValidator()
    is_valid = validator.validate_splits(train_records, val_records, test_records)
    assert is_valid, f"Canonical dataset validation failed: {validator.errors}"
