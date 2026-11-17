"""Unit tests for RUTAMMemoryDatasetBuilder serialization and split loading."""

from pathlib import Path

from rutam.data.memory_dataset import RUTAMMemoryDatasetBuilder


def test_builder_creates_valid_directory_structure(tmp_path: Path):
    builder = RUTAMMemoryDatasetBuilder(seed=42)
    manifest = builder.build_dataset(tmp_path, total_count=50)

    assert (tmp_path / "train.jsonl").is_file()
    assert (tmp_path / "validation.jsonl").is_file()
    assert (tmp_path / "test.jsonl").is_file()
    assert (tmp_path / "statistics.json").is_file()
    assert (tmp_path / "schema.json").is_file()
    assert (tmp_path / "manifest.json").is_file()

    assert manifest["total_count"] == 50
    assert manifest["validation_status"] == "VALID"
    assert "file_hashes" in manifest

    # Load back train split
    train_loaded = RUTAMMemoryDatasetBuilder.load_split(tmp_path / "train.jsonl")
    assert len(train_loaded) == manifest["split_counts"]["train"]
