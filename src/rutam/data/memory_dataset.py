"""Dataset builder serializing versioned JSONL splits, manifest, and statistics."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rutam.data.dataset_splitter import split_dataset
from rutam.data.dataset_statistics import compute_dataset_statistics
from rutam.data.dataset_validator import DatasetValidator
from rutam.data.example_schema import DATASET_SCHEMA_VERSION, MemoryExampleRecord
from rutam.data.scenario_generator import ScenarioGenerator
from rutam.exceptions import ValidationError


class RUTAMMemoryDatasetBuilder:
    """Coordinates generation, validation, splitting, and serialization of memory datasets."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.generator = ScenarioGenerator(seed=seed)
        self.validator = DatasetValidator()

    def build_dataset(
        self,
        output_dir: str | Path,
        *,
        total_count: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> dict[str, Any]:
        """Generate and save the complete structured dataset with manifest and statistics."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Scale category targets proportionally to total_count
        scale = total_count / 1000.0
        target_counts = {
            "creation": int(180 * scale),
            "update": int(140 * scale),
            "link": int(140 * scale),
            "noop": int(140 * scale),
            "duplicate": int(100 * scale),
            "contradiction": int(100 * scale),
            "multihop": int(100 * scale),
            "authorization": int(100 * scale),
        }
        # Adjust remainder to match exact total_count
        current_sum = sum(target_counts.values())
        if current_sum < total_count:
            target_counts["creation"] += total_count - current_sum

        # 1. Generate records
        records = self.generator.generate_all(target_counts)
        if len(records) != total_count:
            raise ValidationError(f"Generated {len(records)} != expected {total_count}")

        # 2. Leakage-safe split
        train_records, val_records, test_records = split_dataset(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=self.seed,
        )

        # 3. Validation
        if not self.validator.validate_splits(train_records, val_records, test_records):
            raise ValidationError(f"Dataset validation failed: {self.validator.errors}")

        # 4. Serialize JSONL splits
        train_file = out_path / "train.jsonl"
        val_file = out_path / "validation.jsonl"
        test_file = out_path / "test.jsonl"

        self._write_jsonl(train_file, train_records)
        self._write_jsonl(val_file, val_records)
        self._write_jsonl(test_file, test_records)

        # 5. Compute statistics
        stats = {
            "overall": compute_dataset_statistics(records, split_name="overall"),
            "train": compute_dataset_statistics(train_records, split_name="train"),
            "validation": compute_dataset_statistics(val_records, split_name="validation"),
            "test": compute_dataset_statistics(test_records, split_name="test"),
        }
        stats_file = out_path / "statistics.json"
        stats_file.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

        # 6. Schema file
        schema = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "fields": [
                "example_id",
                "scenario_id",
                "scenario_family",
                "instruction",
                "context",
                "graph_before",
                "expected_write_decision",
                "expected_operation",
                "expected_target_nodes",
                "expected_relationships",
                "expected_validation_decision",
                "authorization_context",
                "provenance",
                "graph_after_expected",
                "difficulty",
                "schema_version",
            ],
        }
        schema_file = out_path / "schema.json"
        schema_file.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")

        # 7. Compute file hashes for manifest
        file_hashes = {
            "train.jsonl": self._sha256(train_file),
            "validation.jsonl": self._sha256(val_file),
            "test.jsonl": self._sha256(test_file),
            "statistics.json": self._sha256(stats_file),
            "schema.json": self._sha256(schema_file),
        }

        manifest = {
            "dataset_version": "1.0.0",
            "schema_version": DATASET_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "random_seed": self.seed,
            "total_count": len(records),
            "split_counts": {
                "train": len(train_records),
                "validation": len(val_records),
                "test": len(test_records),
            },
            "file_hashes": file_hashes,
            "validation_status": "VALID",
            "known_limitations": [
                "Synthetic scenarios only; does not represent verified real-world facts.",
                "Small bounded graphs designed for deterministic CPU testing.",
            ],
        }
        manifest_file = out_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        return manifest

    @staticmethod
    def load_split(file_path: str | Path) -> list[MemoryExampleRecord]:
        """Load records from a JSONL split file."""
        path = Path(file_path)
        if not path.is_file():
            raise ValidationError(f"File not found: {path}")

        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    record = MemoryExampleRecord.from_json(clean)
                    records.append(record)
                except Exception as exc:
                    raise ValidationError(
                        f"Failed to parse line {line_no} in {path}: {exc}"
                    ) from exc
        return records

    @staticmethod
    def _write_jsonl(path: Path, records: list[MemoryExampleRecord]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(r.to_json() + "\n")

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()
