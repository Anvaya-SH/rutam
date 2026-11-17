"""Data loader for loading, shuffling, and batching memory instruction records."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

from rutam.data.example_schema import MemoryExampleRecord
from rutam.exceptions import ValidationError
from rutam.training.batch_collator import MemoryBatch, MemoryBatchCollator


class MemoryDataLoader:
    """Lightweight deterministic data loader for MemoryExampleRecord datasets."""

    def __init__(
        self,
        dataset: str | Path | Sequence[MemoryExampleRecord],
        batch_size: int = 4,
        shuffle: bool = False,
        collator: MemoryBatchCollator | None = None,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if isinstance(dataset, (str, Path)):
            path = Path(dataset)
            if not path.is_file():
                raise ValidationError(f"Dataset file does not exist: {path}")
            records: list[MemoryExampleRecord] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(MemoryExampleRecord.from_dict(json.loads(line)))
            self.records = records
        else:
            self.records = list(dataset)

        if not self.records:
            raise ValidationError("Cannot create MemoryDataLoader with an empty dataset")

        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.collator = collator if collator is not None else MemoryBatchCollator()
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for deterministic seed advancement."""
        self._epoch = epoch

    def __len__(self) -> int:
        n = len(self.records)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[MemoryBatch]:
        indices = list(range(len(self.records)))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            batch_records = [self.records[idx] for idx in batch_indices]
            yield self.collator(batch_records)
