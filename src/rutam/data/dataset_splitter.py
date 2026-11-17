"""Leakage-safe deterministic train/validation/test splitting grouped by scenario family."""

from __future__ import annotations

import random
from collections import defaultdict

from rutam.data.example_schema import MemoryExampleRecord
from rutam.exceptions import ValidationError


def split_dataset(
    records: list[MemoryExampleRecord],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[MemoryExampleRecord], list[MemoryExampleRecord], list[MemoryExampleRecord]]:
    """Split records into train, validation, and test sets safely grouped by scenario family.

    Prevents train/test leakage by ensuring all examples belonging to the same scenario
    family or entity cluster reside in strictly one split.
    """
    if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
        raise ValidationError("Split ratios must sum to 1.0")

    # Group by scenario_family
    family_groups: dict[str, list[MemoryExampleRecord]] = defaultdict(list)
    for r in records:
        family_groups[r.scenario_family].append(r)

    families = sorted(family_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(families)

    total_records = len(records)
    target_train = int(total_records * train_ratio)
    target_val = int(total_records * val_ratio)

    train_records: list[MemoryExampleRecord] = []
    val_records: list[MemoryExampleRecord] = []
    test_records: list[MemoryExampleRecord] = []

    for fam in families:
        group = family_groups[fam]
        if len(train_records) < target_train:
            train_records.extend(group)
        elif len(val_records) < target_val:
            val_records.extend(group)
        else:
            test_records.extend(group)

    # Verify zero leakage
    train_fams = {r.scenario_family for r in train_records}
    val_fams = {r.scenario_family for r in val_records}
    test_fams = {r.scenario_family for r in test_records}

    if (train_fams & val_fams) or (train_fams & test_fams) or (val_fams & test_fams):
        raise ValidationError("Scenario family leakage detected between dataset splits!")

    return train_records, val_records, test_records
