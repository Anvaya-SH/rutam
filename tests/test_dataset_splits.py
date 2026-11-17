"""Unit tests for leakage-safe grouped dataset splitting."""

import pytest

from rutam.data.dataset_splitter import split_dataset
from rutam.data.example_schema import MemoryExampleRecord
from rutam.data.scenario_generator import ScenarioGenerator
from rutam.exceptions import ValidationError


def test_leakage_safe_split():
    gen = ScenarioGenerator(seed=42)
    records = gen.generate_all(
        {
            "creation": 20,
            "update": 20,
            "link": 20,
            "noop": 20,
            "duplicate": 10,
            "contradiction": 10,
            "multihop": 10,
            "authorization": 10,
        }
    )

    train, val, test = split_dataset(
        records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    assert len(train) + len(val) + len(test) == len(records)

    train_fams = {r.scenario_family for r in train}
    val_fams = {r.scenario_family for r in val}
    test_fams = {r.scenario_family for r in test}

    # Strict zero leakage guarantee
    assert len(train_fams & val_fams) == 0
    assert len(train_fams & test_fams) == 0
    assert len(val_fams & test_fams) == 0


def test_invalid_split_ratios():
    r = MemoryExampleRecord(
        example_id="ex1",
        scenario_id="s1",
        scenario_family="f1",
        instruction="Test",
        context="Context",
        graph_before={"nodes": [], "edges": []},
        expected_write_decision=False,
        expected_operation="NO_OP",
    )
    with pytest.raises(ValidationError, match="ratios must sum to 1.0"):
        split_dataset([r], train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)
