"""Unit tests for training batch collation and dataloader mechanics."""

from __future__ import annotations

import pytest
import torch

from rutam.data.scenario_generator import ScenarioGenerator
from rutam.exceptions import ValidationError
from rutam.training.batch_collator import MemoryBatchCollator, simple_tokenize
from rutam.training.dataloader import MemoryDataLoader


def test_simple_tokenize() -> None:
    """Verify deterministic ascii byte tokenization."""
    toks = simple_tokenize("Hello", vocab_size=100, max_len=10)
    assert len(toks) == 5
    assert all(1 <= t < 100 for t in toks)

    # Empty string fallback
    empty_toks = simple_tokenize("", vocab_size=50)
    assert empty_toks == [1]


def test_batch_collator_shapes_and_masks() -> None:
    """Verify collation into aligned tensors and target instances."""
    gen = ScenarioGenerator(seed=42)
    records = gen.generate_creation_scenarios(3)

    collator = MemoryBatchCollator(vocab_size=100, max_seq_len=32)
    batch = collator(records)

    assert batch.batch_size == 3
    assert batch.input_ids.shape[0] == 3
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.input_ids.dtype == torch.long
    assert batch.attention_mask.dtype == torch.bool

    # Check mask matches positive token positions
    for i in range(3):
        seq_len = (batch.input_ids[i] > 0).sum().item()
        assert batch.attention_mask[i, :seq_len].sum().item() == seq_len

    assert len(batch.targets) == 3
    assert batch.targets[0].example_id == records[0].example_id
    assert len(batch.graph_fixtures) == 3


def test_batch_collator_empty_error() -> None:
    """Verify collation of empty record sequence raises error."""
    collator = MemoryBatchCollator()
    with pytest.raises(ValueError, match="Cannot collate an empty sequence"):
        collator([])


def test_memory_dataloader_batching_and_iteration() -> None:
    """Verify dataloader yields correctly-sized batches and covers dataset."""
    gen = ScenarioGenerator(seed=42)
    records = gen.generate_creation_scenarios(10)

    loader = MemoryDataLoader(records, batch_size=4, shuffle=False)
    assert len(loader) == 3  # 4 + 4 + 2

    batches = list(loader)
    assert len(batches) == 3
    assert batches[0].batch_size == 4
    assert batches[1].batch_size == 4
    assert batches[2].batch_size == 2

    total_examples = sum(b.batch_size for b in batches)
    assert total_examples == 10


def test_memory_dataloader_empty_error() -> None:
    """Verify dataloader rejects empty datasets."""
    with pytest.raises(ValidationError, match="Cannot create MemoryDataLoader with an empty"):
        MemoryDataLoader([])
