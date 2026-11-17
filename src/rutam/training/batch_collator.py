"""Batch collator converting raw dataset records into aligned tensors and training targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rutam.data.example_schema import MemoryExampleRecord
from rutam.training.memory_targets import MemoryTrainingTarget


@dataclass(frozen=True, slots=True)
class MemoryBatch:
    """Collated training batch with tokenized inputs, attention masks, and target labels."""

    batch_size: int
    example_ids: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    targets: list[MemoryTrainingTarget]
    graph_fixtures: list[dict[str, Any]]
    graphs_after: list[dict[str, Any]]
    records: list[MemoryExampleRecord]


def simple_tokenize(text: str, vocab_size: int = 100, max_len: int = 64) -> list[int]:
    """Deterministic tokenization mapping text to integers in [1, vocab_size - 1]."""
    if not text:
        return [1]
    # Map UTF-8 bytes into vocabulary space avoiding 0 (reserved for PAD)
    tokens = [(b % (vocab_size - 1)) + 1 for b in text.encode("utf-8")]
    return tokens[:max_len]


class MemoryBatchCollator:
    """Collates a list of MemoryExampleRecords into a MemoryBatch."""

    def __init__(self, vocab_size: int = 100, max_seq_len: int = 64) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __call__(self, records: Sequence[MemoryExampleRecord]) -> MemoryBatch:
        if not records:
            raise ValueError("Cannot collate an empty sequence of records")

        batch_size = len(records)
        example_ids: list[str] = []
        tokenized_seqs: list[list[int]] = []
        targets: list[MemoryTrainingTarget] = []
        graph_fixtures: list[dict[str, Any]] = []
        graphs_after: list[dict[str, Any]] = []

        for r in records:
            example_ids.append(r.example_id)
            full_text = f"{r.instruction} {r.context}".strip()
            tokens = simple_tokenize(full_text, self.vocab_size, self.max_seq_len)
            tokenized_seqs.append(tokens)

            auth_val = (
                r.authorization_context.get("authorized", True)
                if isinstance(r.authorization_context, dict)
                else True
            )

            rel_nodes = [n["id"] for n in r.graph_before.get("nodes", [])]
            target = MemoryTrainingTarget(
                example_id=r.example_id,
                write_decision=r.expected_write_decision,
                operation_label=r.expected_operation,
                target_node_ids=list(r.expected_target_nodes),
                relevant_node_ids=rel_nodes,
                authorization_label=auth_val,
                validation_label=r.expected_validation_decision,
                provenance=r.provenance,
            )
            targets.append(target)
            graph_fixtures.append(r.graph_before)
            graphs_after.append(r.graph_after_expected)

        # Pad sequences to max length in this batch
        max_batch_len = max(len(s) for s in tokenized_seqs)
        padded_ids = torch.zeros((batch_size, max_batch_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_batch_len), dtype=torch.bool)

        for i, seq in enumerate(tokenized_seqs):
            seq_len = len(seq)
            padded_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = True

        return MemoryBatch(
            batch_size=batch_size,
            example_ids=example_ids,
            input_ids=padded_ids,
            attention_mask=attention_mask,
            targets=targets,
            graph_fixtures=graph_fixtures,
            graphs_after=graphs_after,
            records=list(records),
        )
