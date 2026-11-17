"""Supervised training targets and schema validation for native memory learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rutam.config import nonempty_string
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor
from rutam.memory.write_proposal import MemoryOperation


@dataclass(frozen=True, slots=True)
class MemoryTrainingTarget:
    """Structured training target defining ground-truth supervision for a memory example."""

    example_id: str
    write_decision: bool | None = None
    operation_label: int | str | MemoryOperation | None = None
    target_node_ids: list[int] = field(default_factory=list)
    relevant_node_ids: list[int] = field(default_factory=list)
    relevant_path_ids: list[int] = field(default_factory=list)
    relationship_labels: list[int] = field(default_factory=list)
    reference_embeddings: torch.Tensor | None = None
    authorization_label: bool | None = None
    validation_label: str | None = None
    provenance: str | None = None
    graph_snapshot_id: str | None = None
    curriculum_stage: str = "Stage_A"

    def __post_init__(self) -> None:
        nonempty_string(self.example_id, "example_id")
        if self.write_decision is not None and not isinstance(self.write_decision, bool):
            raise ValidationError("write_decision must be a boolean or None")

        if self.operation_label is not None:
            # Validate operation label using MemoryOperation
            op = MemoryOperation.from_value(self.operation_label)
            object.__setattr__(self, "operation_label", op)

        if self.reference_embeddings is not None:
            validate_tensor(self.reference_embeddings, "reference_embeddings", torch.float32)
            require_finite(self.reference_embeddings, "reference_embeddings")

        if self.authorization_label is not None and not isinstance(self.authorization_label, bool):
            raise ValidationError("authorization_label must be a boolean or None")

        self.validate_consistency()

    def validate_consistency(self) -> None:
        """Validate logical consistency between operation, write decision, and targets."""
        op = self.operation_label
        if op is not None:
            if op == MemoryOperation.NO_OP and self.write_decision is True:
                raise ValidationError("NO_OP operation is incompatible with write_decision=True")
            if (
                op in (MemoryOperation.UPDATE_NODE, MemoryOperation.SUPERSEDE_NODE)
                and self.write_decision is False
            ):
                raise ValidationError(
                    f"{op.name} operation requires write_decision to be True or None"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert target to serializable dictionary."""
        emb = (
            self.reference_embeddings.tolist()
            if self.reference_embeddings is not None
            else None
        )
        return {
            "example_id": self.example_id,
            "write_decision": self.write_decision,
            "operation_label": self.operation_label.name if self.operation_label else None,
            "target_node_ids": list(self.target_node_ids),
            "relevant_node_ids": list(self.relevant_node_ids),
            "relevant_path_ids": list(self.relevant_path_ids),
            "relationship_labels": list(self.relationship_labels),
            "reference_embeddings": emb,
            "authorization_label": self.authorization_label,
            "validation_label": self.validation_label,
            "provenance": self.provenance,
            "graph_snapshot_id": self.graph_snapshot_id,
            "curriculum_stage": self.curriculum_stage,
        }
