"""Unverified neural predictions, never confirmed graph relationships."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch

from rutam.config import bounded_float, nonempty_string, positive_int
from rutam.exceptions import ValidationError
from rutam.graph.schema import identifier
from rutam.graph.schema import timestamp as validate_timestamp


@dataclass(frozen=True, slots=True)
class LinkProposal:
    source_node_id: int
    destination_node_id: int
    relation_id: int  # Persistent graph relation ID, not classifier class ID.
    relation_name: str
    link_score: float
    relation_score: float
    combined_confidence: float
    evidence_references: tuple[str, ...]
    graph_version: int
    topology_fingerprint: str
    relation_schema_version: int
    timestamp: datetime
    validation_status: str
    annotations: tuple[str, ...]
    batch_index: int | None = None

    def __post_init__(self) -> None:
        for value in (self.source_node_id, self.destination_node_id, self.relation_id):
            identifier(value)
        for name in ("link_score", "relation_score", "combined_confidence"):
            bounded_float(getattr(self, name), name, maximum=1)
        positive_int(self.graph_version, "graph_version", minimum=0)
        positive_int(self.relation_schema_version, "relation_schema_version")
        nonempty_string(self.relation_name, "relation_name")
        if self.validation_status not in ("unverified", "conflict_review"):
            raise ValidationError("Neural link proposals cannot be marked confirmed")
        validate_timestamp(self.timestamp, "proposal timestamp")
        if (
            not isinstance(self.topology_fingerprint, str)
            or len(self.topology_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.topology_fingerprint)
        ):
            raise ValidationError("Invalid topology fingerprint")
        if not isinstance(self.evidence_references, tuple) or not isinstance(
            self.annotations, tuple
        ):
            raise ValidationError("Evidence and annotations must be immutable string tuples")
        for value in (*self.evidence_references, *self.annotations):
            nonempty_string(value, "proposal reference/annotation")
        if self.batch_index is not None:
            positive_int(self.batch_index, "batch_index", minimum=0)


@dataclass(frozen=True, slots=True)
class LinkCandidates:
    pairs: torch.Tensor
    node_ids: torch.Tensor
    duplicates_removed: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class NegativeSamples:
    pairs: torch.Tensor
    requested_count: int
    exhausted: bool


@dataclass(frozen=True, slots=True)
class LinkerOutput:
    candidate_pairs: torch.Tensor
    candidate_node_ids: torch.Tensor
    link_logits: torch.Tensor
    link_probabilities: torch.Tensor
    relation_logits: torch.Tensor
    relation_probabilities: torch.Tensor
    predicted_relation_types: torch.Tensor  # Class 0 is NO_RELATION; real class = graph ID + 1.
    predicted_graph_relation_ids: torch.Tensor  # -1 for NO_RELATION or invalid candidate.
    proposal_confidence: torch.Tensor
    proposal_mask: torch.Tensor
    candidate_mask: torch.Tensor
    structural_features: torch.Tensor
    annotations: tuple[tuple[tuple[str, ...], ...], ...]  # [batch][candidate][annotation]
    proposal_batches: tuple[tuple[LinkProposal, ...], ...]
    graph_version: int
    topology_fingerprint: str
    candidate_duplicates_removed: int
    candidates_truncated: bool


@dataclass(frozen=True, slots=True)
class LinkLossOutput:
    total: torch.Tensor
    link: torch.Tensor
    relation: torch.Tensor
    ranking: torch.Tensor
