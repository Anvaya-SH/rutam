"""Typed outputs and status reporting for RUTAMMemoryController."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch

from rutam.config import nonempty_string, positive_int
from rutam.graph.traversal_output import TraversalOutput
from rutam.memory.memory_state import MemoryState
from rutam.memory.validation_output import ValidationBatchResult
from rutam.memory.write_proposal import MemoryProposal, MemoryWriteOutput
from rutam.model.query_output import QueryHeadOutput


@dataclass(frozen=True, slots=True)
class MemoryReadOutput:
    """Output of a differentiable neural memory read pass."""

    memory_context: torch.Tensor
    selected_node_ids: torch.Tensor
    selected_node_states: torch.Tensor
    selected_scores: torch.Tensor
    selection_mask: torch.Tensor
    query_output: QueryHeadOutput | None
    traversal_output: TraversalOutput | None
    graph_version: int
    gated_context: torch.Tensor | None = None

    @property
    def gated_memory_context(self) -> torch.Tensor:
        return self.gated_context if self.gated_context is not None else self.memory_context


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Outcome of a transactional commit of validated memory updates."""

    committed: bool
    transaction_id: str
    applied_mutations: int
    new_graph_version: int
    snapshot_path: str | None = None
    error_message: str | None = None
    rollback_performed: bool = False

    def __post_init__(self) -> None:
        nonempty_string(self.transaction_id, "transaction_id")
        positive_int(self.applied_mutations, "applied_mutations", minimum=0)
        positive_int(self.new_graph_version, "new_graph_version", minimum=0)


@dataclass(frozen=True, slots=True)
class MemoryWritePipelineOutput:
    """Output of a memory proposal generation and validation pipeline pass."""

    proposals: tuple[tuple[MemoryProposal, ...], ...]
    write_output: MemoryWriteOutput
    validation_result: ValidationBatchResult | None = None
    accepted_proposals: tuple[MemoryProposal, ...] = ()
    rejected_proposals: tuple[MemoryProposal, ...] = ()
    deferred_proposals: tuple[MemoryProposal, ...] = ()
    commit_result: CommitResult | None = None


@dataclass(frozen=True, slots=True)
class ControllerStatus:
    """Runtime status report of the memory controller subsystem."""

    state: MemoryState
    graph_version: int
    node_count: int
    edge_count: int
    cached_states_count: int
    pending_proposals_count: int
    total_commits: int
    read_enabled: bool
    write_enabled: bool
    last_commit_timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "graph_version": self.graph_version,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "cached_states_count": self.cached_states_count,
            "pending_proposals_count": self.pending_proposals_count,
            "total_commits": self.total_commits,
            "read_enabled": self.read_enabled,
            "write_enabled": self.write_enabled,
            "last_commit_timestamp": (
                self.last_commit_timestamp.isoformat() if self.last_commit_timestamp else None
            ),
            "metadata": dict(self.metadata),
        }
