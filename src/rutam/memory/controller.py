"""Central controller orchestrating neural reading, write proposals, validation, and commits."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.encoder import GraphEncoderOutput, RUTAMGraphEncoder
from rutam.graph.operations import Graph
from rutam.graph.selector import RUTAMGraphSelector
from rutam.graph.state import GraphState
from rutam.graph.traversal import RUTAMGraphTraversal
from rutam.memory.audit import AuditLogger
from rutam.memory.controller_output import (
    CommitResult,
    ControllerStatus,
    MemoryReadOutput,
    MemoryWritePipelineOutput,
)
from rutam.memory.memory_state import MemoryState
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.read_pipeline import MemoryReadPipeline
from rutam.memory.transaction import GraphTransaction
from rutam.memory.validation_output import ValidationBatchResult
from rutam.memory.validator import RUTAMMemoryValidator
from rutam.memory.write_head import RUTAMMemoryWriteHead
from rutam.memory.write_pipeline import MemoryWritePipeline
from rutam.memory.write_proposal import MemoryProposal
from rutam.model.query_head import RUTAMNeuralQueryHead

if TYPE_CHECKING:
    from rutam.graph.storage import GraphStorage


class RUTAMMemoryController(nn.Module):
    """Central orchestrator for native graph-memory read, proposal, validation, and commits.

    Coordinates neural query pooling, graph selection, multi-hop traversal, write proposals,
    discrete validation, and isolated transactional atomic graph commits.
    """

    def __init__(
        self,
        config: RUTAMConfig | None = None,
        *,
        graph: Graph | None = None,
        storage: GraphStorage | None = None,
        audit_path: Path | str | None = None,
        traversal_enabled: bool = False,
        encoder: RUTAMGraphEncoder | None = None,
        selector: RUTAMGraphSelector | None = None,
        query_head: RUTAMNeuralQueryHead | None = None,
        traversal: RUTAMGraphTraversal | None = None,
        write_head: RUTAMMemoryWriteHead | None = None,
        validator: RUTAMMemoryValidator | None = None,
    ) -> None:
        super().__init__()
        self.config = RUTAMConfig() if config is None else config
        if not isinstance(self.config, RUTAMConfig):
            raise ValidationError("RUTAMMemoryController requires a RUTAMConfig")

        self.graph = graph if graph is not None else Graph(self.config)
        self.storage = storage

        # Auditing
        audit_file = (
            audit_path
            if audit_path is not None
            else Path(self.config.graph.path) / "memory_audit.jsonl"
            if self.config.memory_controller.audit_enabled
            else None
        )
        self.audit_logger = (
            AuditLogger(audit_file) if self.config.memory_controller.audit_enabled else None
        )

        # Core neural modules
        self.encoder = encoder if encoder is not None else RUTAMGraphEncoder(self.config.encoder)
        self.selector = (
            selector if selector is not None else RUTAMGraphSelector(self.config.selector)
        )
        self.query_head = (
            query_head if query_head is not None else RUTAMNeuralQueryHead(self.config.query_head)
        )
        self.traversal = (
            traversal
            if traversal is not None
            else (RUTAMGraphTraversal(self.config.traversal) if traversal_enabled else None)
        )
        self.write_head = (
            write_head if write_head is not None else RUTAMMemoryWriteHead(self.config.memory_write)
        )

        # Validator
        self.validator = validator if validator is not None else RUTAMMemoryValidator(self.config)

        # Pipelines
        self.read_pipeline = MemoryReadPipeline(
            query_head=self.query_head,
            selector=self.selector,
            encoder=self.encoder,
            traversal=self.traversal,
        )
        self.write_pipeline = MemoryWritePipeline(
            write_head=self.write_head,
            validator=self.validator,
        )

        # Lifecycle tracking
        self.state = MemoryState.READY if len(self.graph.nodes) > 0 else MemoryState.EMPTY
        self._cached_state: GraphState | None = None
        self._cached_encoding: GraphEncoderOutput | None = None
        self._total_commits = 0
        self._last_commit_time: datetime | None = None

        if self.config.memory_controller.cache_graph_states:
            self.refresh_graph_state()

    def refresh_graph_state(self) -> None:
        """Invalidate caches and generate a fresh GraphState snapshot."""
        self._cached_state = self.graph.to_state()
        self._cached_encoding = None
        self.state = MemoryState.READY if len(self.graph.nodes) > 0 else MemoryState.EMPTY

    def get_active_state(self) -> GraphState:
        """Retrieve current cached or live GraphState."""
        if self._cached_state is None or self._cached_state.graph_version != self.graph.version:
            self.refresh_graph_state()
        assert self._cached_state is not None
        return self._cached_state

    def read_memory(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        memory_available: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> MemoryReadOutput:
        """Differentiable read pipeline: Query -> Selection -> Traversal -> Memory Context."""
        if not self.config.memory_controller.read_enabled:
            raise ValidationError("Memory read is disabled by configuration")
        if not self.state.can_read and self.state != MemoryState.EMPTY:
            raise ValidationError(f"Cannot read memory in state {self.state}")

        state = self.get_active_state()

        # Update state during read
        prev_state = self.state
        self.state = MemoryState.READ_READY

        try:
            output = self.read_pipeline(
                hidden_states=hidden_states,
                graph_snapshot=state,
                attention_mask=attention_mask,
                memory_available=memory_available,
                encoded_cache=self._cached_encoding,
                authorization_context=authorization_context,
            )
            return output
        finally:
            self.state = prev_state

    def propose_memory_update(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        selected_memory_states: torch.Tensor | None = None,
        selected_memory_ids: torch.Tensor | None = None,
        source_references: Any = None,
        memory_available: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
        auto_validate: bool = True,
    ) -> MemoryWritePipelineOutput:
        """Differentiable proposal generation: LM -> Write Head -> Proposals -> Validator."""
        if not self.config.memory_controller.write_enabled:
            raise ValidationError("Memory write is disabled by configuration")
        if not self.state.can_propose_write:
            raise ValidationError(f"Cannot propose write updates in state {self.state}")

        state = self.get_active_state()
        previous_state = self.state
        self.state = MemoryState.WRITE_PENDING
        try:
            output = self.write_pipeline(
                hidden_states=hidden_states,
                graph_snapshot=state,
                attention_mask=attention_mask,
                graph_summary=(
                    self._cached_encoding.graph_summary if self._cached_encoding else None
                ),
                selected_memory_states=selected_memory_states,
                selected_memory_ids=selected_memory_ids,
                source_references=source_references,
                memory_available=memory_available,
                authorization_context=authorization_context,
                auto_validate=auto_validate,
            )

            commit_res = None
            if self.config.memory_controller.auto_commit and output.accepted_proposals:
                commit_res = self.commit_validated_updates(
                    output.accepted_proposals, authorization_context=authorization_context
                )
                output = MemoryWritePipelineOutput(
                    proposals=output.proposals,
                    write_output=output.write_output,
                    validation_result=output.validation_result,
                    accepted_proposals=output.accepted_proposals,
                    rejected_proposals=output.rejected_proposals,
                    deferred_proposals=output.deferred_proposals,
                    commit_result=commit_res,
                )

            return output

        finally:
            if self.graph.version != state.graph_version:
                self.refresh_graph_state()
            else:
                self.state = previous_state

    def validate_updates(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        authorization_context: AuthorizationContext | None = None,
    ) -> ValidationBatchResult:
        """Discrete validation of staged proposals."""
        state = self.get_active_state()
        prev_state = self.state
        self.state = MemoryState.VALIDATING
        try:
            return self.validator.validate_batch(
                proposals,
                state,
                authorization_context=authorization_context,
            )
        finally:
            self.state = prev_state

    def commit_validated_updates(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        authorization_context: AuthorizationContext | None = None,
    ) -> CommitResult:
        """Transactional commit of validated mutations outside autograd."""
        if not proposals:
            return CommitResult(
                committed=True,
                transaction_id="empty_commit",
                applied_mutations=0,
                new_graph_version=self.graph.version,
            )

        prev_state = self.state
        self.state = MemoryState.COMMITTING

        try:
            with torch.no_grad():
                tx = GraphTransaction(
                    self.graph,
                    storage=self.storage,
                    audit_logger=self.audit_logger,
                    preserve_snapshots=self.config.memory_controller.preserve_snapshots,
                    validator=self.validator,
                    authorization_context=authorization_context,
                    provenance_verifier=self.validator.provenance_verifier,
                )
                with tx:
                    for p in proposals:
                        tx.apply_proposal(p)
                    result = tx.commit()

            if result.committed:
                self._total_commits += 1
                self._last_commit_time = datetime.now()
                self.refresh_graph_state()

            return result
        finally:
            self.state = prev_state if self.state == MemoryState.COMMITTING else self.state

    def get_memory_status(self) -> ControllerStatus:
        """Retrieve current memory system status report."""
        cached_count = 1 if self._cached_state is not None else 0
        return ControllerStatus(
            state=self.state,
            graph_version=self.graph.version,
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            cached_states_count=cached_count,
            pending_proposals_count=0,
            total_commits=self._total_commits,
            read_enabled=self.config.memory_controller.read_enabled,
            write_enabled=self.config.memory_controller.write_enabled,
            last_commit_timestamp=self._last_commit_time,
        )
