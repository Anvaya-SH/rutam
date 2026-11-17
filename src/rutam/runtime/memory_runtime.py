"""Memory runtime coordinator managing reads, proposals, validation, and explicit commits."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rutam.exceptions import ValidationError
from rutam.graph.operations import GraphState
from rutam.memory.controller_output import CommitResult
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.validation_output import ValidationBatchResult, ValidationDecision
from rutam.memory.write_proposal import MemoryProposal
from rutam.model.rutam_model import RUTAMForCausalLM

if TYPE_CHECKING:
    from rutam.graph.storage import GraphStorage
    from rutam.memory.controller_output import MemoryReadOutput


class MemoryRuntime:
    """Coordinates full lifecycle of discrete graph memory updates.

    Enforces RUTAM architectural invariants:
    1. Reads are dynamic and non-destructive.
    2. Model proposes candidate memory mutations outside the graph.
    3. Structural, semantic, and permission validations occur deterministically.
    4. Only validated proposals are explicitly and transactionally committed.
    5. Cache invalidation ensures immediate consistency for subsequent inference turns.

    This is a trusted in-process façade, not a Python sandbox. The host must keep
    the wrapped model private and construct AuthorizationContext from trusted identity
    and policy data rather than accepting caller-supplied role/permission claims.
    """

    def __init__(
        self,
        model: RUTAMForCausalLM,
        storage: GraphStorage | None = None,
    ) -> None:
        if not isinstance(model, RUTAMForCausalLM):
            raise ValidationError(f"model must be a RUTAMForCausalLM, got {type(model).__name__}")
        self.model = model
        self.storage = storage or model.storage

    @property
    def is_available(self) -> bool:
        """Whether graph memory operations are enabled in the active model mode."""
        return self.model.is_graph_enabled

    def get_graph_state(
        self, *, authorization_context: AuthorizationContext | None = None
    ) -> GraphState:
        """Retrieve a full immutable graph snapshot for an explicitly privileged caller.

        Raw GraphState values contain every scope, including node text and embeddings.
        Only admins or callers with ``read_all_scopes`` bound to every active scope
        may retrieve them. Scoped readers should use ``read_memory`` instead. This
        check assumes the host trusted the AuthorizationContext; Python callers who
        can access ``self.model`` are inside the trusted boundary.
        """
        return self._get_privileged_graph_state(authorization_context)

    def _get_privileged_graph_state(
        self, authorization_context: AuthorizationContext | None
    ) -> GraphState:
        if not isinstance(authorization_context, AuthorizationContext):
            raise ValidationError(
                "Full graph snapshot access requires an AuthorizationContext with "
                "role='admin' or read_all_scopes permission"
            )
        if authorization_context.role == "admin":
            return self.model.bridge.get_active_state()
        if "read" not in authorization_context.permissions:
            raise ValidationError("Full graph snapshot access requires read permission")
        if "read_all_scopes" not in authorization_context.permissions:
            raise ValidationError(
                "Full graph snapshot access requires read_all_scopes permission"
            )

        state = self.model.bridge.get_active_state()
        authorized_scopes = set(authorization_context.authorized_scopes)
        active_scopes = {
            record.get("scope", "default") for record in state.node_metadata
        } | {
            record.get("scope", "default") for record in state.edge_metadata
        }
        if not active_scopes.issubset(authorized_scopes):
            raise ValidationError(
                "read_all_scopes permission is not bound to every active graph scope"
            )
        if any(
            not isinstance(record.get("permissions", ("read", "write")), (tuple, list))
            or "read" not in record.get("permissions", ("read", "write"))
            for record in state.node_metadata
        ):
            raise ValidationError(
                "Full graph snapshot includes nodes without node-level read permission"
            )
        return state

    def get_node_count(
        self, *, authorization_context: AuthorizationContext | None = None
    ) -> int:
        """Return the global node count only to a full-snapshot-authorized caller."""
        return int(self._get_privileged_graph_state(authorization_context).node_ids.numel())

    def get_edge_count(
        self, *, authorization_context: AuthorizationContext | None = None
    ) -> int:
        """Return the global edge count only to a full-snapshot-authorized caller."""
        return int(self._get_privileged_graph_state(authorization_context).edge_types.numel())

    def read_memory(
        self,
        query_states: torch.Tensor,
        *,
        top_k: int | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> MemoryReadOutput:
        """Query active graph memory for pertinent node states."""
        if not self.is_available:
            raise ValidationError(
                "Graph memory read is not available when graph memory is disabled"
            )
        return self.model.memory_controller.read_memory(
            query_states,
            top_k=top_k,
            authorization_context=authorization_context,
        )

    def propose(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
        source_references: Any = None,
    ) -> tuple[MemoryProposal, ...]:
        """Execute model forward pass generating candidate memory update proposals.

        Args:
            input_ids: Prompt token IDs [B, T].
            attention_mask: Optional attention mask.
            authorization_context: Security/provenance authorization context.
            source_references: Provenance references.

        Returns:
            Tuple of proposed MemoryProposal instances.
        """
        if not self.is_available:
            return ()

        return self.model.propose_memory_updates(
            input_ids=input_ids,
            attention_mask=attention_mask,
            authorization_context=authorization_context,
            source_references=source_references,
        )

    def validate(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        authorization_context: AuthorizationContext | None = None,
    ) -> ValidationBatchResult:
        """Validate candidate memory proposals against current graph state and schema constraints.

        Args:
            proposals: Sequence of memory proposals to evaluate.
            authorization_context: Security/permission context.

        Returns:
            ValidationBatchResult detailing acceptance, rejections, and conflict detections.
        """
        if not proposals:
            from rutam.memory.validation_output import ValidationBatchResult

            return ValidationBatchResult.from_results(())

        return self.model.validate_memory_updates(
            proposals,
            authorization_context=authorization_context,
        )

    def commit(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        require_valid: bool = True,
        authorization_context: AuthorizationContext | None = None,
    ) -> CommitResult:
        """Transactionally commit memory proposals to active graph memory.

        Args:
            proposals: Sequence of memory proposals to commit.
            require_valid: If True, validates first and filters out rejected proposals.
            authorization_context: Security context for validation if require_valid=True.

        Returns:
            CommitResult indicating nodes/edges modified and updated graph version.
        """
        if not self.is_available:
            raise ValidationError("Cannot commit memory updates when graph memory is disabled")

        if not proposals:
            return CommitResult(
                committed=False,
                transaction_id="empty",
                applied_mutations=0,
                new_graph_version=self.model.graph.version,
            )

        to_commit: Sequence[MemoryProposal] = proposals
        if require_valid:
            validation_batch = self.validate(
                proposals,
                authorization_context=authorization_context,
            )
            valid_proposals = [
                prop
                for prop, result in zip(proposals, validation_batch.results, strict=False)
                if result.decision == ValidationDecision.ACCEPT
            ]
            if not valid_proposals:
                return CommitResult(
                    committed=False,
                    transaction_id="no_valid_proposals",
                    applied_mutations=0,
                    new_graph_version=self.model.graph.version,
                    error_message="No proposals passed discrete validation for commit",
                )
            to_commit = valid_proposals

        # Execute atomic commit and invalidate cache
        result = self.model.commit_memory_updates(to_commit, authorization_context=authorization_context)
        return result

    def reload(self, storage: GraphStorage | None = None) -> None:
        """Reload graph state from storage backend and flush neural caches."""
        target_storage = storage or self.storage
        self.model.reload_graph(target_storage)

    def save_snapshot(self, path: str | Path) -> None:
        """Persist graph state snapshot to specified directory."""
        from rutam.graph.storage import GraphStorage

        p = Path(path)
        storage = GraphStorage(p)
        storage.save(self.model.graph)

    def restore_snapshot(self, path: str | Path) -> None:
        """Restore graph memory state from a persisted snapshot path."""
        from rutam.graph.storage import GraphStorage

        p = Path(path)
        storage = GraphStorage(p)
        loaded_graph = storage.load()
        # Update model's graph contents
        self.model.graph = loaded_graph
        self.model.bridge.graph = loaded_graph
        self.model.bridge.invalidate_caches()
        self.model.memory_controller.graph = loaded_graph
        self.model.memory_controller.refresh_graph_state()
