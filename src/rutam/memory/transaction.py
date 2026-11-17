"""Transactional commit coordinator with all-or-nothing rollback and auditing."""

from __future__ import annotations

import copy
import shutil
import uuid
from dataclasses import replace
from functools import wraps
from typing import TYPE_CHECKING, Any

from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.graph.schema import Edge, Node
from rutam.memory.audit import AuditLogger, AuditRecord
from rutam.memory.controller_output import CommitResult
from rutam.memory.provenance import ProvenanceVerifier
from rutam.memory.validator import RUTAMMemoryValidator
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal

if TYPE_CHECKING:
    from rutam.graph.storage import GraphStorage


def serialized_commit(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._target_graph._mutation_lock:
            return method(self, *args, **kwargs)

    return call


class GraphTransaction:
    """Manages isolated transactional mutations on graph memory with atomic commit or rollback."""

    def __init__(
        self,
        graph: Graph,
        *,
        storage: GraphStorage | None = None,
        audit_logger: AuditLogger | None = None,
        preserve_snapshots: int = 5,
        transaction_id: str | None = None,
        validator: Any | None = None,
        authorization_context: Any | None = None,
        provenance_verifier: ProvenanceVerifier | None = None,
    ) -> None:
        if not isinstance(graph, Graph):
            raise ValidationError("GraphTransaction requires a Graph instance")
        self._target_graph = graph
        with graph._mutation_lock:
            self._base_version = graph.version
            self._base_storage_path = graph._storage_path
            self._base_generation = graph._storage_generation
            base_state = graph.to_state()
        # Custom validators may impose additional policy but must never bypass the
        # built-in structural, provenance and privacy checks at the commit boundary.
        self._validator = RUTAMMemoryValidator(
            graph.config, provenance_verifier=provenance_verifier
        )
        self._additional_validator = validator
        self._authorization_context = authorization_context
        if storage is None and graph.config.graph.auto_save:
            from rutam.graph.storage import GraphStorage

            storage = graph._storage or GraphStorage(graph.config.graph.path)
        self._storage = storage
        self._audit_logger = audit_logger
        self._preserve_snapshots = preserve_snapshots
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self._is_active = True
        self._applied_mutations: list[dict[str, Any]] = []

        # Create isolated working copy
        self._working_graph = Graph.from_state(base_state)
        # Staging must never trigger the graph's configured autosave path.
        # Restore the original configuration immediately before persistence.
        self._working_graph.config = graph.config.with_overrides({"graph": {"auto_save": False}})

    def __enter__(self) -> GraphTransaction:
        self._target_graph._mutation_lock.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is not None and self._is_active:
                self.rollback(error=str(exc_val))
        finally:
            self._target_graph._mutation_lock.release()

    def apply_proposal(self, proposal: MemoryProposal) -> None:
        """Apply a proposal; any failure permanently aborts this transaction."""
        try:
            self._apply_proposal(proposal)
        except Exception:
            # A graph operation or ledger update can fail after mutating the
            # working copy. Never allow a caught exception to publish it.
            self._is_active = False
            raise

    def _apply_proposal(self, proposal: MemoryProposal) -> None:
        """Validate and apply a proposal to the isolated working graph."""
        if not self._is_active:
            raise ValidationError("Cannot apply proposals to an inactive transaction")
        if not isinstance(proposal, MemoryProposal):
            raise ValidationError("proposal must be a MemoryProposal")

        if proposal.operation == MemoryOperation.NO_OP:
            return
        auth = self._authorization_context
        if auth is None or "commit" not in auth.permissions:
            raise ValidationError("COMMIT_NOT_AUTHORIZED")
        if proposal.scope not in auth.authorized_scopes:
            raise ValidationError("MEMORY_SCOPE_NOT_AUTHORIZED")
        for target_id in proposal.target_node_ids:
            node = self._working_graph.get_node(target_id)
            if node.scope not in auth.authorized_scopes or "write" not in node.permissions:
                raise ValidationError("MEMORY_TARGET_NOT_AUTHORIZED")
        if (
            proposal.source_graph_version is not None
            and proposal.source_graph_version != self._base_version
        ):
            raise ValidationError("STALE_GRAPH_VERSION")
        committed_ids = self._working_graph._metadata.get("committed_proposals", [])
        if proposal.proposal_id in committed_ids:
            raise ValidationError("PROPOSAL_ALREADY_COMMITTED")
        if len(committed_ids) >= 10000:
            raise ValidationError("Proposal ledger capacity exceeded")
        # A transaction is a public commit boundary. Require validator approval;
        # callers that already validated still pass the validator for re-checking
        # against this transaction's snapshot.
        verdict = self._validator.validate(
            replace(proposal, source_graph_version=None),
            self._working_graph,
            authorization_context=self._authorization_context,
        )
        if getattr(getattr(verdict, "decision", None), "name", None) != "ACCEPT":
            raise ValidationError(f"Proposal not approved: {getattr(verdict, 'reason_codes', ())}")
        if self._additional_validator is not None:
            verdict = self._additional_validator.validate(
                replace(proposal, source_graph_version=None),
                self._working_graph,
                authorization_context=self._authorization_context,
            )
            if getattr(getattr(verdict, "decision", None), "name", None) != "ACCEPT":
                raise ValidationError(
                    f"Proposal not approved: {getattr(verdict, 'reason_codes', ())}"
                )

        op = proposal.operation
        if op == MemoryOperation.NO_OP:
            return

        cand_emb = (
            tuple(proposal.candidate_embedding.tolist())
            if proposal.candidate_embedding is not None
            else None
        )
        prov_str = (
            proposal.provenance.source_id
            if hasattr(proposal.provenance, "source_id")
            else str(proposal.provenance or "model_proposal")
        )

        if op == MemoryOperation.CREATE_NODE:
            # Create new node in working graph
            new_id = self._working_graph._next_node_id
            node = Node(
                id=new_id,
                node_type=proposal.node_type or "memory_node",
                content=proposal.content or "",
                scope=proposal.scope,
                provenance=prov_str,
                confidence=proposal.confidence,
                embedding=cand_emb,
            )
            created = self._working_graph.add_node(node)
            self._applied_mutations.append(
                {
                    "op": "CREATE_NODE",
                    "target_id": created.id,
                    "proposal_id": proposal.proposal_id,
                }
            )

        elif op == MemoryOperation.UPDATE_NODE:
            if not proposal.target_node_ids:
                raise ValidationError(f"Target node ID required for {op.name}")
            target_id = proposal.target_node_ids[0]
            existing = self._working_graph.get_node(target_id)
            updated_emb = cand_emb if cand_emb is not None else existing.embedding
            self._working_graph.update_node(
                target_id,
                embedding=updated_emb,
                confidence=proposal.confidence,
                **({"content": proposal.content} if proposal.content is not None else {}),
            )
            self._applied_mutations.append(
                {
                    "op": op.name,
                    "target_id": target_id,
                    "proposal_id": proposal.proposal_id,
                }
            )

        elif op == MemoryOperation.PROPOSE_LINK:
            if len(proposal.target_node_ids) != 2:
                raise ValidationError("LINK requires exactly two targets")
            if len(proposal.target_node_ids) == 2:
                src, dst = proposal.target_node_ids[0], proposal.target_node_ids[1]
                edge = Edge(
                    source_id=src,
                    target_id=dst,
                    relationship=proposal.relation_type or "ASSOCIATED_WITH",
                    provenance=prov_str,
                    weight=proposal.confidence,
                    scope=proposal.scope,
                )

                self._working_graph.add_edge(edge)
                self._applied_mutations.append(
                    {
                        "op": "PROPOSE_LINK",
                        "target_id": src,
                        "destination_id": dst,
                        "proposal_id": proposal.proposal_id,
                    }
                )

        elif op == MemoryOperation.DELETE_NODE:
            if len(proposal.target_node_ids) != 1:
                raise ValidationError("DELETE_NODE requires one target")
            self._working_graph.delete_node(proposal.target_node_ids[0])
            self._applied_mutations.append(
                {
                    "op": op.name,
                    "target_id": proposal.target_node_ids[0],
                    "proposal_id": proposal.proposal_id,
                }
            )
        elif op == MemoryOperation.DELETE_LINK:
            if len(proposal.target_node_ids) != 2 or not proposal.relation_type:
                raise ValidationError("DELETE_LINK requires two targets and relation_type")
            self._working_graph.remove_relationship(
                *proposal.target_node_ids, proposal.relation_type
            )
            self._applied_mutations.append({"op": op.name, "proposal_id": proposal.proposal_id})
        else:
            raise ValidationError(f"Unsupported transactional operation: {op.name}")
        committed_ids = list(committed_ids)
        committed_ids.append(proposal.proposal_id)
        self._working_graph._metadata["committed_proposals"] = committed_ids

    @serialized_commit
    def commit(self) -> CommitResult:
        """Atomically commit all working graph mutations to the target graph."""
        if not self._is_active:
            raise ValidationError("Transaction is already closed")

        published = False
        try:
            # Atomic state transfer
            target = self._target_graph
            working = self._working_graph

            if target.version != self._base_version:
                raise ValidationError("STALE_GRAPH_VERSION")
            if (target._storage_path, target._storage_generation) != (
                self._base_storage_path, self._base_generation
            ):
                raise ValidationError("STALE_SNAPSHOT_GENERATION")
            nodes = copy.deepcopy(working._nodes)
            edges = copy.deepcopy(working._edges)
            metadata = copy.deepcopy(working._metadata)
            edge_ids = dict(working._edge_ids)
            relationship_types = dict(working._relationship_types)
            retired_ids = set(working._retired_ids)
            retired_edge_ids = set(working._retired_edge_ids)
            durability_warning = None
            if self._storage is not None:
                from rutam.graph.storage import SnapshotPublicationError

                working.config = target.config
                try:
                    expected = (
                        self._base_generation
                        if self._base_storage_path == str(self._storage.path.resolve())
                        else None
                    )
                    self._storage.save(working, expected_generation=expected)
                except SnapshotPublicationError as exc:
                    # The manifest is already visible; mirror it in memory and
                    # report success with a warning rather than pretending rollback.
                    durability_warning = str(exc)
            target._nodes = nodes
            target._edges = edges
            target._edge_ids = edge_ids
            target._relationship_types = relationship_types
            target._retired_ids = retired_ids
            target._retired_edge_ids = retired_edge_ids
            target._next_node_id = working._next_node_id
            target._version = working._version
            target._metadata = metadata
            if self._storage is not None:
                target._storage_generation = working._storage_generation
                target._storage_path = working._storage_path
                target._storage = self._storage

            published = True
            snapshot_path_str = None
            if self._storage is not None:
                snapshot_path_str = str(self._storage.path)
                if self._preserve_snapshots > 0:
                    self._preserve_snapshot_archive(target.version)

            # Audit logging
            if self._audit_logger is not None:
                for mutation in self._applied_mutations:
                    rec = AuditRecord(
                        transaction_id=self.transaction_id,
                        operation=mutation["op"],
                        target_id=mutation.get("target_id"),
                        details=mutation,
                        status="COMMITTED",
                    )
                    self._audit_logger.log(rec)

            self._is_active = False
            return CommitResult(
                committed=True,
                transaction_id=self.transaction_id,
                applied_mutations=len(self._applied_mutations),
                new_graph_version=target.version,
                snapshot_path=snapshot_path_str,
                error_message=durability_warning,
            )

        except Exception as exc:
            if published:
                self._is_active = False
                return CommitResult(
                    committed=True,
                    transaction_id=self.transaction_id,
                    applied_mutations=len(self._applied_mutations),
                    new_graph_version=self._target_graph.version,
                    error_message=(
                        f"{durability_warning}; " if durability_warning else ""
                    ) + f"Commit succeeded; post-commit maintenance failed: {exc}",
                )
            self.rollback(error=str(exc))
            return CommitResult(
                committed=False,
                transaction_id=self.transaction_id,
                applied_mutations=0,
                new_graph_version=self._target_graph.version,
                error_message=str(exc),
                rollback_performed=True,
            )

    def rollback(self, error: str | None = None) -> CommitResult:
        """Discard working mutations without altering the target graph."""
        self._is_active = False
        if self._audit_logger is not None and self._applied_mutations:
            rec = AuditRecord(
                transaction_id=self.transaction_id,
                operation="ROLLBACK",
                details={
                    "error": error or "Manual rollback",
                    "count": len(self._applied_mutations),
                },
                status="ROLLED_BACK",
            )
            self._audit_logger.log(rec)

        return CommitResult(
            committed=False,
            transaction_id=self.transaction_id,
            applied_mutations=0,
            new_graph_version=self._target_graph.version,
            error_message=error,
            rollback_performed=True,
        )

    def _preserve_snapshot_archive(self, version: int) -> None:
        """Archive committed snapshot and prune older snapshots beyond preserve_snapshots."""
        if self._storage is None:
            return
        base_dir = self._storage.path / "snapshots"
        base_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = base_dir / f"v{version}"
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        # Write a complete independent snapshot, including its immutable
        # generation directory, rather than copying racy compatibility aliases.
        from rutam.graph.storage import GraphStorage

        GraphStorage(archive_dir).save(self._target_graph, bind=False)

        # Prune snapshots keeping only the most recent self._preserve_snapshots
        all_snapshots = sorted(
            [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
            key=lambda d: d.stat().st_mtime,
        )
        while len(all_snapshots) > self._preserve_snapshots:
            oldest = all_snapshots.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
