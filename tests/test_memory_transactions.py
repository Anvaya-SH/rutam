"""Tests for GraphTransaction ACID semantics, atomic commits, rollbacks, and audit logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage
from rutam.memory.audit import AuditLogger
from rutam.memory.controller_output import CommitResult
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.transaction import GraphTransaction
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal


def _make_config() -> RUTAMConfig:
    return RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}, "memory_validation": {"detect_duplicates": False}})


def _make_proposal(
    op: MemoryOperation,
    *,
    target_ids: tuple[int, ...] = (),
    proposal_id: str = "prop_1",
    embedding: list[float] | None = None,
) -> MemoryProposal:
    emb = torch.tensor(embedding, dtype=torch.float32) if embedding is not None else None
    prov = ProvenanceRecord(source_id="test_actor", authorization_category="user_input")
    return MemoryProposal(
        proposal_id=proposal_id,
        operation=op,
        target_node_ids=target_ids,
        candidate_embedding=emb,
        provenance=prov,
        confidence=0.95,
    )


def test_transaction_isolation_and_rollback() -> None:
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Seed Node", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    initial_version = graph.version

    tx = GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
    with tx:
        proposal = _make_proposal(
            MemoryOperation.CREATE_NODE,
            proposal_id="p_add",
            embedding=[0.2] * 8,
        )
        tx.apply_proposal(proposal)

        # Working graph modified, target graph still untouched
        assert len(tx._working_graph.nodes) == 2
        assert len(graph.nodes) == 1
        assert graph.version == initial_version

        # Perform rollback
        res = tx.rollback(error="Manual test abort")
        assert not res.committed
        assert res.rollback_performed
        assert res.error_message == "Manual test abort"

    # Target graph remained unchanged
    assert len(graph.nodes) == 1
    assert graph.version == initial_version


def test_transaction_commit_atomicity() -> None:
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Node 1", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    initial_version = graph.version

    tx = GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
    with tx:
        # Create node
        p_add = _make_proposal(
            MemoryOperation.CREATE_NODE,
            proposal_id="p1",
            embedding=[0.3] * 8,
        )
        tx.apply_proposal(p_add)

        # Link nodes (source node 1 to newly created node 2)
        p_link = _make_proposal(
            MemoryOperation.PROPOSE_LINK,
            target_ids=(1, 2),
            proposal_id="p2",
        )
        tx.apply_proposal(p_link)

        # Update node 1
        p_update = _make_proposal(
            MemoryOperation.UPDATE_NODE,
            target_ids=(1,),
            proposal_id="p3",
            embedding=[0.5] * 8,
        )
        tx.apply_proposal(p_update)

        res = tx.commit()

    assert isinstance(res, CommitResult)
    assert res.committed
    assert res.applied_mutations == 3
    assert res.new_graph_version > initial_version
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.get_node(1).embedding == (0.5,) * 8


def test_transaction_exception_rolls_back_automatically() -> None:
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Seed", node_id=1, embedding=[0.1] * 8, provenance="test:p")

    with pytest.raises(RuntimeError, match="Simulation error"):
        with GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"}))) as tx:
            p = _make_proposal(
                MemoryOperation.CREATE_NODE,
                proposal_id="fail_prop",
                embedding=[0.4] * 8,
            )
            tx.apply_proposal(p)
            raise RuntimeError("Simulation error")

    # Target graph is untouched
    assert len(graph.nodes) == 1


def test_transaction_inactive_operations_raise() -> None:
    config = _make_config()
    graph = Graph(config)
    tx = GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
    with tx:
        tx.commit()

    p = _make_proposal(MemoryOperation.NO_OP)
    with pytest.raises(ValidationError, match="Cannot apply proposals to an inactive transaction"):
        tx.apply_proposal(p)
    with pytest.raises(ValidationError, match="Transaction is already closed"):
        tx.commit()


def test_transaction_with_storage_and_snapshot_pruning(tmp_path: Path) -> None:
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Root", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    storage = GraphStorage(tmp_path / "graph_storage")

    # Perform 4 commits with preserve_snapshots=2
    for i in range(4):
        with GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})), storage=storage, preserve_snapshots=2) as tx:
            p = _make_proposal(
                MemoryOperation.CREATE_NODE,
                proposal_id=f"p_{i}",
                embedding=[float(i + 1)] * 8,
            )
            tx.apply_proposal(p)
            res = tx.commit()
            assert res.committed

    # Canonical snapshot files must exist
    assert (storage.path / "graph.safetensors").exists()
    assert (storage.path / "graph.json").exists()
    assert (storage.path / "graph.manifest.json").exists()

    # Preserved snapshots directory must not exceed 2 snapshots
    snapshots_dir = storage.path / "snapshots"
    assert snapshots_dir.exists()
    preserved = [d for d in snapshots_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    assert len(preserved) == 2


def test_transaction_audit_logging(tmp_path: Path) -> None:
    audit_file = tmp_path / "tx_audit.jsonl"
    logger = AuditLogger(audit_file)
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Seed", node_id=1, embedding=[0.1] * 8, provenance="test:p")

    # Commit transaction
    with GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})), audit_logger=logger) as tx:
        p = _make_proposal(
            MemoryOperation.CREATE_NODE,
            proposal_id="audited_prop",
            embedding=[0.9] * 8,
        )
        tx.apply_proposal(p)
        tx.commit()

    # Rollback transaction
    with GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})), audit_logger=logger) as tx:
        p = _make_proposal(
            MemoryOperation.CREATE_NODE,
            proposal_id="rolled_back_prop",
            embedding=[0.8] * 8,
        )
        tx.apply_proposal(p)
        tx.rollback("Intended abort")

    # Verify audit file
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    rec1 = json.loads(lines[0])
    assert rec1["operation"] == "CREATE_NODE"
    assert rec1["status"] == "COMMITTED"

    rec2 = json.loads(lines[1])
    assert rec2["operation"] == "ROLLBACK"
    assert rec2["status"] == "ROLLED_BACK"
