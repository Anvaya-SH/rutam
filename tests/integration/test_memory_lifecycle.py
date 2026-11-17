"""Integration tests for the complete 14-step memory lifecycle in RUTAM."""

from __future__ import annotations

from pathlib import Path

import torch

from rutam.config import RUTAMConfig
from rutam.diagnostics.memory_checks import run_memory_lifecycle_audit
from rutam.graph.operations import Graph
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.transaction import GraphTransaction
from rutam.memory.validation_output import ValidationDecision
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def test_complete_14_step_memory_lifecycle_audit(tmp_path: Path) -> None:
    """Verify that the full 14-step memory lifecycle audit passes with zero errors."""
    report = run_memory_lifecycle_audit(work_dir=tmp_path / "lifecycle_run")

    assert report.passed, (
        f"Lifecycle failed at steps: {[s.step_name for s in report.steps if not s.passed]}"
    )
    assert report.total_steps == 14
    assert report.passed_steps == 14
    assert report.failed_steps == 0
    assert report.total_duration_ms > 0.0


def test_stale_proposal_rejection() -> None:
    """Verify that proposals created against old graph versions are rejected with STALE."""
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 16}})
    graph = Graph(config)
    backbone = SyntheticBackbone(16, 50, 2)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)

    auth = AuthorizationContext(user_id="lead_user", role="admin")
    prov = ProvenanceRecord(source_id="agent", authorization_category="user_authorized")

    # Step 1: Create node 0 (advances version from 0 to 1)
    p1 = MemoryProposal(
        proposal_id="p1",
        operation=MemoryOperation.CREATE_NODE,
        target_node_ids=(),
        candidate_embedding=torch.tensor([0.1] * 16),
        provenance=prov,
        confidence=0.95,
        source_graph_version=0,
    )
    res1 = model.commit_memory_updates(
        [p1],
        authorization_context=AuthorizationContext(
            user_id="lead_user", role="admin", permissions=frozenset({"commit"})
        ),
    )
    assert res1.committed
    assert model.graph.version == 1

    # Step 2: Attempt update with stale source version 0
    stale_p = MemoryProposal(
        proposal_id="stale_p",
        operation=MemoryOperation.UPDATE_NODE,
        target_node_ids=(0,),
        candidate_embedding=torch.tensor([0.2] * 16),
        provenance=prov,
        confidence=0.9,
        source_graph_version=0,  # Current is 1
    )
    val_res = model.validate_memory_updates([stale_p], authorization_context=auth)
    assert val_res.stale_count == 1
    assert val_res.results[0].decision == ValidationDecision.STALE


def test_transaction_rollback_preserves_target_graph() -> None:
    """Verify that an aborted transaction leaves the target graph in its original intact state."""
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 16}})
    graph = Graph(config)
    graph.create_node("concept", "root", node_id=0, embedding=[0.5] * 16, provenance="test:root")
    initial_version = graph.version

    tx = GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
    with tx:
        bad_prop = MemoryProposal(
            proposal_id="prop_fail",
            operation=MemoryOperation.CREATE_NODE,
            target_node_ids=(),
            candidate_embedding=torch.tensor([1.] + [0.] * 15),
            provenance=ProvenanceRecord(
                source_id="actor", authorization_category="user_input"
            ),
            confidence=0.5,
        )
        tx.apply_proposal(bad_prop)
        assert len(tx._working_graph.nodes) == 2
        assert len(graph.nodes) == 1

        rollback_res = tx.rollback(error="Abort mutation")
        assert not rollback_res.committed
        assert rollback_res.rollback_performed

    # Target graph remained unchanged
    assert len(graph.nodes) == 1
    assert graph.version == initial_version
