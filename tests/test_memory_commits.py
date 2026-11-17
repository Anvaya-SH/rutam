"""Tests for controller commit execution, auto-commit workflows, and storage snapshots."""

from __future__ import annotations

from pathlib import Path

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage
from rutam.memory.controller import RUTAMMemoryController
from rutam.memory.controller_output import CommitResult
from rutam.memory.memory_state import MemoryState
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal


def _make_config(auto_commit: bool = False, preserve_snapshots: int = 3) -> RUTAMConfig:
    return RUTAMConfig().with_overrides(
        {
            "memory_validation": {"detect_duplicates": False},
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 8},
            "selector": {"query_dim": 8, "hidden_dim": 8, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 8,
                "query_dim": 8,
                "graph_hidden_dim": 8,
                "intermediate_dim": 8,
            },
            "memory_write": {
                "backbone_hidden_dim": 8,
                "hidden_dim": 8,
                "projection_dim": 8,
                "max_proposals": 2,
            },
            "memory_controller": {
                "auto_commit": auto_commit,
                "preserve_snapshots": preserve_snapshots,
            },
        }
    )


def test_commit_empty_proposals() -> None:
    config = _make_config()
    controller = RUTAMMemoryController(config)
    result = controller.commit_validated_updates([])
    assert isinstance(result, CommitResult)
    assert result.committed
    assert result.applied_mutations == 0
    assert result.transaction_id == "empty_commit"
    assert controller.get_memory_status().total_commits == 0


def test_commit_validated_mutations_and_refresh() -> None:
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Root", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    controller = RUTAMMemoryController(config, graph=graph)

    prov = ProvenanceRecord(source_id="actor_1", authorization_category="user_input")
    p1 = MemoryProposal(
        proposal_id="prop_create",
        operation=MemoryOperation.CREATE_NODE,
        target_node_ids=(),
        candidate_embedding=torch.tensor([0.2] * 8, dtype=torch.float32),
        provenance=prov,
        confidence=0.9,
    )
    p2 = MemoryProposal(
        proposal_id="prop_link",
        operation=MemoryOperation.PROPOSE_LINK,
        target_node_ids=(1, 2),
        provenance=prov,
        confidence=0.85,
    )

    result = controller.commit_validated_updates([p1, p2], authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
    assert result.committed
    assert result.applied_mutations == 2
    assert result.new_graph_version > 0

    status = controller.get_memory_status()
    assert status.total_commits == 1
    assert status.last_commit_timestamp is not None
    assert status.node_count == 2
    assert status.edge_count == 1
    assert status.state == MemoryState.READY

    active_state = controller.get_active_state()
    assert active_state.node_ids.shape[0] == 2


def test_auto_commit_pipeline() -> None:
    config = _make_config(auto_commit=True)
    graph = Graph(config)
    graph.create_node("entity", "Seed", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    controller = RUTAMMemoryController(config, graph=graph)

    hidden_states = torch.randn(1, 3, 8)
    auth = AuthorizationContext(user_id="admin_user", role="admin")

    output = controller.propose_memory_update(
        hidden_states=hidden_states,
        authorization_context=auth,
        auto_validate=True,
    )
    if output.accepted_proposals:
        assert output.commit_result is not None
        assert output.commit_result.committed
        assert controller.get_memory_status().total_commits >= 1
    assert controller.state == MemoryState.READY


def test_commit_with_storage_and_pruning(tmp_path: Path) -> None:
    config = _make_config(preserve_snapshots=2)
    graph = Graph(config)
    graph.create_node("entity", "Root", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    storage = GraphStorage(tmp_path / "controller_storage")
    controller = RUTAMMemoryController(config, graph=graph, storage=storage)

    prov = ProvenanceRecord(source_id="actor_1", authorization_category="user_input")

    # Perform 4 separate commits
    for i in range(4):
        p = MemoryProposal(
            proposal_id=f"commit_prop_{i}",
            operation=MemoryOperation.CREATE_NODE,
            candidate_embedding=torch.tensor([float(i + 1)] * 8, dtype=torch.float32),
            provenance=prov,
        )
        res = controller.commit_validated_updates([p], authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
        assert res.committed

    # Verify canonical files in storage
    assert (storage.path / "graph.safetensors").exists()
    assert (storage.path / "graph.json").exists()
    assert (storage.path / "graph.manifest.json").exists()

    # Verify pruned snapshots directory
    snapshots_dir = storage.path / "snapshots"
    assert snapshots_dir.exists()
    preserved = [d for d in snapshots_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    assert len(preserved) == 2
