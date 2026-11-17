"""Integration tests for graph state consistency, immutability, and synchronization in RUTAM."""

from __future__ import annotations
from rutam.memory.provenance import AuthorizationContext

from pathlib import Path

import torch

from rutam.checkpoint.graph_snapshot import load_graph_snapshot, save_graph_snapshot
from rutam.config import RUTAMConfig
from rutam.diagnostics.integration import create_verification_model
from rutam.graph.neighborhood import NeighborhoodCache
from rutam.graph.operations import Graph
from rutam.memory.provenance import ProvenanceRecord
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal


def test_forward_inference_preserves_graph_immutability() -> None:
    """Verify that model forward passes never mutate or side-effect live graph storage."""
    model, graph = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=60)

    initial_version = graph.version
    initial_node_count = len(graph.nodes)
    initial_edge_count = len(graph.edges)

    # Execute multiple forward passes with and without proposals
    input_ids = torch.randint(0, 60, (2, 8), dtype=torch.long)
    for _ in range(3):
        model(input_ids, generate_proposals=True)

    assert graph.version == initial_version
    assert len(graph.nodes) == initial_node_count
    assert len(graph.edges) == initial_edge_count


def test_snapshot_roundtrip_parity(tmp_path: Path) -> None:
    """Verify exact numerical and topological parity after saving and loading graph snapshots."""
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 16}})
    graph = Graph(config)
    for i in range(5):
        graph.create_node(
            node_type="entity",
            content=f"node_{i}",
            node_id=i * 10,
            provenance="test:parity",
            embedding=[0.1 * i] * 16,
        )
    graph.add_relationship(0, 10, "links", provenance="test:edge", weight=0.7)
    graph.add_relationship(10, 20, "supports", provenance="test:edge", weight=0.9)

    snap_dir = tmp_path / "parity_snapshot"
    save_graph_snapshot(graph, snap_dir)

    reloaded = load_graph_snapshot(config, snap_dir)

    assert reloaded.version == graph.version
    assert len(reloaded.nodes) == len(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)

    for nid in (0, 10, 20):
        n_orig = graph.get_node(nid)
        n_reloaded = reloaded.get_node(nid)
        assert n_orig.content == n_reloaded.content
        assert n_orig.embedding == n_reloaded.embedding


def test_bridge_cache_invalidation_on_commit() -> None:
    """Verify that GraphMemoryBridge clears cached tensors when new memory updates are committed."""
    model, graph = create_verification_model(num_nodes=2, hidden_dim=16, vocab_size=60)

    # Initial state cached in bridge
    state_v0 = model.bridge.get_active_state()
    assert state_v0.graph_version == graph.version

    # Commit memory proposal
    prov = ProvenanceRecord(source_id="agent", authorization_category="user_input")
    prop = MemoryProposal(
        proposal_id="new_node",
        operation=MemoryOperation.CREATE_NODE,
        target_node_ids=(),
        candidate_embedding=torch.tensor([0.4] * 16),
        provenance=prov,
        confidence=0.9,
    )
    commit_res = model.commit_memory_updates([prop], authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
    assert commit_res.committed

    # Bridge state should automatically reflect updated version
    state_v1 = model.bridge.get_active_state()
    assert state_v1.graph_version == commit_res.new_graph_version
    assert state_v1.node_ids.shape[0] == len(graph.nodes)


def test_neighborhood_cache_consistency_after_mutations() -> None:
    """Verify that NeighborhoodCache updates neighborhood indices after edge additions."""
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 16}})
    graph = Graph(config)
    graph.create_node("concept", "A", node_id=1, embedding=[0.1] * 16, provenance="test:p")
    graph.create_node("concept", "B", node_id=2, embedding=[0.2] * 16, provenance="test:p")

    cache = NeighborhoodCache()
    state1 = graph.to_state()
    index1 = cache.get(state1.node_ids, state1.edge_index, state1.edge_types)
    assert len(index1.typed_edges) == 0

    graph.add_relationship(1, 2, "connects", provenance="test:e", weight=1.0)
    state2 = graph.to_state()
    index2 = cache.get(state2.node_ids, state2.edge_index, state2.edge_types)
    assert len(index2.typed_edges) == 1
    assert len(index2.neighbors(0, direction="outgoing")) == 1
