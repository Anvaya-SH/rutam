"""Tests for contradiction detection, mutually exclusive relations, and concurrency checks."""

from __future__ import annotations

from rutam import Edge, Graph, MemoryProposal, Node, RUTAMConfig
from rutam.memory.conflict_detection import detect_conflicts
from rutam.memory.write_proposal import MemoryOperation


def test_optimistic_concurrency_stale_graph_version() -> None:
    graph = Graph()
    # Add nodes to advance graph version
    graph.add_node(Node(id=1, node_type="concept", content="Node 1", provenance="doc_1"))
    graph.add_node(Node(id=2, node_type="concept", content="Node 2", provenance="doc_2"))
    assert graph.version == 2

    # Proposal created on stale snapshot (version 0)
    stale_proposal = MemoryProposal(
        operation=MemoryOperation.UPDATE_NODE,
        target_node_ids=(1,),
        source_graph_version=0,
    )
    is_conflict, reason, msgs = detect_conflicts(stale_proposal, graph)
    assert is_conflict
    assert reason == "STALE_GRAPH_VERSION"
    assert "older than active" in msgs[0]

    # Proposal created on active snapshot (version 2)
    current_proposal = MemoryProposal(
        operation=MemoryOperation.UPDATE_NODE,
        target_node_ids=(1,),
        source_graph_version=2,
    )
    is_conflict, reason, _ = detect_conflicts(current_proposal, graph)
    assert not is_conflict
    assert reason is None


def test_mutually_exclusive_relationship_conflict() -> None:
    graph = Graph()
    graph.add_node(Node(id=1, node_type="person", content="Alice", provenance="doc"))
    graph.add_node(Node(id=2, node_type="person", content="Bob", provenance="doc"))
    graph.add_edge(Edge(source_id=1, target_id=2, relationship="PARENT_OF", provenance="doc"))

    conflicts = (("PARENT_OF", "CHILD_OF"), ("FRIEND_OF", "ENEMY_OF"))

    # Conflicting link proposal
    class DummyLinkProposal:
        source_node_id = 1
        destination_node_id = 2
        relation_name = "CHILD_OF"

    is_conflict, reason, msgs = detect_conflicts(
        DummyLinkProposal(), graph, conflicting_relations=conflicts
    )
    assert is_conflict
    assert reason == "CONFLICTING_RELATIONSHIP"
    assert len(msgs) == 1

    # Non-conflicting link proposal
    class CompatibleLinkProposal:
        source_node_id = 1
        destination_node_id = 2
        relation_name = "MENTOR_OF"

    is_conflict, reason, _ = detect_conflicts(
        CompatibleLinkProposal(), graph, conflicting_relations=conflicts
    )
    assert not is_conflict
    assert reason is None


def test_undirected_graph_conflict_detection() -> None:
    cfg = RUTAMConfig().with_overrides({"graph": {"directed": False}})
    graph = Graph(cfg)
    graph.add_node(Node(id=10, node_type="entity", content="E1", provenance="p"))
    graph.add_node(Node(id=20, node_type="entity", content="E2", provenance="p"))
    graph.add_edge(Edge(source_id=10, target_id=20, relationship="FRIEND_OF", provenance="p"))

    conflicts = (("FRIEND_OF", "ENEMY_OF"),)

    # Reverse direction proposal should still conflict in undirected mode
    class ReverseProposal:
        source_id = 20
        target_id = 10
        relationship = "ENEMY_OF"

    is_conflict, reason, _ = detect_conflicts(
        ReverseProposal(), graph, conflicting_relations=conflicts
    )
    assert is_conflict
    assert reason == "CONFLICTING_RELATIONSHIP"
