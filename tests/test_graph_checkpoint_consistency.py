"""Unit tests for persistent graph memory snapshot consistency within checkpoints."""

from __future__ import annotations

import pytest

from rutam.checkpoint.graph_snapshot import load_graph_snapshot, save_graph_snapshot
from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph


def test_graph_snapshot_consistency(tmp_path) -> None:
    """Verify exact restoration of graph nodes, edges, embeddings, and metadata."""
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}})
    g1 = Graph(config)

    emb1 = tuple(0.1 * i for i in range(8))
    emb2 = tuple(0.2 * i for i in range(8))
    g1.create_node(
        node_type="concept",
        content="KnowledgeNode_A",
        embedding=emb1,
        node_id=1,
        provenance="test:provenance",
    )
    g1.create_node(
        node_type="concept",
        content="KnowledgeNode_B",
        embedding=emb2,
        node_id=2,
        provenance="test:provenance",
    )
    g1.add_relationship(
        source_id=1,
        target_id=2,
        relationship="relates_to",
        weight=0.85,
        provenance="test:provenance",
    )

    snap_dir = tmp_path / "graph_snapshot"
    save_graph_snapshot(g1, snap_dir)

    assert (snap_dir / "graph.safetensors").is_file()
    assert (snap_dir / "graph.json").is_file()
    assert (snap_dir / "graph.manifest.json").is_file()

    g2 = load_graph_snapshot(config, snap_dir)
    assert len(g2.nodes) == len(g1.nodes)
    assert len(g2.edges) == len(g1.edges)
    assert g2.version == g1.version

    node_a = g2.get_node(1)
    assert node_a is not None
    assert node_a.content == "KnowledgeNode_A"
    assert len(node_a.embedding) == len(emb1)
    for v_actual, v_expected in zip(node_a.embedding, emb1, strict=True):
        assert v_actual == pytest.approx(v_expected, rel=1e-5)
