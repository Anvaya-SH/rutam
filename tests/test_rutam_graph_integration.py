"""Tests for GraphMemoryBridge caching, mutation invalidation, and storage synchronization."""

from __future__ import annotations

from pathlib import Path

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def _create_integration_model(
    tmp_path: Path | None = None,
) -> tuple[RUTAMForCausalLM, Graph, GraphStorage | None]:
    config_dict = {
        "graph": {"embedding_dim": 8, "path": str(tmp_path) if tmp_path else "data/graph"},
        "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
        "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
        "query_head": {
            "backbone_hidden_dim": 16,
            "graph_hidden_dim": 16,
            "query_dim": 8,
            "intermediate_dim": 16,
        },
        "traversal": {
            "hidden_dim": 16,
            "path_hidden_dim": 16,
            "query_dim": 8,
            "num_relation_types": 4,
        },
        "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
        "memory_write": {
            "backbone_hidden_dim": 16,
            "hidden_dim": 16,
            "projection_dim": 16,
        },
    }
    config = RUTAMConfig().with_overrides(config_dict)
    storage = GraphStorage(config.graph.path) if tmp_path else None
    graph = Graph(config)

    gen = torch.Generator().manual_seed(101)
    for i in range(3):
        graph.create_node(
            node_type="concept",
            content=f"item_{i}",
            node_id=(i + 1) * 10,
            provenance="test:source",
            embedding=torch.randn(8, generator=gen).tolist(),
        )
    graph.add_relationship(10, 20, "rel_0", provenance="test:edge")

    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph, storage=storage)
    return model, graph, storage


def test_bridge_caching_behavior() -> None:
    """Verify bridge caches graph states and neural encodings when version is static."""
    torch.manual_seed(42)
    model, _, _ = _create_integration_model()
    bridge = model.bridge

    state1 = bridge.get_active_state()
    state2 = bridge.get_active_state()
    assert state1 is state2

    model.encoder.eval()
    with torch.no_grad():
        enc1 = bridge.get_active_encoding(model.encoder)
        enc2 = bridge.get_active_encoding(model.encoder)
    assert enc1 is enc2


def test_bridge_invalidation_on_mutation() -> None:
    """Verify modifying graph version invalidates bridge caches automatically."""
    torch.manual_seed(42)
    model, graph, _ = _create_integration_model()
    bridge = model.bridge

    initial_version = graph.version
    state1 = bridge.get_active_state()
    enc1 = bridge.get_active_encoding(model.encoder)

    # Mutate graph
    gen = torch.Generator().manual_seed(202)
    graph.create_node(
        node_type="concept",
        content="new_item",
        node_id=99,
        provenance="test:new",
        embedding=torch.randn(8, generator=gen).tolist(),
    )
    assert graph.version > initial_version

    state2 = bridge.get_active_state()
    assert state2 is not state1
    assert state2.graph_version == graph.version

    enc2 = bridge.get_active_encoding(model.encoder)
    assert enc2 is not enc1


def test_reload_from_storage(tmp_path: Path) -> None:
    """Verify persistent storage reload synchronizes active graph state."""
    torch.manual_seed(42)
    model, graph, storage = _create_integration_model(tmp_path=tmp_path)
    assert storage is not None

    # Save current state to storage
    storage.save(graph)

    # Mutate memory state
    gen = torch.Generator().manual_seed(303)
    graph.create_node(
        node_type="concept",
        content="ephemeral",
        node_id=123,
        provenance="test:ephemeral",
        embedding=torch.randn(8, generator=gen).tolist(),
    )
    assert any(n.id == 123 for n in graph.nodes)

    # Reload from storage
    model.reload_graph()
    assert not any(n.id == 123 for n in graph.nodes)


def test_forward_tracks_graph_version() -> None:
    """Verify output contains active graph version and reflects mutations."""
    torch.manual_seed(42)
    model, graph, _ = _create_integration_model()

    input_ids = torch.randint(0, 50, (1, 4), dtype=torch.long)
    out1 = model(input_ids)
    assert out1.graph_version == graph.version

    gen = torch.Generator().manual_seed(404)
    graph.create_node(
        node_type="concept",
        content="another_item",
        node_id=77,
        provenance="test:another",
        embedding=torch.randn(8, generator=gen).tolist(),
    )

    out2 = model(input_ids)
    assert out2.graph_version == graph.version
    assert out2.graph_version > out1.graph_version
