"""Tests for unified graph-aware forward computation under various conditions."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def _create_small_model_and_graph(num_nodes: int = 4) -> tuple[RUTAMForCausalLM, Graph]:
    """Helper creating small model and populated graph for testing."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
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
                "max_visited_nodes": 4,
                "max_candidate_paths": 8,
                "max_selected_paths": 2,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
            },
            "rutam_model": {"return_memory_diagnostics": True},
        }
    )
    graph = Graph(config)
    gen = torch.Generator().manual_seed(42)
    for i in range(num_nodes):
        graph.create_node(
            node_type="concept",
            content=f"node_{i}",
            node_id=(i + 1) * 10,
            provenance="test:provenance",
            embedding=torch.randn(8, generator=gen).tolist(),
        )

    if num_nodes >= 2:
        graph.add_relationship(10, 20, "rel_0", provenance="test:edge")
    if num_nodes >= 3:
        graph.add_relationship(20, 30, "rel_1", provenance="test:edge")

    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=60, num_layers=3)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)
    return model, graph


def test_forward_empty_graph() -> None:
    """Verify forward computation gracefully handles empty graph without NaNs or failures."""
    torch.manual_seed(42)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {"backbone_hidden_dim": 16, "hidden_dim": 16, "projection_dim": 16},
            "rutam_model": {"return_memory_diagnostics": True},
        }
    )
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=60, num_layers=3)
    empty_graph = Graph(config)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=empty_graph)

    input_ids = torch.randint(0, 60, (2, 6), dtype=torch.long)
    out = model(input_ids)

    assert out.logits.shape == (2, 6, 60)
    assert not torch.isnan(out.logits).any()
    assert out.graph_selection_output is None
    assert out.traversal_output is None
    assert out.fusion_output is None
    assert out.diagnostics["has_nodes"] is False


def test_forward_with_populated_graph() -> None:
    """Verify forward pipeline runs query, selection, traversal, and fusion when graph has nodes."""
    torch.manual_seed(42)
    model, _ = _create_small_model_and_graph(num_nodes=4)

    input_ids = torch.randint(0, 60, (2, 6), dtype=torch.long)
    out = model(input_ids)

    assert out.logits.shape == (2, 6, 60)
    assert not torch.isnan(out.logits).any()
    assert out.graph_selection_output is not None
    assert out.traversal_output is not None
    assert out.fusion_output is not None
    assert out.fusion_output.fused_hidden_states.shape == (2, 6, 16)
    assert out.diagnostics["has_nodes"] is True
    assert out.diagnostics["selection_active"] is True
    assert out.diagnostics["traversal_active"] is True
    assert out.diagnostics["fusion_active"] is True


def test_forward_memory_available_masking() -> None:
    """Verify memory_available boolean mask gates memory querying per batch item."""
    torch.manual_seed(42)
    model, _ = _create_small_model_and_graph(num_nodes=4)

    input_ids = torch.randint(0, 60, (2, 6), dtype=torch.long)
    memory_available = torch.tensor([False, True], dtype=torch.bool)

    out = model(input_ids, memory_available=memory_available)
    assert out.graph_selection_output is not None
    # For batch item 0 (memory_available=False), query gate should be 0.0
    gate = out.graph_selection_output.query.query_gate
    assert gate[0, 0].item() == 0.0
    # For batch item 1 (memory_available=True), gate should be positive
    assert gate[1, 0].item() > 0.0


def test_forward_attention_mask() -> None:
    """Verify attention_mask tensor is supported and propagates through layers."""
    torch.manual_seed(42)
    model, _ = _create_small_model_and_graph(num_nodes=4)

    input_ids = torch.randint(0, 60, (2, 6), dtype=torch.long)
    attention_mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]],
        dtype=torch.bool,
    )

    out = model(input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (2, 6, 60)
    assert not torch.isnan(out.logits).any()


def test_forward_traversal_reasoning_context() -> None:
    """Verify multi-hop traversal constructs reasoning context when edges are available."""
    torch.manual_seed(42)
    model, _ = _create_small_model_and_graph(num_nodes=4)

    input_ids = torch.randint(0, 60, (2, 5), dtype=torch.long)
    out = model(input_ids)

    assert out.traversal_output is not None
    assert out.traversal_output.reasoning_context.shape == (2, 16)
    assert not torch.isnan(out.traversal_output.reasoning_context).any()
