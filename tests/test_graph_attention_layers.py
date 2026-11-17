"""Unit tests for RUTAMGraphAttention multi-layer graph attention."""

import torch

from rutam.config import GraphAttentionConfig
from rutam.graph.attention import RUTAMGraphAttention


def test_multi_layer_stacking():
    config = GraphAttentionConfig(
        hidden_dim=32, num_heads=4, num_layers=3, dropout=0.0, return_attention=True
    )
    module = RUTAMGraphAttention(config)
    assert len(module.layers) == 3

    nodes = torch.randn(5, 32)
    # Directed line: 0 -> 1 -> 2 -> 3 -> 4
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    types = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

    out = module(nodes, edges, types, weights)
    assert out.node_states.shape == (5, 32)
    assert not torch.isnan(out.node_states).any()
    assert out.graph_summary is not None
    assert out.graph_summary.shape == (32,)
    assert len(out.diagnostics["layers"]) == 3


def test_subgraph_extraction_by_indices():
    config = GraphAttentionConfig(hidden_dim=16, num_heads=2, num_layers=1)
    module = RUTAMGraphAttention(config)

    # 5 nodes: 0, 1, 2, 3, 4
    # Edges: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4
    nodes = torch.randn(5, 16)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    types = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    node_ids = ["n0", "n1", "n2", "n3", "n4"]

    # Extract subgraph of nodes 1, 2, 3
    out = module(nodes, edges, types, weights, subgraph_nodes=[1, 2, 3], node_ids=node_ids)

    assert out.subgraph_extracted is True
    assert out.node_states.shape == (3, 16)
    assert out.node_id_map == ["n1", "n2", "n3"]
    assert out.diagnostics["num_nodes"] == 3
    # Induced edges between {1, 2, 3} are 1 -> 2 and 2 -> 3 (2 edges)
    assert out.diagnostics["num_edges"] == 2


def test_subgraph_extraction_by_string_ids():
    config = GraphAttentionConfig(hidden_dim=16, num_heads=2, num_layers=1)
    module = RUTAMGraphAttention(config)

    nodes = torch.randn(4, 16)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
    types = torch.tensor([0, 0, 0], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0, 1.0])
    node_ids = ["alpha", "beta", "gamma", "delta"]

    out = module(nodes, edges, types, weights, subgraph_nodes=["alpha", "gamma"], node_ids=node_ids)

    assert out.subgraph_extracted is True
    assert out.node_states.shape == (2, 16)
    assert out.node_id_map == ["alpha", "gamma"]
    # No direct edge between alpha and gamma
    assert out.diagnostics["num_edges"] == 0


def test_diagnostics_content():
    config = GraphAttentionConfig(
        hidden_dim=16, num_heads=2, num_relation_types=4, return_attention=True
    )
    module = RUTAMGraphAttention(config)

    nodes = torch.randn(3, 16)
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    types = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0])

    out = module(nodes, edges, types, weights)
    diag = out.diagnostics
    assert "final_attention_entropy" in diag
    assert "mean_per_relation" in diag
    assert "avg_active_neighbors" in diag
    assert isinstance(diag["mean_per_relation"], dict)
