"""Unit tests for relation-aware attention bias and relation attention layers."""

import torch

from rutam.config import GraphAttentionConfig
from rutam.graph.attention_bias import RelationAttentionBias
from rutam.graph.relation_attention import RelationAttentionLayer


def test_relation_attention_bias_computation():
    config = GraphAttentionConfig(hidden_dim=32, num_heads=4, num_relation_types=8)
    bias_module = RelationAttentionBias(config)

    edge_types = torch.tensor([0, 1, 2], dtype=torch.int64)
    edge_weights = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)

    bias = bias_module.compute_bias(edge_types, edge_weights)
    assert bias.shape == (3, 4)
    assert not torch.isnan(bias).any()
    assert not torch.isinf(bias).any()

    # Different relation types should have different biases
    assert not torch.allclose(bias[0], bias[1])


def test_relation_attention_layer_forward():
    config = GraphAttentionConfig(
        hidden_dim=32, num_heads=4, num_relation_types=8, residual=True, normalization="layer_norm"
    )
    layer = RelationAttentionLayer(config)

    # 4 nodes in a cyclic structure: 0 -> 1 -> 2 -> 3 -> 0
    nodes = torch.randn(4, 32)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int64)
    types = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

    out, alpha, diag = layer(nodes, edges, types, weights)
    assert out.shape == (4, 32)
    assert not torch.isnan(out).any()
    assert alpha is not None
    assert alpha.shape == (4, 4)

    # Each node has 1 incoming edge in a simple cycle, so alpha is 1.0 for each edge
    assert torch.allclose(alpha, torch.ones_like(alpha), atol=1e-5)


def test_relation_attention_solitary_nodes_residual():
    config = GraphAttentionConfig(
        hidden_dim=16, num_heads=2, residual=True, normalization="none", dropout=0.0
    )
    layer = RelationAttentionLayer(config)

    # Node 0 has no incoming or outgoing edges
    # Node 1 -> Node 2
    nodes = torch.randn(3, 16)
    edges = torch.tensor([[1], [2]], dtype=torch.int64)
    types = torch.tensor([0], dtype=torch.int64)
    weights = torch.tensor([1.0])

    out, _, _ = layer(nodes, edges, types, weights)
    assert out.shape == (3, 16)
    assert not torch.isnan(out).any()

    # Node 0 has identity preserved via residual + self_linear
    expected_node0 = nodes[0] + layer.dropout(layer.activation(layer.self_linear(nodes[0])))
    assert torch.allclose(out[0], expected_node0, atol=1e-5)


def test_relation_attention_self_loops():
    config = GraphAttentionConfig(hidden_dim=16, num_heads=2, num_relation_types=4)
    layer = RelationAttentionLayer(config)

    # Node 0 has a self-loop (0 -> 0) and incoming from 1 (1 -> 0)
    nodes = torch.randn(2, 16)
    edges = torch.tensor([[0, 1], [0, 0]], dtype=torch.int64)
    types = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0])

    out, alpha, _ = layer(nodes, edges, types, weights)
    assert out.shape == (2, 16)
    assert alpha is not None
    assert alpha.shape == (2, 2)
    assert torch.allclose(alpha.sum(dim=0), torch.ones(2), atol=1e-5)
