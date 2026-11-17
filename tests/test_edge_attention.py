"""Unit tests for sparse relation-aware edge attention in RUTAM."""

import pytest
import torch

from rutam.config import GraphAttentionConfig
from rutam.exceptions import ValidationError
from rutam.graph.edge_attention import SparseRelationEdgeAttention


def test_edge_attention_empty_graph():
    config = GraphAttentionConfig(hidden_dim=32, num_heads=2)
    module = SparseRelationEdgeAttention(config)

    nodes = torch.zeros((0, 32))
    edges = torch.zeros((2, 0), dtype=torch.int64)
    types = torch.zeros((0,), dtype=torch.int64)
    weights = torch.zeros((0,))

    out, alpha, diag = module(nodes, edges, types, weights)
    assert out.shape == (0, 32)
    assert alpha is None
    assert diag["num_nodes"] == 0


def test_edge_attention_single_node_no_edges():
    config = GraphAttentionConfig(hidden_dim=32, num_heads=2)
    module = SparseRelationEdgeAttention(config)

    nodes = torch.randn(1, 32)
    edges = torch.zeros((2, 0), dtype=torch.int64)
    types = torch.zeros((0,), dtype=torch.int64)
    weights = torch.zeros((0,))

    out, alpha, diag = module(nodes, edges, types, weights)
    assert out.shape == (1, 32)
    assert torch.all(out == 0.0)
    assert alpha is None
    assert diag["num_nodes"] == 1


def test_edge_attention_destination_normalization():
    config = GraphAttentionConfig(hidden_dim=32, num_heads=4)
    module = SparseRelationEdgeAttention(config)

    # 3 nodes: node 0 -> node 2, node 1 -> node 2. Node 2 has 2 incoming edges.
    nodes = torch.randn(3, 32)
    edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.int64)
    types = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0])

    out, alpha, diag = module(nodes, edges, types, weights)
    assert out.shape == (3, 32)
    assert alpha is not None
    assert alpha.shape == (2, 4)

    # Sum of alpha over incoming edges for node 2 must equal 1.0 for each head
    alpha_sum = alpha.sum(dim=0)
    assert torch.allclose(alpha_sum, torch.ones(4), atol=1e-5)

    # Nodes 0 and 1 have no incoming edges; aggregated messages are zero
    assert diag["num_nodes"] == 3
    assert diag["num_edges"] == 2


def test_edge_attention_masked_edges():
    config = GraphAttentionConfig(hidden_dim=16, num_heads=2)
    module = SparseRelationEdgeAttention(config)

    nodes = torch.randn(2, 16)
    edges = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)  # 2 edges to node 1
    types = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0])

    # Mask out second edge
    mask = torch.tensor([True, False], dtype=torch.bool)
    _, alpha, diag = module(nodes, edges, types, weights, edge_mask=mask)

    assert alpha is not None
    assert torch.allclose(alpha[0], torch.ones(2), atol=1e-5)
    assert torch.allclose(alpha[1], torch.zeros(2), atol=1e-5)
    assert diag["masked_edge_pct"] == 50.0


def test_edge_attention_invalid_inputs():
    config = GraphAttentionConfig(hidden_dim=16, num_heads=2)
    module = SparseRelationEdgeAttention(config)

    nodes = torch.randn(2, 16)
    edges = torch.tensor([[0, 5], [1, 1]], dtype=torch.int64)  # Node 5 out of bounds
    types = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0])

    with pytest.raises(ValidationError):
        module(nodes, edges, types, weights)
