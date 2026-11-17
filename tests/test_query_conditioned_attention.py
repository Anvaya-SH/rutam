"""Unit tests for query-conditioned graph attention."""

import pytest
import torch

from rutam.config import GraphAttentionConfig
from rutam.exceptions import ValidationError
from rutam.graph.attention import RUTAMGraphAttention


def test_query_conditioned_scores_influence_attention():
    config = GraphAttentionConfig(
        hidden_dim=32, num_heads=2, num_relation_types=4, query_dim=16, return_attention=True
    )
    module = RUTAMGraphAttention(config)

    # 3 nodes: node 0 -> node 2 (rel 0), node 1 -> node 2 (rel 1)
    nodes = torch.randn(3, 32)
    edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.int64)
    types = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0])

    # 1D Query vector
    query1 = torch.randn(16)
    out1 = module(nodes, edges, types, weights, query=query1)

    query2 = torch.randn(16) * 5.0
    out2 = module(nodes, edges, types, weights, query=query2)

    assert out1.edge_attention is not None
    assert out2.edge_attention is not None

    # Different queries must produce different attention distributions over relation types
    assert not torch.allclose(out1.edge_attention, out2.edge_attention, atol=1e-4)


def test_batched_query_conditioning():
    config = GraphAttentionConfig(
        hidden_dim=32, num_heads=2, num_relation_types=4, query_dim=16, return_attention=True
    )
    module = RUTAMGraphAttention(config)

    nodes = torch.randn(3, 32)
    edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.int64)
    types = torch.tensor([0, 1], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0])

    # Batch of 4 queries: [4, 16]
    queries = torch.randn(4, 16)
    out = module(nodes, edges, types, weights, query=queries)

    # Output node states should be batched: [4, 3, 32]
    assert out.node_states.shape == (4, 3, 32)
    assert out.edge_attention is not None
    assert out.edge_attention.shape == (4, 2, 2)
    assert out.graph_summary is not None
    assert out.graph_summary.shape == (4, 32)


def test_batched_query_memory_budget_check():
    config = GraphAttentionConfig(hidden_dim=128, max_subgraph_nodes=10)
    module = RUTAMGraphAttention(config)

    nodes = torch.randn(20, 128)
    edges = torch.tensor([[0], [1]], dtype=torch.int64)
    types = torch.tensor([0], dtype=torch.int64)
    weights = torch.tensor([1.0])

    # Huge batch that exceeds budget
    huge_queries = torch.randn(1000, 128)
    with pytest.raises(ValidationError, match="exceeds memory budget"):
        module(nodes, edges, types, weights, query=huge_queries)
