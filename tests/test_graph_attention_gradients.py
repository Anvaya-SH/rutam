"""Tests verifying gradient flow through all graph attention components."""

import torch

from rutam.config import GraphAttentionConfig
from rutam.graph.attention import RUTAMGraphAttention


def test_gradient_flow_to_all_parameters():
    config = GraphAttentionConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_relation_types=8,
        query_dim=16,
        relation_bias=True,
        query_conditioning=True,
        dropout=0.0,
    )
    module = RUTAMGraphAttention(config)

    nodes = torch.randn(3, 32, requires_grad=True)
    # 3 nodes fully connected with directed edges, so every node has 2 competing incoming edges
    edges = torch.tensor([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.int64)
    types = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    query = torch.randn(16, requires_grad=True)

    out = module(nodes, edges, types, weights, query=query)
    loss = out.node_states.sum() + out.graph_summary.sum()
    loss.backward()

    # Input gradient
    assert nodes.grad is not None
    assert torch.isfinite(nodes.grad).all()
    assert (nodes.grad != 0).any()

    # Query gradient
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert (query.grad != 0).any()

    # Check layer parameters
    for _i, layer in enumerate(module.layers):
        edge_attn = layer.edge_attention
        assert edge_attn.q_proj.weight.grad is not None
        assert torch.isfinite(edge_attn.q_proj.weight.grad).all()

        assert edge_attn.k_proj.weight.grad is not None
        assert torch.isfinite(edge_attn.k_proj.weight.grad).all()

        assert edge_attn.v_proj.weight.grad is not None
        assert torch.isfinite(edge_attn.v_proj.weight.grad).all()

        assert edge_attn.relation_k_embed.weight.grad is not None
        assert torch.isfinite(edge_attn.relation_k_embed.weight.grad).all()

        assert edge_attn.relation_v_embed.weight.grad is not None
        assert torch.isfinite(edge_attn.relation_v_embed.weight.grad).all()

        assert edge_attn.bias_module.relation_score_bias.grad is not None
        assert torch.isfinite(edge_attn.bias_module.relation_score_bias.grad).all()

        assert edge_attn.bias_module.edge_weight_scale.grad is not None
        assert torch.isfinite(edge_attn.bias_module.edge_weight_scale.grad).all()

        assert edge_attn.bias_module.query_relation_proj.weight.grad is not None
        assert torch.isfinite(edge_attn.bias_module.query_relation_proj.weight.grad).all()

        assert layer.self_linear.weight.grad is not None
        assert torch.isfinite(layer.self_linear.weight.grad).all()

        assert edge_attn.out_proj.weight.grad is not None
        assert torch.isfinite(edge_attn.out_proj.weight.grad).all()
