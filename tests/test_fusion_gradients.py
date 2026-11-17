"""Unit tests for backward pass, gradient flow, and differentiability in RUTAMGraphFusion."""

from __future__ import annotations

import torch

from rutam.config import GraphFusionConfig
from rutam.model.fusion import RUTAMGraphFusion


def test_fusion_end_to_end_gradient_flow() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(
        hidden_dim=16,
        graph_dim=8,
        num_attention_heads=2,
        use_reasoning_context=True,
    )
    fusion = RUTAMGraphFusion(config)

    b, t, k = 2, 4, 3
    hidden_states = torch.randn(b, t, 16, requires_grad=True)
    graph_states = torch.randn(b, k, 8, requires_grad=True)
    reasoning_context = torch.randn(b, 8, requires_grad=True)
    query_gate = torch.rand(b, 1, requires_grad=True)
    graph_mask = torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool)

    output = fusion(
        hidden_states,
        graph_states,
        graph_mask=graph_mask,
        reasoning_context=reasoning_context,
        query_gate=query_gate,
    )

    loss = output.fused_hidden_states.sum()
    loss.backward()

    # Input tensor gradients
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()
    assert hidden_states.grad.abs().sum() > 0.0

    assert graph_states.grad is not None
    assert not torch.isnan(graph_states.grad).any()
    assert graph_states.grad.abs().sum() > 0.0

    assert reasoning_context.grad is not None
    assert not torch.isnan(reasoning_context.grad).any()
    assert reasoning_context.grad.abs().sum() > 0.0

    assert query_gate.grad is not None
    assert not torch.isnan(query_gate.grad).any()

    # Module parameter gradients
    assert fusion.residual_scale.grad is not None
    assert not torch.isnan(fusion.residual_scale.grad).any()

    assert fusion.projections.q_proj.weight.grad is not None
    assert fusion.projections.k_proj.weight.grad is not None
    assert fusion.gate.mlp[0].weight.grad is not None


def test_fusion_empty_graph_gradient_stability() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion = RUTAMGraphFusion(config)

    b, t = 2, 3
    hidden_states = torch.randn(b, t, 16, requires_grad=True)
    empty_graph_states = torch.zeros((b, 0, 8), dtype=torch.float32, requires_grad=True)

    output = fusion(hidden_states, empty_graph_states)
    loss = output.fused_hidden_states.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()
    assert torch.allclose(hidden_states.grad, torch.ones_like(hidden_states))


def test_fusion_residual_identity_at_initialization() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(
        hidden_dim=16,
        graph_dim=8,
        num_attention_heads=2,
        residual_scale_init=0.001,
    )
    fusion = RUTAMGraphFusion(config)

    b, t, k = 2, 4, 3
    hidden_states = torch.randn(b, t, 16)
    graph_states = torch.randn(b, k, 8)

    output = fusion(hidden_states, graph_states)
    # With small residual scale, output should closely match identity
    assert torch.allclose(output.fused_hidden_states, hidden_states, atol=1e-2)
