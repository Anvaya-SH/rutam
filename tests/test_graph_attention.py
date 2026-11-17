"""Unit tests for multi-head cross-attention over graph representations."""

from __future__ import annotations

import pytest
import torch

from rutam.config import GraphFusionConfig
from rutam.exceptions import ValidationError
from rutam.model.graph_attention import RUTAMGraphAttention


def test_graph_attention_dimensions_and_sum() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    attn = RUTAMGraphAttention(config)

    b, t, k = 2, 4, 3
    hidden_states = torch.randn(b, t, 16)
    graph_states = torch.randn(b, k, 8)

    context, weights = attn(hidden_states, graph_states, return_attention=True)
    assert context.shape == (b, t, 16)
    assert weights is not None
    assert weights.shape == (b, 2, t, k)
    assert not torch.isnan(context).any()
    assert not torch.isnan(weights).any()

    # Attention weights over graph slots must sum to 1.0
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_graph_attention_slot_masking() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    attn = RUTAMGraphAttention(config)

    b, t, k = 1, 3, 4
    hidden_states = torch.randn(b, t, 16)
    graph_states = torch.randn(b, k, 8)
    graph_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

    context, weights = attn(
        hidden_states, graph_states, graph_mask=graph_mask, return_attention=True
    )
    assert weights is not None
    assert torch.allclose(weights[:, :, :, 2:], torch.zeros_like(weights[:, :, :, 2:]))
    assert torch.allclose(weights[:, :, :, :2].sum(dim=-1), torch.ones(b, 2, t), atol=1e-5)


def test_graph_attention_all_masked_slots_no_nan() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    attn = RUTAMGraphAttention(config)

    b, t, k = 2, 3, 2
    hidden_states = torch.randn(b, t, 16)
    graph_states = torch.randn(b, k, 8)
    graph_mask = torch.zeros((b, k), dtype=torch.bool)

    context, weights = attn(
        hidden_states, graph_states, graph_mask=graph_mask, return_attention=True
    )
    assert not torch.isnan(context).any()
    assert context.abs().sum() == 0.0


def test_graph_attention_empty_graph() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    attn = RUTAMGraphAttention(config)

    b, t = 2, 3
    hidden_states = torch.randn(b, t, 16)
    empty_graph_states = torch.zeros((b, 0, 8), dtype=torch.float32)

    context, weights = attn(hidden_states, empty_graph_states, return_attention=True)
    assert context.shape == (b, t, 16)
    assert not torch.isnan(context).any()
    assert weights is not None
    assert weights.shape == (b, 2, t, 0)


def test_graph_attention_invalid_mask_shape() -> None:
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    attn = RUTAMGraphAttention(config)

    hidden_states = torch.randn(2, 3, 16)
    graph_states = torch.randn(2, 4, 8)
    bad_mask = torch.ones((2, 2), dtype=torch.bool)

    with pytest.raises(ValidationError, match="graph_mask shape"):
        attn(hidden_states, graph_states, graph_mask=bad_mask)
