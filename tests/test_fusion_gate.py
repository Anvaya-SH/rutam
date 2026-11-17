"""Unit tests for RUTAMFusionGate gating dynamics and mode selections."""

from __future__ import annotations

import pytest
import torch

from rutam.config import GraphFusionConfig
from rutam.exceptions import ValidationError
from rutam.model.fusion_gate import RUTAMFusionGate


def test_fusion_gate_token_mode() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, gate_mode="token")
    gate = RUTAMFusionGate(config)

    b, t, d = 2, 5, 16
    hidden = torch.randn(b, t, d)
    context = torch.randn(b, t, d)

    g = gate(hidden, context)
    assert g.shape == (b, t, 1)
    assert (g >= 0.0).all() and (g <= 1.0).all()


def test_fusion_gate_sequence_mode() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, gate_mode="sequence")
    gate = RUTAMFusionGate(config)

    b, t, d = 2, 5, 16
    hidden = torch.randn(b, t, d)
    context = torch.randn(b, t, d)

    g = gate(hidden, context)
    assert g.shape == (b, 1, 1)
    assert (g >= 0.0).all() and (g <= 1.0).all()


def test_fusion_gate_query_gate_conditioning() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, gate_mode="token")
    gate = RUTAMFusionGate(config)

    b, t, d = 2, 4, 16
    hidden = torch.randn(b, t, d)
    context = torch.randn(b, t, d)

    # Conditioning with zero query gate shuts off fusion gate
    zero_q_gate = torch.zeros((b, 1), dtype=torch.float32)
    g_zero = gate(hidden, context, query_gate=zero_q_gate)
    assert torch.allclose(g_zero, torch.zeros_like(g_zero))

    # Conditioning with active query gate
    ones_q_gate = torch.ones((b, 1), dtype=torch.float32)
    g_active = gate(hidden, context, query_gate=ones_q_gate)
    assert (g_active > 0.0).any()


def test_fusion_gate_sequence_with_token_mask() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, gate_mode="sequence")
    gate = RUTAMFusionGate(config)

    b, t, d = 2, 4, 16
    hidden = torch.randn(b, t, d)
    context = torch.randn(b, t, d)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]], dtype=torch.bool)

    g = gate(hidden, context, token_attention_mask=mask)
    assert g.shape == (b, 1, 1)
    assert not torch.isnan(g).any()


def test_fusion_gate_shape_mismatch_raises() -> None:
    config = GraphFusionConfig(hidden_dim=16)
    gate = RUTAMFusionGate(config)

    hidden = torch.randn(2, 4, 16)
    bad_context = torch.randn(2, 3, 16)

    with pytest.raises(ValidationError, match="shape"):
        gate(hidden, bad_context)
