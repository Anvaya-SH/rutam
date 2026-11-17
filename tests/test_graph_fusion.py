"""Unit tests for RUTAMGraphFusion configuration, initialization, and forward pass."""

from __future__ import annotations

import pytest
import torch

from rutam.config import GraphFusionConfig, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.fusion_output import FusionOutput


def test_graph_fusion_config_defaults() -> None:
    config = GraphFusionConfig()
    assert config.enabled is True
    assert config.hidden_dim == 256
    assert config.graph_dim == 128
    assert config.num_attention_heads == 4
    assert config.attention_dropout == 0.0
    assert config.projection_dropout == 0.1
    assert config.fusion_mode == "cross_attention"
    assert config.gate_mode == "token"
    assert config.residual_scale_init == 0.001
    assert config.use_reasoning_context is True
    assert config.preserve_empty_identity is True
    assert config.return_attention is False


def test_graph_fusion_config_validation() -> None:
    with pytest.raises(ValidationError, match="divisible by num_attention_heads"):
        GraphFusionConfig(hidden_dim=250, num_attention_heads=4)

    with pytest.raises(ValidationError, match="attention_dropout must be less than 1"):
        GraphFusionConfig(attention_dropout=1.0)

    with pytest.raises(ValidationError, match="gate_mode"):
        GraphFusionConfig(gate_mode="invalid_mode")

    with pytest.raises(ValidationError, match="fusion_mode"):
        GraphFusionConfig(fusion_mode="invalid_mode")


def test_graph_fusion_forward_basic() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion = RUTAMGraphFusion(config)

    b, t, k = 2, 4, 3
    hidden_states = torch.randn(b, t, 16)
    graph_states = torch.randn(b, k, 8)
    graph_mask = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)

    output = fusion(hidden_states, graph_states, graph_mask=graph_mask, return_attention=True)
    assert isinstance(output, FusionOutput)
    assert output.fused_hidden_states.shape == (b, t, 16)
    assert output.fusion_gate.shape == (b, t, 1)
    assert output.attention_weights is not None
    assert output.attention_weights.shape == (b, 2, t, k)
    assert not torch.isnan(output.fused_hidden_states).any()
    assert "mean_gate" in output.utilization_metrics
    assert "max_gate" in output.utilization_metrics


def test_graph_fusion_sequence_gate_mode() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(
        hidden_dim=16,
        graph_dim=8,
        num_attention_heads=2,
        gate_mode="sequence",
    )
    fusion = RUTAMGraphFusion(config)

    b, t, k = 2, 4, 3
    hidden_states = torch.randn(b, t, 16)
    graph_states = torch.randn(b, k, 8)

    output = fusion(hidden_states, graph_states)
    assert output.fusion_gate.shape == (b, 1, 1)


def test_graph_fusion_empty_graph_identity_preservation() -> None:
    torch.manual_seed(42)
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion = RUTAMGraphFusion(config)

    b, t = 2, 3
    hidden_states = torch.randn(b, t, 16)
    empty_graph_states = torch.zeros((b, 0, 8), dtype=torch.float32)

    output = fusion(hidden_states, empty_graph_states)
    assert torch.allclose(output.fused_hidden_states, hidden_states)
    assert not torch.isnan(output.fused_hidden_states).any()
    assert output.fusion_gate.abs().sum() == 0.0


def test_graph_fusion_disabled() -> None:
    config = GraphFusionConfig(hidden_dim=16, graph_dim=8, enabled=False)
    fusion = RUTAMGraphFusion(config)

    hidden_states = torch.randn(2, 3, 16)
    graph_states = torch.randn(2, 2, 8)

    output = fusion(hidden_states, graph_states)
    assert torch.equal(output.fused_hidden_states, hidden_states)


def test_graph_fusion_with_rutam_config() -> None:
    rutam_config = RUTAMConfig().with_overrides(
        {"graph_fusion": {"hidden_dim": 32, "graph_dim": 16, "num_attention_heads": 4}}
    )
    fusion = RUTAMGraphFusion(rutam_config)
    assert fusion.config.hidden_dim == 32
    assert fusion.config.graph_dim == 16
