"""Unit tests for RUTAMLFM3Adapter initialization, inspection, and forward execution."""

from __future__ import annotations

import pytest
import torch

from rutam.config import GraphFusionConfig, LFM3AdapterConfig, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.adapter_output import AdapterOutput
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.lfm3_adapter import RUTAMLFM3Adapter


def test_adapter_initialization_defaults() -> None:
    adapter = RUTAMLFM3Adapter()
    assert adapter.config.backbone_id == "LiquidAI/LFM3-1.7B-Base"
    assert adapter.config.backbone_frozen is True
    assert adapter.inspection.verified_integration is True
    assert adapter.inspection.num_layers == 4


def test_adapter_forward_baseline_without_graph() -> None:
    torch.manual_seed(42)
    adapter = RUTAMLFM3Adapter()
    input_ids = torch.randint(0, 100, (2, 5), dtype=torch.long)

    output = adapter(input_ids)
    assert isinstance(output, AdapterOutput)
    assert output.hidden_states.shape == (2, 5, 256)
    assert output.logits is not None
    assert output.logits.shape == (2, 5, 100)
    assert len(output.fusion_outputs) == 0


def test_adapter_forward_with_graph_fusion() -> None:
    torch.manual_seed(42)
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=4)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)

    adapter_cfg = LFM3AdapterConfig(fusion_locations=(1, 3))
    adapter = RUTAMLFM3Adapter(
        adapter_cfg,
        backbone=backbone,
        fusion_modules=fusion_mod,
    )

    b, t, k = 2, 4, 3
    input_ids = torch.randint(0, 50, (b, t), dtype=torch.long)
    graph_states = torch.randn(b, k, 8)
    graph_mask = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)

    output = adapter(input_ids, graph_states=graph_states, graph_mask=graph_mask)
    assert isinstance(output, AdapterOutput)
    assert output.hidden_states.shape == (b, t, 16)
    assert output.logits.shape == (b, t, 50)
    assert len(output.fusion_outputs) == 2
    assert output.fused_hidden_states is not None
    assert output.fused_hidden_states.shape == (b, t, 16)


def test_adapter_invalid_fusion_location_raises() -> None:
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    adapter_cfg = LFM3AdapterConfig(fusion_locations=(5,))  # Layer 5 does not exist

    with pytest.raises(ValidationError, match="exceeds backbone layer count"):
        RUTAMLFM3Adapter(adapter_cfg, backbone=backbone)


def test_adapter_with_rutam_config() -> None:
    rutam_config = RUTAMConfig().with_overrides(
        {
            "lfm3_adapter": {"backbone_id": "CustomTestModel", "fusion_locations": (0,)},
            "graph_fusion": {"hidden_dim": 256, "num_attention_heads": 4},
        }
    )
    adapter = RUTAMLFM3Adapter(rutam_config)
    assert adapter.config.backbone_id == "CustomTestModel"
    assert adapter.config.fusion_locations == (0,)
