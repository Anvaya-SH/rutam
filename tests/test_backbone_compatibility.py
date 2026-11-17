"""Unit tests for parameter freezing, selective unfreezing, and backbone compatibility."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from rutam.config import GraphFusionConfig, LFM3AdapterConfig
from rutam.model.architecture_inspection import inspect_backbone
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.lfm3_adapter import RUTAMLFM3Adapter


def test_parameter_freezing_and_unfreezing() -> None:
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=3)
    adapter = RUTAMLFM3Adapter(backbone=backbone)

    # Initial state: backbone is frozen by default
    for p in adapter.backbone.parameters():
        assert not p.requires_grad

    # Unfreeze backbone
    adapter.unfreeze_backbone()
    for p in adapter.backbone.parameters():
        assert p.requires_grad

    # Freeze backbone again
    adapter.freeze_backbone()
    for p in adapter.backbone.parameters():
        assert not p.requires_grad


def test_selective_layer_unfreezing() -> None:
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=4)
    adapter = RUTAMLFM3Adapter(backbone=backbone)
    adapter.freeze_backbone()

    # Unfreeze only layer 1
    adapter.unfreeze_layers([1])

    for i, layer in enumerate(adapter.backbone.layers):
        for p in layer.parameters():
            if i == 1:
                assert p.requires_grad
            else:
                assert not p.requires_grad


def test_graph_modules_trainable_switch() -> None:
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)
    adapter_cfg = LFM3AdapterConfig(fusion_locations=(0,))
    adapter = RUTAMLFM3Adapter(adapter_cfg, fusion_modules=fusion_mod)

    # Disable fusion training
    adapter.set_graph_modules_trainable(False)
    for p in adapter.fusion_modules.parameters():
        assert not p.requires_grad

    # Enable fusion training
    adapter.set_graph_modules_trainable(True)
    for p in adapter.fusion_modules.parameters():
        assert p.requires_grad


def test_parameter_groups_differentiated_lr() -> None:
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)

    adapter_cfg = LFM3AdapterConfig(fusion_locations=(0,), backbone_frozen=False)
    adapter = RUTAMLFM3Adapter(adapter_cfg, backbone=backbone, fusion_modules=fusion_mod)

    groups = adapter.get_parameter_groups(backbone_lr=2e-5, fusion_lr=1e-3)
    assert len(groups) == 2

    bb_group = next(g for g in groups if g["name"] == "backbone")
    fus_group = next(g for g in groups if g["name"] == "fusion")

    assert bb_group["lr"] == 2e-5
    assert fus_group["lr"] == 1e-3


def test_generic_backbone_attribute_inspection() -> None:
    class MockGenericBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=64, vocab_size=1000)
            self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])

    mock = MockGenericBackbone()
    report = inspect_backbone(mock)

    assert report.hidden_dim == 64
    assert report.vocab_size == 1000
    assert report.num_layers == 3
    assert report.verified_integration is True
    assert len(report.discovered_layers) == 3
