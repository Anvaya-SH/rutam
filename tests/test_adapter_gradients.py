"""Unit tests for backward pass, selective gradients, and frozen backbone isolation."""

from __future__ import annotations

import torch

from rutam.config import GraphFusionConfig, LFM3AdapterConfig
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.lfm3_adapter import RUTAMLFM3Adapter


def test_adapter_gradients_frozen_backbone() -> None:
    torch.manual_seed(42)
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=3)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)

    adapter_cfg = LFM3AdapterConfig(
        fusion_locations=(1,),
        backbone_frozen=True,
        graph_modules_trainable=True,
    )
    adapter = RUTAMLFM3Adapter(adapter_cfg, backbone=backbone, fusion_modules=fusion_mod)

    b, t, k = 2, 4, 3
    input_ids = torch.randint(0, 50, (b, t), dtype=torch.long)
    graph_states = torch.randn(b, k, 8, requires_grad=True)
    reasoning_context = torch.randn(b, 8, requires_grad=True)

    output = adapter(
        input_ids,
        graph_states=graph_states,
        reasoning_context=reasoning_context,
    )
    assert output.logits is not None
    loss = output.logits.sum()
    loss.backward()

    # Backbone parameters must remain untouched by gradients
    for p in adapter.backbone.parameters():
        assert p.grad is None

    # Graph inputs and fusion parameters must receive gradients
    assert graph_states.grad is not None
    assert not torch.isnan(graph_states.grad).any()
    assert graph_states.grad.abs().sum() > 0.0

    assert reasoning_context.grad is not None
    assert not torch.isnan(reasoning_context.grad).any()

    for p in adapter.fusion_modules.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()


def test_adapter_gradients_unfrozen_backbone() -> None:
    torch.manual_seed(42)
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)

    adapter_cfg = LFM3AdapterConfig(
        fusion_locations=(0,),
        backbone_frozen=False,
        graph_modules_trainable=True,
    )
    adapter = RUTAMLFM3Adapter(adapter_cfg, backbone=backbone, fusion_modules=fusion_mod)

    input_ids = torch.randint(0, 50, (2, 3), dtype=torch.long)
    graph_states = torch.randn(2, 2, 8, requires_grad=True)
    reasoning_context = torch.randn(2, 8, requires_grad=True)

    output = adapter(
        input_ids,
        graph_states=graph_states,
        reasoning_context=reasoning_context,
    )
    loss = output.logits.sum()  # type: ignore[union-attr]
    loss.backward()

    # Both backbone and fusion parameters must receive gradients
    for p in adapter.backbone.parameters():
        assert p.grad is not None

    for p in adapter.fusion_modules.parameters():
        assert p.grad is not None


def test_adapter_selective_layer_gradient_flow() -> None:
    torch.manual_seed(42)
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=4)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)

    adapter_cfg = LFM3AdapterConfig(fusion_locations=(2,), backbone_frozen=True)
    adapter = RUTAMLFM3Adapter(adapter_cfg, backbone=backbone, fusion_modules=fusion_mod)

    # Unfreeze only layer 1
    adapter.unfreeze_layers([1])

    input_ids = torch.randint(0, 50, (2, 3), dtype=torch.long)
    graph_states = torch.randn(2, 2, 8, requires_grad=True)

    output = adapter(input_ids, graph_states=graph_states)
    loss = output.logits.sum()  # type: ignore[union-attr]
    loss.backward()

    # Layer 0 must have no gradients
    for p in adapter.backbone.layers[0].parameters():
        assert p.grad is None

    # Layer 1 must have gradients
    for p in adapter.backbone.layers[1].parameters():
        assert p.grad is not None
