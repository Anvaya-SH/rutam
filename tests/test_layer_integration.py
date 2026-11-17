"""Unit tests for layer wrappers, forward hooks, and reversible integration."""

from __future__ import annotations

import torch

from rutam.config import GraphFusionConfig, LFM3AdapterConfig
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.layer_integration import (
    IntegratedLayerWrapper,
    LayerIntegrationHook,
    SyntheticBackbone,
    SyntheticDecoderLayer,
)
from rutam.model.lfm3_adapter import RUTAMLFM3Adapter


def test_synthetic_decoder_layer_forward() -> None:
    torch.manual_seed(42)
    layer = SyntheticDecoderLayer(hidden_dim=16)
    x = torch.randn(2, 4, 16, requires_grad=True)

    out = layer(x)
    assert out.shape == (2, 4, 16)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0.0


def test_integrated_layer_wrapper_without_fusion() -> None:
    layer = SyntheticDecoderLayer(hidden_dim=16)
    wrapper = IntegratedLayerWrapper(layer, fusion=None)

    x = torch.randn(2, 3, 16)
    out = wrapper(x)
    assert out.shape == (2, 3, 16)
    assert wrapper.last_fusion_output is None


def test_integrated_layer_wrapper_with_fusion() -> None:
    torch.manual_seed(42)
    layer = SyntheticDecoderLayer(hidden_dim=16)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion = RUTAMGraphFusion(fusion_cfg)
    wrapper = IntegratedLayerWrapper(layer, fusion=fusion)

    x = torch.randn(2, 3, 16)
    graph_states = torch.randn(2, 2, 8)

    out = wrapper(x, graph_states=graph_states)
    assert out.shape == (2, 3, 16)
    assert wrapper.last_fusion_output is not None


def test_layer_integration_hook_lifecycle() -> None:
    layer = SyntheticDecoderLayer(hidden_dim=16)
    called = []

    def hook_fn(mod: object, inp: object, outp: object) -> None:
        called.append(True)

    hook = LayerIntegrationHook(layer, hook_fn)
    x = torch.randn(1, 2, 16)
    layer(x)
    assert len(called) == 1

    # Remove hook
    hook.remove()
    layer(x)
    assert len(called) == 1  # Not called again


def test_adapter_hook_mode_lifecycle() -> None:
    torch.manual_seed(42)
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=4)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)

    adapter_cfg = LFM3AdapterConfig(fusion_locations=(1,), integration_mode="hook")
    adapter = RUTAMLFM3Adapter(adapter_cfg, backbone=backbone, fusion_modules=fusion_mod)

    b, t, k = 1, 3, 2
    input_ids = torch.randint(0, 50, (b, t), dtype=torch.long)
    graph_states = torch.randn(b, k, 8)

    out = adapter(input_ids, graph_states=graph_states)
    assert len(out.fusion_outputs) == 1

    # Detach hooks
    adapter.detach_hooks()
    out_detached = adapter(input_ids, graph_states=graph_states)
    assert len(out_detached.fusion_outputs) == 0


def test_adapter_explicit_unwrapping() -> None:
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=4)
    fusion_cfg = GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)
    fusion_mod = RUTAMGraphFusion(fusion_cfg)

    adapter_cfg = LFM3AdapterConfig(fusion_locations=(1,), integration_mode="explicit")
    adapter = RUTAMLFM3Adapter(adapter_cfg, backbone=backbone, fusion_modules=fusion_mod)

    # Layer 1 is wrapped
    assert isinstance(adapter.backbone.layers[1], IntegratedLayerWrapper)

    # Unwrap layers
    adapter.unwrap_layers()
    assert isinstance(adapter.backbone.layers[1], SyntheticDecoderLayer)
