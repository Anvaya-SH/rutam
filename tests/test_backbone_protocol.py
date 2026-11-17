"""Unit tests for BackboneProtocol runtime conformance and interface discovery."""

from __future__ import annotations

import torch
from torch import nn

from rutam.model.backbone_protocol import BackboneProtocol
from rutam.model.layer_integration import SyntheticBackbone


class IncompleteBackbone(nn.Module):
    """Backbone missing required protocol methods."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)


def test_synthetic_backbone_implements_protocol() -> None:
    backbone = SyntheticBackbone(hidden_dim=32, vocab_size=150, num_layers=3)
    assert isinstance(backbone, BackboneProtocol)
    assert backbone.get_hidden_dim() == 32
    assert backbone.get_vocab_size() == 150
    assert backbone.get_num_layers() == 3

    embeddings = backbone.get_input_embeddings()
    assert isinstance(embeddings, nn.Module)

    layers = backbone.get_layers()
    assert len(layers) == 3

    h = torch.randn(2, 4, 32)
    logits = backbone.get_output_logits(h)
    assert logits.shape == (2, 4, 150)


def test_incomplete_backbone_fails_protocol() -> None:
    incomplete = IncompleteBackbone()
    assert not isinstance(incomplete, BackboneProtocol)


def test_synthetic_backbone_forward_pass() -> None:
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    input_ids = torch.randint(0, 50, (2, 6), dtype=torch.long)

    out = backbone(input_ids)
    assert "last_hidden_state" in out
    assert "logits" in out
    assert out["last_hidden_state"].shape == (2, 6, 16)
    assert out["logits"].shape == (2, 6, 50)
