"""Tests for backbone protocol compliance, LFM3 adapter compatibility, and HF style interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.backbone_protocol import BackboneProtocol
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.lfm3_adapter import RUTAMLFM3Adapter
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.model.rutam_outputs import RUTAMCausalLMOutput


class CustomDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> torch.Tensor:
        return self.linear(x)


class CustomBackbone(nn.Module):
    """Custom backbone implementing BackboneProtocol via duck typing."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 40, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.dec_layers = nn.ModuleList([CustomDecoderLayer(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, vocab_size)

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_num_layers(self) -> int:
        return self.num_layers

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def get_layers(self) -> Sequence[nn.Module]:
        return tuple(self.dec_layers)

    def get_output_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        x = self.embed(input_ids)
        for layer in self.dec_layers:
            x = layer(x)
        logits = self.head(x)
        return {"last_hidden_state": x, "logits": logits}


def test_backbone_protocol_duck_typing() -> None:
    """Verify arbitrary custom backbones adhering to BackboneProtocol work in RUTAMForCausalLM."""
    assert issubclass(CustomBackbone, BackboneProtocol)
    custom_bb = CustomBackbone(hidden_dim=16, vocab_size=40, num_layers=2)

    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {"backbone_hidden_dim": 16, "hidden_dim": 16, "projection_dim": 16},
        }
    )
    model = RUTAMForCausalLM(config=config, backbone=custom_bb)
    input_ids = torch.randint(0, 40, (1, 5), dtype=torch.long)
    out = model(input_ids)

    assert isinstance(out, RUTAMCausalLMOutput)
    assert out.logits.shape == (1, 5, 40)


def test_lfm3_adapter_integration() -> None:
    """Verify RUTAMLFM3Adapter integration with RUTAMForCausalLM."""
    raw_backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=3)
    adapter = RUTAMLFM3Adapter(backbone=raw_backbone)

    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {"backbone_hidden_dim": 16, "hidden_dim": 16, "projection_dim": 16},
        }
    )
    model = RUTAMForCausalLM(config=config, backbone=adapter.backbone)
    input_ids = torch.randint(0, 50, (2, 4), dtype=torch.long)
    labels = torch.randint(0, 50, (2, 4), dtype=torch.long)
    out = model(input_ids, labels=labels)

    assert out.loss is not None
    assert out.logits.shape == (2, 4, 50)


def test_huggingface_causal_lm_call_signature() -> None:
    """Verify Hugging Face CausalLM signature compatibility: input_ids, labels, attention_mask."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {"backbone_hidden_dim": 16, "hidden_dim": 16, "projection_dim": 16},
        }
    )
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    model = RUTAMForCausalLM(config=config, backbone=backbone)

    input_ids = torch.randint(0, 50, (2, 6), dtype=torch.long)
    labels = torch.randint(0, 50, (2, 6), dtype=torch.long)
    mask = torch.ones(2, 6, dtype=torch.bool)

    out = model(input_ids=input_ids, labels=labels, attention_mask=mask)
    assert hasattr(out, "logits")
    assert hasattr(out, "loss")
    assert hasattr(out, "hidden_states")
    assert out.loss is not None


def test_rutam_output_structure() -> None:
    """Verify immutability and tensor validations in RUTAMCausalLMOutput."""
    logits = torch.randn(2, 4, 10, dtype=torch.float32)
    out = RUTAMCausalLMOutput(logits=logits)

    with pytest.raises(FrozenInstanceError):
        out.logits = torch.randn(2, 4, 10)  # type: ignore[misc]

    # Invalid non-CPU logits
    with pytest.raises(ValidationError, match="logits must be a dense CPU torch.float32 tensor"):
        RUTAMCausalLMOutput(logits=torch.randint(0, 10, (2, 4, 10), dtype=torch.long))  # type: ignore[arg-type]
