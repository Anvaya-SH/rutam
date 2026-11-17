"""Unit tests for RUTAMForCausalLM initialization, parameter management, and input validation."""

from __future__ import annotations

import pytest
import torch

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.model.rutam_outputs import RUTAMCausalLMOutput


def test_model_default_init() -> None:
    """Verify clean instantiation with default configuration and synthetic backbone."""
    torch.manual_seed(42)
    model = RUTAMForCausalLM()
    assert isinstance(model, torch.nn.Module)
    assert model.backbone is not None
    assert model.encoder is not None
    assert model.selector is not None
    assert model.query_head is not None
    assert model.fusion is not None
    assert model.write_head is not None
    assert model.validator is not None
    assert model.bridge is not None
    assert model.memory_controller is not None
    assert model.is_backbone_frozen() is True


def test_model_backbone_freezing() -> None:
    """Verify backbone freezing, unfreezing, and state query API."""
    torch.manual_seed(42)
    model = RUTAMForCausalLM()
    assert model.is_backbone_frozen() is True

    model.unfreeze_backbone()
    assert model.is_backbone_frozen() is False
    assert any(p.requires_grad for p in model.backbone.parameters())

    model.freeze_backbone()
    assert model.is_backbone_frozen() is True
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_parameter_groups_and_counts() -> None:
    """Verify optimizer parameter partitioning and parameter count reporting."""
    torch.manual_seed(42)
    model = RUTAMForCausalLM()

    # Parameter groups without frozen backbone
    groups = model.get_parameter_groups(include_frozen=False)
    assert len(groups) > 0
    group_names = [g["name"] for g in groups]
    assert any("fusion" in name for name in group_names)
    assert any("encoder" in name for name in group_names)

    # Parameter groups including frozen backbone
    groups_all = model.get_parameter_groups(include_frozen=True)
    assert len(groups_all) > len(groups)
    assert any("backbone" in g["name"] for g in groups_all)

    # Parameter counts
    counts = model.count_parameters()
    assert "backbone" in counts
    assert "fusion" in counts
    assert "encoder" in counts
    assert counts["backbone"]["frozen"] > 0
    assert counts["fusion"]["trainable"] > 0


def test_custom_config_and_components() -> None:
    """Verify custom backbone, graph, and configuration are properly registered."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 12, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 24,
                "graph_hidden_dim": 16,
                "query_dim": 12,
                "intermediate_dim": 16,
            },
            "traversal": {
                "hidden_dim": 16,
                "path_hidden_dim": 16,
                "query_dim": 12,
                "num_relation_types": 4,
            },
            "graph_fusion": {"hidden_dim": 24, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {
                "backbone_hidden_dim": 24,
                "hidden_dim": 16,
                "projection_dim": 16,
            },
            "rutam_model": {"backbone_frozen": False},
        }
    )
    backbone = SyntheticBackbone(hidden_dim=24, vocab_size=120, num_layers=2)
    graph = Graph(config)

    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)
    assert model.config.graph.embedding_dim == 8
    assert model.is_backbone_frozen() is False
    assert model.graph is graph


def test_invalid_inputs() -> None:
    """Verify proper validation errors on malformed inputs."""
    model = RUTAMForCausalLM()

    # Float input_ids
    with pytest.raises(ValidationError, match="input_ids must be a rank-2 CPU torch.long tensor"):
        model(torch.randn(2, 4))

    # String input
    with pytest.raises(ValidationError, match="input_ids must be a rank-2 CPU torch.long tensor"):
        model("invalid")  # type: ignore[arg-type]

    # Invalid labels dtype
    with pytest.raises(ValidationError, match="labels must be a torch.long tensor"):
        model(torch.ones(2, 4, dtype=torch.long), labels=torch.ones(2, 4, dtype=torch.float32))


def test_forward_with_and_without_labels() -> None:
    """Verify standard causal forward pass with and without loss calculation."""
    torch.manual_seed(42)
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
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
            },
        }
    )
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    model = RUTAMForCausalLM(config=config, backbone=backbone)

    input_ids = torch.randint(0, 50, (2, 8), dtype=torch.long)

    # Without labels
    out_no_loss = model(input_ids)
    assert isinstance(out_no_loss, RUTAMCausalLMOutput)
    assert out_no_loss.loss is None
    assert out_no_loss.logits.shape == (2, 8, 50)
    assert not torch.isnan(out_no_loss.logits).any()

    # With labels
    labels = torch.randint(0, 50, (2, 8), dtype=torch.long)
    out_with_loss = model(input_ids, labels=labels)
    assert out_with_loss.loss is not None
    assert out_with_loss.loss.ndim == 0
    assert out_with_loss.loss.item() > 0.0
