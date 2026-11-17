"""Tests for backpropagation, gradient flow, and parameter group optimization."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def _create_trainable_model() -> RUTAMForCausalLM:
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
            "traversal": {
                "hidden_dim": 16,
                "path_hidden_dim": 16,
                "query_dim": 8,
                "num_relation_types": 4,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
            },
            "rutam_model": {
                "backbone_frozen": True,
            },
        }
    )
    graph = Graph(config)
    gen = torch.Generator().manual_seed(10)
    for i in range(4):
        graph.create_node(
            node_type="concept",
            content=f"node_{i}",
            node_id=(i + 1) * 10,
            provenance="test:provenance",
            embedding=torch.randn(8, generator=gen).tolist(),
        )
    graph.add_relationship(10, 20, "rel_0", provenance="test:edge")
    graph.add_relationship(20, 30, "rel_1", provenance="test:edge")

    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=3)
    return RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)


def test_frozen_backbone_gradients() -> None:
    """Verify frozen backbone has no gradients while graph modules receive valid gradients."""
    torch.manual_seed(42)
    model = _create_trainable_model()
    assert model.is_backbone_frozen() is True

    input_ids = torch.randint(0, 50, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 50, (2, 8), dtype=torch.long)

    out = model(input_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()

    # Backbone parameters must have no gradients
    for p in model.backbone.parameters():
        assert p.grad is None

    # Graph fusion parameters must have valid gradients without NaNs
    fusion_grads = [p.grad for p in model.fusion.parameters() if p.requires_grad]
    assert len(fusion_grads) > 0
    for grad in fusion_grads:
        if grad is not None:
            assert not torch.isnan(grad).any()
            assert not torch.isinf(grad).any()


def test_unfrozen_backbone_gradients() -> None:
    """Verify unfrozen backbone receives valid gradients during full fine-tuning."""
    torch.manual_seed(42)
    model = _create_trainable_model()
    model.unfreeze_backbone()
    assert model.is_backbone_frozen() is False

    input_ids = torch.randint(0, 50, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 50, (2, 8), dtype=torch.long)

    out = model(input_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()

    # Backbone parameters must now have gradients
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
    assert len(backbone_grads) > 0
    for grad in backbone_grads:
        if grad is not None:
            assert not torch.isnan(grad).any()


def test_optimizer_step_with_parameter_groups() -> None:
    """Verify optimizer step correctly updates trainable parameters."""
    torch.manual_seed(42)
    model = _create_trainable_model()

    groups = model.get_parameter_groups(include_frozen=False)
    optimizer = torch.optim.AdamW(groups)

    # Monitor a fusion parameter
    target_param = next(p for p in model.fusion.parameters() if p.requires_grad)
    initial_val = target_param.detach().clone()

    input_ids = torch.randint(0, 50, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 50, (2, 8), dtype=torch.long)

    out = model(input_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    optimizer.step()

    # Verify parameter updated
    assert not torch.allclose(initial_val, target_param.detach())
