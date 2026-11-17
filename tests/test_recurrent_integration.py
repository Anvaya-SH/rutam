"""Integration tests verifying RUTAMForCausalLM with recurrent reasoning enabled and disabled."""

from __future__ import annotations

import torch

from rutam.config import RecurrentConfig, RUTAMConfig
from rutam.graph.operations import Graph
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.recurrent_integration import RUTAMRecurrentIntegration
from rutam.model.recurrent_output import RecurrentOutput
from rutam.model.rutam_model import RUTAMForCausalLM


def test_default_config_disabled_baseline_identity() -> None:
    """Requirement 7: Recurrence disabled by default; baseline completely unaffected."""
    config = RUTAMConfig()
    assert config.recurrent.enabled is False

    model = RUTAMForCausalLM(config=config)
    assert model.recurrent.is_enabled is False

    input_ids = torch.randint(0, 100, (2, 4), dtype=torch.long)
    out = model(input_ids)

    # Recurrent output is None when disabled
    assert out.recurrent_output is None


def test_recurrent_integration_passthrough_when_disabled() -> None:
    """Requirement 7: Integration layer is strict identity pass-through when disabled."""
    integration = RUTAMRecurrentIntegration(RecurrentConfig(enabled=False))
    assert integration.is_enabled is False

    x = torch.randn(2, 4, 32)
    out_x, rec_out = integration(x)

    assert out_x is x  # Exact same object
    assert rec_out is None


def test_recurrent_enabled_in_full_model() -> None:
    """Verify forward pass with recurrent reasoning enabled populates RecurrentOutput."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16},
            "encoder": {"input_dim": 16, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 16, "hidden_dim": 16, "score_dim": 16, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 16,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "recurrent": {
                "enabled": True,
                "hidden_dim": 16,
                "num_heads": 2,
                "intermediate_dim": 32,
                "graph_dim": 16,
                "loop_embedding_dim": 16,
                "default_loops": 3,
                "max_loops": 5,
            },
            "rutam_model": {"backbone_frozen": False},
        }
    )
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=100, num_layers=4)
    model = RUTAMForCausalLM(config=config, backbone=backbone)
    assert model.recurrent.is_enabled is True

    input_ids = torch.randint(0, 100, (2, 5), dtype=torch.long)
    out = model(input_ids)

    assert out.recurrent_output is not None
    assert isinstance(out.recurrent_output, RecurrentOutput)
    assert out.recurrent_output.num_loops == 3
    assert out.recurrent_output.output_states.shape == (2, 5, 16)
    assert len(out.recurrent_output.loop_states) == 3


def test_parameter_groups_and_counts_with_recurrent() -> None:
    """Verify parameter group management and parameter counting include recurrent when active."""
    # 1. Disabled model
    config_disabled = RUTAMConfig()
    model_disabled = RUTAMForCausalLM(config=config_disabled)
    counts_dis = model_disabled.count_parameters()
    groups_dis = model_disabled.get_parameter_groups(include_frozen=False)

    assert "recurrent" not in counts_dis
    assert not any("recurrent" in g["name"] for g in groups_dis)

    # 2. Enabled model
    config_enabled = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16},
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "recurrent": {
                "enabled": True,
                "hidden_dim": 16,
                "num_heads": 2,
                "intermediate_dim": 32,
                "graph_dim": 16,
                "loop_embedding_dim": 16,
            },
        }
    )
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    model_enabled = RUTAMForCausalLM(config=config_enabled, backbone=backbone)
    counts_en = model_enabled.count_parameters()
    groups_en = model_enabled.get_parameter_groups(include_frozen=False)

    assert "recurrent" in counts_en
    assert counts_en["recurrent"]["trainable"] > 0
    assert any("recurrent" in g["name"] for g in groups_en)


def test_end_to_end_gradient_flow_from_causal_lm_loss() -> None:
    """Verify loss gradients flow end-to-end through LM head, decoder, and recurrent block."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16},
            "encoder": {"input_dim": 16, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 16, "hidden_dim": 16, "score_dim": 16, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 16,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "recurrent": {
                "enabled": True,
                "hidden_dim": 16,
                "num_heads": 2,
                "intermediate_dim": 32,
                "graph_dim": 16,
                "loop_embedding_dim": 16,
                "default_loops": 2,
                "max_loops": 4,
            },
            "rutam_model": {"backbone_frozen": False},
        }
    )
    from rutam.graph.schema import Node

    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=100, num_layers=4)
    graph = Graph(config)
    graph.add_node(Node(id=0, node_type="entity", content="Node 0", provenance="test"))
    graph.add_node(Node(id=1, node_type="entity", content="Node 1", provenance="test"))

    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)

    input_ids = torch.randint(0, 100, (2, 6), dtype=torch.long)
    labels = torch.randint(0, 100, (2, 6), dtype=torch.long)

    out = model(input_ids, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)

    out.loss.backward()

    # Recurrent block weights receive gradients
    assert model.recurrent.engine is not None
    assert model.recurrent.engine.block.q_proj.weight.grad is not None
    assert torch.isfinite(model.recurrent.engine.block.q_proj.weight.grad).all()
    assert model.recurrent.engine.residual_gate.residual_scale.grad is not None
    assert torch.isfinite(model.recurrent.engine.residual_gate.residual_scale.grad).all()


def test_dynamic_halting_in_full_model() -> None:
    """Verify dynamic loop controller operates seamlessly during full causal model forward pass."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16},
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "recurrent": {
                "enabled": True,
                "hidden_dim": 16,
                "num_heads": 2,
                "intermediate_dim": 32,
                "graph_dim": 16,
                "loop_embedding_dim": 16,
                "dynamic_halting": True,
                "min_loops": 1,
                "max_loops": 4,
                "halting_threshold": 0.5,
                "controller_hidden_dim": 16,
            },
        }
    )
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=60, num_layers=2)
    model = RUTAMForCausalLM(config=config, backbone=backbone)

    input_ids = torch.randint(0, 60, (2, 4), dtype=torch.long)
    out = model(input_ids)

    assert out.recurrent_output is not None
    assert 1 <= out.recurrent_output.num_loops <= 4
    assert out.recurrent_output.continue_logits is not None
    assert out.recurrent_output.halting_probabilities is not None
