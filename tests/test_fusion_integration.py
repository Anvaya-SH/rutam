"""End-to-end integration tests for RUTAMGraphFusion with RUTAM pipeline and config roundtrip."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.memory.controller import RUTAMMemoryController
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.fusion_output import FusionOutput


def _make_config() -> RUTAMConfig:
    return RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 8},
            "selector": {
                "query_dim": 8,
                "hidden_dim": 8,
                "score_dim": 8,
                "top_k": 2,
                "selection_mode": "soft",
            },
            "query_head": {
                "backbone_hidden_dim": 16,
                "query_dim": 8,
                "graph_hidden_dim": 8,
                "intermediate_dim": 8,
                "pooling": "mean",
            },
            "traversal": {
                "query_dim": 8,
                "hidden_dim": 8,
                "path_hidden_dim": 8,
                "score_hidden_dim": 8,
                "scoring_chunk_size": 8,
                "max_candidate_paths": 8,
                "max_selected_paths": 2,
            },
            "graph_fusion": {
                "enabled": True,
                "hidden_dim": 16,
                "graph_dim": 8,
                "num_attention_heads": 2,
                "use_reasoning_context": True,
                "residual_scale_init": 0.01,
            },
        }
    )


def test_fusion_integration_with_controller_and_training_step() -> None:
    torch.manual_seed(42)
    config = _make_config()

    # 1. Initialize graph memory
    graph = Graph(config)
    graph.create_node("topic", "Graph Fusion", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    graph.create_node("topic", "Attention", node_id=2, embedding=[0.2] * 8, provenance="test:p")
    graph.add_relationship(1, 2, "CONNECTS_TO", provenance="test:p", weight=0.8)

    controller = RUTAMMemoryController(config, graph=graph, traversal_enabled=True)
    fusion = RUTAMGraphFusion(config.graph_fusion)

    lm_head = nn.Linear(16, 50, bias=False)
    optimizer = torch.optim.Adam(
        list(controller.parameters()) + list(fusion.parameters()) + list(lm_head.parameters()),
        lr=1e-3,
    )

    # 2. LM forward pass simulation
    b, t = 2, 4
    hidden_states = torch.randn(b, t, 16, requires_grad=True)

    # Differentiable read through controller
    read_out = controller.read_memory(hidden_states)

    # Fuse graph memory into LM hidden states
    graph_states = read_out.selected_node_states
    graph_mask = read_out.selection_mask
    reasoning_ctx = (
        read_out.traversal_output.reasoning_context
        if read_out.traversal_output is not None
        else None
    )
    query_gate = read_out.query_output.query_gate if read_out.query_output is not None else None

    fusion_out = fusion(
        hidden_states=hidden_states,
        graph_states=graph_states,
        graph_mask=graph_mask,
        reasoning_context=reasoning_ctx,
        query_gate=query_gate,
        return_attention=True,
    )

    assert isinstance(fusion_out, FusionOutput)
    assert fusion_out.fused_hidden_states.shape == (b, t, 16)

    # Downstream token prediction
    logits = lm_head(fusion_out.fused_hidden_states)
    target_tokens = torch.randint(0, 50, (b, t))
    loss = nn.functional.cross_entropy(logits.view(-1, 50), target_tokens.view(-1))

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss).any()
    assert fusion.residual_scale.grad is not None


def test_fusion_config_roundtrip(tmp_path: Path) -> None:
    config = RUTAMConfig()
    assert config.graph_fusion.enabled
    assert config.graph_fusion.hidden_dim == 256
    assert config.graph_fusion.num_attention_heads == 4

    yaml_text = config.to_yaml()
    assert "graph_fusion:" in yaml_text
    assert "num_attention_heads: 4" in yaml_text

    cfg_file = tmp_path / "fusion_config.yaml"
    cfg_file.write_text(yaml_text, encoding="utf-8")

    loaded = RUTAMConfig.from_yaml(cfg_file)
    assert loaded == config

    overridden = config.with_overrides(
        {"graph_fusion": {"num_attention_heads": 8, "gate_mode": "sequence"}}
    )
    assert overridden.graph_fusion.num_attention_heads == 8
    assert overridden.graph_fusion.gate_mode == "sequence"
