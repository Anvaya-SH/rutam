"""Integration tests for memory write head with RUTAMConfig, graph encoders, and selector."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import Adam

from rutam import (
    EncoderConfig,
    Graph,
    MemoryWriteConfig,
    Node,
    QueryHeadConfig,
    RUTAMConfig,
    RUTAMGraphEncoder,
    RUTAMGraphSelector,
    RUTAMMemoryWriteHead,
    RUTAMNeuralQueryHead,
    SelectorConfig,
)
from rutam.exceptions import ValidationError


def test_config_yaml_serialization_with_memory_write(tmp_path: Path) -> None:
    config = RUTAMConfig()
    assert config.memory_write.enabled
    assert config.memory_write.hidden_dim == 128
    assert config.memory_write.projection_dim == 256
    assert config.memory_write.max_proposals == 4
    assert config.memory_write.write_threshold == 0.70
    assert config.memory_write.operation_count == 7
    assert config.memory_write.commit_automatically is False

    # Roundtrip YAML serialization
    yaml_str = config.to_yaml()
    assert "memory_write:" in yaml_str
    assert "commit_automatically: false" in yaml_str

    path = tmp_path / "test_config.yaml"
    path.write_text(yaml_str, encoding="utf-8")
    loaded = RUTAMConfig.from_yaml(path)
    assert loaded == config

    # Nested override
    overridden = config.with_overrides(
        {"memory_write": {"write_threshold": 0.85, "max_proposals": 2}}
    )
    assert overridden.memory_write.write_threshold == 0.85
    assert overridden.memory_write.max_proposals == 2

    # Invariant: commit_automatically must remain false
    with pytest.raises(ValidationError):
        RUTAMConfig().with_overrides({"memory_write": {"commit_automatically": True}})


def test_end_to_end_read_and_propose_write_pipeline() -> None:
    # 1. Setup small graph
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}}))
    graph.add_node(
        Node(
            id=10,
            node_type="concept",
            content="Memory A",
            embedding=(0.1,) * 8,
            provenance="doc_1",
        )
    )
    graph.add_node(
        Node(
            id=20,
            node_type="concept",
            content="Memory B",
            embedding=(0.2,) * 8,
            provenance="doc_2",
        )
    )
    state = graph.to_state()

    # 2. Setup encoder and selector
    encoder = RUTAMGraphEncoder(EncoderConfig(input_dim=8, hidden_dim=8))
    encoded = encoder(state)

    selector = RUTAMGraphSelector(
        SelectorConfig(
            query_dim=8,
            hidden_dim=8,
            score_dim=8,
            top_k=2,
            selection_mode="hard",
        )
    )

    # 3. Setup query head to query graph
    query_head = RUTAMNeuralQueryHead(
        QueryHeadConfig(
            backbone_hidden_dim=16,
            graph_hidden_dim=8,
            query_dim=8,
            intermediate_dim=16,
        )
    )

    # 4. Setup memory write head
    write_head = RUTAMMemoryWriteHead(
        MemoryWriteConfig(
            backbone_hidden_dim=16,
            hidden_dim=8,
            projection_dim=16,
            max_proposals=2,
            target_selection=True,
            commit_automatically=False,
        )
    )
    write_head.eval()

    # Forward query and graph selection
    b, t, d = 2, 4, 16
    hidden_states = torch.randn(b, t, d)
    query_out = query_head.select_graph(
        hidden_states=hidden_states,
        selector=selector,
        node_states=encoded,
        node_ids=state.node_ids,
        graph_summary=encoded.graph_summary,
    )

    selected_states = query_out.selection.selected_states  # [B, K, H]
    selected_ids = query_out.selection.selected_node_ids  # [B, K]

    # Forward write head
    write_out = write_head(
        hidden_states=hidden_states,
        graph_summary=encoded.graph_summary,
        selected_memory_states=selected_states,
        selected_memory_ids=selected_ids,
        source_graph_version=graph.version,
    )

    # Verification: proposal generation only, no persistent mutation of the graph
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0
    assert write_out.write_decisions is not None
    assert write_out.proposed_embeddings.shape == (b, 2, 8)
    assert write_out.target_memory_scores is not None
    assert write_out.target_memory_scores.shape == (b, 2)

    # Check proposals structure
    for sample_proposals in write_out.proposals:
        for p in sample_proposals:
            assert p.source_graph_version == graph.version
            assert p.origin_category == "neural_write_head"
            assert p.validation_status == "pending"


def test_end_to_end_optimization_step() -> None:
    config = MemoryWriteConfig(
        backbone_hidden_dim=12,
        hidden_dim=8,
        projection_dim=16,
        max_proposals=2,
        target_selection=True,
    )
    write_head = RUTAMMemoryWriteHead(config)
    optimizer = Adam(write_head.parameters(), lr=1e-3)

    b, t, d, h, k = 2, 3, 12, 8, 3
    hidden = torch.randn(b, t, d)
    selected_states = torch.randn(b, k, h)

    # Record initial weights
    w_initial = write_head.write_gate.linear1.weight.clone()

    optimizer.zero_grad()
    out = write_head(hidden, selected_memory_states=selected_states)
    loss = write_head.loss(
        out,
        write_targets=torch.tensor([1.0, 1.0]),
        operation_targets=torch.tensor([1, 2], dtype=torch.int64),
        target_embeddings=torch.randn(b, h),
        target_node_labels=torch.tensor([0, 1], dtype=torch.int64),
    )

    loss.total.backward()
    optimizer.step()

    # Check weights were updated
    assert not torch.equal(write_head.write_gate.linear1.weight, w_initial)
