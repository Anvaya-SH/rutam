"""Tests for differentiable neural memory read pipeline, gradient flow, and traversal."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.memory.controller import RUTAMMemoryController
from rutam.memory.controller_output import MemoryReadOutput
from rutam.memory.memory_state import MemoryState


def _make_config() -> RUTAMConfig:
    return RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 8},
            "selector": {"query_dim": 8, "hidden_dim": 8, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 8,
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
        }
    )


def test_read_memory_differentiability() -> None:
    torch.manual_seed(42)
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Node 1", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    graph.create_node("entity", "Node 2", node_id=2, embedding=[0.2] * 8, provenance="test:p")
    graph.add_relationship(1, 2, "LINKS_TO", provenance="test:p", weight=0.8)

    controller = RUTAMMemoryController(config, graph=graph)
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    read_out = controller.read_memory(hidden_states)
    assert isinstance(read_out, MemoryReadOutput)
    assert read_out.gated_memory_context is not None
    assert read_out.gated_memory_context.shape == (2, 8)
    assert not torch.isnan(read_out.gated_memory_context).any()

    # Verify gradient backprop through the read pipeline
    loss = read_out.gated_memory_context.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.shape == (2, 4, 8)
    assert hidden_states.grad.abs().sum() > 0.0


def test_read_memory_empty_graph() -> None:
    torch.manual_seed(42)
    config = _make_config()
    graph = Graph(config)
    controller = RUTAMMemoryController(config, graph=graph)
    assert controller.state == MemoryState.EMPTY

    hidden_states = torch.randn(2, 3, 8, requires_grad=True)
    read_out = controller.read_memory(hidden_states)

    assert read_out.gated_memory_context is not None
    assert read_out.gated_memory_context.shape == (2, 8)
    assert not torch.isnan(read_out.gated_memory_context).any()

    # Empty graph read must be differentiable without crashing or NaNs
    loss = read_out.gated_memory_context.sum()
    loss.backward()
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()


def test_read_memory_with_attention_mask() -> None:
    torch.manual_seed(42)
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Node 1", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    controller = RUTAMMemoryController(config, graph=graph)

    hidden_states = torch.randn(2, 4, 8, requires_grad=True)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    read_out = controller.read_memory(hidden_states, attention_mask=attention_mask)
    assert read_out.gated_memory_context.shape == (2, 8)

    loss = read_out.gated_memory_context.sum()
    loss.backward()
    assert hidden_states.grad is not None


def test_read_memory_with_memory_availability() -> None:
    torch.manual_seed(42)
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Node 1", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    controller = RUTAMMemoryController(config, graph=graph)

    hidden_states = torch.randn(2, 3, 8)
    memory_available = torch.tensor([False, True], dtype=torch.bool)

    read_out = controller.read_memory(hidden_states, memory_available=memory_available)
    assert read_out.gated_memory_context.shape == (2, 8)
    # Batch item 0 was gated off (memory_available = False)
    assert torch.allclose(
        read_out.gated_memory_context[0],
        torch.zeros(8),
        atol=1e-5,
    )


def test_read_memory_with_traversal_enabled() -> None:
    torch.manual_seed(42)
    config = _make_config()
    graph = Graph(config)
    graph.create_node("entity", "Node 1", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    graph.create_node("entity", "Node 2", node_id=2, embedding=[0.2] * 8, provenance="test:p")
    graph.add_relationship(1, 2, "RELATED_TO", provenance="test:p", weight=0.9)

    controller = RUTAMMemoryController(config, graph=graph, traversal_enabled=True)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    read_out = controller.read_memory(hidden_states)
    assert read_out.traversal_output is not None
    assert read_out.gated_memory_context.shape == (2, 8)

    loss = read_out.gated_memory_context.sum()
    loss.backward()
    assert hidden_states.grad is not None


def test_read_memory_soft_selection_gradients() -> None:
    torch.manual_seed(42)
    config = _make_config().with_overrides({"selector": {"selection_mode": "soft"}})
    graph = Graph(config)
    graph.create_node("entity", "Node 1", node_id=1, embedding=[0.5] * 8, provenance="test:p")
    graph.create_node("entity", "Node 2", node_id=2, embedding=[-0.5] * 8, provenance="test:p")

    controller = RUTAMMemoryController(config, graph=graph)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    read_out = controller.read_memory(hidden_states)
    assert read_out.gated_memory_context.shape == (2, 8)

    loss = read_out.gated_memory_context.sum()
    loss.backward()
    assert hidden_states.grad is not None
    assert hidden_states.grad.abs().sum() > 0.0
