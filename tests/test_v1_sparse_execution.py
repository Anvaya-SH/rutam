"""Tests for bounded sparse-first graph selection and relational subgraph encoding."""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
import torch

from rutam import Graph, GraphState, RUTAMConfig
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM
from test_rutam_forward import _create_small_model_and_graph


def test_sparse_selection_forward_finite_shapes():
    torch.manual_seed(42)
    model, graph = _create_small_model_and_graph(num_nodes=8)
    model.config = model.config.with_overrides({
        "rutam_model": {
            "sparse_selection_enabled": True,
            "max_sparse_nodes": 4,
            "return_memory_diagnostics": True,
        }
    })
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model(ids)

    assert out.logits.shape == (1, 4, 60)
    assert torch.isfinite(out.logits).all()
    assert out.fusion_output is not None
    assert torch.isfinite(out.fusion_output.fused_hidden_states).all()
    assert out.diagnostics.get("sparse_selection_active") is True


def test_sparse_execution_bounds_encoder_workload():
    torch.manual_seed(99)
    # Build a graph with 50 nodes and connected components
    config = RUTAMConfig().with_overrides({
        "graph": {"embedding_dim": 128},
        "encoder": {"input_dim": 128, "hidden_dim": 128, "num_relation_types": 4},
        "selector": {"query_dim": 128, "hidden_dim": 128, "score_dim": 128, "top_k": 2},
        "query_head": {
            "backbone_hidden_dim": 128,
            "graph_hidden_dim": 128,
            "query_dim": 128,
            "intermediate_dim": 128,
        },
        "graph_fusion": {"hidden_dim": 128, "graph_dim": 128, "num_attention_heads": 2},
        "traversal": {
            "hidden_dim": 128,
            "path_hidden_dim": 128,
            "query_dim": 128,
            "num_relation_types": 4,
            "max_visited_nodes": 64,
            "max_candidate_paths": 8,
            "max_selected_paths": 2,
        },
        "rutam_model": {
            "sparse_selection_enabled": True,
            "max_sparse_nodes": 6,
        },
    })
    graph = Graph(config)
    for i in range(50):
        graph.create_node(
            node_type="concept",
            content=f"node_{i}",
            node_id=i,
            provenance="test",
            embedding=torch.randn(128).tolist(),
        )
    for i in range(40):
        graph.add_relationship(i, i + 1, "rel_0", provenance="test")

    backbone = SyntheticBackbone(hidden_dim=128, vocab_size=60, num_layers=2)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph).eval()

    encoded_subgraph_sizes = []
    original_encoder_forward = model.encoder.forward

    def intercepted_encoder_forward(sub_state, *args, **kwargs):
        encoded_subgraph_sizes.append(sub_state.node_ids.shape[0])
        return original_encoder_forward(sub_state, *args, **kwargs)

    with patch.object(model.encoder, "forward", side_effect=intercepted_encoder_forward):
        out = model(torch.tensor([[1, 2, 3]]))

    assert len(encoded_subgraph_sizes) > 0
    # Every relational encoder call must be strictly bounded by max_sparse_nodes (6),
    # even though the graph has 50 nodes.
    assert all(size <= 6 for size in encoded_subgraph_sizes)
    assert max(encoded_subgraph_sizes) <= 6


def test_default_forward_selects_before_encoding_2000_node_graph():
    torch.manual_seed(100)
    config = RUTAMConfig().with_overrides({
        "graph": {"embedding_dim": 8},
        "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 1, "dropout": 0},
        "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 16},
        "query_head": {
            "backbone_hidden_dim": 16,
            "graph_hidden_dim": 16,
            "query_dim": 8,
            "intermediate_dim": 16,
        },
        "traversal": {
            "max_hops": 2,
            "hidden_dim": 16,
            "path_hidden_dim": 16,
            "query_dim": 8,
            "num_relation_types": 1,
            "max_visited_nodes": 32,
            "max_candidate_paths": 8,
            "max_selected_paths": 2,
        },
        "rutam_model": {"max_sparse_nodes": 24, "return_memory_diagnostics": True},
        "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
        "memory_write": {"backbone_hidden_dim": 16, "hidden_dim": 16, "projection_dim": 16},
    })
    assert config.rutam_model.sparse_selection_enabled is True
    graph = Graph(config)
    model = RUTAMForCausalLM(
        config=config,
        backbone=SyntheticBackbone(hidden_dim=16, vocab_size=60, num_layers=2),
        graph=graph,
    ).eval()

    node_count = 2_000
    edge_count = node_count - 1
    state = GraphState(
        node_ids=torch.arange(node_count, dtype=torch.int64),
        node_embeddings=torch.randn(node_count, 8, dtype=torch.float32),
        edge_index=torch.stack((torch.arange(edge_count), torch.arange(1, node_count))),
        edge_types=torch.zeros(edge_count, dtype=torch.int64),
        edge_weights=torch.ones(edge_count, dtype=torch.float32),
        node_metadata=[{} for _ in range(node_count)],
        edge_metadata=[{} for _ in range(edge_count)],
        relationship_types={"rel_0": 0},
        graph_version=1,
        config=config,
        next_node_id=node_count,
        retired_node_ids=(),
        metadata={},
    )

    events = []
    encoded_subgraph_sizes = []
    original_select = model.selector.select_projected
    original_encode = model.encoder.forward

    def intercepted_select(*args, **kwargs):
        events.append("select")
        return original_select(*args, **kwargs)

    def intercepted_encode(sub_state, *args, **kwargs):
        events.append("encode")
        encoded_subgraph_sizes.append(sub_state.node_ids.shape[0])
        return original_encode(sub_state, *args, **kwargs)

    with (
        patch.object(model.selector, "select_projected", side_effect=intercepted_select),
        patch.object(model.encoder, "forward", side_effect=intercepted_encode),
    ):
        out = model(torch.tensor([[1, 2, 3]]), graph_state_snapshot=state)

    assert node_count > config.selector.top_k * 100
    assert events == ["select", "encode"] * 3
    assert len(encoded_subgraph_sizes) == 3
    assert all(size <= config.rutam_model.max_sparse_nodes for size in encoded_subgraph_sizes)
    assert sum(encoded_subgraph_sizes) <= 3 * config.rutam_model.max_sparse_nodes
    assert out.graph_selection_output is not None
    assert int(out.graph_selection_output.selection.selection_mask.sum()) <= config.selector.top_k


def test_sparse_graph_conditioning_is_prefix_causal():
    torch.manual_seed(23)
    model, graph = _create_small_model_and_graph(num_nodes=6)
    model.config = model.config.with_overrides({
        "rutam_model": {
            "sparse_selection_enabled": True,
            "max_sparse_nodes": 4,
        }
    })
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.no_grad():
        full = model(ids).logits
        for end in range(1, 5):
            prefix = model(ids[:, :end]).logits
            torch.testing.assert_close(full[:, :end], prefix, rtol=1e-4, atol=1e-5)


def test_sparse_selection_backward_gradients():
    torch.manual_seed(47)
    model, graph = _create_small_model_and_graph(num_nodes=6)
    model.config = model.config.with_overrides({
        "rutam_model": {
            "sparse_selection_enabled": True,
            "max_sparse_nodes": 4,
        }
    })
    model.train()
    ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[1, 2, 3]])

    model.zero_grad(set_to_none=True)
    out = model(ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()

    # Gradients must reach both node projection and message passing layers
    proj_grads = [p.grad for p in model.encoder.node_projection.parameters() if p.grad is not None]
    layer_grads = [p.grad for p in model.encoder.layers.parameters() if p.grad is not None]
    query_grads = [p.grad for p in model.query_head.parameters() if p.grad is not None]
    selector_grads = [p.grad for p in model.selector.parameters() if p.grad is not None]

    assert len(proj_grads) > 0 and all(torch.isfinite(g).all() for g in proj_grads)
    assert len(layer_grads) > 0 and all(torch.isfinite(g).all() for g in layer_grads)
    assert len(query_grads) > 0 and all(torch.isfinite(g).all() for g in query_grads)
    assert any(g.abs().sum() > 0 for g in layer_grads)
    assert any(g.abs().sum() > 0 for g in query_grads)
    # Top-k ordering and selected scores do not feed the LM loss; scorer params
    # therefore need explicit retrieval-score supervision to receive gradients.
    assert selector_grads == []


def test_edge_validity_filtering_excludes_inactive_edges():
    torch.manual_seed(12)
    model, graph = _create_small_model_and_graph(num_nodes=4)
    # Add an expired edge and a retracted edge
    past = datetime.now(UTC) - timedelta(days=2)
    graph.add_relationship(10, 30, "rel_0", provenance="test", valid_until=past)
    graph.add_relationship(20, 40, "rel_0", provenance="test", status="retracted")

    state = graph.to_state()
    # Check that extract_subgraph with edge_mask excludes the inactive edges
    now = datetime.now(UTC)
    edge_mask = torch.tensor([
        r.get("status", "active") not in ("retracted", "superseded", "inactive")
        and (r.get("valid_until") is None or datetime.fromisoformat(r["valid_until"]) > now)
        for r in state.edge_metadata
    ], dtype=torch.bool)

    sub_state, _ = state.extract_subgraph([0, 1, 2, 3], edge_mask=edge_mask)
    # Active relationships were rel_0 (10->20) and rel_1 (20->30). The 2 expired/retracted ones are excluded.
    assert sub_state.edge_index.shape[1] == 2
