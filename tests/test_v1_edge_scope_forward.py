"""Edge-scope boundaries for model encoding and the graph encoding cache."""

from dataclasses import replace
from unittest.mock import patch

import torch

from rutam import Edge, Graph, Node, RUTAMConfig
from rutam.memory.provenance import AuthorizationContext
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def _model_with_edge(*, sparse: bool, edge_scope: str) -> RUTAMForCausalLM:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {
                "input_dim": 8,
                "hidden_dim": 16,
                "num_relation_types": 1,
                "dropout": 0,
            },
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
                "query_gate_enabled": False,
            },
            "traversal": {
                "hidden_dim": 16,
                "path_hidden_dim": 16,
                "query_dim": 8,
                "num_relation_types": 1,
                "max_visited_nodes": 8,
                "max_candidate_paths": 8,
                "max_selected_paths": 2,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
            },
            "rutam_model": {
                "sparse_selection_enabled": sparse,
                "max_sparse_nodes": 4,
                "sparse_max_hops": 2,
            },
        }
    )
    graph = Graph(config)
    for node_id, coordinate in ((1, 0), (2, 1)):
        embedding = torch.zeros(config.graph.embedding_dim)
        embedding[coordinate] = 1
        graph.add_node(
            Node(
                id=node_id,
                node_type="entity",
                content=f"node-{node_id}",
                provenance="fixture",
                embedding=tuple(embedding.tolist()),
                scope="A" if sparse else "default",
            )
        )
    graph.add_edge(
        Edge(
            source_id=1,
            target_id=2,
            relationship="related",
            provenance="fixture",
            scope=edge_scope,
        )
    )
    return RUTAMForCausalLM(
        config=config,
        backbone=SyntheticBackbone(hidden_dim=16, vocab_size=60, num_layers=2),
        graph=graph,
    ).eval()


def test_forward_excludes_foreign_scope_edge_before_sparse_encoding_and_traversal():
    model = _model_with_edge(sparse=True, edge_scope="B")
    state = model.bridge.get_active_state()
    no_edge_state, _ = state.extract_subgraph(
        torch.arange(state.node_ids.shape[0], dtype=torch.int64),
        edge_mask=torch.zeros(state.edge_types.shape, dtype=torch.bool),
    )
    allowed_edge_state = replace(
        state,
        edge_metadata=[{**state.edge_metadata[0], "scope": "A"}],
    )
    authorization = AuthorizationContext(
        user_id="alice",
        authorized_scopes=("A",),
        permissions=frozenset({"read"}),
    )
    encoded_edges = []
    original_forward = model.encoder.forward

    def capture_edges(sub_state, *args, **kwargs):
        encoded_edges.append(tuple(record["scope"] for record in sub_state.edge_metadata))
        return original_forward(sub_state, *args, **kwargs)

    with patch.object(model.encoder, "forward", side_effect=capture_edges):
        with torch.no_grad():
            scoped = model(
                torch.tensor([[1, 2]]),
                graph_state_snapshot=state,
                authorization_context=authorization,
            )
            baseline = model(
                torch.tensor([[1, 2]]),
                graph_state_snapshot=no_edge_state,
                authorization_context=authorization,
            )
            allowed = model(
                torch.tensor([[1, 2]]),
                graph_state_snapshot=allowed_edge_state,
                authorization_context=authorization,
            )

    assert encoded_edges[:4] == [(), (), (), ()]
    assert ("A",) in encoded_edges[4:]
    torch.testing.assert_close(scoped.logits, baseline.logits)
    assert scoped.traversal_output is None
    assert allowed.graph_selection_output is not None


def test_bridge_encoding_cache_separates_edge_visibility_masks():
    model = _model_with_edge(sparse=False, edge_scope="default")
    state = model.bridge.get_active_state()
    node_mask = torch.ones(state.node_ids.shape, dtype=torch.bool)
    hidden_edge = torch.zeros(state.edge_types.shape, dtype=torch.bool)
    visible_edge = torch.ones(state.edge_types.shape, dtype=torch.bool)

    with torch.no_grad():
        hidden_encoding = model.bridge.get_active_encoding(
            model.encoder, state=state, node_mask=node_mask, edge_mask=hidden_edge
        )
        visible_encoding = model.bridge.get_active_encoding(
            model.encoder, state=state, node_mask=node_mask, edge_mask=visible_edge
        )

    assert hidden_encoding is not visible_encoding
    assert model.bridge._cached_encoding_key[-1] == (True,)
    assert not torch.allclose(hidden_encoding.node_states, visible_encoding.node_states)
