"""GRU order, padding and differentiable node/relation/weight inputs."""

from dataclasses import replace

import pytest
import torch

from rutam import Graph, RUTAMPathEncoder
from rutam.exceptions import ValidationError
from rutam.graph.neighborhood import NeighborhoodCache
from rutam.graph.traversal import enumerate_paths


def test_path_encoding_padding_order_and_gradients(reasoning_graph: Graph) -> None:
    torch.manual_seed(3)
    state = reasoning_graph.to_state()
    c = reasoning_graph.config.traversal
    paths = enumerate_paths(
        NeighborhoodCache().get(state.node_ids, state.edge_index, state.edge_types),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        c,
    )
    nodes = torch.randn((8, 8), requires_grad=True)
    weights = state.edge_weights.clone().requires_grad_()
    encoder = RUTAMPathEncoder(c)
    encoded = encoder(nodes, paths, weights)
    assert encoded.shape == (1, 16, 8)
    assert encoded[~paths.path_mask].count_nonzero() == 0
    changed = paths.relation_types.clone()
    changed[paths.edge_mask] = (changed[paths.edge_mask] + 1) % c.num_relation_types
    assert not torch.allclose(
        encoded, encoder(nodes, replace(paths, relation_types=changed), weights)
    )
    reversed_signs = -paths.directions
    assert not torch.allclose(
        encoded, encoder(nodes, replace(paths, directions=reversed_signs), weights)
    )
    encoded.square().sum().backward()
    assert nodes.grad.count_nonzero() > 0 and weights.grad.count_nonzero() > 0
    assert encoder.relation_embedding.weight.grad.count_nonzero() > 0
    assert encoder.gru.weight_ih.grad.count_nonzero() > 0


def test_padding_positions_do_not_change_short_path(reasoning_graph: Graph) -> None:
    state = reasoning_graph.to_state()
    c = reasoning_graph.config.traversal
    paths = enumerate_paths(
        NeighborhoodCache().get(state.node_ids, state.edge_index, state.edge_types),
        torch.tensor([[7]]),
        torch.tensor([[True]]),
        c,
    )
    encoder = RUTAMPathEncoder(c).eval()
    nodes = torch.ones((8, 8))
    before = encoder(nodes, paths, state.edge_weights)
    with torch.no_grad():
        encoder.hop_embedding.weight[1:].fill_(1000)
    torch.testing.assert_close(before, encoder(nodes, paths, state.edge_weights), rtol=0, atol=0)
    wrong = paths.node_indices.clone()
    wrong[0, 0, 1] = 999
    with pytest.raises(ValidationError):
        encoder(nodes, replace(paths, node_indices=wrong), state.edge_weights)


def test_path_internal_dimension_can_differ_from_graph_hidden(reasoning_graph: Graph) -> None:
    state = reasoning_graph.to_state()
    c = replace(reasoning_graph.config.traversal, path_hidden_dim=5)
    paths = enumerate_paths(
        NeighborhoodCache().get(state.node_ids, state.edge_index, state.edge_types),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        c,
    )
    encoder = RUTAMPathEncoder(c)
    assert encoder(torch.ones((8, 8)), paths, state.edge_weights).shape == (1, 16, 8)
