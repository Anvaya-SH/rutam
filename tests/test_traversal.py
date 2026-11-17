"""Bounded discrete graph paths and stable persistent IDs."""

from dataclasses import replace

import pytest
import torch

from rutam import Graph, RUTAMGraphTraversal, TraversalConfig
from rutam.exceptions import ValidationError
from rutam.graph.neighborhood import NeighborhoodCache


def run(graph: Graph, *, seed: int = 0, config: TraversalConfig | None = None):
    state = graph.to_state()
    settings = config or graph.config.traversal
    module = RUTAMGraphTraversal(settings).eval()
    output = module(
        torch.ones((1, settings.query_dim)),
        torch.ones((len(graph.nodes), settings.hidden_dim)),
        state.edge_index,
        state.edge_types,
        state.edge_weights,
        torch.tensor([[seed]]),
        torch.tensor([[True]]),
        state.node_ids,
        graph_version=state.graph_version,
    )
    return output


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_one_two_three_hops_and_disconnected_components(reasoning_graph: Graph, hops: int) -> None:
    graph = reasoning_graph
    output = run(graph, config=replace(graph.config.traversal, max_hops=hops))
    lengths = output.path_edge_mask.sum(2)[output.path_mask]
    assert lengths.max().item() == hops
    assert (lengths <= hops).all()
    assert output.visited_node_ids[0, 0] == 10
    assert output.visited_mask.sum() <= 8
    graph.create_node("isolated", "disconnected", node_id=999, provenance="test")
    output = run(graph)
    assert 999 not in output.visited_node_ids.tolist()[0]


def test_cycles_dedup_nodes_but_keep_alternative_simple_paths(reasoning_graph: Graph) -> None:
    output = run(reasoning_graph)
    visited = output.visited_node_ids[output.visited_mask].tolist()
    assert len(visited) == len(set(visited))
    paths = [
        tuple(rows[mask].tolist())
        for rows, mask in zip(output.path_node_ids[0], output.path_node_mask[0], strict=True)
        if mask.any()
    ]
    assert (10, 20, 30, 40) in paths
    assert (10, 50, 40) in paths
    assert all(len(path) == len(set(path)) for path in paths)
    assert (10, 20, 30, 10) not in paths


def test_direction_and_relation_order(reasoning_graph: Graph) -> None:
    graph = reasoning_graph
    outgoing = run(graph, seed=7)
    assert outgoing.path_mask.sum() == 1  # Only the seed; no outgoing edges.
    incoming = run(graph, seed=7, config=replace(graph.config.traversal, direction="incoming"))
    assert incoming.visited_node_ids[0, :4].tolist() == [80, 70, 50, 60]
    assert (incoming.path_directions[incoming.path_edge_mask] == -1).all()
    output = run(graph)
    for rows, mask, relations in zip(
        output.path_node_ids[0], output.path_node_mask[0], output.path_relation_ids[0], strict=True
    ):
        if rows[mask].tolist() == [10, 20, 30, 40]:
            assert relations[:3].tolist() == [0, 1, 2]
            break
    else:
        pytest.fail("Expected three-hop dependency path")


def test_duplicate_seeds_masks_and_stable_id_order(reasoning_graph: Graph) -> None:
    graph = reasoning_graph
    state = graph.to_state()
    model = RUTAMGraphTraversal(graph.config.traversal).eval()
    output = model(
        torch.zeros((1, 5)),
        torch.ones((8, 8)),
        state.edge_index,
        state.edge_types,
        state.edge_weights,
        torch.tensor([[4, 0, 4, -999]]),
        torch.tensor([[True, True, True, False]]),
        state.node_ids,
    )
    assert output.visited_node_ids[0, :2].tolist() == [10, 50]
    assert (
        len(set(output.visited_indices[output.visited_mask].tolist())) == output.visited_mask.sum()
    )
    with pytest.raises(ValidationError, match="seed"):
        model(
            torch.zeros((1, 5)),
            torch.ones((8, 8)),
            state.edge_index,
            state.edge_types,
            state.edge_weights,
            torch.tensor([[99]]),
            torch.tensor([[True]]),
            state.node_ids,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_hops": 0},
        {"max_visited_nodes": 2},
        {"max_candidate_paths": 2, "max_selected_paths": 1},
        {"max_neighbors_per_node": 1},
        {"include_seed_nodes": False},
    ],
)
def test_all_budgets(reasoning_graph: Graph, overrides: dict) -> None:
    config = replace(reasoning_graph.config.traversal, **overrides)
    output = run(reasoning_graph, config=config)
    assert output.path_mask.sum() <= config.max_candidate_paths
    assert output.visited_mask.sum() <= config.max_visited_nodes
    assert output.path_edge_mask.sum(2).max() <= config.max_hops
    if not config.include_seed_nodes:
        assert 10 not in output.visited_node_ids[output.visited_mask].tolist()
        assert (output.path_edge_mask.sum(2)[output.path_mask] >= 1).all()
    if config.max_neighbors_per_node == 1:
        assert 50 not in output.visited_node_ids[output.visited_mask].tolist()


def test_cache_mutation_invalidation_even_with_reused_version(reasoning_graph: Graph) -> None:
    state = reasoning_graph.to_state()
    cache = NeighborhoodCache()
    first = cache.get(state.node_ids, state.edge_index, state.edge_types, 2)
    assert cache.get(state.node_ids, state.edge_index, state.edge_types, 2) is first
    assert cache.build_count == 1
    state.edge_index[1, 0] = 7
    second = cache.get(state.node_ids, state.edge_index, state.edge_types, 2)
    assert cache.build_count == 2 and second.fingerprint != first.fingerprint
    cache.get(state.node_ids, state.edge_index, state.edge_types, 3)
    assert cache.build_count == 3


def test_shape_dtype_and_workspace_validation(reasoning_graph: Graph) -> None:
    with pytest.raises(ValidationError):
        TraversalConfig(max_selected_paths=65)
    with pytest.raises(ValidationError):
        TraversalConfig(max_hops=-1)
    with pytest.raises(ValidationError):
        TraversalConfig(temperature=0)
    with pytest.raises(ValidationError):
        run(reasoning_graph, config=replace(reasoning_graph.config.traversal, max_path_elements=1))
    with pytest.warns(UserWarning, match="combinatorially"):
        TraversalConfig(max_hops=9)


def test_invalid_topology_and_weights_are_rejected(reasoning_graph: Graph) -> None:
    state = reasoning_graph.to_state()
    model = RUTAMGraphTraversal(reasoning_graph.config.traversal)
    arguments = [
        torch.ones((1, 5)),
        torch.ones((8, 8)),
        state.edge_index,
        state.edge_types,
        state.edge_weights,
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        state.node_ids,
    ]
    for position, value in (
        (0, torch.ones((1, 4))),
        (1, torch.ones((8, 7))),
        (2, torch.full((2, 10), 99, dtype=torch.int64)),
        (3, torch.full((10,), 99, dtype=torch.int64)),
        (4, torch.ones(9)),
        (4, torch.full((10,), float("nan"))),
    ):
        invalid = list(arguments)
        invalid[position] = value
        with pytest.raises(ValidationError):
            model(*invalid)
