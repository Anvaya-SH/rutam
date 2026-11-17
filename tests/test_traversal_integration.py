"""Native hidden-state/encoder/selector/traversal pipeline and persistence boundaries."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rutam import (
    Graph,
    GraphStorage,
    RUTAMGraphEncoder,
    RUTAMGraphSelector,
    RUTAMGraphTraversal,
    RUTAMNeuralQueryHead,
    path_retrieval_loss,
)


def test_full_neural_pipeline_and_storage_immutability(
    reasoning_graph: Graph, tmp_path: Path
) -> None:
    torch.manual_seed(41)
    graph = reasoning_graph
    c = graph.config
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("graph.*")}
    state = storage.load().to_state()
    encoder = RUTAMGraphEncoder(c.encoder)
    selector = RUTAMGraphSelector(c.selector)
    head = RUTAMNeuralQueryHead(c.query_head, selector_config=c.selector)
    traversal = RUTAMGraphTraversal(c.traversal, relation_embedding=encoder.relation_embeddings)
    assert traversal.path_encoder.relation_embedding is encoder.relation_embeddings
    hidden = torch.randn((2, 4, 6), requires_grad=True)
    with (
        patch.object(GraphStorage, "save", side_effect=AssertionError("hidden write")),
        patch.object(GraphStorage, "load", side_effect=AssertionError("hidden disk read")),
    ):
        encoded = encoder(state)
        seeds = head.select_graph(hidden, selector, encoded, state.node_ids)
        output = traversal.from_state(
            seeds.query.query_vectors,
            encoded,
            state,
            seeds.selection.selected_indices,
            seeds.selection.selection_mask,
        )
        loss = path_retrieval_loss(output.path_scores, torch.tensor([0, 0]), mask=output.path_mask)
        loss = loss + (output.reasoning_context * seeds.query.query_gate).square().mean()
        loss.backward()
    for parameter in (
        head.output_projection.weight,
        traversal.path_scorer.scorer.query_projection.weight,
        traversal.path_encoder.gru.weight_ih,
        encoder.relation_embeddings.weight,
        encoder.node_projection.linear.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    assert hidden.grad.count_nonzero() > 0
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("graph.*")}


def test_empty_single_node_zero_paths_and_empty_batch(reasoning_graph: Graph) -> None:
    c = reasoning_graph.config
    for n in (0, 1):
        graph = Graph(c)
        if n:
            graph.create_node("only", "one", provenance="test", embedding=[0] * 4)
        state = graph.to_state()
        traversal = RUTAMGraphTraversal(c.traversal)
        nodes = torch.ones((n, 8), requires_grad=True)
        queries = torch.ones((2, 5), requires_grad=True)
        output = traversal.from_state(
            queries,
            nodes,
            state,
            torch.tensor([[0], [0]]) if n else torch.full((2, 1), -1),
            torch.full((2, 1), bool(n)),
        )
        assert output.reasoning_context.shape == (2, 8)
        assert torch.isfinite(output.reasoning_context).all()
        if not n:
            assert not output.path_mask.any() and output.reasoning_context.count_nonzero() == 0
            assert (output.visited_node_ids == -1).all()
        output.reasoning_context.sum().backward()
        assert torch.isfinite(queries.grad).all()
        zero = RUTAMGraphTraversal(replace(c.traversal, include_seed_nodes=False, max_hops=0))
        out = zero.from_state(
            queries, nodes, state, torch.full((2, 1), -1), torch.zeros((2, 1), dtype=torch.bool)
        )
        assert not out.path_mask.any()
        empty = traversal.from_state(
            torch.empty((0, 5)),
            nodes,
            state,
            torch.empty((0, 1), dtype=torch.int64),
            torch.empty((0, 1), dtype=torch.bool),
        )
        assert empty.reasoning_context.shape == (0, 8)


def test_sparse_graph_never_allocates_node_adjacency(reasoning_graph: Graph) -> None:
    graph = reasoning_graph
    for row in range(30):
        graph.create_node("extra", str(row), provenance="test", embedding=[0] * 4)
    state = graph.to_state()
    n = len(graph.nodes)

    class Guard(TorchDispatchMode):
        def __torch_dispatch__(
            self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
        ) -> Any:
            result = func(*args, **(kwargs or {}))
            for value in result if isinstance(result, (tuple, list)) else (result,):
                if isinstance(value, torch.Tensor):
                    assert value.shape[-2:] != (n, n)
            return result

    with Guard():
        output = RUTAMGraphTraversal(graph.config.traversal).from_state(
            torch.ones((1, 5)),
            torch.ones((n, 8)),
            state,
            torch.tensor([[0]]),
            torch.tensor([[True]]),
        )
    assert output.path_mask.sum() <= graph.config.traversal.max_candidate_paths
