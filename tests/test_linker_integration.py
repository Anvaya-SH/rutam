"""Encoder/linker gradients, safe negatives, immutable snapshots and stale proposals."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rutam import (
    Graph,
    GraphStorage,
    RUTAMConfig,
    RUTAMGraphEncoder,
    RUTAMGraphLinker,
    binary_link_prediction_loss,
    relation_classification_loss,
    sample_negative_pairs,
    validate_proposal_snapshot,
)
from rutam.exceptions import StaleGraphError, ValidationError
from rutam.graph.neighborhood import NeighborhoodCache


def test_encoder_gradients_and_saved_snapshot_immutability(
    link_graph: Graph, tmp_path: Path
) -> None:
    torch.manual_seed(10)
    graph = link_graph
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("graph.*")}
    state = storage.load().to_state()
    encoder = RUTAMGraphEncoder(graph.config.encoder)
    linker = RUTAMGraphLinker(graph.config.linker)
    with (
        patch.object(GraphStorage, "save", side_effect=AssertionError("hidden write")),
        patch.object(GraphStorage, "load", side_effect=AssertionError("hidden read")),
    ):
        encoded = encoder(state)
        output = linker.from_state(encoded, state, candidate_pairs=torch.tensor([[0, 1], [0, 3]]))
        loss = binary_link_prediction_loss(
            output.link_logits, torch.tensor([1, 0])
        ) + relation_classification_loss(
            output.relation_logits, torch.tensor([1, 0]), link_labels=torch.tensor([1, 0])
        )
        loss.backward()
    assert encoder.node_projection.linear.weight.grad.count_nonzero() > 0
    assert encoder.relation_embeddings.weight.grad.count_nonzero() > 0
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("graph.*")}
    reloaded = storage.load().to_state()
    linker.eval()
    torch.testing.assert_close(
        linker.from_state(encoded, state).link_logits,
        linker.from_state(encoded, reloaded).link_logits,
        rtol=0,
        atol=0,
    )


def test_negative_sampling_excludes_all_known_positive_pairs(link_graph: Graph) -> None:
    state = link_graph.to_state()
    index = NeighborhoodCache().get(state.node_ids, state.edge_index, state.edge_types)
    first = sample_negative_pairs(index, 12, seed=17, max_attempts=500)
    second = sample_negative_pairs(index, 12, seed=17, max_attempts=500)
    assert torch.equal(first.pairs, second.pairs) and first.pairs.shape == (12, 2)
    known = set(map(tuple, state.edge_index.T.tolist()))
    assert not known.intersection(map(tuple, first.pairs.tolist()))
    assert (first.pairs[:, 0] != first.pairs[:, 1]).all()
    undirected = sample_negative_pairs(index, 20, seed=3, directed=False, max_attempts=300)
    assert undirected.exhausted  # Fewer than 20 possible unobserved unordered pairs.
    assert all((s, t) not in known and (t, s) not in known for s, t in undirected.pairs.tolist())
    assert RUTAMGraphLinker(link_graph.config.linker).sample_negatives(state, 2).pairs.shape[0] == 6


def test_stale_proposal_detection_and_valid_provenance(link_graph: Graph) -> None:
    graph = link_graph
    linker = RUTAMGraphLinker(
        replace(graph.config.linker, link_threshold=0, relation_threshold=0)
    ).eval()
    with torch.no_grad():
        linker.link_scorer.output_projection.weight.zero_()
        linker.link_scorer.output_projection.bias.fill_(5)
        linker.relation_predictor.output_projection.weight.zero_()
        linker.relation_predictor.output_projection.bias.fill_(-5)
        linker.relation_predictor.output_projection.bias[1] = 5
    state = graph.to_state()
    output = linker.from_state(torch.ones((6, 8)), state, candidate_pairs=torch.tensor([[1, 0]]))
    proposal = output.proposal_batches[0][0]
    validate_proposal_snapshot(proposal, state, linker.vocabulary)
    assert proposal.evidence_references == ("test:source:1", "test:source:0")
    assert proposal.timestamp.utcoffset() is not None
    graph.update_node(10, content="edited")
    with pytest.raises(StaleGraphError):
        validate_proposal_snapshot(proposal, graph.to_state(), linker.vocabulary)
    with pytest.raises(StaleGraphError):
        validate_proposal_snapshot(proposal, state, linker.vocabulary.extend(("new_type",)))


def test_batched_queries_masks_and_global_budget(link_graph: Graph) -> None:
    torch.manual_seed(8)
    c = replace(
        link_graph.config.linker,
        query_conditioning=True,
        link_threshold=0,
        relation_threshold=0,
        max_proposals_per_call=3,
    )
    model = RUTAMGraphLinker(c).eval()
    nodes = torch.randn((6, 8), requires_grad=True)
    queries = torch.randn((2, 5), requires_grad=True)
    output = model.from_state(nodes, link_graph.to_state(), query_vectors=queries)
    count = output.candidate_pairs.shape[0]
    assert output.link_logits.shape == (2, count) and output.relation_logits.shape == (2, count, 4)
    assert len(output.proposal_batches) == 2
    assert sum(map(len, output.proposal_batches)) <= 3
    assert not torch.allclose(output.link_logits[0], output.link_logits[1])
    output.link_logits.square().sum().backward()
    assert queries.grad.count_nonzero() > 0
    masked = model.from_state(
        nodes, link_graph.to_state(), query_vectors=queries, query_mask=torch.tensor([True, False])
    )
    assert masked.link_probabilities[1].count_nonzero() == 0
    assert masked.relation_probabilities[1].count_nonzero() == 0
    assert not masked.proposal_mask[1].any() and not masked.proposal_batches[1]
    empty = model.from_state(nodes, link_graph.to_state(), query_vectors=torch.empty((0, 5)))
    assert empty.link_logits.shape == (0, count)


def test_linker_never_allocates_all_node_pairs(link_graph: Graph) -> None:
    graph = link_graph
    for row in range(40):
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
        output = RUTAMGraphLinker(graph.config.linker).from_state(torch.ones((n, 8)), state)
    assert output.candidate_pairs.shape[0] <= graph.config.linker.max_candidate_pairs


def test_config_roundtrip_and_previous_defaults(tmp_path: Path) -> None:
    config = RUTAMConfig().with_overrides(
        {"linker": {"loss_weights": {"link": 2}, "relation_labels": ["a", "b"]}}
    )
    path = tmp_path / "config.yaml"
    path.write_text(config.to_yaml())
    assert RUTAMConfig.from_yaml(path) == config
    old = config.to_dict()
    old.pop("traversal")
    old.pop("linker")
    loaded = RUTAMConfig.from_dict(old)
    assert loaded.traversal.max_hops == 3 and loaded.linker.max_candidate_pairs == 512
    assert loaded.to_yaml() == loaded.to_yaml()
    with pytest.raises(ValidationError):
        config.with_overrides({"linker": {"loss_weights": {"typo": 1}}})
