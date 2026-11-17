"""Bounded candidates, typed duplicate/conflict annotations and proposal-only behavior."""

from dataclasses import replace

import pytest
import torch

from rutam import Graph, LinkerConfig, RUTAMGraphLinker
from rutam.exceptions import StaleGraphError, ValidationError
from rutam.graph.link_candidates import generate_link_candidates, structural_link_features
from rutam.graph.neighborhood import NeighborhoodCache


def force_scores(model: RUTAMGraphLinker, class_id: int) -> None:
    """Controlled unit-test scores, not a learning or semantic-accuracy demonstration."""
    with torch.no_grad():
        model.link_scorer.output_projection.weight.zero_()
        model.link_scorer.output_projection.bias.fill_(8)
        model.relation_predictor.output_projection.weight.zero_()
        model.relation_predictor.output_projection.bias.fill_(-8)
        model.relation_predictor.output_projection.bias[class_id] = 8


def test_candidate_dedup_order_self_rejection_and_invalid_references(link_graph: Graph) -> None:
    state = link_graph.to_state()
    index = NeighborhoodCache().get(state.node_ids, state.edge_index, state.edge_types)
    result = generate_link_candidates(
        index,
        link_graph.config.linker,
        candidate_pairs=torch.tensor([[1, 0], [0, 2], [0, 1], [0, 2]]),
    )
    assert result.pairs.tolist() == [[0, 1], [0, 2], [1, 0]]
    assert result.node_ids.tolist() == [[10, 20], [10, 30], [20, 10]]
    assert result.duplicates_removed == 1
    for pairs in (torch.tensor([[0, 0]]), torch.tensor([[0, 99]]), torch.tensor([[-1, 0]])):
        with pytest.raises(ValidationError):
            generate_link_candidates(index, link_graph.config.linker, candidate_pairs=pairs)


@pytest.mark.parametrize("strategy", ["neighborhood", "two_hop", "selected", "similarity"])
def test_generation_strategies_are_bounded_and_repeatable(link_graph: Graph, strategy: str) -> None:
    state = link_graph.to_state()
    index = NeighborhoodCache().get(state.node_ids, state.edge_index, state.edge_types)
    config = replace(
        link_graph.config.linker,
        candidate_strategy=strategy,
        max_candidate_pairs=3,
        similarity_chunk_size=2,
        max_neighbors_per_node=2,
    )
    nodes = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    arguments = dict(node_states=nodes, selected_indices=torch.tensor([1, 0]))
    first = generate_link_candidates(index, config, **arguments)
    second = generate_link_candidates(index, config, **arguments)
    assert first.pairs.shape[0] <= 3 and first.pairs.shape[1] == 2
    assert torch.equal(first.pairs, second.pairs)
    assert first.node_ids.tolist() == sorted(first.node_ids.tolist())
    assert (first.pairs[:, 0] != first.pairs[:, 1]).all()


def test_structural_features_are_numerical_evidence_not_hard_rules(link_graph: Graph) -> None:
    state = link_graph.to_state()
    index = NeighborhoodCache().get(state.node_ids, state.edge_index, state.edge_types)
    features = structural_link_features(
        index,
        torch.tensor([[0, 1], [0, 2], [0, 5]]),
        3,
        node_types=tuple(x["node_type"] for x in state.node_metadata),
        compatible_types=(("fact", "fact"), ("fact", "concept")),
    )
    assert features.shape == (3, 7)
    assert features[0, 0].item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert features[1, 1] == 1 and features[2, 2] == 1
    assert features[:, 3].tolist() == [1.0, 1.0, 1.0]
    torch.testing.assert_close(features[:, 4:], torch.tensor([[3 / 8, 3 / 8, 2 / 8]]).expand(3, -1))


def test_typed_duplicates_and_reverse_edges_are_distinct(link_graph: Graph) -> None:
    model = RUTAMGraphLinker(
        replace(link_graph.config.linker, link_threshold=0, relation_threshold=0)
    ).eval()
    force_scores(model, 1)  # Class 1 -> persistent related_to relation ID 0.
    output = model.from_state(
        torch.ones((6, 8)), link_graph.to_state(), candidate_pairs=torch.tensor([[0, 1], [1, 0]])
    )
    assert output.proposal_mask.tolist() == [False, True]
    assert "identical_typed_edge" in output.annotations[0][0]
    assert "opposite_direction_same_relation" in output.annotations[0][1]
    proposal = output.proposal_batches[0][0]
    assert (proposal.source_node_id, proposal.destination_node_id, proposal.relation_id) == (
        20,
        10,
        0,
    )
    assert proposal.evidence_references and proposal.validation_status == "unverified"


def test_conflict_annotations_never_destroy_existing_relationships(link_graph: Graph) -> None:
    config = replace(
        link_graph.config.linker,
        relation_labels=(*link_graph.config.linker.relation_labels, "contradicts"),
        conflicting_relations=(("supports", "contradicts"),),
        link_threshold=0,
        relation_threshold=0,
    )
    model = RUTAMGraphLinker(config).eval()
    force_scores(model, 4)
    before = link_graph.edges
    output = model.from_state(
        torch.ones((6, 8)), link_graph.to_state(), candidate_pairs=torch.tensor([[2, 3]])
    )
    proposal = output.proposal_batches[0][0]
    assert proposal.validation_status == "conflict_review"
    assert "conflicting_relation_evidence" in proposal.annotations
    assert link_graph.edges == before


def test_no_relation_thresholds_and_global_proposal_budget(link_graph: Graph) -> None:
    model = RUTAMGraphLinker(
        replace(
            link_graph.config.linker,
            link_threshold=0,
            relation_threshold=0,
            max_proposals_per_call=2,
        )
    ).eval()
    force_scores(model, 0)
    output = model.from_state(torch.ones((6, 8)), link_graph.to_state())
    assert output.proposal_confidence.count_nonzero() == 0 and not output.proposal_mask.any()
    force_scores(model, 3)
    output = model.from_state(torch.ones((6, 8)), link_graph.to_state())
    assert output.proposal_mask.sum() == 2 and len(output.proposal_batches[0]) == 2
    strict = RUTAMGraphLinker(replace(model.config, link_threshold=1)).eval()
    strict.load_state_dict(model.state_dict())
    assert not strict.from_state(torch.ones((6, 8)), link_graph.to_state()).proposal_mask.any()
    with pytest.raises(StaleGraphError):
        model.from_state(torch.ones((6, 8)), link_graph.to_state(), expected_graph_version=999)


def test_empty_single_node_and_zero_candidates(link_graph: Graph) -> None:
    for n in (0, 1):
        graph = Graph(link_graph.config)
        if n:
            graph.create_node("single", "node", provenance="test", embedding=[0] * 4)
        model = RUTAMGraphLinker(graph.config.linker)
        output = model.from_state(torch.ones((n, 8)), graph.to_state())
        assert output.candidate_pairs.shape == (0, 2)
        assert output.link_logits.shape == (0,) and output.relation_logits.shape == (0, 4)
        assert not output.proposal_batches[0]
        assert torch.isfinite(output.link_probabilities).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"max_candidate_pairs": 0},
        {"max_candidate_hops": 3},
        {"link_threshold": 1.1},
        {"relation_threshold": -1},
        {"dropout": 1},
        {"directed": 1},
        {"candidate_strategy": "all_pairs"},
        {"relation_labels": ["a", "a"]},
        {"relation_labels": ["NO_RELATION"]},
        {"positive_weight": 0},
        {"link_temperature": 0},
        {"loss_weights": {"link": 0, "relation": 0, "ranking": 0}},
    ],
)
def test_linker_configuration_rejection(settings: dict) -> None:
    with pytest.raises(ValidationError):
        LinkerConfig(**settings)
