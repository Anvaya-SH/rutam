"""Directed neural pair scores, gradient flow and reusable loss calculations."""

from dataclasses import replace

import pytest
import torch

from rutam import (
    Graph,
    LinkerConfig,
    RUTAMGraphLinker,
    RUTAMLinkScorer,
    binary_link_prediction_loss,
    combined_link_loss,
    link_ranking_loss,
)
from rutam.exceptions import ValidationError


def test_directed_asymmetry_and_undirected_symmetric_features() -> None:
    torch.manual_seed(6)
    config = LinkerConfig(
        hidden_dim=3, scoring_hidden_dim=5, dropout=0, use_structural_features=False
    )
    nodes = torch.tensor([[1.0, 2.0, -0.5], [-2.0, 0.1, 3.0]], requires_grad=True)
    pairs = torch.tensor([[0, 1], [1, 0]])
    scorer = RUTAMLinkScorer(config)
    logits = scorer(nodes, pairs)
    assert not torch.allclose(logits[0], logits[1])
    logits.square().sum().backward()
    assert nodes.grad.count_nonzero() > 0
    for parameter in scorer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    symmetric = RUTAMLinkScorer(replace(config, directed=False))
    values = symmetric(nodes, pairs)
    torch.testing.assert_close(values[0], values[1])


def test_query_conditioning_keeps_explicit_batch_dimension() -> None:
    torch.manual_seed(2)
    config = LinkerConfig(
        hidden_dim=3,
        scoring_hidden_dim=5,
        query_dim=2,
        query_conditioning=True,
        use_structural_features=False,
        dropout=0,
    )
    model = RUTAMLinkScorer(config)
    nodes = torch.randn((4, 3), requires_grad=True)
    queries = torch.randn((2, 2), requires_grad=True)
    output = model(nodes, torch.tensor([[0, 1], [2, 3]]), query_vectors=queries)
    assert output.shape == (2, 2) and not torch.allclose(output[0], output[1])
    output.square().sum().backward()
    assert queries.grad.count_nonzero() > 0
    assert model.features.query_projection.weight.grad.count_nonzero() > 0
    assert model(nodes, torch.empty((0, 2), dtype=torch.int64), query_vectors=queries).shape == (
        2,
        0,
    )


def test_binary_loss_positive_weight_margin_and_invalid_labels() -> None:
    logits = torch.tensor([0.2, -0.5], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    actual = binary_link_prediction_loss(logits, labels, positive_weight=3)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=torch.tensor(3.0)
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert (
        link_ranking_loss(torch.tensor([0.0, 2.0]), torch.tensor([0]), torch.tensor([1])).item()
        == 3
    )
    with pytest.raises(ValidationError):
        binary_link_prediction_loss(logits, torch.tensor([0.5, 0.0]))


def test_combined_objective_and_safe_label_semantics(link_graph: Graph) -> None:
    torch.manual_seed(5)
    model = RUTAMGraphLinker(link_graph.config.linker)
    pairs = torch.tensor([[0, 1], [0, 3]])
    output = model.from_state(
        torch.randn((6, 8), requires_grad=True), link_graph.to_state(), candidate_pairs=pairs
    )
    labels, relations = torch.tensor([1.0, 0.0]), torch.tensor([1, 0])
    loss = combined_link_loss(
        output,
        labels,
        relations,
        config=model.config,
        positive_indices=torch.tensor([0]),
        negative_indices=torch.tensor([1]),
    )
    torch.testing.assert_close(loss.total, loss.link + loss.relation + 0.2 * loss.ranking)
    loss.total.backward()
    assert model.link_scorer.output_projection.weight.grad.count_nonzero() > 0
    assert model.relation_predictor.output_projection.weight.grad.count_nonzero() > 0
    with pytest.raises(ValidationError, match="Nonexistent"):
        combined_link_loss(output, labels, torch.tensor([1, 2]), config=model.config)


def test_default_linker_parameter_count() -> None:
    model = RUTAMGraphLinker()
    assert sum(p.numel() for p in model.link_scorer.parameters()) == 136913
    assert sum(p.numel() for p in model.relation_predictor.parameters()) == 138969
    assert sum(p.numel() for p in model.parameters()) == 275882
