"""Analytical scoring, masked probabilities and projection derivatives on CPU."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from rutam import NeuralRelevanceScorer, SelectorConfig, masked_softmax
from rutam.exceptions import ValidationError


def identity_scorer(**settings: Any) -> NeuralRelevanceScorer:
    scorer = NeuralRelevanceScorer(
        SelectorConfig(query_dim=2, hidden_dim=2, score_dim=2, **settings)
    )
    with torch.no_grad():
        scorer.query_projection.weight.copy_(torch.eye(2))
        scorer.node_projection.weight.copy_(torch.eye(2))
    return scorer


def test_scaled_dot_matches_hand_calculation() -> None:
    scorer = identity_scorer()
    queries = torch.tensor([[1.0, 2.0], [-1.0, 0.0]])
    nodes = torch.tensor([[3.0, 4.0], [0.0, -2.0], [1.0, 1.0]])
    expected = torch.tensor([[11.0, -4.0, 3.0], [-3.0, 0.0, -1.0]]) / math.sqrt(2)
    torch.testing.assert_close(scorer(queries, nodes), expected)


def test_additive_matches_hand_calculation() -> None:
    scorer = identity_scorer(scoring_method="additive")
    with torch.no_grad():
        scorer.bias.copy_(torch.tensor([0.1, -0.2]))
        scorer.value.weight.copy_(torch.tensor([[0.5, -0.8]]))
    actual = scorer(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    expected = 0.5 * math.tanh(1.1) - 0.8 * math.tanh(0.8)
    assert actual.item() == pytest.approx(expected, abs=1e-7)


def test_normalization_zero_vectors_and_large_finite_norms() -> None:
    scorer = identity_scorer(normalize_vectors=True)
    nodes = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1e30, 1e30]])
    output = scorer(torch.tensor([[3.0, 4.0]]), nodes)
    assert torch.isfinite(output).all()
    assert output[0, 0].item() == pytest.approx(1 / math.sqrt(2), abs=1e-7)
    assert output[0, 1].item() == 0
    torch.testing.assert_close(scorer(torch.tensor([[30.0, 40.0]]), nodes), output)


def test_temperature_changes_attention_concentration() -> None:
    queries, nodes = torch.tensor([[1.0, 0.0]]), torch.eye(2)
    cold = masked_softmax(identity_scorer(temperature=0.1)(queries, nodes))
    warm = masked_softmax(identity_scorer(temperature=2.0)(queries, nodes))
    assert cold[0, 0] > warm[0, 0] > 0.5
    torch.testing.assert_close(cold.sum(1), torch.ones(1))


def test_masked_softmax_all_masked_empty_extremes_and_gradients() -> None:
    logits = torch.tensor([[1000.0, -1000.0, 999.0], [1.0, 2.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    probabilities = masked_softmax(logits, mask)
    assert torch.isfinite(probabilities).all()
    assert probabilities[0, 1] == 0 and torch.equal(probabilities[1], torch.zeros(3))
    assert probabilities[0, 0].item() == pytest.approx(0.7310586)
    probabilities[:, 0].sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0 and logits.grad[1].count_nonzero() == 0
    empty = torch.empty((2, 0), requires_grad=True)
    masked_softmax(empty).sum().backward()
    assert empty.grad.shape == (2, 0)


@pytest.mark.parametrize("method", ["scaled_dot", "additive"])
def test_paired_scores_and_gradients_match_full_scoring(method: str) -> None:
    torch.manual_seed(9)
    scorer = identity_scorer(scoring_method=method)
    queries = torch.randn((2, 2), requires_grad=True)
    nodes = torch.randn((3, 2), requires_grad=True)
    indices = torch.tensor([[0, 2], [1, 0]])
    expected = scorer(queries, nodes).gather(1, indices)
    actual = scorer.score_pairs(queries, nodes[indices])
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    for parameter in scorer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    assert queries.grad.count_nonzero() > 0 and nodes.grad.count_nonzero() > 0


@pytest.mark.parametrize("method,expected", [("scaled_dot", 32768), ("additive", 33024)])
def test_default_trainable_parameter_counts(method: str, expected: int) -> None:
    model = NeuralRelevanceScorer(SelectorConfig(scoring_method=method))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"query_dim": 0},
        {"hidden_dim": -1},
        {"score_dim": True},
        {"top_k": 0},
        {"temperature": 0},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": True},
        {"chunk_size": 0},
        {"chunk_size": 1.5},
        {"max_dense_elements": 0},
        {"return_attention": 1},
        {"normalize_vectors": "true"},
        {"selection_mode": "hybrid"},
        {"scoring_method": "text"},
    ],
)
def test_selector_config_validation(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SelectorConfig(**settings)


def test_scoring_shape_dtype_and_finite_validation() -> None:
    scorer = identity_scorer()
    for queries, nodes in [
        (torch.ones(2), torch.ones((1, 2))),
        (torch.ones((1, 3)), torch.ones((1, 2))),
        (torch.ones((1, 2)), torch.ones((1, 3))),
        (torch.ones((1, 2), dtype=torch.float64), torch.ones((1, 2))),
        (torch.tensor([[float("nan"), 1.0]]), torch.ones((1, 2))),
    ]:
        with pytest.raises(ValidationError):
            scorer(queries, nodes)
    with pytest.raises(ValidationError):
        masked_softmax(torch.ones((2, 3)), torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValidationError):
        masked_softmax(torch.tensor([[float("inf")]]))
    with pytest.raises(ValidationError, match="nonfinite"):
        identity_scorer(temperature=1e-40)(torch.ones((1, 2)), torch.ones((1, 2)))
    assert replace(SelectorConfig(), chunk_size=None).chunk_size is None
