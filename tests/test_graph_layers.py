"""Small analytical checks for sparse aggregation and its derivatives."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn

from rutam import EncoderConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import NodeProjection, RelationEmbedding
from rutam.graph.layers import RelationAwareMessagePassing


def analytical_layer(
    *, aggregation: str = "mean", residual: bool = False
) -> RelationAwareMessagePassing:
    config = EncoderConfig(
        input_dim=2,
        hidden_dim=2,
        num_relation_types=2,
        dropout=0,
        activation="relu",
        layer_norm=False,
        residual=residual,
        aggregation=aggregation,
    )
    layer = RelationAwareMessagePassing(config)
    with torch.no_grad():
        layer.source_linear.weight.copy_(torch.eye(2))
        layer.relation_linear.weight.copy_(torch.eye(2))
        layer.self_linear.weight.zero_()
        layer.self_linear.bias.zero_()
    return layer


def analytical_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [9.0, 9.0]]),
        torch.tensor([[0, 1], [2, 2]], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([2.0, 0.5]),
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
    )


@pytest.mark.parametrize("aggregation,expected", [("mean", [2.75, 3.5]), ("sum", [5.5, 7.0])])
def test_hand_computed_weighted_typed_aggregation(aggregation: str, expected: list[float]) -> None:
    output = analytical_layer(aggregation=aggregation)(*analytical_inputs())
    torch.testing.assert_close(output[2], torch.tensor(expected))
    torch.testing.assert_close(output[[0, 1, 3]], torch.zeros((3, 2)))


def test_residual_preserves_isolated_and_source_nodes() -> None:
    inputs = analytical_inputs()
    output = analytical_layer(residual=True)(*inputs)
    torch.testing.assert_close(output[[0, 1, 3]], inputs[0][[0, 1, 3]])
    torch.testing.assert_close(output[2], torch.tensor([2.75, 3.5]))


def test_analytical_input_relation_and_edge_weight_gradients() -> None:
    nodes, edges, types, weights, relations = analytical_inputs()
    for tensor in (nodes, weights, relations):
        tensor.requires_grad_()
    layer = analytical_layer()
    layer(nodes, edges, types, weights, relations)[2].sum().backward()
    torch.testing.assert_close(
        nodes.grad, torch.tensor([[1.0, 1.0], [0.25, 0.25], [0.0, 0.0], [0.0, 0.0]])
    )
    torch.testing.assert_close(relations.grad, torch.tensor([[1.0, 1.0], [0.25, 0.25]]))
    torch.testing.assert_close(weights.grad, torch.tensor([2.0, 4.5]))
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_zero_weights_zero_degree_empty_edges_and_empty_nodes() -> None:
    layer = analytical_layer()
    nodes, edges, types, weights, relations = analytical_inputs()
    output = layer(nodes, edges, types, weights * 0, relations)
    assert torch.count_nonzero(output).item() == 0
    empty_edges = torch.empty((2, 0), dtype=torch.int64)
    empty_types = torch.empty(0, dtype=torch.int64)
    empty_weights = torch.empty(0, dtype=torch.float32)
    assert torch.equal(layer(nodes, empty_edges, empty_types, empty_weights, relations), nodes * 0)
    empty = layer(nodes[:0], empty_edges, empty_types, empty_weights, relations)
    assert empty.shape == (0, 2)
    assert torch.isfinite(empty).all()


def test_mean_counts_incoming_edges_instead_of_normalizing_weights_away() -> None:
    nodes, _, _, _, relations = analytical_inputs()
    edges = torch.tensor([[0], [2]])
    types = torch.tensor([0])
    layer = analytical_layer()
    one = layer(nodes, edges, types, torch.tensor([1.0]), relations)
    two = layer(nodes, edges, types, torch.tensor([2.0]), relations)
    torch.testing.assert_close(two[2], 2 * one[2])


def test_node_projection_and_relation_embedding_shapes() -> None:
    config = EncoderConfig(input_dim=3, hidden_dim=7, num_relation_types=4, dropout=0)
    projection = NodeProjection(config)
    relations = RelationEmbedding(4, 7)
    assert isinstance(relations, nn.Embedding)
    assert projection(torch.ones((5, 3))).shape == (5, 7)
    assert projection(torch.empty((0, 3))).shape == (0, 7)
    assert relations(torch.tensor([0, 3, 0])).shape == (3, 7)
    assert relations(torch.empty(0, dtype=torch.int64)).shape == (0, 7)
    torch.testing.assert_close(relations(torch.tensor([0, 0]))[0], relations.weight[0])


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([-1]),
        torch.tensor([4]),
        torch.tensor([0.0]),
        torch.tensor([[0]]),
        torch.tensor([0], dtype=torch.int32),
    ],
)
def test_relation_embedding_rejects_invalid_ids(ids: torch.Tensor) -> None:
    with pytest.raises(ValidationError):
        RelationEmbedding(4, 3)(ids)


@pytest.mark.parametrize(
    "position,invalid",
    [
        (0, torch.ones((4, 3))),
        (0, torch.ones((4, 2), dtype=torch.float64)),
        (1, torch.tensor([[0], [8]])),
        (1, torch.tensor([[0], [-1]])),
        (1, torch.tensor([[0, 1]])),
        (1, torch.tensor([[0, 1], [2, 2]], dtype=torch.int32)),
        (2, torch.tensor([0, 2])),
        (2, torch.tensor([-1, 0])),
        (3, torch.tensor([1.0])),
        (3, torch.tensor([-1.0, 1.0])),
        (3, torch.tensor([float("nan"), 1.0])),
        (3, torch.tensor([1, 1])),
        (4, torch.zeros((1, 2))),
        (4, torch.zeros((2, 3))),
    ],
)
def test_layer_rejects_invalid_tensor_contract(position: int, invalid: torch.Tensor) -> None:
    inputs = list(analytical_inputs())
    inputs[position] = invalid
    with pytest.raises(ValidationError):
        analytical_layer()(*inputs)


def test_nonfinite_arithmetic_is_reported_instead_of_returning_nans() -> None:
    nodes, edges, types, weights, relations = analytical_inputs()
    weights[:] = torch.finfo(torch.float32).max
    with pytest.raises(ValidationError, match="nonfinite"):
        analytical_layer(aggregation="sum")(nodes, edges, types, weights, relations)


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_layer_options(activation: str, layer_norm: bool, residual: bool) -> None:
    config = replace(
        EncoderConfig(),
        hidden_dim=2,
        num_relation_types=2,
        dropout=0,
        activation=activation,
        layer_norm=layer_norm,
        residual=residual,
    )
    result = RelationAwareMessagePassing(config)(*analytical_inputs())
    assert result.shape == (4, 2) and result.dtype == torch.float32
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0},
        {"hidden_dim": -2},
        {"num_layers": 0},
        {"num_layers": True},
        {"num_relation_types": 0},
        {"hidden_dim": 1.5},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"dropout": float("inf")},
        {"dropout": "0.1"},
        {"dropout": True},
        {"activation": "unknown"},
        {"aggregation": "weighted_mean"},
        {"residual": 1},
        {"layer_norm": "true"},
    ],
)
def test_invalid_encoder_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EncoderConfig(**kwargs)
