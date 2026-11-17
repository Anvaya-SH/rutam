"""Analytical token pooling and padding behavior, entirely on CPU."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rutam import QueryHeadConfig, QueryPooling
from rutam.exceptions import ValidationError


def config(method: str, *, policy: str = "error") -> QueryHeadConfig:
    return QueryHeadConfig(
        backbone_hidden_dim=3,
        graph_hidden_dim=4,
        query_dim=2,
        intermediate_dim=5,
        pooling=method,
        empty_sequence_policy=policy,
    )


def test_last_valid_handles_left_right_and_noncontiguous_padding() -> None:
    hidden = torch.arange(45, dtype=torch.float32).reshape(3, 5, 3).requires_grad_()
    mask = torch.tensor(
        [
            [False, False, True, True, False],
            [True, True, False, False, False],
            [True, False, True, False, True],
        ]
    )
    output = QueryPooling(config("last_valid"))(hidden, mask, return_attention=True)
    torch.testing.assert_close(output.context, hidden[torch.arange(3), torch.tensor([3, 1, 4])])
    assert output.attention_weights.tolist() == [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    output.context.sum().backward()
    expected = output.attention_weights[:, :, None].expand_as(hidden)
    torch.testing.assert_close(hidden.grad, expected)


def test_masked_mean_matches_hand_calculation_and_gradient() -> None:
    hidden = torch.tensor([[[1.0, 2.0, 3.0], [1e30] * 3, [3.0, 4.0, 5.0]]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    output = QueryPooling(config("mean"))(hidden, mask, return_attention=True)
    torch.testing.assert_close(output.context, torch.tensor([[2.0, 3.0, 4.0]]))
    torch.testing.assert_close(output.attention_weights, torch.tensor([[0.5, 0.0, 0.5]]))
    output.context.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.tensor([[[0.5] * 3, [0.0] * 3, [0.5] * 3]]))


def test_learnable_attention_matches_formula_and_receives_gradients() -> None:
    pooling = QueryPooling(config("attention"))
    with torch.no_grad():
        pooling.attention_score.weight.copy_(torch.tensor([[0.5, -0.25, 0.1]]))
    hidden = torch.tensor(
        [[[1.0, 0.0, 2.0], [100.0, 200.0, 300.0], [0.0, 1.0, 1.0]]], requires_grad=True
    )
    output = pooling(hidden, torch.tensor([[True, False, True]]), return_attention=True)
    expected = torch.softmax(torch.tensor([0.7, -0.15]), dim=0)
    torch.testing.assert_close(output.attention_weights[0, [0, 2]], expected)
    assert output.attention_weights[0, 1] == 0
    torch.testing.assert_close(
        output.context[0], expected[0] * hidden[0, 0] + expected[1] * hidden[0, 2]
    )
    output.context.square().sum().backward()
    assert pooling.attention_score.weight.grad.count_nonzero() > 0
    assert hidden.grad[0, 1].count_nonzero() == 0
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("method", ["last_valid", "mean", "attention"])
@pytest.mark.parametrize("t", [0, 4])
def test_no_valid_tokens_default_rejection_and_explicit_zero_fallback(method: str, t: int) -> None:
    hidden = torch.ones((2, t, 3), requires_grad=True)
    mask = torch.zeros((2, t), dtype=torch.bool)
    with pytest.raises(ValidationError, match="valid token"):
        QueryPooling(config(method))(hidden, mask)
    output = QueryPooling(config(method, policy="zeros"))(hidden, mask, return_attention=True)
    assert output.context.shape == (2, 3) and output.context.count_nonzero() == 0
    assert (
        output.attention_weights.shape == (2, t) and output.attention_weights.count_nonzero() == 0
    )
    assert not output.valid_context.any()
    output.context.sum().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.count_nonzero() == 0


@pytest.mark.parametrize("method", ["last_valid", "mean", "attention"])
def test_missing_mask_and_optional_inspection_weights(method: str) -> None:
    torch.manual_seed(1)
    pooling = QueryPooling(config(method))
    hidden = torch.randn((2, 5, 3))
    output = pooling(hidden)
    inspected = pooling(hidden, return_attention=True)
    assert output.attention_weights is None and output.valid_context.all()
    torch.testing.assert_close(output.context, inspected.context)
    torch.testing.assert_close(inspected.attention_weights.sum(1), torch.ones(2))


def test_noncontiguous_storage_and_masks_work() -> None:
    hidden = torch.arange(48, dtype=torch.float32).reshape(2, 8, 3)[:, ::2]
    mask = torch.tensor(
        [
            [True, False, False, False, True, False, False, False],
            [False, False, True, False, False, False, True, False],
        ]
    )[:, ::2]
    assert not hidden.is_contiguous() and not mask.is_contiguous()
    output = QueryPooling(config("last_valid"))(hidden, mask)
    torch.testing.assert_close(output.context, torch.stack((hidden[0, 2], hidden[1, 3])))


@pytest.mark.parametrize(
    "hidden,mask",
    [
        (torch.ones((2, 3)), None),
        (torch.ones((2, 4, 2)), None),
        (torch.ones((2, 4, 3), dtype=torch.float64), None),
        (torch.full((2, 4, 3), float("nan")), None),
        (torch.ones((2, 4, 3)), torch.ones((2, 4))),
        (torch.ones((2, 4, 3)), torch.ones((2, 3), dtype=torch.bool)),
    ],
)
def test_pooling_rejects_invalid_tensor_contracts(
    hidden: torch.Tensor, mask: torch.Tensor | None
) -> None:
    with pytest.raises(ValidationError):
        QueryPooling(config("mean"))(hidden, mask)


@pytest.mark.parametrize("method", ["last_valid", "mean", "attention"])
@pytest.mark.parametrize("t", [0, 3])
def test_empty_batch(method: str, t: int) -> None:
    output = QueryPooling(replace(config(method), return_attention=True))(torch.empty((0, t, 3)))
    assert output.context.shape == (0, 3)
    assert output.attention_weights.shape == (0, t)
