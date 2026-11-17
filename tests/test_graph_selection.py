"""Stable row/ID selection and analytically checked retrieval/KL objectives."""

from __future__ import annotations

import math

import pytest
import torch

from rutam import attention_supervision_loss, node_retrieval_loss, stable_topk
from rutam.exceptions import ValidationError


def test_stable_ties_use_ids_without_score_perturbation() -> None:
    scores = torch.tensor([[1.0, 1.0, 1.0, 2.0], [3.0, 3.0, 3.0, 3.0]], requires_grad=True)
    ids = torch.tensor([90, 7, 2**63 - 1, 31])
    result = stable_topk(scores, ids, 3)
    assert result.indices.tolist() == [[3, 1, 0], [1, 3, 0]]
    assert result.node_ids.tolist() == [[31, 7, 90], [7, 31, 90]]
    torch.testing.assert_close(result.scores, scores.gather(1, result.indices))
    result.scores.sum().backward()
    assert scores.grad[:, 2].count_nonzero() == 0
    assert not result.indices.requires_grad and not result.mask.requires_grad


def test_fixed_size_padding_masks_and_global_row_indices() -> None:
    result = stable_topk(
        torch.tensor([[-5.0, 100.0]]),
        torch.tensor([80, 90]),
        4,
        mask=torch.tensor([[True, False]]),
        indices=torch.tensor([[9, 14]]),
    )
    assert result.indices.tolist() == [[9, -1, -1, -1]]
    assert result.node_ids.tolist() == [[80, -1, -1, -1]]
    assert result.scores.tolist() == [[-5.0, 0.0, 0.0, 0.0]]
    assert result.mask.tolist() == [[True, False, False, False]]
    empty = stable_topk(torch.empty((2, 0)), torch.empty(0, dtype=torch.int64), 3)
    assert empty.indices.shape == (2, 3) and (empty.indices == -1).all()
    assert not empty.mask.any() and empty.scores.count_nonzero() == 0


def test_near_ties_remain_ordered_by_score() -> None:
    scores = torch.tensor([[1.0, 1.0 + 2**-23]])
    assert stable_topk(scores, torch.tensor([1, 1000]), 1).node_ids.item() == 1000


def test_invalid_topk_inputs() -> None:
    for scores, ids, k in [
        (torch.ones((1, 2)), torch.tensor([1, 1]), 1),
        (torch.ones((1, 2)), torch.tensor([-1, 2]), 1),
        (torch.ones((1, 2)), torch.tensor([1, 2]), 0),
        (torch.ones((1, 2)), torch.tensor([1]), 1),
    ]:
        with pytest.raises(ValidationError):
            stable_topk(scores, ids, k)


def test_single_target_retrieval_matches_cross_entropy() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.3, 2.0]], requires_grad=True)
    targets = torch.tensor([2, 1])
    actual = node_retrieval_loss(logits, targets)
    torch.testing.assert_close(actual, torch.nn.functional.cross_entropy(logits, targets))
    actual.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.count_nonzero() > 0


def test_multi_positive_loss_is_marginal_log_likelihood() -> None:
    logits = torch.tensor([[0.0, math.log(2), math.log(3)]], requires_grad=True)
    positive = torch.tensor([[False, True, True]])
    loss = node_retrieval_loss(logits, positive)
    assert loss.item() == pytest.approx(-math.log(5 / 6), abs=1e-7)
    loss.backward()
    assert logits.grad[0, 0] > 0 and (logits.grad[0, 1:] < 0).all()


def test_retrieval_masks_and_explicit_unlabeled_policy() -> None:
    logits = torch.tensor([[0.0, math.log(2), 1e3], [2.0, 3.0, 4.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    targets = torch.tensor([1, -1])
    with pytest.raises(ValidationError, match="ignore_unlabeled"):
        node_retrieval_loss(logits, targets, mask=mask)
    loss = node_retrieval_loss(logits, targets, mask=mask, ignore_unlabeled=True)
    assert loss.item() == pytest.approx(-math.log(2 / 3), abs=1e-7)
    loss.backward()
    assert logits.grad[0, 2] == 0 and logits.grad[1].count_nonzero() == 0
    for invalid in (
        torch.tensor([2, -1]),
        torch.tensor([3, -1]),
        torch.tensor([-2, -1]),
        torch.ones((2, 3)),
        torch.tensor([[False, False, True], [False] * 3]),
    ):
        with pytest.raises(ValidationError):
            node_retrieval_loss(logits, invalid, mask=mask, ignore_unlabeled=True)


def test_all_unlabeled_empty_and_extreme_logits_have_finite_zero_gradients() -> None:
    for n in (0, 2):
        logits = torch.full((2, n), 3e38, requires_grad=True)
        loss = node_retrieval_loss(logits, torch.tensor([-1, -1]), ignore_unlabeled=True)
        assert loss.item() == 0 and torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all() and logits.grad.count_nonzero() == 0
    extreme = torch.tensor([[1e30, -1e30]], requires_grad=True)
    loss = node_retrieval_loss(extreme, torch.tensor([1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(extreme.grad).all()


def test_losses_are_stable_under_extreme_equal_logit_offsets() -> None:
    logits = torch.full((1, 2), 1e30, requires_grad=True)
    loss = node_retrieval_loss(logits, torch.tensor([0]))
    assert loss.item() == pytest.approx(math.log(2))
    teacher = torch.tensor([[1.0, 0.0]])
    kl = attention_supervision_loss(logits, teacher)
    assert kl.item() == pytest.approx(math.log(2))
    (loss + kl).backward()
    assert torch.isfinite(logits.grad).all()


def test_attention_supervision_matches_hand_calculated_kl() -> None:
    logits = torch.tensor([[math.log(0.8), math.log(0.2), 100.0]], requires_grad=True)
    teacher = torch.tensor([[0.25, 0.75, 0.0]])
    mask = torch.tensor([[True, True, False]])
    loss = attention_supervision_loss(logits, teacher, mask=mask)
    expected = 0.25 * math.log(0.25 / 0.8) + 0.75 * math.log(0.75 / 0.2)
    assert loss.item() == pytest.approx(expected, abs=1e-7)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.55, -0.55, 0.0]]))


def test_teacher_zeros_equal_distributions_and_unlabeled_policy() -> None:
    logits = torch.tensor([[0.0, 1.0], [3.0, 4.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValidationError):
        attention_supervision_loss(logits, teacher)
    loss = attention_supervision_loss(logits, teacher, ignore_unlabeled=True)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-1)))
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[1].count_nonzero() == 0
    assert attention_supervision_loss(torch.zeros((1, 2)), torch.full((1, 2), 0.5)).item() == 0


@pytest.mark.parametrize(
    "teacher",
    [
        torch.tensor([[0.2, 0.2]]),
        torch.tensor([[-0.1, 1.1]]),
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        torch.tensor([[1.0]]),
        torch.tensor([[0.0, 1.0]]),
    ],
)
def test_invalid_teacher_distributions(teacher: torch.Tensor) -> None:
    with pytest.raises(ValidationError):
        attention_supervision_loss(torch.zeros((1, 2)), teacher, mask=torch.tensor([[True, False]]))
