"""Stable tensor top-k and reusable supervised objectives; no training engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rutam.config import boolean, positive_int
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor
from rutam.graph.scoring import validate_logits, validity_mask


@dataclass(frozen=True, slots=True)
class TopKSelection:
    indices: torch.Tensor
    node_ids: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor


def stable_topk(
    scores: torch.Tensor,
    node_ids: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
) -> TopKSelection:
    """Rank descending score then ascending persistent ID; pad to [B,K].

    IDs can be [C] or [B,C]. Optional indices [B,C] preserve original row offsets
    while merging chunks. Padded IDs/indices -1 are permitted only where masked.
    Gathered scores stay differentiable; ordering/indices are discrete.
    """
    validate_logits(scores)
    positive_int(k, "k")
    b, c = scores.shape
    valid = validity_mask(scores, mask)
    validate_tensor(node_ids, "node_ids", torch.int64)
    if node_ids.shape == (c,):
        node_ids = node_ids.unsqueeze(0).expand(b, -1)
    if node_ids.shape != scores.shape:
        raise ValidationError("node_ids must have shape [C] or [B,C]")
    if (node_ids[valid] < 0).any().item():
        raise ValidationError("Eligible node IDs must be nonnegative")
    if indices is None:
        indices = torch.arange(c, dtype=torch.int64).unsqueeze(0).expand(b, -1)
    validate_tensor(indices, "indices", torch.int64)
    if indices.shape != scores.shape or (indices[valid] < 0).any().item():
        raise ValidationError("indices must have shape [B,C] and eligible row offsets >= 0")
    if c == 0:
        return TopKSelection(
            torch.full((b, k), -1, dtype=torch.int64),
            torch.full((b, k), -1, dtype=torch.int64),
            scores.sum(dim=1, keepdim=True).expand(b, k) * 0,
            torch.zeros((b, k), dtype=torch.bool),
        )
    # Stable secondary sort makes ID tie-breaking exact, without epsilon perturbations.
    id_order = torch.argsort(node_ids, dim=1, stable=True)
    sorted_ids, sorted_valid = node_ids.gather(1, id_order), valid.gather(1, id_order)
    if (
        ((sorted_ids[:, 1:] == sorted_ids[:, :-1]) & sorted_valid[:, 1:] & sorted_valid[:, :-1])
        .any()
        .item()
    ):
        raise ValidationError("Eligible persistent IDs must be unique")
    ordered_scores = scores.gather(1, id_order).masked_fill(~valid.gather(1, id_order), -torch.inf)
    order = id_order.gather(1, torch.argsort(ordered_scores, dim=1, descending=True, stable=True))
    order = order[:, : min(k, c)]
    selected_mask = valid.gather(1, order)
    selected_indices = torch.where(selected_mask, indices.gather(1, order), -1)
    selected_ids = torch.where(selected_mask, node_ids.gather(1, order), -1)
    selected_scores = torch.where(selected_mask, scores.gather(1, order), 0)
    if c < k:
        padding = k - c
        selected_indices = torch.nn.functional.pad(selected_indices, (0, padding), value=-1)
        selected_ids = torch.nn.functional.pad(selected_ids, (0, padding), value=-1)
        selected_scores = torch.nn.functional.pad(selected_scores, (0, padding))
        selected_mask = torch.nn.functional.pad(selected_mask, (0, padding), value=False)
    return TopKSelection(selected_indices, selected_ids, selected_scores, selected_mask)


def _positive_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    b, n = logits.shape
    if not isinstance(targets, torch.Tensor):
        raise ValidationError("targets must be an int64 [B] or boolean [B,N] tensor")
    if targets.dtype == torch.bool:
        validate_tensor(targets, "targets", torch.bool)
        if targets.shape != logits.shape:
            raise ValidationError("Multi-positive targets must have shape [B,N]")
        return targets
    validate_tensor(targets, "targets", torch.int64)
    if targets.shape != (b,) or ((targets < -1) | (targets >= n)).any().item():
        raise ValidationError("Single targets must be row offsets [B] in [0,N), or -1 if unlabeled")
    positive = torch.zeros_like(logits, dtype=torch.bool)
    labeled = targets >= 0
    positive[torch.arange(b)[labeled], targets[labeled]] = True
    return positive


def _active_rows(
    positive: torch.Tensor, valid: torch.Tensor, ignore_unlabeled: bool
) -> torch.Tensor:
    boolean(ignore_unlabeled, "ignore_unlabeled")
    if (positive & ~valid).any().item():
        raise ValidationError("Positive targets must refer only to eligible nodes")
    active = positive.any(dim=1)
    if not ignore_unlabeled and not active.all().item():
        raise ValidationError("Unlabeled samples require explicit ignore_unlabeled=True")
    return active


def node_retrieval_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    ignore_unlabeled: bool = False,
) -> torch.Tensor:
    """Mean single-target CE or multi-positive marginal negative log-likelihood.

    Integer targets are tensor row positions, not persistent IDs. Boolean targets
    mark all positive rows. Invalid targets never become silently ignored labels.
    """
    validate_logits(logits)
    valid = validity_mask(logits, mask)
    positive = _positive_targets(logits, targets)
    active = _active_rows(positive, valid, ignore_unlabeled)
    if not active.any().item():
        return (logits * 0).sum()
    values = logits[active].double()
    values = values - values.masked_fill(~valid[active], -torch.inf).amax(dim=1, keepdim=True)
    denominator = values.masked_fill(~valid[active], -torch.inf).logsumexp(dim=1)
    numerator = values.masked_fill(~positive[active], -torch.inf).logsumexp(dim=1)
    loss = (denominator - numerator).mean().float()
    require_finite(loss, "retrieval loss")
    return loss


def attention_supervision_loss(
    logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    ignore_unlabeled: bool = False,
) -> torch.Tensor:
    """Mean KL(teacher || masked student); teacher rows must already be normalized.

    Zero teacher rows are unlabeled and allowed only with ignore_unlabeled=True.
    Student probabilities are never clamped/detached to manufacture gradients.
    """
    validate_logits(logits)
    validate_tensor(teacher_probs, "teacher_probs", torch.float32)
    if teacher_probs.shape != logits.shape:
        raise ValidationError("teacher_probs must have shape [B,N]")
    require_finite(teacher_probs, "teacher_probs")
    if (teacher_probs < 0).any().item():
        raise ValidationError("Teacher probabilities must be nonnegative")
    valid = validity_mask(logits, mask)
    active = _active_rows(teacher_probs > 0, valid, ignore_unlabeled)
    totals = teacher_probs.double().sum(dim=1)[active]
    if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-5, atol=1e-6):
        raise ValidationError("Teacher probabilities must sum to one per labeled sample")
    if not active.any().item():
        return (logits * 0).sum()
    values = logits[active].double()
    values = values - values.masked_fill(~valid[active], -torch.inf).amax(dim=1, keepdim=True)
    denominator = values.masked_fill(~valid[active], -torch.inf).logsumexp(dim=1, keepdim=True)
    log_probs = values - denominator
    teacher = teacher_probs[active].double()
    teacher_log = teacher.clamp_min(torch.finfo(torch.float64).tiny).log()
    loss = (teacher * (teacher_log - log_probs)).sum(dim=1).mean().float()
    require_finite(loss, "attention supervision loss")
    return loss
