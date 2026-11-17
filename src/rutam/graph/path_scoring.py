"""Trainable path relevance, global attention and reusable supervised objectives."""

from __future__ import annotations

import math

import torch
from torch import nn

from rutam.config import SelectorConfig, TraversalConfig, bounded_float
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor
from rutam.graph.scoring import (
    NeuralRelevanceScorer,
    masked_softmax,
    validate_logits,
    validity_mask,
)
from rutam.graph.selection import node_retrieval_loss, stable_topk
from rutam.graph.traversal_output import PathScoringOutput


class RUTAMPathScorer(nn.Module):
    def __init__(self, config: TraversalConfig | None = None) -> None:
        super().__init__()
        self.config = TraversalConfig() if config is None else config
        if not isinstance(self.config, TraversalConfig):
            raise ValidationError("Path scorer requires TraversalConfig")
        c = self.config
        self.scorer = NeuralRelevanceScorer(
            SelectorConfig(
                query_dim=c.query_dim,
                hidden_dim=c.hidden_dim,
                score_dim=c.score_hidden_dim,
                scoring_method="scaled_dot" if c.path_scoring == "cosine" else c.path_scoring,
                normalize_vectors=c.path_scoring == "cosine",
                temperature=c.temperature,
            )
        )

    def forward(
        self, queries: torch.Tensor, paths: torch.Tensor, mask: torch.Tensor
    ) -> PathScoringOutput:
        validate_features(queries, self.config.query_dim, "queries")
        validate_tensor(paths, "paths", torch.float32)
        if (
            paths.ndim != 3
            or paths.shape[2] != self.config.hidden_dim
            or paths.shape[0] != queries.shape[0]
        ):
            raise ValidationError("Path representations must have shape [B,P,H]")
        if (
            paths.shape[0]
            * paths.shape[1]
            * max(self.config.hidden_dim, self.config.score_hidden_dim)
            > self.config.max_path_elements
        ):
            raise ValidationError("Path scoring workspace exceeds max_path_elements")
        chunks = [
            self.scorer.score_pairs(queries, chunk)
            for chunk in paths.split(self.config.scoring_chunk_size, dim=1)
        ]
        logits = torch.cat(chunks, 1) if chunks else self.scorer.score_pairs(queries, paths)
        if self.config.path_scoring == "cosine":
            logits = logits * math.sqrt(self.config.score_hidden_dim)
        require_finite(logits, "path logits")
        valid = validity_mask(logits, mask)
        logits = torch.where(valid, logits, 0)
        attention = masked_softmax(logits, valid)
        ids = torch.arange(paths.shape[1], dtype=torch.int64)
        selected = stable_topk(logits, ids, self.config.max_selected_paths, mask=valid)
        return PathScoringOutput(logits, attention, selected.indices, selected.mask)


def _target_mask(ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    validate_tensor(ids, "candidate IDs", torch.int64)
    validate_tensor(targets, "target IDs", torch.int64)
    if ids.ndim != 2 or targets.ndim not in (1, 2) or targets.shape[0] != ids.shape[0]:
        raise ValidationError("IDs must be [B,P] and targets [B] or [B,L]")
    if targets.ndim == 1:
        targets = targets[:, None]
    if (targets < -1).any().item():
        raise ValidationError("Only -1 can pad target IDs")
    positive = torch.zeros_like(ids, dtype=torch.bool)
    for column in range(targets.shape[1]):
        matches = ids == targets[:, column, None]
        requested = targets[:, column] >= 0
        if (requested & ~matches.any(dim=1)).any().item():
            raise ValidationError("Target ID is absent from candidate paths")
        positive |= matches & requested[:, None]
    return positive


def path_retrieval_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    path_ids: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    ignore_unlabeled: bool = False,
) -> torch.Tensor:
    positives = (
        _target_mask(path_ids, targets)
        if path_ids is not None and targets.dtype != torch.bool
        else targets
    )
    return node_retrieval_loss(logits, positives, mask=mask, ignore_unlabeled=ignore_unlabeled)


def endpoint_supervision_loss(
    logits: torch.Tensor,
    endpoint_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    ignore_unlabeled: bool = False,
) -> torch.Tensor:
    """Marginalize all paths reaching the target; does not label one reasoning path as correct."""
    positive = _target_mask(endpoint_node_ids, target_node_ids)
    return node_retrieval_loss(logits, positive, mask=mask, ignore_unlabeled=ignore_unlabeled)


def pairwise_path_ranking_loss(
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    margin: float = 1.0,
) -> torch.Tensor:
    validate_logits(logits)
    bounded_float(margin, "ranking margin")
    valid = validity_mask(logits, mask)
    for tensor in (positive_indices, negative_indices):
        validate_tensor(tensor, "ranking indices", torch.int64)
        if tensor.ndim != 2 or tensor.shape[0] != logits.shape[0]:
            raise ValidationError("Ranking indices must be [B,L]")
    if positive_indices.shape != negative_indices.shape:
        raise ValidationError("Ranking index shapes differ")
    active = (positive_indices >= 0) & (negative_indices >= 0)
    if not torch.equal(positive_indices == -1, negative_indices == -1):
        raise ValidationError("Ranking pairs must be padded together with -1")
    if (
        (
            (positive_indices < -1)
            | (negative_indices < -1)
            | (positive_indices >= logits.shape[1])
            | (negative_indices >= logits.shape[1])
        )
        .any()
        .item()
    ):
        raise ValidationError("Invalid ranking row index")
    if not active.any().item():
        return (logits * 0).sum()
    rows = torch.arange(logits.shape[0])[:, None].expand_as(positive_indices)[active]
    pos, neg = positive_indices[active], negative_indices[active]
    if not (valid[rows, pos] & valid[rows, neg]).all().item():
        raise ValidationError("Ranking targets must be eligible")
    loss = (
        (margin - logits[rows, pos].double() + logits[rows, neg].double())
        .clamp_min(0)
        .mean()
        .float()
    )
    require_finite(loss, "ranking loss")
    return loss
