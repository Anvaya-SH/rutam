"""Trainable neural scoring and stable masked probabilities, without text retrieval."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import SelectorConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor


def validate_logits(logits: torch.Tensor) -> None:
    validate_tensor(logits, "logits", torch.float32)
    if logits.ndim != 2:
        raise ValidationError("logits must have shape [B,N]")
    require_finite(logits, "logits")


def validity_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(logits, dtype=torch.bool)
    validate_tensor(mask, "mask", torch.bool)
    if mask.shape != logits.shape:
        raise ValidationError("mask must have the same [B,N] shape as logits")
    return mask


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Float32 probabilities, exactly zero on invalid/empty/all-masked rows.

    Temperature is applied by NeuralRelevanceScorer, not a second time here.
    Internal -inf sentinels never escape this API or poison all-masked gradients.
    """
    validate_logits(logits)
    valid = validity_mask(logits, mask)
    if logits.shape[1] == 0:
        return logits * 0
    masked = logits.masked_fill(~valid, -torch.inf)
    safe = torch.where(valid.any(dim=1, keepdim=True), masked, torch.zeros_like(logits))
    return torch.where(valid, torch.softmax(safe, dim=1), torch.zeros_like(logits))


class NeuralRelevanceScorer(nn.Module):
    """Shared API for scaled-dot or additive logits, including optional L2 normalization."""

    def __init__(self, config: SelectorConfig | None = None) -> None:
        super().__init__()
        self.config = SelectorConfig() if config is None else config
        if not isinstance(self.config, SelectorConfig):
            raise ValidationError("NeuralRelevanceScorer requires a SelectorConfig")
        c = self.config
        self.query_projection = nn.Linear(
            c.query_dim, c.score_dim, bias=False, device="cpu", dtype=torch.float32
        )
        self.node_projection = nn.Linear(
            c.hidden_dim, c.score_dim, bias=False, device="cpu", dtype=torch.float32
        )
        if c.scoring_method == "additive":
            self.bias = nn.Parameter(torch.zeros(c.score_dim, device="cpu", dtype=torch.float32))
            self.value = nn.Linear(c.score_dim, 1, bias=False, device="cpu", dtype=torch.float32)
        else:
            self.register_parameter("bias", None)
            self.value = None

    def project_queries(self, query_states: torch.Tensor) -> torch.Tensor:
        validate_features(query_states, self.config.query_dim, "query_states")
        validate_tensor(self.query_projection.weight, "scorer parameters", torch.float32)
        return self._normalize(self.query_projection(query_states))

    def project_nodes(self, node_states: torch.Tensor) -> torch.Tensor:
        validate_features(node_states, self.config.hidden_dim, "node_states")
        validate_tensor(self.node_projection.weight, "scorer parameters", torch.float32)
        return self._normalize(self.node_projection(node_states))

    def _normalize(self, projected: torch.Tensor) -> torch.Tensor:
        require_finite(projected, "scoring projection")
        if self.config.normalize_vectors:
            # Normalize in float64 to avoid overflow in finite float32 vector norms.
            projected = F.normalize(projected.double(), dim=-1, eps=1e-12).float()
        return projected

    def _finish(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores / self.config.temperature
        require_finite(scores, "relevance logits")
        return scores

    def score_projected(self, queries: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """[B,S] and [C,S] -> [B,C]; suitable for scoring a node chunk."""
        validate_features(queries, self.config.score_dim, "projected queries")
        validate_features(nodes, self.config.score_dim, "projected nodes")
        if self.config.scoring_method == "scaled_dot":
            scores = (queries / math.sqrt(self.config.score_dim)) @ nodes.T
        else:
            scores = self.value(torch.tanh(queries[:, None, :] + nodes[None, :, :] + self.bias))
            scores = scores.squeeze(-1)
        return self._finish(scores)

    def forward(self, query_states: torch.Tensor, node_states: torch.Tensor) -> torch.Tensor:
        return self.score_projected(
            self.project_queries(query_states), self.project_nodes(node_states)
        )

    def score_pairs(self, query_states: torch.Tensor, node_states: torch.Tensor) -> torch.Tensor:
        """[B,Q] and gathered [B,K,H] -> [B,K], retaining selected-score gradients."""
        queries = self.project_queries(query_states)
        validate_tensor(node_states, "selected node states", torch.float32)
        if node_states.ndim != 3 or node_states.shape[0] != queries.shape[0]:
            raise ValidationError("selected node states must have shape [B,K,H]")
        b, k, h = node_states.shape
        nodes = self.project_nodes(node_states.reshape(b * k, h)).reshape(
            b, k, self.config.score_dim
        )
        if self.config.scoring_method == "scaled_dot":
            scores = ((queries[:, None, :] / math.sqrt(self.config.score_dim)) * nodes).sum(-1)
        else:
            scores = self.value(torch.tanh(queries[:, None, :] + nodes + self.bias)).squeeze(-1)
        return self._finish(scores)
