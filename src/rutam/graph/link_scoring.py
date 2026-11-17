"""Directed neural pair scoring and reusable, explicitly supervised link objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import LinkerConfig, bounded_float
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor
from rutam.graph.link_output import LinkerOutput, LinkLossOutput
from rutam.graph.path_scoring import pairwise_path_ranking_loss


class PairFeatureNetwork(nn.Module):
    """Ordered [u,v,u*v,|u-v|] -> normalized MLP,
    with optional numerical structure/query features."""

    def __init__(self, config: LinkerConfig) -> None:
        super().__init__()
        self.config = config
        h, s = config.hidden_dim, config.scoring_hidden_dim
        self.norm = nn.LayerNorm(4 * h, device="cpu", dtype=torch.float32)
        self.structural_projection = (
            nn.Linear(
                4 + len(config.relation_labels),
                config.structural_feature_dim,
                device="cpu",
                dtype=torch.float32,
            )
            if config.use_structural_features
            else None
        )
        self.query_projection = (
            nn.Linear(config.query_dim, h, device="cpu", dtype=torch.float32)
            if config.query_conditioning
            else None
        )
        width = (
            4 * h
            + (config.structural_feature_dim if config.use_structural_features else 0)
            + (h if config.query_conditioning else 0)
        )
        self.input_projection = nn.Linear(width, s, device="cpu", dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        pairs: torch.Tensor,
        structural_features: torch.Tensor | None = None,
        query_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        validate_features(node_states, c.hidden_dim, "node_states")
        validate_tensor(pairs, "candidate pairs", torch.int64)
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or ((pairs < 0) | (pairs >= node_states.shape[0])).any().item()
        ):
            raise ValidationError("Pairs must be valid row offsets [C,2]")
        validate_tensor(self.input_projection.weight, "link model parameters", torch.float32)
        batch = 1
        if c.query_conditioning:
            if query_vectors is None:
                raise ValidationError("query_conditioning requires query_vectors [B,Q]")
            validate_features(query_vectors, c.query_dim, "query_vectors")
            batch = max(1, query_vectors.shape[0])
        width = (
            4 * c.hidden_dim
            + (c.structural_feature_dim if c.use_structural_features else 0)
            + (c.hidden_dim if c.query_conditioning else 0)
        )
        if batch * pairs.shape[0] * max(width, c.scoring_hidden_dim) > c.max_link_elements:
            raise ValidationError("Pair feature workspace exceeds max_link_elements")
        source, target = node_states[pairs[:, 0]], node_states[pairs[:, 1]]
        product, difference = source * target, (source - target).abs()
        features = (
            torch.cat((source, target, product, difference), -1)
            if c.directed
            else (
                torch.cat(
                    (source + target, product, difference, source.square() + target.square()), -1
                )
            )
        )
        require_finite(features, "pair interactions")
        features = self.norm(features)
        if self.structural_projection is not None:
            if structural_features is None:
                raise ValidationError("Configured structural features are required")
            validate_features(
                structural_features, 4 + len(c.relation_labels), "structural_features"
            )
            if structural_features.shape[0] != pairs.shape[0]:
                raise ValidationError("Structural feature candidate count differs")
            features = torch.cat(
                (features, F.gelu(self.structural_projection(structural_features))), -1
            )
        if self.query_projection is not None:
            if query_vectors is None:
                raise ValidationError("query_conditioning requires query_vectors [B,Q]")
            validate_features(query_vectors, c.query_dim, "query_vectors")
            projected_query = self.query_projection(query_vectors)
            features = torch.cat(
                (
                    features[None].expand(query_vectors.shape[0], -1, -1),
                    projected_query[:, None].expand(-1, pairs.shape[0], -1),
                ),
                -1,
            )
        elif query_vectors is not None:
            raise ValidationError("Queries supplied while query_conditioning is disabled")
        result = self.dropout(F.gelu(self.input_projection(features)))
        require_finite(result, "link hidden states")
        return result


class RUTAMLinkScorer(nn.Module):
    def __init__(self, config: LinkerConfig | None = None) -> None:
        super().__init__()
        self.config = LinkerConfig() if config is None else config
        if not isinstance(self.config, LinkerConfig):
            raise ValidationError("Link scorer requires LinkerConfig")
        self.features = PairFeatureNetwork(self.config)
        self.output_projection = nn.Linear(
            self.config.scoring_hidden_dim, 1, device="cpu", dtype=torch.float32
        )

    def forward(
        self,
        node_states: torch.Tensor,
        pairs: torch.Tensor,
        structural_features: torch.Tensor | None = None,
        query_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.output_projection(
            self.features(node_states, pairs, structural_features, query_vectors)
        ).squeeze(-1)
        require_finite(logits, "link logits")
        return logits


def binary_link_prediction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    validate_tensor(logits, "link logits", torch.float32)
    if logits.ndim not in (1, 2):
        raise ValidationError("Link logits must be [C] or [B,C]")
    require_finite(logits, "link logits")
    if not isinstance(labels, torch.Tensor) or labels.dtype not in (
        torch.bool,
        torch.int64,
        torch.float32,
    ):
        raise ValidationError("Link labels must be binary tensors")
    validate_tensor(labels, "link labels", labels.dtype)
    if labels.shape != logits.shape or not ((labels == 0) | (labels == 1)).all().item():
        raise ValidationError("Binary labels must match link logits and contain only 0/1")
    bounded_float(positive_weight, "positive_weight")
    if positive_weight == 0:
        raise ValidationError("positive_weight must be positive")
    valid = _loss_mask(logits, mask)
    if not valid.any().item():
        return (logits * 0).sum()
    loss = F.binary_cross_entropy_with_logits(
        logits[valid].double(),
        labels[valid].double(),
        pos_weight=torch.tensor(positive_weight, dtype=torch.float64),
    ).float()
    require_finite(loss, "binary link loss")
    return loss


def _loss_mask(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    validate_tensor(mask, "loss mask", torch.bool)
    if mask.shape != values.shape:
        raise ValidationError("Loss mask shape differs")
    return mask


def link_ranking_loss(
    logits: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    margin: float = 1.0,
) -> torch.Tensor:
    if logits.ndim == 1:
        return pairwise_path_ranking_loss(
            logits[None],
            positives[None],
            negatives[None],
            mask=None if mask is None else mask[None],
            margin=margin,
        )
    return pairwise_path_ranking_loss(logits, positives, negatives, mask=mask, margin=margin)


def combined_link_loss(
    output: LinkerOutput,
    link_labels: torch.Tensor,
    relation_targets: torch.Tensor,
    *,
    config: LinkerConfig | None = None,
    mask: torch.Tensor | None = None,
    positive_indices: torch.Tensor | None = None,
    negative_indices: torch.Tensor | None = None,
    allow_unlinked_relations: bool = False,
) -> LinkLossOutput:
    from rutam.graph.relation_predictor import relation_classification_loss

    c = LinkerConfig() if config is None else config
    valid = (
        output.candidate_mask
        if mask is None
        else output.candidate_mask & _loss_mask(output.link_logits, mask)
    )
    link = binary_link_prediction_loss(
        output.link_logits, link_labels, mask=valid, positive_weight=c.positive_weight
    )
    relation = relation_classification_loss(
        output.relation_logits,
        relation_targets,
        link_labels=link_labels,
        mask=valid,
        allow_unlinked_relations=allow_unlinked_relations,
    )
    ranking = (output.link_logits * 0).sum()
    if positive_indices is not None or negative_indices is not None:
        if positive_indices is None or negative_indices is None:
            raise ValidationError("Both ranking index tensors are required")
        ranking = link_ranking_loss(
            output.link_logits,
            positive_indices,
            negative_indices,
            mask=valid,
            margin=c.ranking_margin,
        )
    total = (
        c.loss_weights.link * link
        + c.loss_weights.relation * relation
        + c.loss_weights.ranking * ranking
    )
    require_finite(total, "combined linker loss")
    return LinkLossOutput(total, link, relation, ranking)


def copy_expanded_parameters(source: nn.Module, destination: nn.Module) -> None:
    """Explicit vocabulary migration: preserve existing rows/columns, zero new feature columns."""
    old = source.state_dict()
    with torch.no_grad():
        for name, parameter in destination.state_dict().items():
            if name.endswith("relation_schema_signature"):
                continue
            if name not in old:
                continue
            previous = old[name]
            if previous.shape == parameter.shape:
                parameter.copy_(previous)
            elif (
                name.endswith("structural_projection.weight")
                and previous.shape[0] == parameter.shape[0]
            ):
                parameter.zero_()
                parameter[:, : previous.shape[1]].copy_(previous)
            elif (
                name in ("output_projection.weight", "output_projection.bias")
                and previous.shape[0] < parameter.shape[0]
            ):
                parameter[: previous.shape[0]].copy_(previous)
