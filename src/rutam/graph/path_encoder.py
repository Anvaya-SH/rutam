"""Ordered relation/direction/position-aware GRU encoding of bounded tensor paths."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import TraversalConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import (
    RelationEmbedding,
    require_finite,
    validate_features,
    validate_tensor,
)
from rutam.graph.traversal_output import PathCandidates


class RUTAMPathEncoder(nn.Module):
    def __init__(
        self,
        config: TraversalConfig | None = None,
        *,
        relation_embedding: RelationEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.config = TraversalConfig() if config is None else config
        if not isinstance(self.config, TraversalConfig):
            raise ValidationError("Path encoder requires TraversalConfig")
        c = self.config
        h, d = c.hidden_dim, c.path_hidden_dim
        self.node_projection = nn.Linear(h, d, device="cpu", dtype=torch.float32)
        self.relation_embedding = (
            relation_embedding
            if relation_embedding is not None
            else RelationEmbedding(c.num_relation_types, h)
        )
        if not isinstance(self.relation_embedding, RelationEmbedding) or (
            self.relation_embedding.num_embeddings != c.num_relation_types
            or self.relation_embedding.embedding_dim != h
        ):
            raise ValidationError(
                "Shared relation embeddings must match traversal vocabulary/hidden dimension"
            )
        self.relation_projection = nn.Linear(h, d, bias=False, device="cpu", dtype=torch.float32)
        self.hop_embedding = nn.Embedding(c.max_hops + 1, d, device="cpu", dtype=torch.float32)
        self.direction_embedding = nn.Embedding(3, d, device="cpu", dtype=torch.float32)
        self.weight_projection = (
            nn.Linear(1, d, bias=False, device="cpu", dtype=torch.float32)
            if c.include_edge_weights
            else None
        )
        self.gru = nn.GRUCell(d, d, device="cpu", dtype=torch.float32)
        self.output_projection = (
            nn.Linear(d, h, bias=False, device="cpu", dtype=torch.float32)
            if d != h
            else nn.Identity()
        )
        self.dropout = nn.Dropout(c.dropout)

    def forward(
        self, node_states: torch.Tensor, paths: PathCandidates, edge_weights: torch.Tensor
    ) -> torch.Tensor:
        c = self.config
        validate_features(node_states, c.hidden_dim, "node_states")
        validate_tensor(self.node_projection.weight, "path encoder parameters", torch.float32)
        validate_tensor(edge_weights, "edge_weights", torch.float32)
        if edge_weights.ndim != 1 or (edge_weights < 0).any().item():
            raise ValidationError("edge_weights must be nonnegative [E]")
        require_finite(edge_weights, "edge_weights")
        self._validate_paths(paths, node_states.shape[0], edge_weights.numel())
        b, p, length = paths.node_indices.shape
        if p > self.config.max_candidate_paths:
            raise ValidationError("Candidate path count exceeds max_candidate_paths")
        if b * p * length * max(c.hidden_dim, c.path_hidden_dim) > c.max_path_elements:
            raise ValidationError("Path tensor workspace exceeds max_path_elements")
        if node_states.shape[0]:
            nodes = node_states[paths.node_indices.clamp_min(0)]
            nodes = torch.where(paths.node_mask[..., None], nodes, 0)
        else:
            nodes = node_states.sum(0).view(1, 1, 1, c.hidden_dim).expand(b, p, length, -1)
        projected = self.node_projection(nodes)
        hidden = projected.new_zeros((b * p, c.path_hidden_dim))
        for hop in range(length):
            token = projected[:, :, hop] + self.hop_embedding.weight[hop]
            if hop:
                relations = self.relation_embedding(
                    paths.relation_types[:, :, hop - 1].clamp_min(0).reshape(-1)
                )
                token = token + self.relation_projection(relations).reshape(b, p, c.path_hidden_dim)
                direction = torch.where(paths.directions[:, :, hop - 1] == 1, 1, 2)
                token = token + self.direction_embedding(direction)
                if self.weight_projection is not None:
                    weights = (
                        edge_weights[paths.edge_indices[:, :, hop - 1].clamp_min(0)]
                        if edge_weights.numel()
                        else edge_weights.sum().expand(b, p)
                    )
                    weights = torch.where(paths.edge_mask[:, :, hop - 1], weights, 0)
                    token = token + self.weight_projection(weights.log1p()[..., None])
            require_finite(token, "path token features")
            update = self.gru(self.dropout(token).reshape(b * p, c.path_hidden_dim), hidden)
            hidden = torch.where(paths.node_mask[:, :, hop].reshape(-1, 1), update, hidden)
        result = self.output_projection(hidden).reshape(b, p, c.hidden_dim)
        require_finite(result, "path representations")
        return result

    def _validate_paths(self, paths: PathCandidates, n: int, e: int) -> None:
        if not isinstance(paths, PathCandidates):
            raise ValidationError("paths must be PathCandidates")
        for name in ("node_indices", "edge_indices", "relation_types", "directions"):
            validate_tensor(getattr(paths, name), name, torch.int64)
        for name in ("node_mask", "edge_mask", "path_mask"):
            validate_tensor(getattr(paths, name), name, torch.bool)
        if paths.node_indices.ndim != 3:
            raise ValidationError("path node indices must have shape [B,P,L+1]")
        b, p, length = paths.node_indices.shape
        if (
            length != self.config.max_hops + 1
            or paths.node_mask.shape != (b, p, length)
            or paths.path_mask.shape != (b, p)
        ):
            raise ValidationError("Path node/mask shapes do not match configured hop count")
        for name in ("edge_indices", "relation_types", "directions", "edge_mask"):
            if getattr(paths, name).shape != (b, p, length - 1):
                raise ValidationError(f"{name} must have shape [B,P,L]")
        if (
            not torch.equal(paths.node_mask[:, :, 0], paths.path_mask)
            or (paths.node_mask[:, :, 1:] & ~paths.node_mask[:, :, :-1]).any().item()
            or not torch.equal(paths.edge_mask, paths.node_mask[:, :, 1:])
        ):
            raise ValidationError(
                "Paths require contiguous prefix masks and one edge per non-root node"
            )
        for indices, mask, bound in (
            (paths.node_indices, paths.node_mask, n),
            (paths.edge_indices, paths.edge_mask, e),
            (paths.relation_types, paths.edge_mask, self.config.num_relation_types),
        ):
            if ((indices[mask] < 0) | (indices[mask] >= bound)).any().item() or (
                indices[~mask] != -1
            ).any().item():
                raise ValidationError("Invalid path reference or non--1 padding")
        if (
            not (
                (paths.directions[paths.edge_mask] == 1) | (paths.directions[paths.edge_mask] == -1)
            )
            .all()
            .item()
        ):
            raise ValidationError("Valid path directions must be +1 or -1")
        if (paths.directions[~paths.edge_mask] != 0).any().item():
            raise ValidationError("Padded directions must be zero")
