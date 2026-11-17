"""Relation-aware and query-conditioned attention biases for graph attention."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import GraphAttentionConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor


class RelationAttentionBias(nn.Module):
    """Computes relation-specific score biases, edge scaling, and query-relation conditioning."""

    def __init__(self, config: GraphAttentionConfig) -> None:
        super().__init__()
        if not isinstance(config, GraphAttentionConfig):
            raise ValidationError("RelationAttentionBias requires a GraphAttentionConfig")
        self.config = config
        self.num_heads = config.num_heads
        self.num_relation_types = config.num_relation_types

        # Relation score bias: [R, num_heads]
        if config.relation_bias:
            self.relation_score_bias = nn.Parameter(
                torch.zeros((config.num_relation_types, config.num_heads), dtype=torch.float32)
            )
            nn.init.normal_(self.relation_score_bias, mean=0.0, std=0.02)
        else:
            self.register_parameter("relation_score_bias", None)

        # Edge weight scale: learned positive scale per head
        self.edge_weight_scale = nn.Parameter(torch.ones((config.num_heads,), dtype=torch.float32))

        # Query conditioning projection: maps [Q] -> [R, num_heads]
        if config.query_conditioning:
            self.query_relation_proj = nn.Linear(
                config.query_dim,
                config.num_relation_types * config.num_heads,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            )
            nn.init.normal_(self.query_relation_proj.weight, mean=0.0, std=0.02)
        else:
            self.query_relation_proj = None

    def compute_bias(
        self,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute total attention score bias for edges.

        Args:
            edge_types: [E] int64 tensor of relationship type IDs.
            edge_weights: [E] float32 tensor of edge weights.
            query: Optional [Q] or [B, Q] float32 tensor of neural query vectors.

        Returns:
            [E, num_heads] tensor (when query is None or [Q]) or
            [B, E, num_heads] tensor (when query is [B, Q] with B > 1).
        """
        validate_tensor(edge_types, "edge_types", torch.int64)
        validate_tensor(edge_weights, "edge_weights", torch.float32)
        if edge_types.ndim != 1:
            raise ValidationError("edge_types must have shape [E]")
        if edge_weights.ndim != 1:
            raise ValidationError("edge_weights must have shape [E]")

        e = edge_types.shape[0]

        # Base structural bias from edge weights: [E, num_heads]
        # Using softplus/log-linear scaling for numerical stability
        weight_factor = torch.clamp_min(edge_weights, 0.0).unsqueeze(-1)  # [E, 1]
        weight_bias = weight_factor * self.edge_weight_scale.unsqueeze(0)  # [E, num_heads]

        # Relation score bias: [E, num_heads]
        if self.relation_score_bias is not None and e > 0:
            rel_bias = self.relation_score_bias.index_select(0, edge_types)
            total_bias = weight_bias + rel_bias
        else:
            total_bias = weight_bias

        if query is None or self.query_relation_proj is None or e == 0:
            require_finite(total_bias, "attention score bias")
            return total_bias

        # Query-relation conditioning
        if query.dim() == 1:
            q_bias = self._query_bias_1d(query, edge_types)  # [E, num_heads]
            total_bias = total_bias + q_bias
            require_finite(total_bias, "query-conditioned attention score bias")
            return total_bias
        elif query.dim() == 2:
            b = query.shape[0]
            # [B, R * H] -> [B, R, H]
            q_rel = self.query_relation_proj(query).view(b, self.num_relation_types, self.num_heads)
            # Select for each edge: [B, E, H]
            q_bias = q_rel[:, edge_types, :]
            total_bias = total_bias.unsqueeze(0) + q_bias  # [B, E, H]
            require_finite(total_bias, "batched query-conditioned attention score bias")
            return total_bias
        else:
            raise ValidationError(f"query must be 1D or 2D, got shape {tuple(query.shape)}")

    def _query_bias_1d(self, query: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        q_rel = self.query_relation_proj(query).view(self.num_relation_types, self.num_heads)
        return q_rel.index_select(0, edge_types)
