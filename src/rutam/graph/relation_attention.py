"""Relation-aware graph attention combining edge attention, self-transform, and residuals."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rutam.config import GraphAttentionConfig
from rutam.exceptions import ValidationError
from rutam.graph.edge_attention import SparseRelationEdgeAttention
from rutam.graph.embeddings import require_finite


class RelationAttentionLayer(nn.Module):
    """Relation-aware graph attention layer with residual connections and normalization.

    Combines:
        1. Multi-head sparse relation-aware attention over graph edges.
        2. Self-node projection preserving identity and solitary node representations.
        3. Configurable residual addition and layer normalization.
    """

    def __init__(self, config: GraphAttentionConfig) -> None:
        super().__init__()
        if not isinstance(config, GraphAttentionConfig):
            raise ValidationError("RelationAttentionLayer requires a GraphAttentionConfig")
        self.config = config
        self.edge_attention = SparseRelationEdgeAttention(config)
        self.self_linear = nn.Linear(
            config.hidden_dim, config.hidden_dim, device="cpu", dtype=torch.float32
        )
        self.norm = (
            nn.LayerNorm(config.hidden_dim, device="cpu", dtype=torch.float32)
            if config.normalization == "layer_norm"
            else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        query: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Execute layer forward pass.

        Args:
            node_states: [N, H] float32 tensor of node representations.
            edge_index: [2, E] int64 edge tensor.
            edge_types: [E] int64 relation IDs.
            edge_weights: [E] float32 weights.
            query: Optional [Q] or [B, Q] neural query vector.
            edge_mask: Optional [E] boolean mask.

        Returns:
            updated_states: [N, H] (or [B, N, H] if query is batched).
            attention_weights: [E, heads] or [B, E, heads] or None.
            diagnostics: dictionary of diagnostic metrics.
        """
        if node_states.dim() == 3:
            n = node_states.shape[1]
            batch_size = node_states.shape[0]
            is_batched = True
        else:
            n = node_states.shape[0]
            is_batched = query is not None and query.dim() == 2
            batch_size = query.shape[0] if is_batched else 1

        if n == 0:
            shape = (
                (batch_size, 0, self.config.hidden_dim)
                if is_batched
                else (0, self.config.hidden_dim)
            )
            return node_states.new_zeros(shape), None, {"num_nodes": 0}

        attn_out, alpha, diagnostics = self.edge_attention(
            node_states, edge_index, edge_types, edge_weights, query=query, edge_mask=edge_mask
        )

        if node_states.dim() == 3:
            self_out = self.self_linear(node_states)
            base = node_states
        elif is_batched:
            self_out = self.self_linear(node_states).unsqueeze(0).expand(batch_size, -1, -1)
            base = node_states.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            self_out = self.self_linear(node_states)
            base = node_states

        update = self.dropout(self.activation(self_out + attn_out))

        if self.config.residual:
            update = base + update

        out = self.norm(update)
        require_finite(out, "relation attention layer output")
        return out, alpha, diagnostics
