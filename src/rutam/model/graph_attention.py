"""Multi-head cross-attention layer attending from LM hidden states over graph memory states."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import GraphFusionConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor
from rutam.model.fusion_projection import FusionProjections


class RUTAMGraphAttention(nn.Module):
    """Multi-head cross-attention querying external graph node representations."""

    def __init__(
        self,
        config: GraphFusionConfig,
        projections: FusionProjections | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, GraphFusionConfig):
            raise ValidationError("RUTAMGraphAttention requires a GraphFusionConfig")
        self.config = config
        self.projections = projections if projections is not None else FusionProjections(config)

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_dim // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        graph_states: torch.Tensor,
        *,
        graph_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute multi-head cross-attention context and optional attention weights."""
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        validate_tensor(graph_states, "graph_states", torch.float32)

        b, t, d = hidden_states.shape
        _, k, _ = graph_states.shape

        requested_att = (
            self.config.return_attention if return_attention is None else return_attention
        )

        # Handle empty graph cleanly without NaNs
        if k == 0 or (graph_mask is not None and not graph_mask.any()):
            zero_ctx = torch.zeros((b, t, d), dtype=torch.float32, device="cpu")
            zero_att = (
                torch.zeros((b, self.num_heads, t, k), dtype=torch.float32, device="cpu")
                if requested_att
                else None
            )
            return zero_ctx, zero_att

        if graph_mask is not None:
            validate_tensor(graph_mask, "graph_mask", torch.bool)
            if graph_mask.shape != (b, k):
                raise ValidationError(
                    f"graph_mask shape {tuple(graph_mask.shape)} != expected {(b, k)}"
                )

        # 1. Projections: Q from LM hidden states, K/V from graph states
        q = self.projections.project_queries(hidden_states)
        keys = self.projections.project_keys(graph_states)
        values = self.projections.project_values(graph_states)

        # 2. Reshape into multi-head format: [B, Heads, SeqLen, HeadDim]
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        scores = torch.matmul(q, keys.transpose(-2, -1)) * self.scale

        # 4. Masking for padded / inactive graph slots
        if graph_mask is not None:
            mask = graph_mask[:, None, None, :]  # [B, 1, 1, K]
            scores = scores.masked_fill(~mask, -1e4)
            attn_weights = F.softmax(scores, dim=-1)
            # Ensure rows where all graph slots were masked don't propagate NaNs
            attn_weights = torch.where(mask, attn_weights, torch.zeros_like(attn_weights))
        else:
            attn_weights = F.softmax(scores, dim=-1)

        dropped_weights = self.dropout(attn_weights)

        # 5. Attention context and output projection
        context = torch.matmul(dropped_weights, values)
        context = context.transpose(1, 2).contiguous().view(b, t, d)
        projected_context = self.projections.project_output(context)

        return projected_context, (attn_weights if requested_att else None)
