"""Learned gating mechanism balancing LM hidden states against graph context."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import GraphFusionConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


class RUTAMFusionGate(nn.Module):
    """Dynamic gating network balancing backbone representations and graph context."""

    def __init__(self, config: GraphFusionConfig) -> None:
        super().__init__()
        if not isinstance(config, GraphFusionConfig):
            raise ValidationError("RUTAMFusionGate requires a GraphFusionConfig")
        self.config = config

        d = config.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, d, bias=True),
            nn.GELU(),
            nn.Linear(d, 1, bias=True),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        projected_context: torch.Tensor,
        *,
        query_gate: torch.Tensor | None = None,
        token_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token-level or sequence-level fusion gate in [0, 1]."""
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        validate_tensor(projected_context, "projected_context", torch.float32)

        if hidden_states.shape != projected_context.shape:
            raise ValidationError(
                f"hidden_states shape {tuple(hidden_states.shape)} != "
                f"projected_context shape {tuple(projected_context.shape)}"
            )

        b, t, d = hidden_states.shape
        cat_input = torch.cat([hidden_states, projected_context], dim=-1)  # [B, T, 2D]
        gate_logits = self.mlp(cat_input)  # [B, T, 1]

        if self.config.gate_mode == "sequence":
            if token_attention_mask is not None:
                validate_tensor(token_attention_mask, "token_attention_mask", torch.bool)
                mask = token_attention_mask.unsqueeze(-1)  # [B, T, 1]
                masked_logits = gate_logits.masked_fill(~mask, 0.0)
                counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                gate_logits = masked_logits.sum(dim=1, keepdim=True) / counts  # [B, 1, 1]
            else:
                gate_logits = gate_logits.mean(dim=1, keepdim=True)  # [B, 1, 1]

        gate = torch.sigmoid(gate_logits)

        # Condition on optional query_gate
        if query_gate is not None:
            validate_tensor(query_gate, "query_gate", torch.float32)
            if query_gate.ndim == 2:
                q_gate = query_gate.unsqueeze(1)  # [B, 1, 1]
            elif query_gate.ndim == 3:
                q_gate = query_gate
            else:
                raise ValidationError("query_gate must have shape [B, 1] or [B, 1, 1]")
            gate = gate * q_gate

        return gate
