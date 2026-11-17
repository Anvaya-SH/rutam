"""Stability-preserving residual gating for recurrent reasoning updates."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import RecurrentConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


class RecurrentResidualGate(nn.Module):
    """Residual gating with small-scale parameter initialization and LayerNorm.

    Ensures numerical stability and identity-preserving behavior across up to 8+
    recurrent reasoning iterations, preventing gradient explosion/vanishing or
    activation drift without NaN/Inf.
    """

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        if not isinstance(config, RecurrentConfig):
            raise ValidationError("RecurrentResidualGate requires a RecurrentConfig")
        self.config = config

        d = config.hidden_dim
        out_dim = d if config.gate_type == "elementwise" else 1

        self.gate_proj = nn.Sequential(
            nn.Linear(2 * d, d, bias=True),
            nn.GELU(),
            nn.Linear(d, out_dim, bias=True),
        )

        # Residual scale initialized near zero (default 0.001) for identity preservation
        self.residual_scale = nn.Parameter(
            torch.tensor(config.residual_scale_init, dtype=torch.float32)
        )

        # Normalization layer stabilizing deep recurrence
        self.norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def forward(
        self,
        previous_state: torch.Tensor,
        update_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gated residual update between previous recurrent state and candidate.

        Args:
            previous_state: Hidden representation from previous iteration [B, T, D].
            update_candidate: Proposed refinement candidate [B, T, D].

        Returns:
            Tuple of (next_state [B, T, D], gate [B, T, out_dim]).
        """
        validate_tensor(previous_state, "previous_state", torch.float32)
        validate_tensor(update_candidate, "update_candidate", torch.float32)

        if previous_state.shape != update_candidate.shape:
            raise ValidationError(
                f"previous_state shape {tuple(previous_state.shape)} != "
                f"update_candidate shape {tuple(update_candidate.shape)}"
            )

        cat_states = torch.cat([previous_state, update_candidate], dim=-1)  # [B, T, 2D]
        gate = torch.sigmoid(self.gate_proj(cat_states))  # [B, T, out_dim]

        # Scaled residual update
        scaled_update = self.residual_scale * gate * update_candidate
        next_state = self.norm(previous_state + scaled_update)

        return next_state, gate
