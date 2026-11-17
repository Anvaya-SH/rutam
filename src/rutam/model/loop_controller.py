"""Dynamic loop controller for adaptive computation time in recurrent reasoning."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import RecurrentConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


class DynamicLoopController(nn.Module):
    """Predicts dynamic continue/halt logits across recurrent reasoning loops.

    Enforces budget constraints:
    - Never halts before `min_loops` iterations have completed.
    - Always halts when `max_loops` budget is reached.
    - Evaluates continue probabilities against `halting_threshold` during intermediate loops.
    """

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        if not isinstance(config, RecurrentConfig):
            raise ValidationError("DynamicLoopController requires a RecurrentConfig")
        self.config = config

        d = config.hidden_dim
        h = config.controller_hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(d, h, bias=True),
            nn.GELU(),
            nn.Linear(h, 1, bias=True),
        )

    def pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool token representations across sequence dimension [B, T, D] -> [B, D]."""
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        if hidden_states.ndim != 3:
            raise ValidationError("hidden_states must have shape [B, T, D]")

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValidationError("attention_mask must be a torch.Tensor")
            if attention_mask.dtype == torch.bool:
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            elif attention_mask.dtype == torch.float32:
                mask = attention_mask.unsqueeze(-1) if attention_mask.ndim == 2 else attention_mask
            else:
                raise ValidationError("attention_mask must be bool or float32")
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden_states * mask).sum(dim=1) / counts  # [B, D]
        else:
            pooled = hidden_states.mean(dim=1)  # [B, D]

        return pooled

    def forward(
        self,
        hidden_states: torch.Tensor,
        loop_idx: int,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Predict continue logit, halt probability, and continuation decision.

        Args:
            hidden_states: Hidden representations after current loop [B, T, D].
            loop_idx: 0-indexed loop index currently completed (0 to max_loops - 1).
            attention_mask: Optional boolean or float attention mask [B, T].

        Returns:
            Tuple of:
            - continue_logits: Raw MLP logits [B, 1].
            - halt_probs: Halting probabilities in [0, 1] [B, 1].
            - should_continue: Boolean decision whether to execute the next loop.
        """
        if type(loop_idx) is not int:
            raise ValidationError(f"loop_idx must be an integer, got {type(loop_idx).__name__}")
        if not (0 <= loop_idx < self.config.max_loops):
            raise ValidationError(
                f"loop_idx {loop_idx} out of range [0, {self.config.max_loops})"
            )

        pooled = self.pool_sequence(hidden_states, attention_mask=attention_mask)
        continue_logits = self.mlp(pooled)  # [B, 1]
        continue_probs = torch.sigmoid(continue_logits)
        halt_probs = 1.0 - continue_probs

        completed_loops = loop_idx + 1

        # Budget constraint 1: Must run at least min_loops
        if completed_loops < self.config.min_loops:
            should_continue = True
        # Budget constraint 2: Must not exceed max_loops
        elif completed_loops >= self.config.max_loops:
            should_continue = False
        else:
            # Intermediate loop: check against halting threshold
            should_continue = bool(
                (continue_probs >= self.config.halting_threshold).any().item()
            )

        return continue_logits, halt_probs, should_continue

    def predict_continue(
        self,
        hidden_states: torch.Tensor,
        loop_idx: int,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Explicit alias for controller forward pass."""
        return self.forward(hidden_states, loop_idx, attention_mask=attention_mask)
