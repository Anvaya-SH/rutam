"""Masked token-context extraction with left/right/arbitrary padding support."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import QueryHeadConfig, boolean
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor
from rutam.graph.scoring import masked_softmax
from rutam.model.query_output import QueryPoolingOutput


class QueryPooling(nn.Module):
    """Pool [B,T,D] to [B,D]; optional weights describe each pooling rule."""

    def __init__(self, config: QueryHeadConfig) -> None:
        super().__init__()
        if not isinstance(config, QueryHeadConfig):
            raise ValidationError("QueryPooling requires a QueryHeadConfig")
        self.config = config
        self.attention_score = (
            nn.Linear(config.backbone_hidden_dim, 1, bias=False, device="cpu", dtype=torch.float32)
            if config.pooling == "attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attention: bool | None = None,
    ) -> QueryPoolingOutput:
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        if hidden_states.ndim != 3 or hidden_states.shape[2] != self.config.backbone_hidden_dim:
            raise ValidationError("hidden_states must have shape [B,T,backbone_hidden_dim]")
        require_finite(hidden_states, "hidden_states")
        b, t, _ = hidden_states.shape
        requested = self.config.return_attention if return_attention is None else return_attention
        boolean(requested, "return_attention")
        if attention_mask is None:
            mask = torch.ones((b, t), dtype=torch.bool, device="cpu")
        else:
            validate_tensor(attention_mask, "attention_mask", torch.bool)
            if attention_mask.shape != (b, t):
                raise ValidationError("attention_mask must have shape [B,T]")
            mask = attention_mask
        valid = mask.any(dim=1)
        if not valid.all().item() and self.config.empty_sequence_policy == "error":
            raise ValidationError(
                "Every sequence needs a valid token; choose empty_sequence_policy=zeros explicitly"
            )
        weights = None
        if t == 0:
            context = hidden_states.sum(dim=1)
            weights = hidden_states.new_zeros((b, 0))
        elif self.config.pooling == "last_valid":
            positions = torch.arange(t, dtype=torch.int64, device="cpu").expand(b, -1)
            last = torch.where(mask, positions, -1).amax(dim=1).clamp_min(0)
            context = hidden_states[torch.arange(b, device="cpu"), last]
            context = torch.where(valid[:, None], context, 0)
            if requested:
                weights = torch.nn.functional.one_hot(last, num_classes=t).float() * valid[:, None]
        else:
            if self.attention_score is not None:
                validate_tensor(self.attention_score.weight, "pooling parameters", torch.float32)
                weights = masked_softmax(self.attention_score(hidden_states).squeeze(-1), mask)
            else:
                weights = mask.float() / mask.sum(dim=1, keepdim=True).clamp_min(1)
            # Divide/normalize before summation; masked tokens contribute exactly zero.
            context = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        require_finite(context, "pooled query context")
        return QueryPoolingOutput(context, valid, weights if requested else None)
