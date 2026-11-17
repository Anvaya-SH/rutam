"""Trainable write decision gate predicting write utility and binary decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import MemoryWriteConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor


def availability_mask(mask: torch.Tensor | None, batch_size: int) -> torch.Tensor:
    """Validate or construct the boolean memory availability mask [B]."""
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device="cpu")
    validate_tensor(mask, "memory_available", torch.bool)
    if mask.shape != (batch_size,):
        raise ValidationError("memory_available must have shape [B]")
    return mask


@dataclass(frozen=True, slots=True)
class WriteGateOutput:
    """Output of the memory write gate."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    decisions: torch.Tensor | None


class MemoryWriteGate(nn.Module):
    """Predicts continuous write utility logits and binary write decisions.

    Architecture: Pooled Hidden State -> LayerNorm -> Linear -> GELU -> Linear -> Write Logit.
    In inference mode, decisions are determined by write_threshold and availability.
    """

    def __init__(self, config: MemoryWriteConfig) -> None:
        super().__init__()
        if not isinstance(config, MemoryWriteConfig):
            raise ValidationError("MemoryWriteGate requires a MemoryWriteConfig")
        self.config = config
        d = config.backbone_hidden_dim
        inter = config.projection_dim
        self.layer_norm = nn.LayerNorm(d, device="cpu", dtype=torch.float32)
        self.linear1 = nn.Linear(d, inter, device="cpu", dtype=torch.float32)
        self.linear2 = nn.Linear(inter, 1, device="cpu", dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        pooled_context: torch.Tensor,
        memory_available: torch.Tensor | None = None,
    ) -> WriteGateOutput:
        validate_features(pooled_context, self.config.backbone_hidden_dim, "pooled_context")
        require_finite(pooled_context, "pooled_context")
        available = availability_mask(memory_available, pooled_context.shape[0])

        validate_tensor(self.linear1.weight, "write gate parameters", torch.float32)
        normed = self.layer_norm(pooled_context)
        require_finite(normed, "write gate normed state")

        hidden = self.dropout(F.gelu(self.linear1(normed)))
        require_finite(hidden, "write gate hidden state")

        logits = self.linear2(hidden)
        require_finite(logits, "write gate logits")

        probabilities = torch.sigmoid(logits) * available[:, None]
        decisions = (
            None
            if self.training
            else (available & (probabilities[:, 0] >= self.config.write_threshold))
        )
        return WriteGateOutput(logits, probabilities, decisions)
