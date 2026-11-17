"""Trainable memory operation predictor classifying intended graph mutations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import MemoryWriteConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor


@dataclass(frozen=True, slots=True)
class OperationPredictorOutput:
    """Output of the operation predictor."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_operations: torch.Tensor


class MemoryOperationPredictor(nn.Module):
    """Predicts discrete operation logits over supported memory operations.

    Architecture: Pooled Hidden State -> LayerNorm -> Linear -> GELU -> Linear -> Operation Logits.
    Outputs operation logits [B, O] and softmax probabilities [B, O].
    """

    def __init__(self, config: MemoryWriteConfig) -> None:
        super().__init__()
        if not isinstance(config, MemoryWriteConfig):
            raise ValidationError("MemoryOperationPredictor requires a MemoryWriteConfig")
        self.config = config
        d = config.backbone_hidden_dim
        inter = config.projection_dim
        o = config.operation_count
        self.layer_norm = nn.LayerNorm(d, device="cpu", dtype=torch.float32)
        self.linear1 = nn.Linear(d, inter, device="cpu", dtype=torch.float32)
        self.linear2 = nn.Linear(inter, o, device="cpu", dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pooled_context: torch.Tensor) -> OperationPredictorOutput:
        validate_features(pooled_context, self.config.backbone_hidden_dim, "pooled_context")
        require_finite(pooled_context, "pooled_context")

        validate_tensor(self.linear1.weight, "operation predictor parameters", torch.float32)
        normed = self.layer_norm(pooled_context)
        require_finite(normed, "operation predictor normed state")

        hidden = self.dropout(F.gelu(self.linear1(normed)))
        require_finite(hidden, "operation predictor hidden state")

        logits = self.linear2(hidden)
        require_finite(logits, "operation predictor logits")

        probabilities = F.softmax(logits, dim=-1)
        predicted = logits.argmax(dim=-1)
        return OperationPredictorOutput(logits, probabilities, predicted)
