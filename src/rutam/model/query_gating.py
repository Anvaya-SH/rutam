"""Trainable graph-use decision and stable supervised binary objective."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import QueryHeadConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor
from rutam.model.query_output import QueryGateOutput


def availability_mask(mask: torch.Tensor | None, batch_size: int) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device="cpu")
    validate_tensor(mask, "memory_available", torch.bool)
    if mask.shape != (batch_size,):
        raise ValidationError("memory_available must have shape [B]")
    return mask


class QueryDecisionGate(nn.Module):
    """Continuous sigmoid utility score; hard threshold only when .eval() is set."""

    def __init__(self, config: QueryHeadConfig) -> None:
        super().__init__()
        if not isinstance(config, QueryHeadConfig):
            raise ValidationError("QueryDecisionGate requires a QueryHeadConfig")
        self.config = config
        if config.query_gate_enabled:
            self.input_projection = nn.Linear(
                config.backbone_hidden_dim + int(config.gate_use_availability),
                config.intermediate_dim,
                device="cpu",
                dtype=torch.float32,
            )
            self.output_projection = nn.Linear(
                config.intermediate_dim, 1, device="cpu", dtype=torch.float32
            )
        else:
            self.input_projection = None
            self.output_projection = None

    def forward(
        self,
        context: torch.Tensor,
        memory_available: torch.Tensor | None = None,
    ) -> QueryGateOutput:
        validate_features(context, self.config.backbone_hidden_dim, "query context")
        available = availability_mask(memory_available, context.shape[0])
        logits = None
        if self.input_projection is not None:
            validate_tensor(self.input_projection.weight, "gate parameters", torch.float32)
            features = (
                torch.cat((context, available[:, None].float()), dim=1)
                if (self.config.gate_use_availability)
                else context
            )
            hidden = F.gelu(self.input_projection(features))
            require_finite(hidden, "gate hidden states")
            logits = self.output_projection(hidden)
            require_finite(logits, "query gate logits")
            gate = torch.sigmoid(logits) * available[:, None]
        else:
            gate = available[:, None].float()
        decision = (
            None if self.training else (available & (gate[:, 0] >= self.config.gate_threshold))
        )
        return QueryGateOutput(gate, logits, decision)


def query_decision_loss(
    gate_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean BCE-with-logits on explicitly included samples; labels are exactly 0/1.

    Supply raw, unmasked logits. Forced gate masking does not automatically mask
    this objective. Pass mask to explicitly exclude fallback/unavailable samples.
    """
    validate_tensor(gate_logits, "gate_logits", torch.float32)
    if gate_logits.ndim != 2 or gate_logits.shape[1] != 1:
        raise ValidationError("gate_logits must have shape [B,1]")
    require_finite(gate_logits, "gate_logits")
    if not isinstance(labels, torch.Tensor) or labels.dtype not in (
        torch.bool,
        torch.int64,
        torch.float32,
    ):
        raise ValidationError("Decision labels must be boolean, int64 or float32")
    validate_tensor(labels, "labels", labels.dtype)
    if labels.shape == (gate_logits.shape[0],):
        labels = labels[:, None]
    if labels.shape != gate_logits.shape or not ((labels == 0) | (labels == 1)).all().item():
        raise ValidationError("Decision labels must be binary [B] or [B,1]")
    active = availability_mask(mask, gate_logits.shape[0])
    if not active.any().item():
        return (gate_logits * 0).sum()
    loss = F.binary_cross_entropy_with_logits(
        gate_logits[active].double(), labels[active].double()
    ).float()
    require_finite(loss, "query decision loss")
    return loss
