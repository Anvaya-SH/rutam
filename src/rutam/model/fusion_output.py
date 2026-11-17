"""Typed output dataclasses and metrics reporting for RUTAMGraphFusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


@dataclass(frozen=True, slots=True)
class FusionOutput:
    """Output of a native graph fusion pass integrating graph memory into LM hidden states."""

    fused_hidden_states: torch.Tensor
    fusion_gate: torch.Tensor
    attention_weights: torch.Tensor | None = None
    utilization_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_tensor(self.fused_hidden_states, "fused_hidden_states", torch.float32)
        if self.fused_hidden_states.ndim != 3:
            shp = tuple(self.fused_hidden_states.shape)
            raise ValidationError(f"fused_hidden_states must have shape [B, T, D], got {shp}")

        validate_tensor(self.fusion_gate, "fusion_gate", torch.float32)
        if self.fusion_gate.ndim not in (2, 3):
            shp = tuple(self.fusion_gate.shape)
            raise ValidationError(
                f"fusion_gate must have shape [B, 1], [B, T, 1], or [B, 1, 1], got {shp}"
            )

        if self.attention_weights is not None:
            validate_tensor(self.attention_weights, "attention_weights", torch.float32)
            if self.attention_weights.ndim != 4:
                shp = tuple(self.attention_weights.shape)
                raise ValidationError(
                    f"attention_weights must have shape [B, Heads, T, K], got {shp}"
                )

        if not isinstance(self.utilization_metrics, dict):
            raise ValidationError("utilization_metrics must be a dictionary")
