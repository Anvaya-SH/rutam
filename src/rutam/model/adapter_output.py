"""Typed output dataclasses and inspection reports for LFM3 backbone adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rutam.config import nonempty_string, positive_int
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor
from rutam.model.fusion_output import FusionOutput


@dataclass(frozen=True, slots=True)
class BackboneInspectionReport:
    """Diagnostic report documenting verified backbone layer boundaries and capabilities."""

    backbone_id: str
    hidden_dim: int
    vocab_size: int
    num_layers: int
    layer_type: str
    verified_integration: bool
    discovered_layers: tuple[str, ...] = ()
    parameter_counts: dict[str, int] = field(default_factory=dict)
    validation_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nonempty_string(self.backbone_id, "backbone_id")
        positive_int(self.hidden_dim, "hidden_dim")
        positive_int(self.vocab_size, "vocab_size")
        positive_int(self.num_layers, "num_layers", minimum=0)
        nonempty_string(self.layer_type, "layer_type")
        if not isinstance(self.verified_integration, bool):
            raise ValidationError("verified_integration must be a boolean")
        if not isinstance(self.discovered_layers, (tuple, list)):
            raise ValidationError("discovered_layers must be a sequence of strings")
        object.__setattr__(self, "discovered_layers", tuple(self.discovered_layers))
        if not isinstance(self.parameter_counts, dict):
            raise ValidationError("parameter_counts must be a dictionary")


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    """Output of RUTAMLFM3Adapter forward pass across backbone layers and graph fusion."""

    hidden_states: torch.Tensor
    logits: torch.Tensor | None = None
    fused_hidden_states: torch.Tensor | None = None
    fusion_outputs: tuple[FusionOutput, ...] = ()
    layer_hidden_states: tuple[torch.Tensor, ...] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_tensor(self.hidden_states, "hidden_states", torch.float32)
        if self.hidden_states.ndim != 3:
            shp = tuple(self.hidden_states.shape)
            raise ValidationError(f"hidden_states must have shape [B, T, D], got {shp}")

        if self.logits is not None:
            validate_tensor(self.logits, "logits", torch.float32)
            if self.logits.ndim != 3:
                shp = tuple(self.logits.shape)
                raise ValidationError(f"logits must have shape [B, T, V], got {shp}")

        if self.fused_hidden_states is not None:
            validate_tensor(self.fused_hidden_states, "fused_hidden_states", torch.float32)
            if self.fused_hidden_states.ndim != 3:
                shp = tuple(self.fused_hidden_states.shape)
                raise ValidationError(f"fused_hidden_states must have shape [B, T, D], got {shp}")

        if not isinstance(self.fusion_outputs, (tuple, list)):
            raise ValidationError("fusion_outputs must be a sequence of FusionOutput")
        object.__setattr__(self, "fusion_outputs", tuple(self.fusion_outputs))

        if not isinstance(self.metrics, dict):
            raise ValidationError("metrics must be a dictionary")
