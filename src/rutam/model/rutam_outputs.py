"""Typed output structures for RUTAM unified graph-aware causal language models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from rutam.config import positive_int
from rutam.exceptions import ValidationError
from rutam.graph.traversal_output import TraversalOutput
from rutam.model.fusion_output import FusionOutput
from rutam.model.query_output import QuerySelectionOutput
from rutam.model.recurrent_output import RecurrentOutput

if TYPE_CHECKING:
    from rutam.memory.write_proposal import MemoryProposal, MemoryWriteOutput


@dataclass(frozen=True, slots=True)
class RUTAMCausalLMOutput:
    """Unified output from RUTAMForCausalLM forward pass.

    Aggregates language model output logits and loss with intermediate neural graph
    selection, multi-hop path traversal, graph fusion, recurrent reasoning, and write proposals.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    graph_selection_output: QuerySelectionOutput | None = None
    traversal_output: TraversalOutput | None = None
    fusion_output: FusionOutput | None = None
    recurrent_output: RecurrentOutput | None = None
    write_output: MemoryWriteOutput | None = None
    write_proposals: tuple[MemoryProposal, ...] = ()
    graph_version: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.logits, torch.Tensor)
            or self.logits.device.type != "cpu"
            or self.logits.dtype != torch.float32
        ):
            raise ValidationError("logits must be a dense CPU torch.float32 tensor")
        if self.logits.ndim != 3:
            shp = tuple(self.logits.shape)
            raise ValidationError(f"logits must have shape [B, T, V], got {shp}")

        if self.loss is not None:
            if (
                not isinstance(self.loss, torch.Tensor)
                or self.loss.device.type != "cpu"
                or self.loss.dtype != torch.float32
            ):
                raise ValidationError("loss must be a CPU torch.float32 tensor")

        positive_int(self.graph_version, "graph_version", minimum=0)

        if not isinstance(self.write_proposals, (tuple, list)):
            raise ValidationError("write_proposals must be a sequence of MemoryProposal")
        object.__setattr__(self, "write_proposals", tuple(self.write_proposals))

        if not isinstance(self.diagnostics, dict):
            raise ValidationError("diagnostics must be a dictionary")
