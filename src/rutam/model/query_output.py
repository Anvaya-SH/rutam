"""Typed query, gate and selector integration outputs; no backbone dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rutam.graph.selector import GraphSelectorOutput


@dataclass(frozen=True, slots=True)
class QueryPoolingOutput:
    context: torch.Tensor
    valid_context: torch.Tensor
    attention_weights: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class QueryGateOutput:
    gate: torch.Tensor
    logits: torch.Tensor | None
    should_query: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class QueryHeadOutput:
    """Continuous queries/gates remain differentiable, including in eval mode.

    should_query is a discrete [B] decision only in eval mode. Raw gate logits
    are unmasked for stable supervision, or None when the gate is disabled.
    valid_context identifies real token contexts versus explicit zero fallbacks.
    """

    query_vectors: torch.Tensor
    query_gate: torch.Tensor
    query_gate_logits: torch.Tensor | None
    should_query: torch.Tensor | None
    token_attention_weights: torch.Tensor | None
    valid_context: torch.Tensor

    @property
    def query_confidence(self) -> torch.Tensor:
        """Uncalibrated alias of query_gate, not a second confidence estimate."""
        return self.query_gate


@dataclass(frozen=True, slots=True)
class QuerySelectionOutput:
    query: QueryHeadOutput
    selection: GraphSelectorOutput
    gated_memory_context: torch.Tensor | None

    @property
    def query_output(self) -> QueryHeadOutput:
        """Alias for query."""
        return self.query
