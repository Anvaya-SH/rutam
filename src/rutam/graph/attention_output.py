"""Typed outputs for relation-aware native graph attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class GraphAttentionOutput:
    """Output container for RUTAM graph attention computations."""

    node_states: torch.Tensor
    edge_attention: torch.Tensor | None = None
    graph_summary: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    node_id_map: list[str] | None = None
    subgraph_extracted: bool = False
