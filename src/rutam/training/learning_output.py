"""Typed learning output dataclasses for RUTAM training objectives."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class MemoryLearningOutput:
    """Output container for joint multi-objective memory training."""

    total_loss: torch.Tensor
    loss_components: dict[str, torch.Tensor] = field(default_factory=dict)
    valid_counts: dict[str, int] = field(default_factory=dict)
    loss_weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
