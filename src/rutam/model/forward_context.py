"""Per-forward pass context, execution tracking, and graph memory diagnostic flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rutam.config import positive_int
from rutam.exceptions import ValidationError


@dataclass(slots=True)
class ForwardContext:
    """Carries dynamic per-forward execution state, caching metadata, and diagnostics."""

    batch_size: int
    sequence_length: int
    graph_available: bool = True
    graph_version: int = 0
    cache_hit: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_int(self.batch_size, "batch_size")
        positive_int(self.sequence_length, "sequence_length")
        positive_int(self.graph_version, "graph_version", minimum=0)
        if not isinstance(self.graph_available, bool):
            raise ValidationError("graph_available must be a boolean")
        if not isinstance(self.cache_hit, bool):
            raise ValidationError("cache_hit must be a boolean")
        if not isinstance(self.diagnostics, dict):
            raise ValidationError("diagnostics must be a dictionary")
        if not isinstance(self.timings, dict):
            raise ValidationError("timings must be a dictionary")

    def record_diagnostic(self, key: str, value: Any) -> None:
        """Record diagnostic observation for analysis."""
        self.diagnostics[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert context summary to dictionary."""
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "graph_available": self.graph_available,
            "graph_version": self.graph_version,
            "cache_hit": self.cache_hit,
            "diagnostics": dict(self.diagnostics),
            "timings": dict(self.timings),
        }
