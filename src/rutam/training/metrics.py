"""Metric tracking and aggregation for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from rutam.training.memory_metrics import (
    binary_classification_metrics,
    mean_reciprocal_rank,
    multi_class_accuracy,
    recall_at_k,
)

__all__ = [
    "MetricTracker",
    "binary_classification_metrics",
    "mean_reciprocal_rank",
    "multi_class_accuracy",
    "recall_at_k",
]


@dataclass
class MetricTracker:
    """Accumulates step metrics and computes running averages."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, metrics: dict[str, float], count: int = 1) -> None:
        """Update tracked metric totals."""
        for k, v in metrics.items():
            self.totals[k] = self.totals.get(k, 0.0) + (v * count)
            self.counts[k] = self.counts.get(k, 0) + count

    def summary(self) -> dict[str, float]:
        """Compute average value for each tracked metric."""
        return {k: self.totals[k] / max(1, self.counts[k]) for k in self.totals}

    def reset(self) -> None:
        """Reset all tracked totals and counts."""
        self.totals.clear()
        self.counts.clear()
