"""Loss balancing mechanisms: fixed configuration weights and learned homoscedastic uncertainty."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from rutam.config import MemoryLearningConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite


class FixedLossBalancer(nn.Module):
    """Combines multi-objective losses using fixed configured loss weights."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        super().__init__()
        self.weights = dict(weights)
        for k, v in self.weights.items():
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or v < 0.0
                or not math.isfinite(v)
            ):
                raise ValidationError(f"Fixed weight for {k} must be a nonnegative finite number")

    def forward(
        self, loss_components: Mapping[str, torch.Tensor], valid_counts: Mapping[str, int]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """Combine valid losses.

        Returns:
            total_loss: scalar tensor.
            scaled_components: dict of individual weighted loss tensors.
            effective_weights: dict of float weights applied.
        """
        scaled: dict[str, torch.Tensor] = {}
        effective_weights: dict[str, float] = {}
        total = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        has_device = False

        for name, loss in loss_components.items():
            count = valid_counts.get(name, 0)
            if count > 0 and loss is not None:
                if not has_device:
                    total = total.to(loss.device)
                    has_device = True
                w = self.weights.get(name, 1.0)
                component = loss * w
                scaled[name] = component
                effective_weights[name] = float(w)
                total = total + component

        require_finite(total, "balanced total loss")
        return total, scaled, effective_weights


class UncertaintyLossBalancer(nn.Module):
    """Learned multi-task homoscedastic uncertainty loss balancing (Kendall & Gal).

    Formulation:
        L_total = sum_i (0.5 * exp(-s_i) * L_i + 0.5 * s_i)
    where s_i = log(sigma_i^2) is a learnable parameter.
    """

    def __init__(self, task_names: list[str]) -> None:
        super().__init__()
        self.task_names = list(task_names)
        # Initialize log-variances to 0 (sigma^2 = 1)
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros((), dtype=torch.float32)) for name in self.task_names}
        )

    def forward(
        self, loss_components: Mapping[str, torch.Tensor], valid_counts: Mapping[str, int]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """Compute uncertainty-weighted total loss."""
        scaled: dict[str, torch.Tensor] = {}
        effective_weights: dict[str, float] = {}
        total = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        has_device = False

        for name, loss in loss_components.items():
            count = valid_counts.get(name, 0)
            if count > 0 and loss is not None and name in self.log_vars:
                if not has_device:
                    total = total.to(loss.device)
                    has_device = True
                s = self.log_vars[name]
                precision = torch.exp(-s)
                # Weighted component: 0.5 * exp(-s) * loss + 0.5 * s
                component = 0.5 * precision * loss + 0.5 * s
                scaled[name] = component
                effective_weights[name] = float((0.5 * precision).item())
                total = total + component

        require_finite(total, "uncertainty total loss")
        return total, scaled, effective_weights


def create_loss_balancer(config: MemoryLearningConfig) -> nn.Module:
    """Instantiate appropriate loss balancer according to configuration."""
    weights = {
        "write_decision": config.write_loss_weight,
        "operation": config.operation_loss_weight,
        "retrieval": config.retrieval_loss_weight,
        "path": config.path_loss_weight,
        "link": config.link_loss_weight,
        "embedding": config.embedding_loss_weight,
        "query_gate": config.query_gate_loss_weight,
        "update_target": config.update_target_loss_weight,
    }
    if config.use_fixed_loss_weights:
        return FixedLossBalancer(weights)
    return UncertaintyLossBalancer(list(weights.keys()))
