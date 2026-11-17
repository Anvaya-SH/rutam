"""Learning rate scheduler factory supporting constant, linear, and cosine schedules."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from rutam.training.train_config import TrainingConfig


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
) -> LRScheduler | None:
    """Create learning rate scheduler according to TrainingConfig."""
    if total_steps <= 0:
        return None

    warmup_steps = max(0, config.warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if config.lr_scheduler_type == "constant":
            return 1.0
        elif config.lr_scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        elif config.lr_scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    if config.lr_scheduler_type == "constant" and warmup_steps == 0:
        return None

    return LambdaLR(optimizer, lr_lambda)
