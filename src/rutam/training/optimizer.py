"""Optimizer factory configuring parameter groups, weight decay, and optimization algorithm."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rutam.exceptions import ValidationError
from rutam.training.train_config import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer with decay/no-decay parameter groups for trainable weights."""
    trainable_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValidationError("Cannot create optimizer: model has no trainable parameters")

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    seen_params: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen_params:
            continue
        seen_params.add(param_id)

        # Biases and 1D normalization weights do not receive weight decay
        if param.ndim < 2 or "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": config.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if config.optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups, lr=config.learning_rate)
    elif config.optimizer_type == "sgd":
        return torch.optim.SGD(param_groups, lr=config.learning_rate, momentum=config.momentum)
    else:
        raise ValidationError(f"Unsupported optimizer type: {config.optimizer_type}")
