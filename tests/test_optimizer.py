"""Unit tests for optimizer and learning rate scheduler creation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rutam.exceptions import ValidationError
from rutam.training.optimizer import create_optimizer
from rutam.training.scheduler import create_lr_scheduler
from rutam.training.train_config import TrainingConfig


class DummyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8, bias=True)
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(x))


def test_optimizer_parameter_groups() -> None:
    """Verify decay and no_decay parameter partitioning without duplicates."""
    mod = DummyModule()
    config = TrainingConfig(learning_rate=0.01, weight_decay=0.05, optimizer_type="adamw")
    opt = create_optimizer(mod, config)

    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2

    decay_group = opt.param_groups[0]
    no_decay_group = opt.param_groups[1]

    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0

    # Ensure no duplicate parameters
    p_ids_decay = {id(p) for p in decay_group["params"]}
    p_ids_nodecay = {id(p) for p in no_decay_group["params"]}
    assert len(p_ids_decay & p_ids_nodecay) == 0

    # Linear weight in decay group
    assert id(mod.fc.weight) in p_ids_decay
    # Bias and norm in no-decay group
    assert id(mod.fc.bias) in p_ids_nodecay
    assert id(mod.norm.weight) in p_ids_nodecay


def test_optimizer_sgd() -> None:
    """Verify SGD optimizer creation with momentum."""
    mod = DummyModule()
    config = TrainingConfig(optimizer_type="sgd", momentum=0.85)
    opt = create_optimizer(mod, config)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["momentum"] == 0.85


def test_optimizer_no_trainable_params_error() -> None:
    """Verify error when model has no trainable parameters."""
    mod = DummyModule()
    for p in mod.parameters():
        p.requires_grad = False

    config = TrainingConfig()
    with pytest.raises(ValidationError, match="no trainable parameters"):
        create_optimizer(mod, config)


def test_lr_scheduler_schedules() -> None:
    """Verify step behavior for cosine, linear, and constant schedules."""
    mod = DummyModule()
    opt = create_optimizer(mod, TrainingConfig())

    # Cosine scheduler
    cfg_cos = TrainingConfig(lr_scheduler_type="cosine", warmup_steps=5, max_steps=20)
    sched = create_lr_scheduler(opt, cfg_cos, total_steps=20)
    assert sched is not None

    # Step through warmup
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    # LRs should start at 0, increase during warmup, and decay
    assert lrs[0] == 0.0
    assert max(lrs) > lrs[0]
