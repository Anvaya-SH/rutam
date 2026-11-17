"""Unit tests for fixed and learned uncertainty loss balancing."""

import torch

from rutam.training.loss_balancer import FixedLossBalancer, UncertaintyLossBalancer


def test_fixed_loss_balancer():
    weights = {"task_a": 1.0, "task_b": 0.5}
    balancer = FixedLossBalancer(weights)

    losses = {
        "task_a": torch.tensor(2.0),
        "task_b": torch.tensor(4.0),
        "task_c": torch.tensor(10.0),  # Not supervised or count is 0
    }
    counts = {"task_a": 5, "task_b": 2, "task_c": 0}

    total, scaled, eff = balancer(losses, counts)

    # task_a: 2.0 * 1.0 = 2.0, task_b: 4.0 * 0.5 = 2.0, task_c: omitted (count=0)
    assert torch.isclose(total, torch.tensor(4.0))
    assert "task_c" not in scaled
    assert eff["task_a"] == 1.0
    assert eff["task_b"] == 0.5


def test_uncertainty_loss_balancer():
    balancer = UncertaintyLossBalancer(["loss_x", "loss_y"])

    losses = {
        "loss_x": torch.tensor(1.0, requires_grad=True),
        "loss_y": torch.tensor(2.0, requires_grad=True),
    }
    counts = {"loss_x": 1, "loss_y": 1}

    total, scaled, eff = balancer(losses, counts)

    assert total.requires_grad
    assert not torch.isnan(total).any()
    assert len(scaled) == 2
    assert "loss_x" in eff
    assert "loss_y" in eff

    # Backprop through uncertainty parameters
    total.backward()
    assert balancer.log_vars["loss_x"].grad is not None
    assert balancer.log_vars["loss_y"].grad is not None
