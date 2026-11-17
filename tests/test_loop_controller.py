"""Unit tests for DynamicLoopController budget enforcement, halting decisions, and pooling."""

from __future__ import annotations

import pytest
import torch

from rutam.config import RecurrentConfig
from rutam.exceptions import ValidationError
from rutam.model.loop_controller import DynamicLoopController


def test_loop_controller_init() -> None:
    """Verify clean instantiation of DynamicLoopController."""
    config = RecurrentConfig(hidden_dim=32, controller_hidden_dim=16)
    controller = DynamicLoopController(config)
    assert isinstance(controller, torch.nn.Module)
    assert controller.config == config

    with pytest.raises(ValidationError):
        DynamicLoopController(None)  # type: ignore[arg-type]


def test_loop_controller_forward_shapes_and_probs() -> None:
    """Verify output shapes and probability ranges."""
    config = RecurrentConfig(hidden_dim=32, controller_hidden_dim=16, min_loops=2, max_loops=5)
    controller = DynamicLoopController(config)

    x = torch.randn(3, 8, 32)
    logits, probs, cont = controller(x, loop_idx=1)

    assert logits.shape == (3, 1)
    assert probs.shape == (3, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert isinstance(cont, bool)


def test_controller_strictly_respects_min_loops() -> None:
    """Requirement 4: Never halts before min_loops budget is reached."""
    config = RecurrentConfig(hidden_dim=32, min_loops=3, max_loops=6, halting_threshold=0.99)
    controller = DynamicLoopController(config)

    # Force MLP weights to strongly predict halt (large negative continue logits)
    with torch.no_grad():
        for p in controller.parameters():
            p.fill_(-100.0)

    x = torch.randn(2, 4, 32)

    # For loop_idx=0 (completed 1 loop) and loop_idx=1 (completed 2 loops),
    # must continue because min_loops=3
    _, halt_p0, cont0 = controller(x, loop_idx=0)
    assert cont0 is True
    assert halt_p0.mean().item() > 0.9  # strongly wants to halt, but cannot

    _, halt_p1, cont1 = controller(x, loop_idx=1)
    assert cont1 is True

    # For loop_idx=2 (completed 3 loops), min_loops is satisfied, so it is allowed to halt
    _, _, cont2 = controller(x, loop_idx=2)
    assert cont2 is False


def test_controller_strictly_respects_max_loops() -> None:
    """Requirement 4: Always halts when max_loops is reached."""
    config = RecurrentConfig(hidden_dim=32, min_loops=1, max_loops=4, halting_threshold=0.01)
    controller = DynamicLoopController(config)

    # Force MLP weights to strongly predict continue
    with torch.no_grad():
        for p in controller.parameters():
            p.fill_(100.0)

    x = torch.randn(2, 4, 32)

    # For loop_idx=3 (completed 4 loops = max_loops), must halt
    _, _, cont3 = controller(x, loop_idx=3)
    assert cont3 is False


def test_controller_halting_threshold_behavior() -> None:
    """Verify halting threshold gating in intermediate loops."""
    config = RecurrentConfig(hidden_dim=16, min_loops=1, max_loops=5, halting_threshold=0.6)
    controller = DynamicLoopController(config)

    x = torch.randn(1, 4, 16)

    # Force continue probability = 0.8 (> 0.6)
    with torch.no_grad():
        controller.mlp[-1].bias.fill_(1.386)  # sigmoid(1.386) ≈ 0.8
        controller.mlp[0].weight.zero_()
        controller.mlp[0].bias.zero_()
        controller.mlp[-1].weight.zero_()

    _, _, cont_high = controller(x, loop_idx=1)
    assert cont_high is True

    # Force continue probability = 0.4 (< 0.6)
    with torch.no_grad():
        controller.mlp[-1].bias.fill_(-0.405)  # sigmoid(-0.405) ≈ 0.4

    _, _, cont_low = controller(x, loop_idx=1)
    assert cont_low is False


def test_masked_sequence_pooling() -> None:
    """Verify sequence pooling ignores padded tokens."""
    config = RecurrentConfig(hidden_dim=8)
    controller = DynamicLoopController(config)

    x = torch.ones(1, 4, 8)
    # Corrupt last 2 tokens with huge values
    x[0, 2:] = 1000.0

    mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    pooled = controller.pool_sequence(x, attention_mask=mask)

    # Mean of first two tokens (which are all ones) should be exactly 1.0
    assert torch.allclose(pooled, torch.ones(1, 8), atol=1e-5)


def test_invalid_loop_index() -> None:
    """Verify validation on loop indices out of bounds."""
    config = RecurrentConfig(hidden_dim=16, max_loops=4)
    controller = DynamicLoopController(config)
    x = torch.randn(1, 2, 16)

    with pytest.raises(ValidationError):
        controller(x, loop_idx=-1)

    with pytest.raises(ValidationError):
        controller(x, loop_idx=4)

    with pytest.raises(ValidationError):
        controller(x, loop_idx="0")  # type: ignore[arg-type]
