"""Tests for memory write gate predicting continuous write utility and binary decisions."""

from __future__ import annotations

import pytest
import torch

from rutam.config import MemoryWriteConfig
from rutam.exceptions import ValidationError
from rutam.memory.write_gate import MemoryWriteGate, availability_mask


def test_availability_mask_helper() -> None:
    mask = availability_mask(None, 4)
    assert mask.shape == (4,)
    assert mask.dtype == torch.bool
    assert mask.all().item()

    explicit = torch.tensor([True, False, True], dtype=torch.bool)
    assert (availability_mask(explicit, 3) == explicit).all()

    with pytest.raises(ValidationError):
        availability_mask(torch.tensor([True, False]), 3)

    with pytest.raises(ValidationError):
        availability_mask(torch.tensor([1, 0]), 2)  # must be bool


def test_write_gate_architecture_and_output_shapes() -> None:
    config = MemoryWriteConfig(backbone_hidden_dim=16, projection_dim=32, write_threshold=0.6)
    gate = MemoryWriteGate(config)

    x = torch.randn(5, 16)
    out = gate(x)

    assert out.logits.shape == (5, 1)
    assert out.probabilities.shape == (5, 1)
    assert out.logits.dtype == torch.float32
    assert out.probabilities.dtype == torch.float32
    assert out.logits.device.type == "cpu"
    assert out.probabilities.device.type == "cpu"
    assert out.decisions is None  # in train mode

    # Check sigmoid relationship
    expected_probs = torch.sigmoid(out.logits)
    torch.testing.assert_close(out.probabilities, expected_probs)


def test_write_gate_eval_vs_train_mode() -> None:
    config = MemoryWriteConfig(backbone_hidden_dim=8, projection_dim=16, write_threshold=0.7)
    gate = MemoryWriteGate(config)

    x = torch.randn(6, 8)

    # Train mode
    gate.train()
    out_train = gate(x)
    assert out_train.decisions is None

    # Eval mode
    gate.eval()
    out_eval = gate(x)
    assert out_eval.decisions is not None
    assert out_eval.decisions.shape == (6,)
    assert out_eval.decisions.dtype == torch.bool

    expected_decisions = out_eval.probabilities[:, 0] >= 0.7
    assert (out_eval.decisions == expected_decisions).all().item()


def test_write_gate_availability_masking() -> None:
    config = MemoryWriteConfig(backbone_hidden_dim=8, projection_dim=16, write_threshold=0.1)
    gate = MemoryWriteGate(config)
    gate.eval()

    x = torch.randn(4, 8)
    available = torch.tensor([True, False, True, False], dtype=torch.bool)

    out = gate(x, memory_available=available)
    assert out.probabilities[1, 0].item() == 0.0
    assert out.probabilities[3, 0].item() == 0.0
    assert out.decisions is not None
    assert not out.decisions[1].item()
    assert not out.decisions[3].item()


def test_write_gate_finite_and_shape_validation() -> None:
    config = MemoryWriteConfig(backbone_hidden_dim=8, projection_dim=16)
    gate = MemoryWriteGate(config)

    # NaN / Inf rejection
    nan_tensor = torch.randn(3, 8)
    nan_tensor[0, 0] = float("nan")
    with pytest.raises(ValidationError):
        gate(nan_tensor)

    inf_tensor = torch.randn(3, 8)
    inf_tensor[1, 1] = float("inf")
    with pytest.raises(ValidationError):
        gate(inf_tensor)

    # Dimensionality mismatch
    with pytest.raises(ValidationError):
        gate(torch.randn(8))  # 1D

    with pytest.raises(ValidationError):
        gate(torch.randn(2, 4, 8))  # 3D

    with pytest.raises(ValidationError):
        gate(torch.randn(3, 10))  # wrong dim

    # Invalid config type
    with pytest.raises(ValidationError):
        MemoryWriteGate("not_a_config")  # type: ignore[arg-type]


def test_write_gate_gradient_flow() -> None:
    config = MemoryWriteConfig(backbone_hidden_dim=8, projection_dim=16)
    gate = MemoryWriteGate(config)

    x = torch.randn(4, 8, requires_grad=True)
    out = gate(x)
    loss = out.logits.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert gate.linear1.weight.grad is not None
    assert gate.linear2.weight.grad is not None
    assert gate.layer_norm.weight.grad is not None
