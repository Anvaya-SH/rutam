"""Unit tests for safetensors model weight serialization and loading."""

from __future__ import annotations

import torch
from torch import nn

from rutam.checkpoint.model_weights import load_model_weights, save_model_weights


class SmallTestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.ln = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.ln(self.fc1(x)))


def test_model_weight_roundtrip(tmp_path) -> None:
    """Verify that saving and loading model weights preserves exact parameter values."""
    torch.manual_seed(42)
    net1 = SmallTestNet()
    save_path = tmp_path / "model.safetensors"

    save_model_weights(net1, save_path, metadata={"test": "true"})
    assert save_path.is_file()

    net2 = SmallTestNet()
    # Weights should initially differ
    assert not torch.equal(net1.fc1.weight, net2.fc1.weight)

    missing, unexpected = load_model_weights(net2, save_path, strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0

    # Weights must now match exactly
    for (n1, p1), (n2, p2) in zip(net1.named_parameters(), net2.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_model_weight_incompatible_keys(tmp_path) -> None:
    """Verify missing and unexpected key reporting during weight loading."""
    net1 = SmallTestNet()
    save_path = tmp_path / "model.safetensors"
    save_model_weights(net1, save_path)

    class IncompatibleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.other = nn.Linear(16, 16)

    net3 = IncompatibleNet()
    missing, unexpected = load_model_weights(net3, save_path, strict=False)
    assert "other.weight" in missing
    assert "fc2.weight" in unexpected
