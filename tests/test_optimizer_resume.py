"""Unit tests for pickle-free optimizer state serialization and resumption."""

from __future__ import annotations

import torch
from torch import nn

from rutam.checkpoint.optimizer_state import load_optimizer_state, save_optimizer_state


def test_optimizer_state_roundtrip(tmp_path) -> None:
    """Verify bit-exact resumption of optimizer state buffers."""
    torch.manual_seed(42)
    fc = nn.Linear(4, 2)
    opt1 = torch.optim.AdamW(fc.parameters(), lr=0.01, weight_decay=0.01)

    # Perform a few optimization steps to build up momentum states
    for _ in range(3):
        x = torch.randn(2, 4)
        loss = fc(x).sum()
        loss.backward()
        opt1.step()
        opt1.zero_grad()

    t_path = tmp_path / "optimizer.safetensors"
    m_path = tmp_path / "optimizer_metadata.json"

    save_optimizer_state(opt1, t_path, m_path)
    assert t_path.is_file()
    assert m_path.is_file()

    # Create fresh optimizer
    opt2 = torch.optim.AdamW(fc.parameters(), lr=0.01, weight_decay=0.01)
    load_optimizer_state(opt2, t_path, m_path)

    # Compare state buffers
    s1 = opt1.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert len(s1) == len(s2)

    for p_id in s1:
        assert p_id in s2
        for k in s1[p_id]:
            v1 = s1[p_id][k]
            v2 = s2[p_id][k]
            if isinstance(v1, torch.Tensor):
                assert torch.equal(v1, v2)
            else:
                assert v1 == v2
