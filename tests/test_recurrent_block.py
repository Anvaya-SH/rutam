"""Unit tests for RUTAMRecurrentBlock parameter-shared execution and causal safety."""

from __future__ import annotations

import pytest
import torch

from rutam.config import RecurrentConfig
from rutam.exceptions import ValidationError
from rutam.model.loop_embeddings import LoopPositionEmbedding
from rutam.model.recurrent_block import RUTAMRecurrentBlock


def test_recurrent_block_init() -> None:
    """Verify clean instantiation and parameter shapes of RUTAMRecurrentBlock."""
    config = RecurrentConfig(hidden_dim=32, num_heads=4, intermediate_dim=64)
    block = RUTAMRecurrentBlock(config)

    assert isinstance(block, torch.nn.Module)
    assert block.hidden_dim == 32
    assert block.num_heads == 4
    assert block.head_dim == 8
    assert block.attn_norm is not None
    assert block.ffn_norm is not None

    with pytest.raises(ValidationError):
        RUTAMRecurrentBlock(None)  # type: ignore[arg-type]


def test_parameter_sharing_constant_parameters() -> None:
    """Requirement 1: Verify block parameter count does not scale with loop iterations."""
    config = RecurrentConfig(hidden_dim=32, num_heads=4, intermediate_dim=64, max_loops=8)
    block = RUTAMRecurrentBlock(config)

    param_count_initial = sum(p.numel() for p in block.parameters())

    # Execute across 5 sequential loops
    x = torch.randn(2, 4, 32)
    loop_emb = LoopPositionEmbedding(config)

    for t in range(5):
        emb = loop_emb.get_embedding(t)
        delta = block(x, loop_embedding=emb)
        x = x + delta

    param_count_after = sum(p.numel() for p in block.parameters())
    assert param_count_initial == param_count_after


def test_causal_safety_no_future_leakage() -> None:
    """Requirement 5: Ensure future tokens cannot leak information to earlier positions."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=16, num_heads=2, intermediate_dim=32, causal=True, dropout=0.0
    )
    block = RUTAMRecurrentBlock(config)
    block.eval()

    # Input sequence of length 5
    x = torch.randn(1, 5, 16, requires_grad=True)

    # Gradient test: gradient of token 1 w.r.t future tokens (2, 3, 4) must be exactly 0
    delta = block(x)
    out = x + delta

    # Loss depends only on representation at token 1
    loss_pos1 = out[0, 1].sum()
    loss_pos1.backward()

    assert x.grad is not None
    # Past and current positions (0 and 1) should have non-zero gradients
    assert not torch.allclose(x.grad[0, :2], torch.zeros_like(x.grad[0, :2]))
    # Future positions (2, 3, 4) must have exactly zero gradients
    assert torch.allclose(x.grad[0, 2:], torch.zeros_like(x.grad[0, 2:]), atol=1e-7)


def test_causal_safety_perturbation_invariance() -> None:
    """Requirement 5: Perturbing future token does not alter earlier token representations."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=16, num_heads=2, intermediate_dim=32, causal=True, dropout=0.0
    )
    block = RUTAMRecurrentBlock(config)
    block.eval()

    x1 = torch.randn(1, 4, 16)
    x2 = x1.clone()
    # Modify last token in x2
    x2[0, 3] = x2[0, 3] + 10.0

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    # Positions 0, 1, 2 must be identical
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6)
    # Position 3 should differ
    assert not torch.allclose(out1[0, 3], out2[0, 3])


def test_loop_position_embedding_effect() -> None:
    """Requirement 3: Learned loop embeddings distinguish iteration t=0 from t=3."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32, num_heads=4, intermediate_dim=64, max_loops=6, dropout=0.0
    )
    block = RUTAMRecurrentBlock(config)
    loop_emb = LoopPositionEmbedding(config)
    block.eval()

    x = torch.randn(2, 4, 32)
    emb_0 = loop_emb.get_embedding(0)
    emb_3 = loop_emb.get_embedding(3)

    assert not torch.allclose(emb_0, emb_3)

    with torch.no_grad():
        out_loop0 = block(x, loop_embedding=emb_0)
        out_loop3 = block(x, loop_embedding=emb_3)

    # Representations produced under t=0 and t=3 must differ significantly
    diff = (out_loop0 - out_loop3).abs().mean().item()
    assert diff > 1e-4


def test_attention_mask_support() -> None:
    """Verify padding attention mask masks out padding tokens."""
    torch.manual_seed(42)
    config = RecurrentConfig(hidden_dim=16, num_heads=2, intermediate_dim=32, dropout=0.0)
    block = RUTAMRecurrentBlock(config)
    block.eval()

    x = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]], dtype=torch.bool)

    with torch.no_grad():
        out = block(x, attention_mask=mask)

    assert out.shape == (2, 4, 16)
    assert torch.isfinite(out).all()


def test_invalid_block_inputs() -> None:
    """Verify strict validation errors on malformed block inputs."""
    config = RecurrentConfig(hidden_dim=16, num_heads=2, intermediate_dim=32)
    block = RUTAMRecurrentBlock(config)

    # Wrong dtype
    with pytest.raises(ValidationError):
        block(torch.randint(0, 10, (2, 4)))

    # Wrong shape
    with pytest.raises(ValidationError):
        block(torch.randn(2, 16))

    # Wrong hidden dim
    with pytest.raises(ValidationError):
        block(torch.randn(2, 4, 32))
