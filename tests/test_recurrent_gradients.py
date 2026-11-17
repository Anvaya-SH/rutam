"""Gradient flow, backpropagation, and non-vanishing/exploding stability tests for recurrence."""

from __future__ import annotations

import pytest
import torch

from rutam.config import RecurrentConfig
from rutam.model.recurrent import RUTAMRecurrentEngine


@pytest.mark.parametrize("loops", [1, 3, 5, 8])
def test_recurrent_gradient_flow_across_loops(loops: int) -> None:
    """Requirement 1 & 6: Verify full gradient flow across unrolled recurrent loops (1 to 8)."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        max_loops=8,
        min_loops=1,
        default_loops=loops,
        dynamic_halting=False,
        use_graph_conditioning=True,
        graph_dim=32,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)

    x = torch.randn(2, 4, 32, requires_grad=True)
    g_states = torch.randn(2, 3, 32, requires_grad=True)
    r_ctx = torch.randn(2, 32, requires_grad=True)

    target = torch.randn_like(x)
    out = engine(x, num_loops=loops, graph_states=g_states, reasoning_context=r_ctx)
    loss = (out.output_states * target).sum()
    loss.backward()

    # Input gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    assert g_states.grad is not None
    assert torch.isfinite(g_states.grad).all()

    assert r_ctx.grad is not None
    assert torch.isfinite(r_ctx.grad).all()

    # Submodule parameter gradients
    assert engine.block.q_proj.weight.grad is not None
    assert torch.isfinite(engine.block.q_proj.weight.grad).all()
    assert not torch.allclose(
        engine.block.q_proj.weight.grad, torch.zeros_like(engine.block.q_proj.weight.grad)
    )

    assert engine.block.ffn_up.weight.grad is not None
    assert torch.isfinite(engine.block.ffn_up.weight.grad).all()

    assert engine.residual_gate.residual_scale.grad is not None
    assert torch.isfinite(engine.residual_gate.residual_scale.grad).all()

    # Loop position embeddings: indices 0 to loops-1 must have non-zero grad
    emb_grad = engine.loop_embeddings.embeddings.weight.grad
    assert emb_grad is not None
    for t in range(loops):
        assert not torch.allclose(emb_grad[t], torch.zeros_like(emb_grad[t]))


def test_gradient_norm_stability_across_loop_depths() -> None:
    """Requirement 6: Verify gradient norm remains bounded across 1 vs 8 loops without exploding."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        max_loops=8,
        min_loops=1,
        dropout=0.0,
    )

    grad_norms = []
    for loops in [1, 2, 4, 8]:
        engine = RUTAMRecurrentEngine(config)
        x = torch.randn(2, 4, 32, requires_grad=True)
        target = torch.randn_like(x)
        out = engine(x, num_loops=loops)
        loss = (out.output_states * target).sum()
        loss.backward()

        total_norm = torch.norm(
            torch.stack([p.grad.norm() for p in engine.parameters() if p.grad is not None])
        ).item()

        assert torch.isfinite(torch.tensor(total_norm))
        assert total_norm > 1e-6  # not vanished
        assert total_norm < 1000.0  # not exploded
        grad_norms.append(total_norm)

    # Norms across depths should stay in a reasonable, stable envelope
    ratio = grad_norms[-1] / grad_norms[0]
    assert 0.1 <= ratio <= 25.0


def test_dynamic_controller_gradients() -> None:
    """Verify gradients flow to DynamicLoopController MLP when dynamic halting is active."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        dynamic_halting=True,
        min_loops=1,
        max_loops=4,
        controller_hidden_dim=16,
    )
    engine = RUTAMRecurrentEngine(config)
    assert engine.controller is not None

    x = torch.randn(2, 4, 32)
    out = engine(x)

    assert out.continue_logits is not None
    # Train controller via pseudo-ACT or continue logits loss
    loss = out.continue_logits.sum() + out.output_states.sum()
    loss.backward()

    for p in engine.controller.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
