"""Tests for graph-conditioned recurrence, reasoning context injection, and node masking."""

from __future__ import annotations

import torch

from rutam.config import RecurrentConfig
from rutam.model.recurrent import RUTAMRecurrentEngine


def test_graph_conditioning_modulates_recurrent_states() -> None:
    """Requirement 2: Graph reasoning context and states modulate recurrent representations."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        use_graph_conditioning=True,
        graph_dim=32,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    x = torch.randn(2, 4, 32)
    g_states = torch.randn(2, 5, 32)
    r_ctx = torch.randn(2, 32)

    with torch.no_grad():
        out_no_graph = engine(x, graph_states=None, reasoning_context=None)
        out_with_graph = engine(x, graph_states=g_states, reasoning_context=r_ctx)

    assert not torch.allclose(out_no_graph.output_states, out_with_graph.output_states)
    diff = (out_no_graph.output_states - out_with_graph.output_states).abs().mean().item()
    assert diff > 1e-7


def test_empty_graph_graceful_handling() -> None:
    """Verify empty graph representations (0 nodes or all-false mask) execute without NaN."""
    torch.manual_seed(42)
    config = RecurrentConfig(hidden_dim=32, num_heads=4, intermediate_dim=64, graph_dim=32)
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    x = torch.randn(2, 4, 32)

    # 1. Zero nodes [B, 0, H]
    g_empty = torch.zeros(2, 0, 32)
    with torch.no_grad():
        out_zero = engine(x, graph_states=g_empty)
    assert torch.isfinite(out_zero.output_states).all()

    # 2. All-False graph mask
    g_states = torch.randn(2, 4, 32)
    all_false_mask = torch.zeros((2, 4), dtype=torch.bool)
    with torch.no_grad():
        out_masked = engine(x, graph_states=g_states, graph_mask=all_false_mask)
    assert torch.isfinite(out_masked.output_states).all()


def test_graph_node_masking_invariance() -> None:
    """Verify masked-out graph nodes do not affect recurrent representations."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        use_graph_conditioning=True,
        graph_dim=32,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    x = torch.randn(1, 3, 32)
    g1 = torch.randn(1, 4, 32)
    g2 = g1.clone()
    # Heavily corrupt masked node 3
    g2[0, 3] = g2[0, 3] + 50.0

    mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)

    with torch.no_grad():
        out1 = engine(x, graph_states=g1, graph_mask=mask)
        out2 = engine(x, graph_states=g2, graph_mask=mask)

    assert torch.allclose(out1.output_states, out2.output_states, atol=1e-5)


def test_reasoning_context_injection_and_gradients() -> None:
    """Verify path reasoning context projection and gradient backpropagation."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        use_graph_conditioning=True,
        graph_dim=16,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)

    x = torch.randn(2, 4, 32)
    r_ctx_2d = torch.randn(2, 16, requires_grad=True)

    out = engine(x, reasoning_context=r_ctx_2d)
    loss = (out.output_states * torch.randn_like(out.output_states)).sum()
    loss.backward()

    assert r_ctx_2d.grad is not None
    assert torch.isfinite(r_ctx_2d.grad).all()
    assert not torch.allclose(r_ctx_2d.grad, torch.zeros_like(r_ctx_2d.grad))


def test_multi_loop_reasoning_progression() -> None:
    """Verify representations evolve across loops as graph memory is processed."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        max_loops=4,
        default_loops=4,
        use_graph_conditioning=True,
        graph_dim=32,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    x = torch.randn(2, 4, 32)
    g_states = torch.randn(2, 4, 32)

    with torch.no_grad():
        out = engine(x, graph_states=g_states)

    assert len(out.loop_states) == 4
    # Check consecutive loop states differ
    for i in range(3):
        diff = (out.loop_states[i + 1] - out.loop_states[i]).abs().mean().item()
        assert diff > 1e-5


def test_graph_conditioning_disabled_configuration() -> None:
    """Verify when use_graph_conditioning is False, graph states are ignored."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        use_graph_conditioning=False,
        graph_dim=32,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    x = torch.randn(1, 3, 32)
    g_states = torch.randn(1, 4, 32)

    with torch.no_grad():
        out_no_g = engine(x, graph_states=None)
        out_with_g = engine(x, graph_states=g_states)

    assert torch.allclose(out_no_g.output_states, out_with_g.output_states, atol=1e-6)
