"""Numerical stability, identity preservation, and bounded activation tests for deep recurrence."""

from __future__ import annotations

import pytest
import torch

from rutam.config import RecurrentConfig
from rutam.model.recurrent import RUTAMRecurrentEngine


@pytest.mark.parametrize("loops", [1, 2, 4, 6, 8])
def test_stability_no_nan_inf_across_loop_budgets(loops: int) -> None:
    """Requirement 6: Verify no NaN or Inf occurs in outputs or gates across up to 8 loops."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        max_loops=8,
        min_loops=1,
        default_loops=loops,
        use_graph_conditioning=True,
        graph_dim=32,
    )
    engine = RUTAMRecurrentEngine(config)

    x = torch.randn(4, 8, 32)
    g_states = torch.randn(4, 6, 32)
    r_ctx = torch.randn(4, 32)

    out = engine(x, num_loops=loops, graph_states=g_states, reasoning_context=r_ctx)

    assert torch.isfinite(out.output_states).all()
    assert not torch.isnan(out.output_states).any()
    assert not torch.isinf(out.output_states).any()

    for s in out.loop_states:
        assert torch.isfinite(s).all()

    for g in out.residual_gates:
        assert torch.isfinite(g).all()
        assert (g >= 0.0).all() and (g <= 1.0).all()


def test_identity_preservation_at_initialization() -> None:
    """Requirement 6: Small-scale initialization (0.001) preserves input identity at step 0."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        residual_scale_init=0.001,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    # Input pre-normalized like transformer layer outputs
    norm = torch.nn.LayerNorm(32)
    x = norm(torch.randn(2, 4, 32))

    with torch.no_grad():
        out = engine(x, num_loops=1)

    # Initial refined state should be extremely close to normalized input
    diff = (out.output_states - x).abs().mean().item()
    assert diff < 0.01, f"Expected near-identity diff < 0.01, got {diff}"


def test_bounded_activation_growth_across_depths() -> None:
    """Requirement 6: Ensure activation norm remains strictly bounded across up to 8 loops."""
    torch.manual_seed(42)
    config = RecurrentConfig(
        hidden_dim=32,
        num_heads=4,
        intermediate_dim=64,
        max_loops=8,
        min_loops=1,
        dropout=0.0,
    )
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out = engine(x, num_loops=8)

    assert len(out.loop_states) == 8
    norms = [s.norm(dim=-1).mean().item() for s in out.loop_states]

    # LayerNorm guarantees unit variance across features, so token vector norm ≈ sqrt(32) ≈ 5.65
    for n in norms:
        assert 4.0 <= n <= 8.0, f"Activation norm {n} outside bounded stable range [4.0, 8.0]"

    # Relative drift between first and last loop should be minimal
    drift = abs(norms[-1] - norms[0]) / norms[0]
    assert drift < 0.2


def test_extreme_input_numerical_resilience() -> None:
    """Verify stability under extreme input magnitudes without overflow or NaN."""
    torch.manual_seed(42)
    config = RecurrentConfig(hidden_dim=16, num_heads=2, intermediate_dim=32, max_loops=4)
    engine = RUTAMRecurrentEngine(config)
    engine.eval()

    # 1. Very large inputs
    x_large = torch.full((1, 3, 16), 1000.0)
    with torch.no_grad():
        out_large = engine(x_large, num_loops=4)
    assert torch.isfinite(out_large.output_states).all()

    # 2. Very small inputs
    x_small = torch.full((1, 3, 16), 1e-6)
    with torch.no_grad():
        out_small = engine(x_small, num_loops=4)
    assert torch.isfinite(out_small.output_states).all()
