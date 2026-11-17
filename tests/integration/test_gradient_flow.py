"""Integration tests for backward pass and gradient flow through RUTAM submodules and full model."""

from __future__ import annotations

import torch

from rutam.diagnostics.gradient_checks import (
    check_end_to_end_gradients,
    check_fusion_gradients,
    check_graph_attention_gradients,
    check_graph_encoder_gradients,
    check_graph_selector_gradients,
    check_link_predictor_gradients,
    check_path_scorer_gradients,
    check_query_head_gradients,
    check_write_head_gradients,
    run_all_gradient_checks,
)
from rutam.diagnostics.integration import create_verification_model


def test_all_submodule_gradient_flow_suite() -> None:
    """Verify that all 8 submodules compute finite, non-zero gradients across trainable weights."""
    report = run_all_gradient_checks()

    assert report.passed, f"Gradient checks failed: {report.summary}"
    assert report.summary.get("passed_components") == 8
    assert report.frozen_backbone_verified is True
    assert report.unfrozen_backbone_verified is True
    assert report.end_to_end_passed is True


def test_query_head_backward_pass() -> None:
    """Verify gradient flow through query projections and decision gate in RUTAMNeuralQueryHead."""
    report = check_query_head_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero
    assert report.params_with_gradients_count == report.trainable_params_count


def test_graph_encoder_backward_pass() -> None:
    """Verify gradient flow through message passing layers in RUTAMGraphEncoder."""
    report = check_graph_encoder_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero


def test_graph_attention_backward_pass() -> None:
    """Verify gradient flow through multi-head attention in RUTAMGraphAttention."""
    report = check_graph_attention_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero


def test_graph_selector_backward_pass() -> None:
    """Verify gradient flow through relevance scorer in RUTAMGraphSelector."""
    report = check_graph_selector_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero


def test_path_scorer_backward_pass() -> None:
    """Verify gradient flow through path scoring in RUTAMPathScorer."""
    report = check_path_scorer_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero


def test_graph_fusion_backward_pass() -> None:
    """Verify gradient flow through cross-attention and fusion gate in RUTAMGraphFusion."""
    report = check_fusion_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero


def test_memory_write_head_backward_pass() -> None:
    """Verify gradient flow through write gate, operation predictor, and memory projections."""
    report = check_write_head_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero


def test_link_predictor_backward_pass() -> None:
    """Verify gradient flow through link scorer and relation classifier in RUTAMGraphLinker."""
    report = check_link_predictor_gradients()
    assert report.passed
    assert report.all_finite
    assert report.all_non_zero


def test_frozen_backbone_isolation() -> None:
    """Verify strict gradient isolation: backbone parameters receive zero gradients when frozen."""
    e2e_ok, frozen_ok, _ = check_end_to_end_gradients()
    assert frozen_ok is True
    assert e2e_ok is True


def test_combined_causal_and_write_loss_backward() -> None:
    """Verify simultaneous backpropagation of causal language loss and memory write loss."""
    model, _ = create_verification_model(
        num_nodes=4, hidden_dim=16, vocab_size=80, backbone_frozen=True
    )

    input_ids = torch.randint(0, 80, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 80, (2, 8), dtype=torch.long)

    out = model(input_ids, labels=labels, generate_proposals=True)
    assert out.loss is not None
    assert out.write_output is not None

    write_targets = torch.tensor([1.0, 0.0])
    op_targets = torch.tensor([1, 0], dtype=torch.long)
    write_losses = model.write_head.loss(
        out.write_output,
        write_targets=write_targets,
        operation_targets=op_targets,
    )

    combined_loss = out.loss + write_losses.total
    combined_loss.backward()

    # Backbone remains frozen
    for p in model.backbone.parameters():
        assert p.grad is None

    # Fusion and write head received gradients
    fusion_grads = [p.grad for p in model.fusion.parameters() if p.requires_grad]
    assert any(g is not None and g.norm().item() > 0.0 for g in fusion_grads)

    write_grads = [p.grad for p in model.write_head.parameters() if p.requires_grad]
    assert any(g is not None and g.norm().item() > 0.0 for g in write_grads)
