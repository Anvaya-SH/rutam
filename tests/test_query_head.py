"""Query projection, graph conditioning, decision supervision and strict configuration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from rutam import (
    QueryDecisionGate,
    QueryHeadConfig,
    RUTAMConfig,
    RUTAMNeuralQueryHead,
    SelectorConfig,
    query_decision_loss,
    query_norm_loss,
)
from rutam.exceptions import ValidationError


def small_config(**overrides: Any) -> QueryHeadConfig:
    return replace(
        QueryHeadConfig(
            backbone_hidden_dim=6, graph_hidden_dim=4, query_dim=3, intermediate_dim=8, dropout=0
        ),
        **overrides,
    )


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(12)
    return torch.randn((2, 5, 6)), torch.randn((2, 4))


def test_output_contract_normalization_and_uncalibrated_confidence_alias() -> None:
    hidden, summary = inputs()
    head = RUTAMNeuralQueryHead(small_config(return_attention=True))
    output = head(hidden, graph_summary=summary)
    assert output.query_vectors.shape == (2, 3)
    assert (
        output.query_gate.shape
        == output.query_confidence.shape
        == output.query_gate_logits.shape
        == (2, 1)
    )
    assert output.query_confidence is output.query_gate
    assert output.token_attention_weights.shape == (2, 5)
    assert output.query_vectors.dtype == output.query_gate.dtype == torch.float32
    assert output.query_vectors.device.type == "cpu"
    assert output.should_query is None and output.valid_context.all()
    assert ((output.query_gate >= 0) & (output.query_gate <= 1)).all()
    torch.testing.assert_close(output.query_vectors.norm(dim=1), torch.ones(2))


def test_batched_and_broadcast_graph_summary_equivalence() -> None:
    hidden, summary = inputs()
    head = RUTAMNeuralQueryHead(small_config()).eval()
    single = head(hidden, graph_summary=summary[0]).query_vectors
    batched = head(hidden, graph_summary=summary[0].expand(2, -1)).query_vectors
    torch.testing.assert_close(single, batched, rtol=0, atol=0)
    changed = summary[0].expand(2, -1).clone()
    changed[1] = summary[1]
    different = head(hidden, graph_summary=changed).query_vectors
    torch.testing.assert_close(different[0], single[0], rtol=0, atol=0)
    assert not torch.allclose(different[1], single[1])


@pytest.mark.parametrize("mode", ["add", "gated_add"])
def test_conditioning_formula_and_unavailable_memory_is_ignored(mode: str) -> None:
    hidden, summary = inputs()
    head = RUTAMNeuralQueryHead(
        small_config(normalize_queries=False, conditioning_mode=mode)
    ).eval()
    base = head(hidden).query_vectors
    projected = head.graph_projection(summary)
    scale = (
        1
        if mode == "add"
        else torch.sigmoid(head.conditioning_gate(torch.cat((base, projected), 1)))
    )
    torch.testing.assert_close(
        head(hidden, graph_summary=summary).query_vectors, base + scale * projected
    )
    available = torch.tensor([True, False])
    output = head(hidden, graph_summary=summary, memory_available=available)
    torch.testing.assert_close(output.query_vectors[1], base[1], rtol=0, atol=0)
    assert output.query_gate[1] == 0 and not output.should_query[1]
    assert not torch.allclose(output.query_vectors[0], base[0])


def test_conditioning_can_be_disabled_without_graph_parameters() -> None:
    hidden, summary = inputs()
    head = RUTAMNeuralQueryHead(small_config(graph_conditioning=False)).eval()
    assert head.graph_projection is None and head.conditioning_gate is None
    torch.testing.assert_close(
        head(hidden, graph_summary=summary).query_vectors, head(hidden).query_vectors
    )


def test_gate_training_thresholds_and_unavailable_masking() -> None:
    config = small_config(gate_threshold=0.9)
    gate = QueryDecisionGate(config)
    with torch.no_grad():
        gate.input_projection.weight.zero_()
        gate.input_projection.bias.zero_()
        gate.output_projection.weight.zero_()
        gate.output_projection.bias.fill_(3)
    context = torch.randn((2, 6), requires_grad=True)
    available = torch.tensor([True, False])
    output = gate(context, available)
    assert output.should_query is None
    assert output.gate[0].item() == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())
    assert output.gate[1].item() == 0
    assert output.logits.tolist() == [[3.0], [3.0]]  # Raw logits remain usable by BCE.
    gate.eval()
    assert gate(context, available).should_query.tolist() == [True, False]
    strict = QueryDecisionGate(replace(config, gate_threshold=0.99)).eval()
    strict.load_state_dict(gate.state_dict())
    assert strict(context, available).should_query.tolist() == [False, False]
    zero_threshold = QueryDecisionGate(replace(config, gate_threshold=0)).eval()
    zero_threshold.load_state_dict(gate.state_dict())
    assert zero_threshold(context, available).should_query.tolist() == [True, False]


def test_disabled_gate_is_explicit_available_only_with_no_fake_logits() -> None:
    hidden, _ = inputs()
    head = RUTAMNeuralQueryHead(small_config(query_gate_enabled=False)).eval()
    output = head(hidden, memory_available=torch.tensor([True, False]))
    assert output.query_gate.tolist() == [[1.0], [0.0]]
    assert output.query_gate_logits is None
    assert output.should_query.tolist() == [True, False]
    assert not tuple(head.decision_gate.parameters())


def test_query_and_gate_gradients_are_finite_and_not_detached() -> None:
    hidden, summary = inputs()
    hidden.requires_grad_()
    summary.requires_grad_()
    head = RUTAMNeuralQueryHead(small_config())
    output = head(hidden, graph_summary=summary)
    target = torch.tensor([[0.1, 0.7, -0.2], [-0.3, 0.2, 0.5]])
    loss = (
        (output.query_vectors - target).square().mean()
        + query_decision_loss(output.query_gate_logits, torch.tensor([1, 0]))
        + output.query_gate.square().mean()
    )
    loss.backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.count_nonzero() > 0, name
    assert hidden.grad.count_nonzero() > 0 and summary.grad.count_nonzero() > 0


def test_decision_loss_matches_bce_uses_explicit_mask_and_handles_extremes() -> None:
    logits = torch.tensor([[0.3], [-0.7]], requires_grad=True)
    labels = torch.tensor([1, 0])
    loss = query_decision_loss(logits, labels)
    torch.testing.assert_close(
        loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[:, None].float())
    )
    masked = query_decision_loss(logits, labels, mask=torch.tensor([True, False]))
    masked.backward()
    assert logits.grad[0] != 0 and logits.grad[1] == 0
    extreme = torch.tensor([[-1e30], [1e30]], requires_grad=True)
    query_decision_loss(extreme, torch.tensor([1.0, 0.0])).backward()
    assert torch.isfinite(extreme.grad).all()
    assert query_decision_loss(extreme, labels, mask=torch.zeros(2, dtype=torch.bool)).item() == 0


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([2, 0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([float("nan"), 0.0]),
        torch.zeros(3),
        torch.zeros((2, 2)),
        torch.zeros(2, dtype=torch.float64),
    ],
)
def test_invalid_decision_labels(labels: torch.Tensor) -> None:
    with pytest.raises(ValidationError):
        query_decision_loss(torch.zeros((2, 1)), labels)


@pytest.mark.parametrize("method", ["last_valid", "mean", "attention"])
def test_all_masked_fallback_zeros_queries_and_disables_gate(method: str) -> None:
    head = RUTAMNeuralQueryHead(small_config(pooling=method, empty_sequence_policy="zeros")).eval()
    hidden = torch.randn((2, 5, 6), requires_grad=True)
    output = head(hidden, torch.zeros((2, 5), dtype=torch.bool), return_attention=True)
    assert output.query_vectors.count_nonzero() == output.query_gate.count_nonzero() == 0
    assert not output.should_query.any() and not output.valid_context.any()
    assert output.token_attention_weights.count_nonzero() == 0
    (output.query_vectors.sum() + output.query_gate.sum()).backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.count_nonzero() == 0


def test_normalization_toggle_and_residual_projection_are_explicit() -> None:
    hidden, _ = inputs()
    plain = RUTAMNeuralQueryHead(small_config(normalize_queries=False)).eval()
    normed = RUTAMNeuralQueryHead(small_config()).eval()
    normed.load_state_dict(plain.state_dict())
    torch.testing.assert_close(
        normed(hidden).query_vectors,
        torch.nn.functional.normalize(plain(hidden).query_vectors, dim=1),
    )
    residual = RUTAMNeuralQueryHead(
        small_config(normalize_queries=False, residual=True, residual_projection=True)
    ).eval()
    residual.load_state_dict(plain.state_dict(), strict=False)
    context = residual.pooling(hidden).context
    torch.testing.assert_close(
        residual(hidden).query_vectors,
        plain(hidden).query_vectors + residual.residual_projection(context),
    )
    identity = RUTAMNeuralQueryHead(small_config(query_dim=6, residual=True))
    assert isinstance(identity.residual_projection, torch.nn.Identity)


def test_query_norm_regularization_targets_unit_norm_instead_of_zero() -> None:
    vectors = torch.tensor([[1.0, 0.0], [2.0, 0.0]], requires_grad=True)
    assert query_norm_loss(vectors, weight=0.1).item() == pytest.approx(0.45)
    assert query_norm_loss(torch.zeros((1, 2))).item() == 1
    query_norm_loss(vectors, mask=torch.tensor([False, True])).backward()
    assert vectors.grad[0].count_nonzero() == 0 and vectors.grad[1, 0] > 0
    head = RUTAMNeuralQueryHead(small_config(norm_regularization_weight=0.1))
    output = head(inputs()[0])
    assert head.regularization_loss(output).item() < 1e-10


def test_eval_is_deterministic_and_training_dropout_is_active() -> None:
    hidden, summary = inputs()
    head = RUTAMNeuralQueryHead(small_config(dropout=0.5)).eval()
    torch.testing.assert_close(
        head(hidden, graph_summary=summary).query_vectors,
        head(hidden, graph_summary=summary).query_vectors,
        rtol=0,
        atol=0,
    )
    head.train()
    assert not torch.equal(head(hidden).query_vectors, head(hidden).query_vectors)


@pytest.mark.parametrize(
    "settings",
    [
        {"backbone_hidden_dim": 0},
        {"graph_hidden_dim": -1},
        {"query_dim": True},
        {"intermediate_dim": 1.5},
        {"pooling": "last"},
        {"dropout": -1},
        {"dropout": 1},
        {"dropout": float("nan")},
        {"gate_threshold": 1.1},
        {"gate_threshold": -1},
        {"gate_threshold": True},
        {"query_gate_enabled": 1},
        {"conditioning_mode": "concat"},
        {"empty_sequence_policy": "ignore"},
        {"residual": True},
        {"residual_projection": True},
        {"norm_regularization_weight": -1},
        {"norm_regularization_weight": float("inf")},
    ],
)
def test_invalid_query_configuration(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        small_config(**settings)


def test_selector_compatibility_config_round_trip_and_legacy_defaults(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="query_dim"):
        RUTAMNeuralQueryHead(small_config(), selector_config=SelectorConfig(query_dim=4))
    head = RUTAMNeuralQueryHead(small_config(), selector_config=SelectorConfig(query_dim=3))
    assert head.config.query_dim == 3
    config = RUTAMConfig().with_overrides({"query_head": {"backbone_hidden_dim": 12}})
    path = tmp_path / "config.yaml"
    path.write_text(config.to_yaml())
    assert RUTAMConfig.from_yaml(path) == config
    old = config.to_dict()
    old.pop("query_head")
    assert RUTAMConfig.from_dict(old).query_head == QueryHeadConfig()
    with pytest.raises(ValidationError):
        config.with_overrides({"query_head": {"unsupported": 1}})


def test_invalid_summary_availability_and_model_precision() -> None:
    hidden, _ = inputs()
    head = RUTAMNeuralQueryHead(small_config())
    for summary in (
        torch.ones(3),
        torch.ones((1, 4)),
        torch.ones((2, 5)),
        torch.full((2, 4), float("inf")),
    ):
        with pytest.raises(ValidationError):
            head(hidden, graph_summary=summary)
    for mask in (torch.ones(2), torch.ones(1, dtype=torch.bool)):
        with pytest.raises(ValidationError):
            head(hidden, memory_available=mask)
    with pytest.raises(ValidationError):
        head.double()(hidden)


def test_default_trainable_parameter_count() -> None:
    head = RUTAMNeuralQueryHead()
    assert (
        sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
        == 182402
    )
