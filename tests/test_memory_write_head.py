"""Tests for RUTAMMemoryWriteHead, neural projections, target scoring, and write losses."""

from __future__ import annotations

import pytest
import torch

from rutam.config import MemoryWriteConfig
from rutam.exceptions import ValidationError
from rutam.memory.write_head import RUTAMMemoryWriteHead
from rutam.memory.write_losses import memory_write_loss
from rutam.memory.write_proposal import MemoryOperation


def test_write_head_forward_output_shapes_and_normalization() -> None:
    config = MemoryWriteConfig(
        backbone_hidden_dim=16,
        hidden_dim=8,
        projection_dim=32,
        max_proposals=3,
        operation_count=7,
        normalize_embeddings=True,
    )
    head = RUTAMMemoryWriteHead(config)

    b, t, d, h, m, o = 4, 10, 16, 8, 3, 7
    hidden = torch.randn(b, t, d)
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, 8:] = False  # padding on sample 0

    out = head(hidden, attention_mask=mask)

    assert out.write_logits.shape == (b, 1)
    assert out.write_probabilities.shape == (b, 1)
    assert out.operation_logits.shape == (b, o)
    assert out.operation_probabilities.shape == (b, o)
    assert out.proposed_embeddings.shape == (b, m, h)
    assert out.proposal_mask.shape == (b, m)
    assert out.target_memory_scores is None
    assert out.write_decisions is None  # training mode

    # Check softmax on operations
    torch.testing.assert_close(
        out.operation_probabilities.sum(dim=-1), torch.ones(b, dtype=torch.float32)
    )

    # Check L2 unit normalization of embeddings
    norms = out.proposed_embeddings.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_write_head_pooling_variants() -> None:
    for mode in ("last_valid", "mean", "attention"):
        config = MemoryWriteConfig(
            backbone_hidden_dim=16,
            hidden_dim=8,
            pooling=mode,
        )
        head = RUTAMMemoryWriteHead(config)
        hidden = torch.randn(2, 6, 16)
        mask = torch.tensor(
            [[True, True, True, False, False, False], [True, True, True, True, True, True]]
        )

        out = head(hidden, attention_mask=mask)
        assert out.write_logits.shape == (2, 1)
        assert out.proposed_embeddings.shape == (2, config.max_proposals, 8)


def test_write_head_graph_summary_conditioning() -> None:
    config = MemoryWriteConfig(
        backbone_hidden_dim=16,
        hidden_dim=8,
        use_graph_context=True,
    )
    head = RUTAMMemoryWriteHead(config)

    hidden = torch.randn(3, 5, 16)

    # Broadcast single summary [H]
    summary_1d = torch.randn(8)
    out_1d = head(hidden, graph_summary=summary_1d)
    assert out_1d.write_logits.shape == (3, 1)

    # Batched summary [B, H]
    summary_2d = torch.randn(3, 8)
    out_2d = head(hidden, graph_summary=summary_2d)
    assert out_2d.write_logits.shape == (3, 1)

    # Mismatched summary dimension
    with pytest.raises(ValidationError):
        head(hidden, graph_summary=torch.randn(3, 12))


def test_write_head_target_memory_selection() -> None:
    config = MemoryWriteConfig(
        backbone_hidden_dim=16,
        hidden_dim=8,
        target_selection=True,
    )
    head = RUTAMMemoryWriteHead(config)

    b, k, h = 2, 5, 8
    hidden = torch.randn(b, 4, 16)
    states = torch.randn(b, k, h)
    ids = torch.tensor([[101, 102, 103, 104, 105], [201, 202, 203, 204, 205]], dtype=torch.int64)

    out = head(hidden, selected_memory_states=states, selected_memory_ids=ids)
    assert out.target_memory_scores is not None
    assert out.target_memory_scores.shape == (b, k)
    assert out.target_memory_scores.dtype == torch.float32

    # Check validation errors on target memory
    with pytest.raises(ValidationError):
        head(hidden, selected_memory_states=torch.randn(b, k, 12))  # wrong H

    with pytest.raises(ValidationError):
        head(hidden, selected_memory_states=states, selected_memory_ids=torch.tensor([1, 2]))


def test_write_head_eval_decisions_and_proposals() -> None:
    config = MemoryWriteConfig(
        backbone_hidden_dim=16,
        hidden_dim=8,
        max_proposals=2,
        write_threshold=0.5,
    )
    head = RUTAMMemoryWriteHead(config)
    head.eval()

    hidden = torch.randn(2, 4, 16)
    out = head(hidden)

    assert out.write_decisions is not None
    assert out.write_decisions.shape == (2,)
    assert isinstance(out.proposals, tuple)
    assert len(out.proposals) == 2

    for sample_props in out.proposals:
        assert isinstance(sample_props, tuple)
        for p in sample_props:
            assert isinstance(p.operation, MemoryOperation)
            assert p.confidence >= 0.0
            assert p.candidate_embedding is not None


def test_write_head_gradient_flow_and_backprop() -> None:
    config = MemoryWriteConfig(
        backbone_hidden_dim=16,
        hidden_dim=8,
        projection_dim=32,
        max_proposals=2,
        target_selection=True,
    )
    head = RUTAMMemoryWriteHead(config)

    b, t, d, h, k = 3, 5, 16, 8, 4
    hidden = torch.randn(b, t, d, requires_grad=True)
    states = torch.randn(b, k, h, requires_grad=True)
    summary = torch.randn(b, h, requires_grad=True)

    out = head(hidden, graph_summary=summary, selected_memory_states=states)

    write_targets = torch.tensor([1.0, 0.0, 1.0])
    op_targets = torch.tensor([1, 0, 2], dtype=torch.int64)
    target_embeddings = torch.randn(b, h)
    target_node_labels = torch.tensor([0, 1, 2], dtype=torch.int64)

    losses = head.loss(
        out,
        write_targets=write_targets,
        operation_targets=op_targets,
        target_embeddings=target_embeddings,
        target_node_labels=target_node_labels,
    )

    assert losses.total.item() > 0
    assert losses.write.item() >= 0
    assert losses.operation.item() >= 0
    assert losses.embedding.item() >= 0
    assert losses.target.item() >= 0

    losses.total.backward()

    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert summary.grad is not None
    assert states.grad is not None


def test_memory_write_losses_isolated_and_weights() -> None:
    b, o, h, k = 4, 7, 8, 3
    write_logits = torch.randn(b, 1)
    write_targets = torch.tensor([1, 1, 0, 0], dtype=torch.int64)
    op_logits = torch.randn(b, o)
    op_targets = torch.tensor([1, 2, 0, 0], dtype=torch.int64)
    proposed_emb = torch.randn(b, 2, h)
    target_emb = torch.randn(b, h)
    target_scores = torch.randn(b, k)
    target_labels = torch.tensor([0, 1, 0, 2], dtype=torch.int64)

    # Test alpha weight scaling
    out1 = memory_write_loss(
        write_logits,
        write_targets,
        op_logits,
        op_targets,
        proposed_emb,
        target_emb,
        target_scores,
        target_labels,
        alpha=2.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
    )
    torch.testing.assert_close(out1.total, 2.0 * out1.write)

    # Test multi-label target scores BCE
    multi_labels = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    out2 = memory_write_loss(
        write_logits,
        write_targets,
        op_logits,
        target_memory_scores=target_scores,
        target_node_labels=multi_labels,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        delta=1.0,
    )
    assert out2.target.item() > 0

    # Error handling: non-binary write target
    with pytest.raises(ValidationError):
        memory_write_loss(write_logits, torch.tensor([2, 0, 1, 0]), op_logits)

    # Error handling: out of range operation target
    with pytest.raises(ValidationError):
        memory_write_loss(
            write_logits,
            write_targets,
            op_logits,
            operation_targets=torch.tensor([10, 0, 0, 0]),
        )
