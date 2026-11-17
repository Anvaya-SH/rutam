"""Unit tests for RUTAMMemoryLearningObjective and individual memory losses."""

import torch

from rutam.config import MemoryLearningConfig
from rutam.training.memory_losses import (
    embedding_alignment_loss,
    operation_classification_loss,
    write_decision_loss,
)
from rutam.training.memory_metrics import (
    binary_classification_metrics,
    mean_reciprocal_rank,
    multi_class_accuracy,
    recall_at_k,
)
from rutam.training.memory_objectives import RUTAMMemoryLearningObjective
from rutam.training.memory_targets import MemoryTrainingTarget


def test_write_decision_loss_and_metrics():
    logits = torch.tensor([2.0, -2.0, 1.5, -1.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss, count = write_decision_loss(logits, targets)
    assert count == 4
    assert loss > 0
    assert not torch.isnan(loss)

    probs = torch.sigmoid(logits)
    metrics = binary_classification_metrics(probs, targets)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_operation_classification_loss_and_metrics():
    # 3 samples, 7 operation classes
    logits = torch.randn(3, 7)
    targets = torch.tensor([1, 0, 2], dtype=torch.int64)

    loss, count = operation_classification_loss(logits, targets)
    assert count == 3
    assert loss > 0

    acc = multi_class_accuracy(logits, targets)
    assert 0.0 <= acc <= 1.0


def test_embedding_alignment_loss():
    # Identical proposed and reference embeddings should produce near-zero cosine loss
    ref = torch.randn(4, 32)
    prop = ref.clone()

    loss, count = embedding_alignment_loss(prop, ref, method="cosine")
    assert count == 4
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_graph_retrieval_metrics():
    logits = torch.tensor([[10.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    targets = torch.tensor([0, 1], dtype=torch.int64)

    assert recall_at_k(logits, targets, k=1) == 1.0
    assert recall_at_k(logits, targets, k=3) == 1.0
    assert mean_reciprocal_rank(logits, targets) == 1.0


def test_memory_learning_objective_missing_supervision():
    objective = RUTAMMemoryLearningObjective(MemoryLearningConfig())

    # Targets with only write_decision, missing operation_label
    targets = [
        MemoryTrainingTarget(example_id="t1", write_decision=True),
        MemoryTrainingTarget(example_id="t2", write_decision=False),
    ]

    from rutam.memory.write_proposal import MemoryWriteOutput

    write_out = MemoryWriteOutput(
        write_logits=torch.randn(2, 1),
        write_probabilities=torch.sigmoid(torch.randn(2, 1)),
        operation_logits=torch.randn(2, 7),
        operation_probabilities=torch.softmax(torch.randn(2, 7), dim=-1),
        proposed_embeddings=torch.randn(2, 1, 32),
        target_memory_scores=None,
        proposal_mask=torch.ones((2, 1), dtype=torch.bool),
        write_decisions=None,
    )

    out = objective(write_output=write_out, targets=targets)

    assert "write_decision" in out.loss_components
    assert "operation" not in out.loss_components
    assert out.valid_counts["write_decision"] == 2
    assert "No valid operation supervision in batch" in out.warnings
