"""Integration tests for joint memory learning architecture."""

import torch

from rutam.config import MemoryLearningConfig, MemoryWriteConfig, QueryHeadConfig
from rutam.memory.write_head import RUTAMMemoryWriteHead
from rutam.memory.write_proposal import MemoryOperation
from rutam.model.query_head import RUTAMNeuralQueryHead
from rutam.training.memory_objectives import RUTAMMemoryLearningObjective
from rutam.training.memory_targets import MemoryTrainingTarget


def test_joint_memory_learning_forward():
    write_head = RUTAMMemoryWriteHead(
        MemoryWriteConfig(backbone_hidden_dim=32, hidden_dim=32, projection_dim=32)
    )
    query_head = RUTAMNeuralQueryHead(
        QueryHeadConfig(backbone_hidden_dim=32, graph_hidden_dim=32, query_dim=32)
    )
    objective = RUTAMMemoryLearningObjective(MemoryLearningConfig())

    hidden = torch.randn(2, 6, 32)

    write_out = write_head(hidden)
    query_out = query_head(hidden)

    targets = [
        MemoryTrainingTarget(
            example_id="s1",
            write_decision=True,
            operation_label=MemoryOperation.CREATE_NODE,
            relevant_node_ids=[0, 1],
            reference_embeddings=torch.randn(32),
        ),
        MemoryTrainingTarget(
            example_id="s2",
            write_decision=False,
            operation_label=MemoryOperation.NO_OP,
            relevant_node_ids=[],
            reference_embeddings=None,
        ),
    ]

    retrieval_logits = torch.randn(2, 8)
    retrieval_targets = torch.tensor([1, 4], dtype=torch.int64)

    out = objective(
        write_output=write_out,
        query_output=query_out,
        targets=targets,
        retrieval_logits=retrieval_logits,
        retrieval_targets=retrieval_targets,
    )

    assert out.total_loss.item() > 0.0
    assert not torch.isnan(out.total_loss).any()
    assert "write_decision" in out.loss_components
    assert "operation" in out.loss_components
    assert "embedding" in out.loss_components
    assert "query_gate" in out.loss_components
    assert "retrieval" in out.loss_components
    assert "write_precision" in out.metrics
    assert "operation_accuracy" in out.metrics
