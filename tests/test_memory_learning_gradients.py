"""Unit tests for backward gradient flow through memory learning objectives."""

import torch

from rutam.config import MemoryLearningConfig, MemoryWriteConfig
from rutam.memory.write_head import RUTAMMemoryWriteHead
from rutam.memory.write_proposal import MemoryOperation
from rutam.training.memory_objectives import RUTAMMemoryLearningObjective
from rutam.training.memory_targets import MemoryTrainingTarget


def test_gradient_flow_to_write_head():
    write_config = MemoryWriteConfig(backbone_hidden_dim=32, hidden_dim=32, projection_dim=32)
    write_head = RUTAMMemoryWriteHead(write_config)

    obj_config = MemoryLearningConfig()
    objective = RUTAMMemoryLearningObjective(obj_config)

    # Synthetic backbone hidden representations
    backbone_hidden = torch.randn(2, 4, 32, requires_grad=True)

    write_output = write_head(backbone_hidden)

    targets = [
        MemoryTrainingTarget(
            example_id="t1",
            write_decision=True,
            operation_label=MemoryOperation.CREATE_NODE,
            reference_embeddings=torch.randn(32),
        ),
        MemoryTrainingTarget(
            example_id="t2",
            write_decision=False,
            operation_label=MemoryOperation.NO_OP,
            reference_embeddings=torch.randn(32),
        ),
    ]

    out = objective(write_output=write_output, targets=targets)
    assert out.total_loss.requires_grad

    out.total_loss.backward()

    # Backbone gradient
    assert backbone_hidden.grad is not None
    assert torch.isfinite(backbone_hidden.grad).all()

    # Write gate gradient
    assert write_head.write_gate.linear1.weight.grad is not None
    assert torch.isfinite(write_head.write_gate.linear1.weight.grad).all()

    # Operation predictor gradient
    assert write_head.operation_predictor.linear1.weight.grad is not None
    assert torch.isfinite(write_head.operation_predictor.linear1.weight.grad).all()

    # Memory projection gradient
    assert write_head.memory_projection.linear1.weight.grad is not None
    assert torch.isfinite(write_head.memory_projection.linear1.weight.grad).all()
