"""Unit tests for MemoryTrainingTarget schema and consistency validation."""

import pytest
import torch

from rutam.exceptions import ValidationError
from rutam.memory.write_proposal import MemoryOperation
from rutam.training.memory_targets import MemoryTrainingTarget


def test_valid_target_instantiation():
    target = MemoryTrainingTarget(
        example_id="ex_001",
        write_decision=True,
        operation_label=MemoryOperation.CREATE_NODE,
        target_node_ids=[1],
        relevant_node_ids=[1, 2],
        reference_embeddings=torch.randn(128),
        authorization_label=True,
        provenance="source:manual",
        graph_snapshot_id="snap_123",
        curriculum_stage="Stage_A",
    )
    assert target.example_id == "ex_001"
    assert target.operation_label == MemoryOperation.CREATE_NODE
    d = target.to_dict()
    assert d["example_id"] == "ex_001"
    assert d["operation_label"] == "CREATE_NODE"


def test_inconsistent_target_rejection():
    # NO_OP with write_decision=True must be rejected
    with pytest.raises(ValidationError, match="NO_OP operation is incompatible"):
        MemoryTrainingTarget(
            example_id="ex_inv",
            write_decision=True,
            operation_label=MemoryOperation.NO_OP,
        )

    # UPDATE_NODE with write_decision=False must be rejected
    with pytest.raises(ValidationError, match="UPDATE_NODE operation requires"):
        MemoryTrainingTarget(
            example_id="ex_inv2",
            write_decision=False,
            operation_label=MemoryOperation.UPDATE_NODE,
        )


def test_missing_and_partial_fields():
    # Targets can have optional and absent fields
    target = MemoryTrainingTarget(example_id="ex_minimal")
    assert target.write_decision is None
    assert target.operation_label is None
    assert target.reference_embeddings is None
    assert target.target_node_ids == []
