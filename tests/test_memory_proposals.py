"""Tests for MemoryProposal data structures, MemoryOperation enum, and ProvenanceRecord."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest
import torch

from rutam.exceptions import ValidationError
from rutam.memory.write_proposal import (
    MemoryOperation,
    MemoryProposal,
    ProvenanceRecord,
)


def test_memory_operation_enum_mapping_and_parsing() -> None:
    # Stable versioned values 0..6
    assert MemoryOperation.NO_OP.value == 0
    assert MemoryOperation.CREATE_NODE.value == 1
    assert MemoryOperation.UPDATE_NODE.value == 2
    assert MemoryOperation.PROPOSE_LINK.value == 3
    assert MemoryOperation.PROPOSE_MERGE.value == 4
    assert MemoryOperation.SUPERSEDE_NODE.value == 5
    assert MemoryOperation.REQUEST_VALIDATION.value == 6

    # from_value parsing
    assert MemoryOperation.from_value(1) == MemoryOperation.CREATE_NODE
    assert MemoryOperation.from_value("create_node") == MemoryOperation.CREATE_NODE
    assert MemoryOperation.from_value("SUPERSEDE_NODE") == MemoryOperation.SUPERSEDE_NODE
    assert MemoryOperation.from_value(MemoryOperation.NO_OP) == MemoryOperation.NO_OP

    with pytest.raises(ValidationError):
        MemoryOperation.from_value(99)

    with pytest.raises(ValidationError):
        MemoryOperation.from_value(-1)

    with pytest.raises(ValidationError):
        MemoryOperation.from_value("UNKNOWN_OP")

    with pytest.raises(ValidationError):
        MemoryOperation.from_value(True)  # booleans are not valid op codes


def test_provenance_record_creation_and_serialization() -> None:
    rec = ProvenanceRecord(source_id="model_turn_42")
    assert rec.source_id == "model_turn_42"
    assert rec.source_category == "neural_write_head"
    assert rec.authorization_category == "unverified_model_hypothesis"
    assert rec.observation_timestamp.utcoffset() is not None
    assert rec.creation_timestamp.utcoffset() is not None

    d = rec.to_dict()
    assert d["source_id"] == "model_turn_42"
    assert d["source_category"] == "neural_write_head"
    assert "observation_timestamp" in d

    with pytest.raises(ValidationError):
        ProvenanceRecord(source_id="")

    with pytest.raises(ValidationError):
        ProvenanceRecord(source_id="valid", observation_timestamp=datetime.now())  # naive dt


def test_memory_proposal_construction_and_immutability() -> None:
    emb = torch.randn(16, dtype=torch.float32)
    prop = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        candidate_embedding=emb,
        target_node_ids=(10, 20),
        proposed_link_ids=(1,),
        confidence=0.85,
        source_graph_version=3,
    )

    assert prop.operation == MemoryOperation.CREATE_NODE
    assert prop.candidate_embedding is not None
    assert prop.candidate_embedding.shape == (16,)
    assert prop.target_node_ids == (10, 20)
    assert prop.proposed_link_ids == (1,)
    assert prop.confidence == 0.85
    assert prop.source_graph_version == 3
    assert prop.origin_category == "neural_write_head"
    assert prop.validation_status == "pending"
    assert prop.timestamp.utcoffset() is not None

    # Immutability
    with pytest.raises(FrozenInstanceError):
        prop.confidence = 0.5  # type: ignore[misc]

    # Serialization
    d = prop.to_dict()
    assert d["operation"] == "CREATE_NODE"
    assert len(d["candidate_embedding"]) == 16
    assert d["target_node_ids"] == [10, 20]
    assert d["confidence"] == 0.85


def test_memory_proposal_candidate_embedding_validation() -> None:
    # 2D candidate embedding should be rejected
    with pytest.raises(ValidationError):
        MemoryProposal(candidate_embedding=torch.randn(2, 16))

    # Non-float32 candidate embedding should be rejected
    with pytest.raises(ValidationError):
        MemoryProposal(candidate_embedding=torch.zeros(16, dtype=torch.float64))

    # NaN / Inf in candidate embedding should be rejected
    nan_emb = torch.randn(16)
    nan_emb[0] = float("nan")
    with pytest.raises(ValidationError):
        MemoryProposal(candidate_embedding=nan_emb)

    inf_emb = torch.randn(16)
    inf_emb[1] = float("inf")
    with pytest.raises(ValidationError):
        MemoryProposal(candidate_embedding=inf_emb)


def test_memory_proposal_bounds_and_error_cases() -> None:
    with pytest.raises(ValidationError):
        MemoryProposal(proposal_id="")

    with pytest.raises(ValidationError):
        MemoryProposal(confidence=-0.1)

    with pytest.raises(ValidationError):
        MemoryProposal(confidence=1.1)

    with pytest.raises(ValidationError):
        MemoryProposal(confidence=float("nan"))

    with pytest.raises(ValidationError):
        MemoryProposal(source_graph_version=-1)

    with pytest.raises(ValidationError):
        MemoryProposal(target_node_ids=(-5,))

    with pytest.raises(ValidationError):
        MemoryProposal(target_node_ids=("not_an_int",))  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        MemoryProposal(timestamp=datetime.now())  # naive datetime
