"""Tests for RUTAMMemoryValidator end-to-end validation, decisions, and no-commit invariant."""

from __future__ import annotations

import pytest

from rutam import (
    Graph,
    MemoryOperation,
    MemoryProposal,
    MemoryValidationConfig,
    MemoryValidationResult,
    Node,
    ProvenanceRecord,
    RUTAMConfig,
    RUTAMMemoryValidator,
    ValidationBatchResult,
    ValidationDecision,
)
from rutam.exceptions import ValidationError


def test_validator_initialization_and_config() -> None:
    # Default init
    v1 = RUTAMMemoryValidator()
    assert v1.config.enabled
    assert v1.config.require_provenance

    # Init from RUTAMConfig
    cfg = RUTAMConfig()
    v2 = RUTAMMemoryValidator(cfg)
    assert v2.config == cfg.memory_validation

    # Init from MemoryValidationConfig
    v3 = RUTAMMemoryValidator(MemoryValidationConfig(min_confidence=0.8))
    assert v3.config.min_confidence == 0.8

    with pytest.raises(ValidationError):
        RUTAMMemoryValidator("invalid_config")  # type: ignore[arg-type]


def test_validator_accept_valid_proposal() -> None:
    graph = Graph()
    validator = RUTAMMemoryValidator(MemoryValidationConfig(allow_unverified_hypotheses=True))

    prop = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        provenance=ProvenanceRecord(
            source_id="doc_1",
            authorization_category="unverified_model_hypothesis",
        ),
        confidence=0.95,
        source_graph_version=graph.version,
    )

    res = validator.validate(prop, graph)
    assert isinstance(res, MemoryValidationResult)
    assert res.decision == ValidationDecision.ACCEPT
    assert res.is_accepted
    assert "PASSED_ALL_CHECKS" in res.reason_codes
    assert res.sanitized_proposal is prop


def test_validator_rejection_hierarchy_and_confidence() -> None:
    graph = Graph()
    validator = RUTAMMemoryValidator(MemoryValidationConfig(allow_unverified_hypotheses=False))

    # 1. Invalid target node should REJECT even if confidence is 1.0
    prop_invalid_target = MemoryProposal(
        operation=MemoryOperation.UPDATE_NODE,
        target_node_ids=(999,),  # missing
        confidence=1.0,
        provenance=ProvenanceRecord(source_id="valid", authorization_category="user_input"),
    )
    res = validator.validate(prop_invalid_target, graph)
    assert res.decision == ValidationDecision.REJECT
    assert "TARGET_NODE_NOT_FOUND" in res.reason_codes

    # 2. Missing authorization should REJECT even with confidence 1.0
    prop_unverified = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        confidence=1.0,
        provenance=ProvenanceRecord(
            source_id="hallucinated",
            authorization_category="unverified_model_hypothesis",
        ),
    )
    res_auth = validator.validate(prop_unverified, graph)
    assert res_auth.decision == ValidationDecision.REJECT
    assert "UNVERIFIED_HYPOTHESIS_DISALLOWED" in res_auth.reason_codes

    # 3. Low confidence should DEFER when all else is valid
    validator_lenient = RUTAMMemoryValidator(
        MemoryValidationConfig(allow_unverified_hypotheses=True, min_confidence=0.80)
    )
    prop_low_conf = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        confidence=0.60,
        provenance=ProvenanceRecord(
            source_id="doc", authorization_category="unverified_model_hypothesis"
        ),
        source_graph_version=graph.version,
    )
    res_defer = validator_lenient.validate(prop_low_conf, graph)
    assert res_defer.decision == ValidationDecision.DEFER
    assert "LOW_CONFIDENCE" in res_defer.reason_codes
    assert res_defer.requires_further_verification


def test_validator_batch_validation_and_bounds() -> None:
    graph = Graph()
    v_cfg = MemoryValidationConfig(
        allow_unverified_hypotheses=True,
        max_proposals_per_batch=3,
    )
    validator = RUTAMMemoryValidator(v_cfg)

    proposals = [
        MemoryProposal(
            operation=MemoryOperation.CREATE_NODE,
            provenance="doc_1",
            confidence=0.9,
            source_graph_version=graph.version,
        ),
        MemoryProposal(
            operation=MemoryOperation.UPDATE_NODE,
            target_node_ids=(999,),  # will fail
            provenance="doc_2",
            confidence=0.9,
            source_graph_version=graph.version,
        ),
    ]

    batch_res = validator.validate_batch(proposals, graph)
    assert isinstance(batch_res, ValidationBatchResult)
    assert batch_res.accepted_count == 1
    assert batch_res.rejected_count == 1
    assert not batch_res.all_valid

    # Exceeding batch bounds
    with pytest.raises(ValidationError):
        validator.validate_batch(proposals * 3, graph)  # 6 > 3


def test_validator_no_commit_invariant() -> None:
    graph = Graph()
    graph.add_node(Node(id=1, node_type="entity", content="Node 1", provenance="doc"))
    initial_version = graph.version
    initial_node_count = len(graph.nodes)
    initial_edge_count = len(graph.edges)

    validator = RUTAMMemoryValidator()

    # Run validation on proposals
    prop = MemoryProposal(operation=MemoryOperation.CREATE_NODE, provenance="doc")
    validator.validate(prop, graph)

    # Verify zero mutations occurred
    assert graph.version == initial_version
    assert len(graph.nodes) == initial_node_count
    assert len(graph.edges) == initial_edge_count
