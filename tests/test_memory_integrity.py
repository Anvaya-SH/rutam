"""Tests for structural invariants, sensitive data filtering, and duplicate detection."""

from __future__ import annotations

import torch

from rutam import Graph, MemoryOperation, MemoryProposal, Node, RUTAMConfig
from rutam.config import GraphConfig, MemoryValidationConfig
from rutam.memory.duplicate_detection import detect_duplicates
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.validation_rules import (
    check_sensitive_data,
    check_structural_invariants,
)


def test_structural_invariants_missing_target_node() -> None:
    graph = Graph()
    g_cfg = GraphConfig()
    v_cfg = MemoryValidationConfig()

    # Empty graph, update node 99 -> target node not found
    prop = MemoryProposal(
        operation=MemoryOperation.UPDATE_NODE,
        target_node_ids=(99,),
    )
    valid, reason = check_structural_invariants(prop, graph, g_cfg, v_cfg)
    assert not valid
    assert reason == "TARGET_NODE_NOT_FOUND"


def test_structural_invariants_embedding_checks() -> None:
    graph = Graph()
    g_cfg = GraphConfig(embedding_dim=8)
    v_cfg = MemoryValidationConfig()

    # Mismatched dimension (16 instead of 8)
    prop_wrong_dim = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        candidate_embedding=torch.randn(16),
    )
    valid, reason = check_structural_invariants(prop_wrong_dim, graph, g_cfg, v_cfg)
    assert not valid
    assert reason == "EMBEDDING_DIMENSION_MISMATCH"

    # Non-finite embedding
    nan_tensor = torch.randn(8)
    nan_tensor[0] = float("nan")

    # Note: MemoryProposal.__post_init__ catches NaN, so test duck-typed proposal object
    class NonFiniteProposal:
        operation = MemoryOperation.CREATE_NODE
        candidate_embedding = nan_tensor

    valid, reason = check_structural_invariants(NonFiniteProposal(), graph, g_cfg, v_cfg)
    assert not valid
    assert reason == "NON_FINITE_EMBEDDING"


def test_structural_invariants_capacity_limits() -> None:
    g_cfg = GraphConfig(max_nodes=1)
    v_cfg = MemoryValidationConfig()
    cfg = RUTAMConfig().with_overrides({"graph": {"max_nodes": 1}})
    graph = Graph(cfg)
    graph.add_node(Node(id=1, node_type="entity", content="E1", provenance="doc"))

    # Attempt to create node when capacity is full
    prop = MemoryProposal(operation=MemoryOperation.CREATE_NODE)
    valid, reason = check_structural_invariants(prop, graph, g_cfg, v_cfg)
    assert not valid
    assert reason == "GRAPH_MAX_NODES_EXCEEDED"


def test_sensitive_data_detection() -> None:
    # 1. AWS API Key detection
    prop_aws = MemoryProposal(
        provenance="Found in source with key AKIAIOSFODNN7EXAMPLE",
    )
    valid, reason = check_sensitive_data(prop_aws)
    assert not valid
    assert reason == "SENSITIVE_DATA_DETECTED"

    # 2. Bearer token detection
    prop_bearer = MemoryProposal(
        provenance="Authorization: Bearer ya29.a0AfH6SMAexampletokenwithsufficientlength",
    )
    valid, reason = check_sensitive_data(prop_bearer)
    assert not valid
    assert reason == "SENSITIVE_DATA_DETECTED"

    # 3. Password assignment detection
    prop_pw = MemoryProposal(
        provenance="DB config password := mysecretpassword123",
    )
    valid, reason = check_sensitive_data(prop_pw)
    assert not valid
    assert reason == "SENSITIVE_DATA_DETECTED"

    # 4. Allowed when authorization context explicitly permits
    auth_allowed = AuthorizationContext(allow_sensitive_data=True)
    valid, reason = check_sensitive_data(prop_pw, auth_allowed)
    assert valid
    assert reason is None


def test_duplicate_detection_scenarios() -> None:
    cfg = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}})
    graph = Graph(cfg)
    emb = (0.5,) * 8
    graph.add_node(
        Node(
            id=10,
            node_type="concept",
            content="Quantum Computing",
            embedding=emb,
            provenance="doc",
        )
    )

    # 1. Duplicate content
    class DuplicateContentProposal:
        content = "  quantum   computing  "

    is_dup, reason, matches = detect_duplicates(DuplicateContentProposal(), graph)
    assert is_dup
    assert reason == "DUPLICATE_CONTENT"
    assert matches == (10,)

    # 2. Duplicate vector similarity
    similar_tensor = torch.tensor(emb, dtype=torch.float32)  # similarity = 1.0
    prop_similar = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        candidate_embedding=similar_tensor,
    )
    is_dup, reason, matches = detect_duplicates(prop_similar, graph, similarity_threshold=0.98)
    assert is_dup
    assert reason == "DUPLICATE_SIMILARITY"
    assert matches == (10,)
