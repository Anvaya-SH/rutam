"""Tests for memory write proposals, discrete validation, and transactional commits."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.memory.controller_output import CommitResult
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.validation_output import ValidationBatchResult
from rutam.memory.write_proposal import MemoryProposal
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def _create_memory_model() -> tuple[RUTAMForCausalLM, Graph]:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
            },
            "traversal": {
                "hidden_dim": 16,
                "path_hidden_dim": 16,
                "query_dim": 8,
                "num_relation_types": 4,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
                "write_threshold": 0.01,  # Lower threshold to reliably trigger proposals
            },
            "rutam_model": {
                "memory_write_enabled": True,
                "memory_auto_commit": False,
            },
        }
    )
    graph = Graph(config)
    gen = torch.Generator().manual_seed(55)
    for i in range(2):
        graph.create_node(
            node_type="concept",
            content=f"initial_{i}",
            node_id=(i + 1) * 10,
            provenance="test:initial",
            embedding=torch.randn(8, generator=gen).tolist(),
        )

    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)
    return model, graph


def test_propose_memory_updates_api() -> None:
    """Verify propose_memory_updates generates structured MemoryProposal instances."""
    torch.manual_seed(42)
    model, _ = _create_memory_model()

    auth_ctx = AuthorizationContext(user_id="user_admin", role="admin")
    input_ids = torch.randint(0, 50, (1, 6), dtype=torch.long)
    proposals = model.propose_memory_updates(input_ids, authorization_context=auth_ctx)

    assert isinstance(proposals, tuple)
    assert len(proposals) > 0
    for prop in proposals:
        assert isinstance(prop, MemoryProposal)
        assert prop.operation is not None


def test_validate_and_commit_memory_updates() -> None:
    """Verify validation and transactional commitment of staged proposals."""
    torch.manual_seed(42)
    model, graph = _create_memory_model()
    initial_version = graph.version

    auth_ctx = AuthorizationContext(user_id="user_admin", role="admin")
    input_ids = torch.randint(0, 50, (1, 6), dtype=torch.long)
    proposals = model.propose_memory_updates(input_ids, authorization_context=auth_ctx)

    assert len(proposals) > 0
    validation = model.validate_memory_updates(proposals, authorization_context=auth_ctx)
    assert isinstance(validation, ValidationBatchResult)

    accepted_proposals = [
        p for p, r in zip(proposals, validation.results, strict=True) if r.is_accepted
    ]
    if accepted_proposals:
        commit_res = model.commit_memory_updates(accepted_proposals)
        assert isinstance(commit_res, CommitResult)
        assert commit_res.committed is True
        assert graph.version > initial_version
        assert commit_res.new_graph_version == graph.version


def test_explicit_proposal_validation_rejection() -> None:
    """Verify unauthorized or invalid proposals are properly rejected by validator."""
    torch.manual_seed(42)
    model, _ = _create_memory_model()

    input_ids = torch.randint(0, 50, (1, 6), dtype=torch.long)
    proposals = model.propose_memory_updates(input_ids)
    assert len(proposals) > 0

    # Validate without valid authorization context (unverified model hypothesis)
    unauthorized_ctx = AuthorizationContext(
        user_id="untrusted",
        role="guest",
        authorized_categories=(),
        allow_unverified_hypotheses=False,
    )
    validation = model.validate_memory_updates(proposals, authorization_context=unauthorized_ctx)
    # Since config enforces require_authorization and disallows unverified hypotheses
    assert validation.rejected_count > 0
