"""End-to-end integration tests for RUTAMMemoryValidator with WriteHead, GraphState, and Config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rutam import (
    AuthorizationContext,
    Graph,
    MemoryOperation,
    MemoryProposal,
    MemoryValidationConfig,
    MemoryWriteConfig,
    Node,
    AuthorizationContext,
    ProvenanceRecord,
    RUTAMConfig,
    RUTAMMemoryValidator,
    RUTAMMemoryWriteHead,
    ValidationDecision,
)
from rutam.exceptions import ValidationError


def test_config_roundtrip_with_memory_validation(tmp_path: Path) -> None:
    config = RUTAMConfig()
    assert config.memory_validation.enabled
    assert config.memory_validation.require_provenance
    assert not config.memory_validation.allow_unverified_hypotheses
    assert config.memory_validation.detect_duplicates
    assert config.memory_validation.detect_structural_conflicts
    assert config.memory_validation.reject_stale_updates
    assert config.memory_validation.require_authorization
    assert config.memory_validation.max_proposals_per_batch == 64
    assert config.memory_validation.max_proposal_bytes == 65536
    assert not config.memory_validation.semantic_verification_enabled
    assert config.memory_validation.default_decision == "defer"

    yaml_text = config.to_yaml()
    assert "memory_validation:" in yaml_text
    assert "max_proposals_per_batch: 64" in yaml_text

    path = tmp_path / "val_config.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    loaded = RUTAMConfig.from_yaml(path)
    assert loaded == config

    # Overrides
    overridden = config.with_overrides(
        {"memory_validation": {"max_proposals_per_batch": 32, "default_decision": "reject"}}
    )
    assert overridden.memory_validation.max_proposals_per_batch == 32
    assert overridden.memory_validation.default_decision == "reject"

    with pytest.raises(ValidationError):
        config.with_overrides({"memory_validation": {"default_decision": "unknown"}})


def test_end_to_end_write_head_proposals_to_validator() -> None:
    # 1. Setup Graph and state
    cfg = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}})
    graph = Graph(cfg)
    graph.add_node(
        Node(
            id=1,
            node_type="concept",
            content="Graph Theory",
            embedding=(0.1,) * 8,
            provenance="doc1",
        )
    )
    graph.add_node(
        Node(
            id=2,
            node_type="concept",
            content="Neural Memory",
            embedding=(0.2,) * 8,
            provenance="doc2",
        )
    )
    state = graph.to_state()

    # 2. Setup write head with authorized source references
    write_head = RUTAMMemoryWriteHead(
        MemoryWriteConfig(
            backbone_hidden_dim=16,
            hidden_dim=8,
            projection_dim=16,
            max_proposals=2,
            target_selection=True,
            write_threshold=0.0,
        )
    )
    with torch.no_grad():
        write_head.operation_predictor.linear2.bias[MemoryOperation.CREATE_NODE.value] = 10.0
    write_head.eval()

    hidden = torch.randn(2, 4, 16)
    selected_states = torch.randn(2, 2, 8)
    selected_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.int64)

    # Supply structured provenance reference
    auth_prov = ProvenanceRecord(
        source_id="model_verified_prompt",
        source_category="user_input",
        authorization_category="user_authorized",
    )

    write_out = write_head(
        hidden_states=hidden,
        selected_memory_states=selected_states,
        selected_memory_ids=selected_ids,
        source_references=[auth_prov, auth_prov],
        source_graph_version=graph.version,
    )

    # 3. Setup validator
    validator = RUTAMMemoryValidator(
        MemoryValidationConfig(
            allow_unverified_hypotheses=False,
            min_confidence=0.0,  # accept valid generated confidence
        ),
        graph_config=cfg.graph,
    )

    # Flatten proposals
    all_proposals = [p for batch in write_out.proposals for p in batch]
    assert len(all_proposals) > 0

    # 4. Validate against GraphState snapshot
    trusted_context = AuthorizationContext(
        trusted_provenance_fingerprints=(auth_prov.fingerprint(),)
    )
    batch_res = validator.validate_batch(
        all_proposals, state, authorization_context=trusted_context
    )
    assert batch_res.accepted_count > 0
    assert batch_res.rejected_count == 0

    # 5. Inject a stale proposal and verify rejection
    stale_prop = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        provenance=auth_prov,
        source_graph_version=-1 if graph.version == 0 else graph.version - 1,
    )
    stale_res = validator.validate(stale_prop, state, authorization_context=trusted_context)
    assert stale_res.decision == ValidationDecision.STALE
    assert "STALE_GRAPH_VERSION" in stale_res.reason_codes

    # 6. Inject unverified hypothesis
    unverified_prop = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        provenance=ProvenanceRecord(
            source_id="unauthorized",
            authorization_category="unverified_model_hypothesis",
        ),
        source_graph_version=graph.version,
    )
    unverified_res = validator.validate(unverified_prop, state)
    assert unverified_res.decision == ValidationDecision.REJECT
    assert "UNVERIFIED_HYPOTHESIS_DISALLOWED" in unverified_res.reason_codes

    # 7. Inject sensitive data
    secret_prop = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        provenance=ProvenanceRecord(
            source_id="leaked",
            authorization_category="user_authorized",
            evidence="Here is password := supersecret123",
        ),
        source_graph_version=graph.version,
    )
    secret_res = validator.validate(secret_prop, state)
    assert secret_res.decision == ValidationDecision.REJECT
    assert "SENSITIVE_DATA_DETECTED" in secret_res.reason_codes

    # 8. Security verification: confidence cannot override security
    secret_high_conf = MemoryProposal(
        operation=MemoryOperation.CREATE_NODE,
        confidence=1.0,
        provenance=ProvenanceRecord(
            source_id="leaked2",
            authorization_category="user_authorized",
            evidence="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.supertoken",
        ),
        source_graph_version=graph.version,
    )
    sec_high_res = validator.validate(secret_high_conf, state)
    assert sec_high_res.decision == ValidationDecision.REJECT
    assert "SENSITIVE_DATA_DETECTED" in sec_high_res.reason_codes

    # 9. Invariant verification: Graph and GraphState completely unmutated
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0
    assert graph.version == 2
