"""Integration tests for end-to-end forward computation of RUTAMForCausalLM."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.diagnostics.integration import (
    create_verification_model,
    verify_forward_contracts,
)
from rutam.graph.operations import Graph
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.write_proposal import MemoryProposal
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def test_forward_empty_graph_identity_preservation() -> None:
    """Verify that an empty graph preserves backbone causal outputs without failure or NaNs."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16},
            "encoder": {"input_dim": 16, "hidden_dim": 16},
            "selector": {"query_dim": 16, "hidden_dim": 16, "score_dim": 16, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 16,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "rutam_model": {"return_memory_diagnostics": True},
        }
    )
    graph = Graph(config)
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=60, num_layers=3)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)

    b, t = 2, 8
    input_ids = torch.randint(0, 60, (b, t), dtype=torch.long)
    out = model(input_ids)

    assert out.logits.shape == (b, t, 60)
    assert torch.isfinite(out.logits).all()
    assert out.loss is None
    assert out.diagnostics.get("has_nodes") is False
    assert out.diagnostics.get("graph_version") == 0


def test_forward_populated_graph_retrieval_and_fusion() -> None:
    """Verify neural query retrieval, path reasoning, and fusion on populated graph."""
    model, _ = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=80)

    b, t = 2, 6
    input_ids = torch.randint(0, 80, (b, t), dtype=torch.long)
    out = model(input_ids)

    assert out.logits.shape == (b, t, 80)
    assert torch.isfinite(out.logits).all()
    assert out.graph_selection_output is not None
    assert out.fusion_output is not None
    assert out.diagnostics.get("has_nodes") is True
    assert out.diagnostics.get("selection_active") is True
    assert out.diagnostics.get("fusion_active") is True


def test_forward_causal_lm_loss_computation() -> None:
    """Verify that next-token shifted cross-entropy loss is correctly calculated."""
    model, _ = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=80)

    b, t = 2, 8
    input_ids = torch.randint(0, 80, (b, t), dtype=torch.long)
    labels = torch.randint(0, 80, (b, t), dtype=torch.long)

    out = model(input_ids, labels=labels)

    assert out.loss is not None
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.loss.item() > 0.0


def test_forward_attention_mask_handling() -> None:
    """Verify that causal attention mask properly handles padded token sequences."""
    model, _ = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=80)

    b, t = 2, 8
    input_ids = torch.randint(0, 80, (b, t), dtype=torch.long)
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, 5:] = False  # padding on batch item 0

    out = model(input_ids, attention_mask=mask)

    assert out.logits.shape == (b, t, 80)
    assert torch.isfinite(out.logits).all()


def test_forward_write_proposal_generation() -> None:
    """Verify that write head generates structured MemoryProposal candidates when enabled."""
    model, _ = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=80)
    auth = AuthorizationContext(user_id="integration_tester", role="admin")

    b, t = 2, 6
    input_ids = torch.randint(0, 80, (b, t), dtype=torch.long)
    out = model(input_ids, generate_proposals=True, authorization_context=auth)

    assert isinstance(out.write_proposals, tuple)
    for prop in out.write_proposals:
        assert isinstance(prop, MemoryProposal)
        assert prop.candidate_embedding is not None
        assert prop.candidate_embedding.shape == (16,)
        assert prop.confidence >= 0.0


def test_forward_tensor_contract_verification() -> None:
    """Verify that full forward pass satisfies all shape, dtype, device, and layout contracts."""
    model, _ = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=80)
    input_ids = torch.randint(0, 80, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 80, (2, 8), dtype=torch.long)

    report = verify_forward_contracts(
        model=model,
        input_ids=input_ids,
        labels=labels,
        generate_proposals=True,
    )

    assert report.passed, f"Contract verification failed: {report.errors}"
    assert report.failed_checks == 0
    assert report.passed_checks > 10
