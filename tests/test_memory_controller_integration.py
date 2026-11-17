"""End-to-end integration tests for RUTAMMemoryController lifecycle across reading, proposals,
validation, transactional commits, storage persistence, and auditing.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage
from rutam.memory.controller import RUTAMMemoryController
from rutam.memory.memory_state import MemoryState
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.validation_output import ValidationDecision
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal


def _make_config(storage_path: Path, audit_path: Path) -> RUTAMConfig:
    return RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8, "path": str(storage_path)},
            "encoder": {"input_dim": 8, "hidden_dim": 8},
            "selector": {
                "query_dim": 8,
                "hidden_dim": 8,
                "score_dim": 8,
                "top_k": 2,
                "selection_mode": "soft",
            },
            "query_head": {
                "backbone_hidden_dim": 8,
                "query_dim": 8,
                "graph_hidden_dim": 8,
                "intermediate_dim": 8,
                "pooling": "mean",
            },
            "traversal": {
                "query_dim": 8,
                "hidden_dim": 8,
                "path_hidden_dim": 8,
                "score_hidden_dim": 8,
                "scoring_chunk_size": 8,
                "max_candidate_paths": 8,
                "max_selected_paths": 2,
            },
            "memory_write": {
                "backbone_hidden_dim": 8,
                "hidden_dim": 8,
                "projection_dim": 8,
                "max_proposals": 2,
            },
            "memory_controller": {
                "enabled": True,
                "read_enabled": True,
                "write_enabled": True,
                "auto_commit": False,
                "audit_enabled": True,
                "preserve_snapshots": 3,
            },
        }
    )


def test_complete_memory_controller_integration(tmp_path: Path) -> None:
    torch.manual_seed(42)
    storage_dir = tmp_path / "graph_store"
    audit_file = tmp_path / "controller_audit.jsonl"
    config = _make_config(storage_dir, audit_file)

    # 1. Initialize graph with seed knowledge
    graph = Graph(config)
    graph.create_node(
        "concept",
        "Artificial Intelligence",
        node_id=1,
        embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        provenance="seed:init",
    )
    graph.create_node(
        "concept",
        "Neural Networks",
        node_id=2,
        embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        provenance="seed:init",
    )
    graph.add_relationship(2, 1, "SUBFIELD_OF", provenance="seed:init", weight=0.95)

    storage = GraphStorage(storage_dir)
    controller = RUTAMMemoryController(
        config,
        graph=graph,
        storage=storage,
        audit_path=audit_file,
        traversal_enabled=True,
    )

    assert controller.state == MemoryState.READY
    status_0 = controller.get_memory_status()
    assert status_0.node_count == 2
    assert status_0.edge_count == 1
    assert status_0.total_commits == 0

    # 2. Differentiable read pass
    lm_hidden = torch.randn(2, 5, 8, requires_grad=True)
    read_output = controller.read_memory(lm_hidden)

    assert read_output.gated_memory_context is not None
    assert read_output.gated_memory_context.shape == (2, 8)
    assert read_output.traversal_output is not None

    # Verify gradient flow back through query head and graph selection
    task_loss = read_output.gated_memory_context.sum()
    task_loss.backward()

    assert lm_hidden.grad is not None
    assert lm_hidden.grad.shape == (2, 5, 8)
    assert lm_hidden.grad.abs().sum() > 0.0

    # 3. Propose updates via neural write pipeline
    write_lm_hidden = torch.randn(1, 4, 8)
    auth = AuthorizationContext(user_id="researcher_1", role="admin")

    write_pipeline_output = controller.propose_memory_update(
        hidden_states=write_lm_hidden,
        selected_memory_states=read_output.selected_node_states[:1],
        selected_memory_ids=read_output.selected_node_ids[:1],
        authorization_context=auth,
        auto_validate=True,
    )
    assert write_pipeline_output.write_output is not None
    assert write_pipeline_output.validation_result is not None

    # 4. Construct a validated proposal to commit
    prov = ProvenanceRecord(
        source_id="researcher_1",
        authorization_category="user_authorized",
        evidence="Observed relationship in text corpus",
    )
    cand_emb = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    new_proposal = MemoryProposal(
        proposal_id="proposal_deep_learning",
        operation=MemoryOperation.CREATE_NODE,
        candidate_embedding=cand_emb,
        provenance=prov,
        confidence=0.92,
    )
    val_result = controller.validate_updates([new_proposal], authorization_context=auth)
    assert val_result.results[0].decision == ValidationDecision.ACCEPT

    # 5. Atomic commit
    commit_result = controller.commit_validated_updates(
        [new_proposal],
        authorization_context=AuthorizationContext(
            user_id="researcher_1",
            role="admin",
            permissions=frozenset({"commit"}),
        ),
    )
    assert commit_result.committed
    assert commit_result.applied_mutations == 1
    assert commit_result.new_graph_version > 0

    # 6. Verify controller state and status
    status_1 = controller.get_memory_status()
    assert status_1.node_count == 3
    assert status_1.total_commits == 1
    assert status_1.last_commit_timestamp is not None

    active_state = controller.get_active_state()
    assert active_state.node_ids.shape[0] == 3

    # 7. Subsequent read reflects new node
    read_output_2 = controller.read_memory(lm_hidden.detach())
    assert read_output_2.gated_memory_context.shape == (2, 8)

    # 8. Verify audit trail on disk
    assert audit_file.exists()
    audit_lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) >= 1
    record = json.loads(audit_lines[0])
    assert record["operation"] == "CREATE_NODE"
    assert record["status"] == "COMMITTED"
    assert record["transaction_id"] == commit_result.transaction_id

    # 9. Verify storage persistence
    assert (storage_dir / "graph.safetensors").exists()
    assert (storage_dir / "graph.json").exists()
    assert (storage_dir / "graph.manifest.json").exists()
