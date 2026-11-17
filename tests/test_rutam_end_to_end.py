"""End-to-end multi-step integration tests across the complete RUTAM graph memory stack."""

from __future__ import annotations

from pathlib import Path

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.validation_output import ValidationDecision
from rutam.memory.write_proposal import MemoryOperation
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def test_complete_end_to_end_memory_lifecycle(tmp_path: Path) -> None:
    """Verify complete lifecycle: inference -> proposal -> validation -> commit -> reload."""
    torch.manual_seed(42)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16, "path": str(tmp_path / "graph")},
            "encoder": {"input_dim": 16, "hidden_dim": 16, "num_relation_types": 4},
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
            "graph_fusion": {
                "hidden_dim": 16,
                "graph_dim": 16,
                "num_attention_heads": 2,
                "residual_scale_init": 0.5,
            },
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
                "write_threshold": 0.01,
            },
            "rutam_model": {
                "return_memory_diagnostics": True,
            },
        }
    )

    storage = GraphStorage(config.graph.path)
    graph = Graph(config)

    # Initial graph population
    gen = torch.Generator().manual_seed(123)
    graph.create_node(
        node_type="concept",
        content="solar_cell",
        node_id=1,
        provenance="doc:pv_intro",
        embedding=torch.randn(16, generator=gen).tolist(),
    )
    graph.create_node(
        node_type="concept",
        content="silicon_wafer",
        node_id=2,
        provenance="doc:pv_intro",
        embedding=torch.randn(16, generator=gen).tolist(),
    )
    graph.add_relationship(1, 2, "composed_of", provenance="doc:pv_intro")
    storage.save(graph)

    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=80, num_layers=3)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph, storage=storage)
    with torch.no_grad():
        model.write_head.operation_predictor.linear2.bias[MemoryOperation.CREATE_NODE.value] = 10.0
    model.write_head.eval()

    # STEP 1: Initial forward pass using knowledge graph
    input_ids = torch.randint(0, 80, (1, 6), dtype=torch.long)
    out1 = model(input_ids)
    assert out1.graph_selection_output is not None
    assert out1.graph_version == graph.version
    assert out1.fusion_output is not None
    logits_before = out1.logits.clone()

    # STEP 2: Generate memory update proposals
    auth_ctx = AuthorizationContext(user_id="researcher", role="admin")
    proposals = model.propose_memory_updates(input_ids, authorization_context=auth_ctx)
    assert len(proposals) > 0

    # STEP 3: Validate memory update proposals
    validation = model.validate_memory_updates(proposals, authorization_context=auth_ctx)
    accepted = [p for p, r in zip(proposals, validation.results, strict=True) if r.is_accepted]
    assert len(accepted) > 0

    # STEP 4: Transactionally commit proposals
    commit_res = model.commit_memory_updates(accepted, authorization_context=AuthorizationContext(permissions=frozenset({"commit"})))
    assert commit_res.committed is True
    assert graph.version > out1.graph_version

    # Persist updated graph to storage
    storage.save(graph)

    # STEP 5: Reload model from persistent storage
    model.reload_graph()
    assert model.graph.version == commit_res.new_graph_version

    # STEP 6: Subsequent forward pass with updated graph
    out2 = model(input_ids)
    assert out2.graph_version == commit_res.new_graph_version
    # Representations are modulated by the updated graph state
    assert not torch.allclose(logits_before, out2.logits, atol=1e-5)


def test_stale_update_detection(tmp_path: Path) -> None:
    """Verify validator flags and rejects proposals based on stale graph versions."""
    torch.manual_seed(42)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16, "path": str(tmp_path / "graph")},
            "encoder": {"input_dim": 16, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
            },
            "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
                "write_threshold": 0.01,
            },
        }
    )
    graph = Graph(config)
    gen = torch.Generator().manual_seed(99)
    graph.create_node(
        node_type="concept",
        content="base_node",
        node_id=1,
        provenance="test:base",
        embedding=torch.randn(16, generator=gen).tolist(),
    )
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=60, num_layers=2)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)
    with torch.no_grad():
        model.write_head.operation_predictor.linear2.bias[MemoryOperation.CREATE_NODE.value] = 10.0
    model.write_head.eval()

    auth_ctx = AuthorizationContext(user_id="user_admin", role="admin")
    input_ids = torch.randint(0, 60, (1, 4), dtype=torch.long)
    proposals = model.propose_memory_updates(input_ids, authorization_context=auth_ctx)
    assert len(proposals) > 0

    # Advance graph version externally before validation
    graph.create_node(
        node_type="concept",
        content="concurrent_node",
        node_id=2,
        provenance="test:concurrent",
        embedding=torch.randn(16, generator=gen).tolist(),
    )

    # Validate proposals generated against previous graph version
    validation = model.validate_memory_updates(proposals, authorization_context=auth_ctx)
    assert validation.stale_count > 0 or any(
        r.decision == ValidationDecision.STALE for r in validation.results
    )
