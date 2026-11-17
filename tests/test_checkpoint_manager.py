"""Unit tests for RUTAMCheckpointManager coordinating model, graph, and optimizer persistence."""

from __future__ import annotations

import pytest
import torch

from rutam.checkpoint.manager import RUTAMCheckpointManager
from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.model.rutam_model import RUTAMForCausalLM


def _make_small_model_and_graph() -> tuple[RUTAMForCausalLM, Graph]:
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
            "graph_fusion": {
                "hidden_dim": 16,
                "graph_dim": 16,
                "num_attention_heads": 2,
            },
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
            },
            "rutam_model": {"backbone_frozen": True, "return_memory_diagnostics": True},
        }
    )
    graph = Graph(config)
    graph.create_node(
        node_type="concept",
        content="Alpha",
        embedding=tuple([0.1] * 8),
        node_id=1,
        provenance="test:provenance",
    )
    graph.create_node(
        node_type="concept",
        content="Beta",
        embedding=tuple([0.2] * 8),
        node_id=2,
        provenance="test:provenance",
    )
    graph.add_relationship(
        source_id=1,
        target_id=2,
        relationship="connects_to",
        provenance="test:provenance",
    )

    model = RUTAMForCausalLM(config=config, graph=graph)
    return model, graph


def test_checkpoint_manager_full_lifecycle(tmp_path) -> None:
    """Verify complete save, verify, list, load, and restore lifecycle."""
    model, graph = _make_small_model_and_graph()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )

    manager = RUTAMCheckpointManager(base_dir=tmp_path, run_id="run_lifecycle")

    # 1. Save checkpoint step 10
    c_path = manager.save_checkpoint(
        model=model,
        graph=graph,
        optimizer=opt,
        step=10,
        rutam_config=model.config,
        metrics={"eval_loss": 1.25},
        trainer_state={"epoch": 1},
    )
    assert c_path.is_dir()
    assert (c_path / "model.safetensors").is_file()
    assert (c_path / "graph_snapshot" / "graph.safetensors").is_file()
    assert (c_path / "optimizer.safetensors").is_file()
    assert (c_path / "manifest.json").is_file()

    # 2. Verify checkpoint
    assert manager.verify_checkpoint(c_path) is True

    # 3. List checkpoints
    checkpoints = manager.list_checkpoints()
    assert len(checkpoints) == 1
    assert checkpoints[0]["step"] == 10
    assert checkpoints[0]["is_valid"] is True

    # 4. Get latest checkpoint
    latest = manager.get_latest_checkpoint()
    assert latest is not None
    assert latest.name == "checkpoint_000010"

    # 5. Restore into fresh model, graph, and optimizer
    fresh_model, _ = _make_small_model_and_graph()
    fresh_opt = torch.optim.AdamW(
        [p for p in fresh_model.parameters() if p.requires_grad], lr=0.001
    )

    restored = manager.restore_checkpoint(
        model=fresh_model,
        optimizer=fresh_opt,
    )
    assert restored["step"] == 10
    assert restored["metrics"]["eval_loss"] == 1.25
    assert len(restored["graph"].nodes) == 2


def test_checkpoint_manager_delete_safety(tmp_path) -> None:
    """Verify delete requires confirmed=True to prevent accidental deletion."""
    model, graph = _make_small_model_and_graph()
    manager = RUTAMCheckpointManager(base_dir=tmp_path, run_id="run_delete")

    manager.save_checkpoint(model=model, graph=graph, step=5)

    with pytest.raises(ValidationError, match="requires confirmed=True"):
        manager.delete_checkpoint(step=5, confirmed=False)

    # With confirmed=True
    deleted = manager.delete_checkpoint(step=5, confirmed=True)
    assert deleted is True
    assert len(manager.list_checkpoints()) == 0


def test_export_architecture(tmp_path) -> None:
    """Verify architecture export to JSON specification."""
    model, _ = _make_small_model_and_graph()
    manager = RUTAMCheckpointManager(base_dir=tmp_path, run_id="run_export")
    out_path = manager.export_architecture(tmp_path / "architecture.json", model.config)
    assert out_path.is_file()
