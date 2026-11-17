"""Unit tests for RUTAMMemoryController initialization, state transitions, and status reporting."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.memory.controller import RUTAMMemoryController
from rutam.memory.controller_output import ControllerStatus, MemoryWritePipelineOutput
from rutam.memory.memory_state import MemoryState
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal


def test_controller_initialization_default() -> None:
    torch.manual_seed(42)
    controller = RUTAMMemoryController()
    assert isinstance(controller, RUTAMMemoryController)
    assert controller.state == MemoryState.EMPTY
    status = controller.get_memory_status()
    assert isinstance(status, ControllerStatus)
    assert status.state == MemoryState.EMPTY
    assert status.node_count == 0
    assert status.edge_count == 0
    assert status.read_enabled is True
    assert status.write_enabled is True
    assert status.total_commits == 0
    assert status.last_commit_timestamp is None


def test_controller_initialization_with_populated_graph() -> None:
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}})
    graph = Graph(config)
    graph.create_node("entity", "Apple Inc.", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    controller = RUTAMMemoryController(config, graph=graph)
    assert controller.state == MemoryState.READY
    status = controller.get_memory_status()
    assert status.state == MemoryState.READY
    assert status.node_count == 1
    assert status.graph_version == graph.version


def test_controller_invalid_config_raises() -> None:
    with pytest.raises(ValidationError, match="requires a RUTAMConfig"):
        RUTAMMemoryController(config="invalid_config")  # type: ignore[arg-type]


def test_controller_custom_audit_path(tmp_path: Path) -> None:
    audit_file = tmp_path / "custom_audit.jsonl"
    controller = RUTAMMemoryController(audit_path=audit_file)
    assert controller.audit_logger is not None
    assert controller.audit_logger.path == audit_file


def test_controller_disabled_audit() -> None:
    config = RUTAMConfig().with_overrides({"memory_controller": {"audit_enabled": False}})
    controller = RUTAMMemoryController(config)
    assert controller.audit_logger is None


def test_controller_refresh_graph_state() -> None:
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}})
    graph = Graph(config)
    controller = RUTAMMemoryController(config, graph=graph)
    assert controller.state == MemoryState.EMPTY

    graph.create_node("entity", "Rutam", node_id=1, embedding=[0.2] * 8, provenance="test:p")
    assert controller.state == MemoryState.EMPTY  # Not refreshed yet
    controller.refresh_graph_state()
    assert controller.state == MemoryState.READY
    active_state = controller.get_active_state()
    assert active_state.node_ids.shape[0] == 1


def test_controller_propose_memory_update_disabled() -> None:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "memory_controller": {"write_enabled": False},
        }
    )
    controller = RUTAMMemoryController(config)
    hidden_states = torch.randn(1, 4, config.memory_write.hidden_dim)
    with pytest.raises(ValidationError, match="Memory write is disabled"):
        controller.propose_memory_update(hidden_states)


def test_controller_read_memory_disabled() -> None:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "memory_controller": {"read_enabled": False},
        }
    )
    controller = RUTAMMemoryController(config)
    hidden_states = torch.randn(1, 4, config.query_head.backbone_hidden_dim)
    with pytest.raises(ValidationError, match="Memory read is disabled"):
        controller.read_memory(hidden_states)


def test_controller_propose_memory_update_pipeline() -> None:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 8},
            "selector": {"query_dim": 8, "hidden_dim": 8, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 8,
                "query_dim": 8,
                "graph_hidden_dim": 8,
                "intermediate_dim": 8,
            },
            "memory_write": {
                "backbone_hidden_dim": 8,
                "hidden_dim": 8,
                "projection_dim": 8,
                "max_proposals": 2,
            },
        }
    )
    graph = Graph(config)
    graph.create_node("entity", "Alpha", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    controller = RUTAMMemoryController(config, graph=graph)

    hidden_states = torch.randn(1, 3, 8)
    auth = AuthorizationContext(user_id="test_actor", role="admin")

    output = controller.propose_memory_update(
        hidden_states=hidden_states,
        authorization_context=auth,
        auto_validate=True,
    )
    assert isinstance(output, MemoryWritePipelineOutput)
    assert output.write_output is not None
    assert output.validation_result is not None
    assert controller.state == MemoryState.READY


def test_controller_validate_updates_lifecycle() -> None:
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 8}})
    graph = Graph(config)
    graph.create_node("entity", "Alpha", node_id=1, embedding=[0.1] * 8, provenance="test:p")
    controller = RUTAMMemoryController(config, graph=graph)

    proposal = MemoryProposal(
        proposal_id="prop_val_1",
        operation=MemoryOperation.CREATE_NODE,
        target_node_ids=(),
        candidate_embedding=torch.tensor([0.2] * 8, dtype=torch.float32),
        confidence=0.9,
    )
    val_res = controller.validate_updates([proposal])
    assert len(val_res.results) == 1
    assert controller.state == MemoryState.READY
