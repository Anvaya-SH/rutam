"""Integration tests for failure recovery, corruption detection, and automatic rollback in RUTAM."""

from __future__ import annotations
from rutam.memory.provenance import AuthorizationContext

from pathlib import Path

import pytest
import torch

from rutam.checkpoint.integrity import verify_checkpoint_integrity
from rutam.checkpoint.manager import RUTAMCheckpointManager
from rutam.checkpoint.recovery import clean_incomplete_checkpoints
from rutam.diagnostics.integration import create_verification_model
from rutam.exceptions import CorruptSnapshotError
from rutam.memory.provenance import ProvenanceRecord
from rutam.memory.transaction import GraphTransaction
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal


def test_checkpoint_corruption_detection_and_rollback(tmp_path: Path) -> None:
    """Verify that bit-level corruption triggers detection and automatic rollback to valid step."""
    model, graph = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=60)
    manager = RUTAMCheckpointManager(base_dir=tmp_path, run_id="fault_tolerant_run")

    # Step 10: Valid checkpoint
    c10 = manager.save_checkpoint(model=model, graph=graph, step=10)
    assert verify_checkpoint_integrity(c10)[0] is True

    # Step 20: Valid checkpoint
    c20 = manager.save_checkpoint(model=model, graph=graph, step=20)
    assert verify_checkpoint_integrity(c20)[0] is True

    # Step 30: Valid checkpoint, then corrupted
    c30 = manager.save_checkpoint(model=model, graph=graph, step=30)
    (c30 / "model.safetensors").write_bytes(b"injected_bit_flip_tampering")

    # Verify that corruption is detected
    is_valid, errors = verify_checkpoint_integrity(c30)
    assert is_valid is False
    assert len(errors) > 0

    # Loading corrupted checkpoint directly must raise CorruptSnapshotError
    with pytest.raises(CorruptSnapshotError):
        manager.load_checkpoint(c30)

    # Automatic recovery must find latest valid checkpoint (step 20)
    latest = manager.get_latest_checkpoint()
    assert latest is not None
    assert latest.name == "checkpoint_000020"


def test_graph_snapshot_tamper_detection(tmp_path: Path) -> None:
    """Verify that tampering with graph.safetensors is caught by checkpoint verification."""
    model, graph = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=60)
    manager = RUTAMCheckpointManager(base_dir=tmp_path, run_id="graph_tamper_run")

    c_path = manager.save_checkpoint(model=model, graph=graph, step=5)
    graph_tensor_file = c_path / "graph_snapshot" / "graph.safetensors"
    assert graph_tensor_file.is_file()

    # Corrupt graph snapshot
    graph_tensor_file.write_bytes(b"corrupt_graph_data")

    is_valid, errors = verify_checkpoint_integrity(c_path)
    assert is_valid is False
    assert any("graph_snapshot/graph.safetensors" in err for err in errors)


def test_transaction_exception_leaves_graph_unaltered() -> None:
    """Verify that unhandled exceptions during transaction commit safely roll back all changes."""
    model, graph = create_verification_model(num_nodes=3, hidden_dim=16, vocab_size=60)
    initial_version = graph.version
    initial_nodes = len(graph.nodes)

    prov = ProvenanceRecord(source_id="agent", authorization_category="user_input")
    bad_proposal = MemoryProposal(
        proposal_id="explode_prop",
        operation=MemoryOperation.CREATE_NODE,
        target_node_ids=(),
        candidate_embedding=torch.tensor([0.3] * 16),
        provenance=prov,
        confidence=0.9,
    )

    with pytest.raises(RuntimeError, match="Simulated crash during transaction"):
        with GraphTransaction(graph, authorization_context=AuthorizationContext(permissions=frozenset({"commit"}))) as tx:
            tx.apply_proposal(bad_proposal)
            assert len(tx._working_graph.nodes) == initial_nodes + 1
            raise RuntimeError("Simulated crash during transaction")

    # Graph remains completely intact
    assert len(graph.nodes) == initial_nodes
    assert graph.version == initial_version


def test_clean_orphaned_temporary_checkpoints(tmp_path: Path) -> None:
    """Verify that abandoned temporary directories from interrupted writes are cleaned up."""
    run_dir = tmp_path / "recovery_clean"
    run_dir.mkdir(parents=True, exist_ok=True)

    orphan1 = run_dir / "tmp_checkpoint_000050_xyz"
    orphan1.mkdir()
    (orphan1 / "partial.tmp").write_text("partial")

    orphan2 = run_dir / ".tmp_checkpoint_000060_abc"
    orphan2.mkdir()

    cleaned = clean_incomplete_checkpoints(run_dir)
    assert len(cleaned) == 2
    assert not orphan1.exists()
    assert not orphan2.exists()
