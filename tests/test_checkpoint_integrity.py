"""Unit tests for cryptographic integrity verification and corruption detection."""

from __future__ import annotations

from torch import nn

from rutam.checkpoint.integrity import verify_checkpoint_integrity
from rutam.checkpoint.manifest import CheckpointManifest
from rutam.checkpoint.model_weights import save_model_weights


def test_integrity_detects_tampering(tmp_path) -> None:
    """Verify that any modified byte causes integrity verification to fail."""
    model = nn.Linear(4, 2)
    checkpoint_dir = tmp_path / "checkpoint_000001"
    checkpoint_dir.mkdir()

    weight_file = checkpoint_dir / "model.safetensors"
    save_model_weights(model, weight_file)

    manifest = CheckpointManifest(
        checkpoint_id="checkpoint_000001",
        run_id="run_test",
        step=1,
        backbone_model_id="test_backbone",
    )
    manifest.save(checkpoint_dir / "manifest.json")

    # Re-save manifest with correct file entry
    from rutam.checkpoint.integrity import create_file_record

    manifest = CheckpointManifest(
        checkpoint_id="checkpoint_000001",
        run_id="run_test",
        step=1,
        backbone_model_id="test_backbone",
        files={"model.safetensors": create_file_record(weight_file)},
    )
    manifest.save(checkpoint_dir / "manifest.json")

    # Initial check passes
    valid, errors = verify_checkpoint_integrity(checkpoint_dir)
    assert valid is True
    assert len(errors) == 0

    # Tamper with file
    with weight_file.open("ab") as f:
        f.write(b"\x00")

    # Check must fail now
    tampered_valid, tampered_errors = verify_checkpoint_integrity(checkpoint_dir)
    assert tampered_valid is False
    assert any("mismatch" in err.lower() for err in tampered_errors)


def test_integrity_detects_missing_file(tmp_path) -> None:
    """Verify missing file reported as integrity error."""
    checkpoint_dir = tmp_path / "checkpoint_missing"
    checkpoint_dir.mkdir()

    from rutam.checkpoint.manifest import CheckpointFileRecord

    manifest = CheckpointManifest(
        checkpoint_id="checkpoint_missing",
        run_id="run_test",
        step=1,
        backbone_model_id="test_backbone",
        files={"ghost.txt": CheckpointFileRecord(sha256="abc", size_bytes=10)},
    )
    manifest.save(checkpoint_dir / "manifest.json")

    valid, errors = verify_checkpoint_integrity(checkpoint_dir)
    assert valid is False
    assert any("missing" in err.lower() for err in errors)
