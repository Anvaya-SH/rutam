"""Unit tests for automatic recovery and rollback to previous valid checkpoints."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from rutam.checkpoint.integrity import create_file_record
from rutam.checkpoint.manifest import CheckpointManifest
from rutam.checkpoint.model_weights import save_model_weights
from rutam.checkpoint.recovery import clean_incomplete_checkpoints, find_latest_valid_checkpoint


def _create_valid_checkpoint(base_dir: Path, step: int) -> Path:
    c_dir = base_dir / f"checkpoint_{step:06d}"
    c_dir.mkdir(parents=True, exist_ok=True)
    model = nn.Linear(4, 2)
    save_model_weights(model, c_dir / "model.safetensors")
    manifest = CheckpointManifest(
        checkpoint_id=f"checkpoint_{step:06d}",
        run_id="run_recovery",
        step=step,
        backbone_model_id="test_backbone",
        files={"model.safetensors": create_file_record(c_dir / "model.safetensors")},
    )
    manifest.save(c_dir / "manifest.json")
    return c_dir


def test_rollback_on_corrupted_latest(tmp_path) -> None:
    """Verify recovery engine skips a corrupted step and rolls back to latest valid step."""
    run_dir = tmp_path / "run_recovery"
    run_dir.mkdir()

    _create_valid_checkpoint(run_dir, step=10)
    _create_valid_checkpoint(run_dir, step=20)
    c30 = _create_valid_checkpoint(run_dir, step=30)

    # Corrupt step 30
    (c30 / "model.safetensors").write_bytes(b"corrupt")

    latest = find_latest_valid_checkpoint(run_dir)
    assert latest is not None
    assert latest.name == "checkpoint_000020"


def test_clean_incomplete_temporary_directories(tmp_path) -> None:
    """Verify cleanup of orphaned temporary directories."""
    run_dir = tmp_path / "run_clean"
    run_dir.mkdir()

    tmp1 = run_dir / "tmp_checkpoint_000050_abc"
    tmp1.mkdir()
    (tmp1 / "dummy.txt").write_text("partial")

    _create_valid_checkpoint(run_dir, step=40)

    cleaned = clean_incomplete_checkpoints(run_dir)
    assert len(cleaned) == 1
    assert not tmp1.exists()
