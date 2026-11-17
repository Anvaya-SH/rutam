"""Integration tests for model, graph, and checkpoint serialization in RUTAM."""

from __future__ import annotations

from pathlib import Path

import torch

from rutam.checkpoint.integrity import verify_checkpoint_integrity
from rutam.checkpoint.manager import RUTAMCheckpointManager
from rutam.diagnostics.integration import create_verification_model


def test_end_to_end_checkpoint_serialization_and_restore(tmp_path: Path) -> None:
    """Verify complete checkpoint save, cryptographic verification, and load into a fresh model."""
    model, graph = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=60)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)

    manager = RUTAMCheckpointManager(base_dir=tmp_path, run_id="test_run")

    # 1. Save checkpoint at step 25
    c_path = manager.save_checkpoint(
        model=model,
        graph=graph,
        optimizer=opt,
        step=25,
        rutam_config=model.config,
        metrics={"eval_loss": 1.85},
    )

    assert c_path.is_dir()
    assert (c_path / "model.safetensors").is_file()
    assert (c_path / "graph_snapshot" / "graph.safetensors").is_file()
    assert (c_path / "optimizer.safetensors").is_file()
    assert (c_path / "manifest.json").is_file()

    # 2. Cryptographic manifest check
    is_valid, errors = verify_checkpoint_integrity(c_path)
    assert is_valid is True, f"Checkpoint integrity check failed: {errors}"
    assert len(errors) == 0

    # 3. Restore into a fresh model instance
    fresh_model, _ = create_verification_model(num_nodes=0, hidden_dim=16, vocab_size=60)
    fresh_opt = torch.optim.AdamW(
        [p for p in fresh_model.parameters() if p.requires_grad], lr=0.001
    )

    result = manager.load_checkpoint(
        checkpoint_dir=c_path,
        model=fresh_model,
        optimizer=fresh_opt,
        rutam_config=fresh_model.config,
    )

    assert result["step"] == 25
    assert len(result.get("missing_keys", [])) == 0
    assert len(result.get("unexpected_keys", [])) == 0
    assert "graph" in result

    # 4. Compare outputs before and after restoration
    fresh_model.graph = result["graph"]
    fresh_model.bridge.graph = result["graph"]
    fresh_model.bridge.invalidate_caches()
    model.eval()
    fresh_model.eval()

    input_ids = torch.randint(0, 60, (2, 6), dtype=torch.long)
    with torch.no_grad():
        orig_out = model(input_ids)
        restored_out = fresh_model(input_ids)

    assert torch.allclose(orig_out.logits, restored_out.logits, atol=1e-4)


def test_checkpoint_manifest_records_complete_file_hashes(tmp_path: Path) -> None:
    """Verify that CheckpointManifest contains SHA-256 digests and file sizes for all payloads."""
    model, graph = create_verification_model(num_nodes=2, hidden_dim=16, vocab_size=50)
    manager = RUTAMCheckpointManager(base_dir=tmp_path, run_id="manifest_run")

    c_path = manager.save_checkpoint(model=model, graph=graph, step=1)
    manifest = manager.load_checkpoint(c_path, strict=False)["manifest"]

    assert manifest.checkpoint_id == "checkpoint_000001"
    assert manifest.step == 1
    assert "model.safetensors" in manifest.files
    assert len(manifest.files["model.safetensors"].sha256) == 64
    assert manifest.files["model.safetensors"].size_bytes > 0
