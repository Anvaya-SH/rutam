"""Cryptographic checksum calculation and checkpoint verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rutam.checkpoint.manifest import CheckpointFileRecord, CheckpointManifest


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in 64KB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_file_record(
    path: Path, content_type: str = "application/octet-stream"
) -> CheckpointFileRecord:
    """Create a CheckpointFileRecord with real hash and size."""
    size = path.stat().st_size
    sha256_hash = compute_file_sha256(path)
    return CheckpointFileRecord(sha256=sha256_hash, size_bytes=size, content_type=content_type)


def verify_checkpoint_integrity(checkpoint_dir: str | Path) -> tuple[bool, list[str]]:
    """Verify all files recorded in manifest match their recorded SHA-256 hashes."""
    base = Path(checkpoint_dir)
    manifest_path = base / "manifest.json"

    if not manifest_path.is_file():
        return False, [f"Manifest file missing: {manifest_path}"]

    try:
        manifest = CheckpointManifest.load(manifest_path)
    except Exception as exc:
        return False, [f"Corrupted manifest file: {exc}"]

    errors: list[str] = []

    for rel_path, rec in manifest.files.items():
        file_path = base / rel_path
        if not file_path.is_file():
            errors.append(f"Missing recorded file: {rel_path}")
            continue

        real_size = file_path.stat().st_size
        if real_size != rec.size_bytes:
            errors.append(
                f"File size mismatch for {rel_path}: expected {rec.size_bytes}, got {real_size}"
            )
            continue

        real_sha256 = compute_file_sha256(file_path)
        if real_sha256 != rec.sha256:
            errors.append(
                f"SHA-256 hash mismatch for {rel_path}: expected {rec.sha256}, got {real_sha256}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
