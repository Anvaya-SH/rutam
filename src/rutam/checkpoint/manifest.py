"""Cryptographic manifest tracking all checkpoint files, hashes, and metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rutam.checkpoint.versioning import CHECKPOINT_SCHEMA_VERSION, is_compatible_version
from rutam.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class CheckpointFileRecord:
    """Entry describing an individual file in a checkpoint."""

    sha256: str
    size_bytes: int
    content_type: str = "application/octet-stream"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointFileRecord:
        return cls(
            sha256=data["sha256"],
            size_bytes=int(data["size_bytes"]),
            content_type=data.get("content_type", "application/octet-stream"),
        )


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Cryptographic manifest documenting checkpoint components, hashes, and provenance."""

    checkpoint_id: str
    run_id: str
    step: int
    backbone_model_id: str
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    architecture_metadata: dict[str, Any] = field(default_factory=dict)
    files: dict[str, CheckpointFileRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_compatible_version(self.schema_version):
            raise ValidationError(
                f"Incompatible checkpoint schema version: {self.schema_version}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "step": self.step,
            "created_at": self.created_at,
            "backbone_model_id": self.backbone_model_id,
            "architecture_metadata": self.architecture_metadata,
            "files": {rel_path: rec.to_dict() for rel_path, rec in self.files.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManifest:
        files_data = data.get("files", {})
        parsed_files = {
            rel_path: CheckpointFileRecord.from_dict(rec)
            for rel_path, rec in files_data.items()
        }
        return cls(
            schema_version=data.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
            checkpoint_id=data["checkpoint_id"],
            run_id=data["run_id"],
            step=int(data["step"]),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            backbone_model_id=data.get("backbone_model_id", "LiquidAI/LFM3-1.7B-Base"),
            architecture_metadata=data.get("architecture_metadata", {}),
            files=parsed_files,
        )

    @classmethod
    def load(cls, path: str | Path) -> CheckpointManifest:
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls.from_dict(content)

    def save(self, path: str | Path) -> None:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(self.to_json(), encoding="utf-8")
