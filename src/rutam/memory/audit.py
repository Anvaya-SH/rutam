"""Persistent audit logging for committed memory transactions and graph mutations."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from rutam.config import nonempty_string
from rutam.exceptions import ValidationError
from rutam.graph.schema import timestamp, utc_now


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Immutable audit trail record documenting a committed or rolled back mutation."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=utc_now)
    transaction_id: str = ""
    operation: str = ""
    target_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    actor_id: str | None = None
    status: str = "COMMITTED"

    def __post_init__(self) -> None:
        nonempty_string(self.record_id, "record_id")
        timestamp(self.timestamp, "timestamp")
        nonempty_string(self.transaction_id, "transaction_id")
        nonempty_string(self.operation, "operation")
        nonempty_string(self.status, "status")
        if not isinstance(self.details, dict):
            raise ValidationError("details must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "target_id": self.target_id,
            "details": dict(self.details),
            "actor_id": self.actor_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditRecord:
        if not isinstance(data, dict):
            raise ValidationError("Audit record data must be a dictionary")
        try:
            ts = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else utc_now()
            return cls(
                record_id=data["record_id"],
                timestamp=ts,
                transaction_id=data["transaction_id"],
                operation=data["operation"],
                target_id=data.get("target_id"),
                details=data.get("details", {}),
                actor_id=data.get("actor_id"),
                status=data.get("status", "COMMITTED"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid audit record dictionary: {exc}") from exc


class AuditLogger:
    """Thread-safe append-only audit logger supporting in-memory and disk persistence."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._path = Path(log_path) if log_path else None
        self._records: list[AuditRecord] = []
        self._lock = Lock()
        if self._path and self._path.exists():
            self._load_existing()

    @property
    def path(self) -> Path | None:
        return self._path

    def _load_existing(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record_dict = json.loads(line)
                        self._records.append(AuditRecord.from_dict(record_dict))
        except (OSError, json.JSONDecodeError, ValidationError):
            pass

    def log(self, record: AuditRecord) -> None:
        """Record an audit entry in memory and append to disk if configured."""
        if not isinstance(record, AuditRecord):
            raise ValidationError("record must be an AuditRecord")
        with self._lock:
            self._records.append(record)
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")

    def get_records(self, limit: int | None = None) -> tuple[AuditRecord, ...]:
        """Retrieve recorded audit entries (most recent first if limit specified)."""
        with self._lock:
            if limit is not None and limit > 0:
                return tuple(self._records[-limit:])
            return tuple(self._records)

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            if self._path and self._path.exists():
                self._path.unlink()
