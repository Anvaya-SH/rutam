"""Immutable node/edge records; graph-level constraints live in operations.py."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rutam.config import nonempty_string, positive_int
from rutam.exceptions import ValidationError

MAX_ID = 2**63 - 1


def utc_now() -> datetime:
    return datetime.now(UTC)


def identifier(value: object, name: str = "node_id") -> None:
    positive_int(value, name, minimum=0)
    if value > MAX_ID:
        raise ValidationError(f"{name} must fit in signed int64")


def timestamp(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValidationError(f"{name} must be a timezone-aware datetime")


def finite_number(value: object, name: str) -> None:
    try:
        valid = (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError(f"{name} must be a finite number")


def provenance_reference(value: object) -> None:
    if value is not None:
        nonempty_string(value, "provenance")


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    node_type: str
    content: str
    provenance: str | None = None
    confidence: float = 1.0
    embedding: tuple[float, ...] | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        identifier(self.id)
        nonempty_string(self.node_type, "node_type")
        if not isinstance(self.content, str):
            raise ValidationError("content must be a string")
        provenance_reference(self.provenance)
        finite_number(self.confidence, "confidence")
        if not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be between 0 and 1")
        timestamp(self.created_at, "created_at")
        timestamp(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValidationError("updated_at cannot precede created_at")
        if self.embedding is not None:
            try:
                values = tuple(self.embedding)
            except TypeError as exc:
                raise ValidationError("embedding must be a sequence of finite numbers") from exc
            for value in values:
                finite_number(value, "embedding value")
                if abs(value) > 3.4028234663852886e38:
                    raise ValidationError("embedding values must fit in float32")
            object.__setattr__(self, "embedding", tuple(float(x) for x in values))


@dataclass(frozen=True, slots=True)
class Edge:
    source_id: int
    target_id: int
    relationship: str
    provenance: str | None = None
    weight: float = 1.0
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        identifier(self.source_id, "source_id")
        identifier(self.target_id, "target_id")
        nonempty_string(self.relationship, "relationship")
        provenance_reference(self.provenance)
        finite_number(self.weight, "weight")
        if not 0 <= self.weight <= 3.4028234663852886e38:
            raise ValidationError("weight must be nonnegative and fit in float32")
        timestamp(self.timestamp, "timestamp")

    @property
    def key(self) -> tuple[int, int, str]:
        return self.source_id, self.target_id, self.relationship
