"""Typed validation output and result records for memory validation decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rutam.config import nonempty_string, positive_int
from rutam.exceptions import ValidationError


class ValidationDecision(StrEnum):
    """Discrete validation decision categories."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    DEFER = "DEFER"
    DUPLICATE = "DUPLICATE"
    CONFLICT = "CONFLICT"
    STALE = "STALE"

    @classmethod
    def from_value(cls, val: object) -> ValidationDecision:
        if isinstance(val, ValidationDecision):
            return val
        if isinstance(val, str):
            try:
                return cls(val.strip().upper())
            except ValueError as exc:
                raise ValidationError(f"Unknown validation decision: {val}") from exc
        raise ValidationError(f"Cannot convert {type(val).__name__} to ValidationDecision")


@dataclass(frozen=True, slots=True)
class MemoryValidationResult:
    """Individual validation decision for a memory proposal."""

    proposal_id: str
    decision: ValidationDecision | str
    reason_codes: tuple[str, ...] = ()
    confidence_metadata: dict[str, Any] = field(default_factory=dict)
    detected_conflicts: tuple[str, ...] = ()
    existing_match_ids: tuple[int, ...] = ()
    source_graph_version: int | None = None
    sanitized_proposal: Any | None = None
    requires_further_verification: bool = False

    def __post_init__(self) -> None:
        nonempty_string(self.proposal_id, "proposal_id")
        dec = ValidationDecision.from_value(self.decision)
        object.__setattr__(self, "decision", dec)
        if not isinstance(self.reason_codes, (tuple, list)):
            raise ValidationError("reason_codes must be a sequence of strings")
        object.__setattr__(self, "reason_codes", tuple(str(x) for x in self.reason_codes))
        if not isinstance(self.detected_conflicts, (tuple, list)):
            raise ValidationError("detected_conflicts must be a sequence of strings")
        object.__setattr__(
            self, "detected_conflicts", tuple(str(x) for x in self.detected_conflicts)
        )
        if not isinstance(self.existing_match_ids, (tuple, list)):
            raise ValidationError("existing_match_ids must be a sequence of integers")
        for mid in self.existing_match_ids:
            positive_int(mid, "match_id", minimum=0)
        object.__setattr__(self, "existing_match_ids", tuple(self.existing_match_ids))
        if self.source_graph_version is not None:
            positive_int(self.source_graph_version, "source_graph_version", minimum=0)
        if not isinstance(self.confidence_metadata, dict):
            raise ValidationError("confidence_metadata must be a dictionary")
        if not isinstance(self.requires_further_verification, bool):
            raise ValidationError("requires_further_verification must be a boolean")

    @property
    def is_accepted(self) -> bool:
        """Convenience property indicating whether proposal was accepted."""
        return self.decision == ValidationDecision.ACCEPT

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "confidence_metadata": dict(self.confidence_metadata),
            "detected_conflicts": list(self.detected_conflicts),
            "existing_match_ids": list(self.existing_match_ids),
            "source_graph_version": self.source_graph_version,
            "requires_further_verification": self.requires_further_verification,
        }


@dataclass(frozen=True, slots=True)
class ValidationBatchResult:
    """Aggregated validation output for a batch of proposals."""

    results: tuple[MemoryValidationResult, ...]
    accepted_count: int
    rejected_count: int
    deferred_count: int
    duplicate_count: int
    conflict_count: int
    stale_count: int
    all_valid: bool

    @classmethod
    def from_results(cls, results: tuple[MemoryValidationResult, ...]) -> ValidationBatchResult:
        accepted = sum(1 for r in results if r.decision == ValidationDecision.ACCEPT)
        rejected = sum(1 for r in results if r.decision == ValidationDecision.REJECT)
        deferred = sum(1 for r in results if r.decision == ValidationDecision.DEFER)
        duplicate = sum(1 for r in results if r.decision == ValidationDecision.DUPLICATE)
        conflict = sum(1 for r in results if r.decision == ValidationDecision.CONFLICT)
        stale = sum(1 for r in results if r.decision == ValidationDecision.STALE)
        return cls(
            results=results,
            accepted_count=accepted,
            rejected_count=rejected,
            deferred_count=deferred,
            duplicate_count=duplicate,
            conflict_count=conflict,
            stale_count=stale,
            all_valid=(accepted == len(results) and len(results) > 0),
        )
