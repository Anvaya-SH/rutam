"""State machine enumerations and transition policies for RUTAMMemoryController."""

from __future__ import annotations

from enum import StrEnum

from rutam.exceptions import ValidationError


class MemoryState(StrEnum):
    """Lifecycle states of the native neural memory subsystem."""

    EMPTY = "EMPTY"
    LOADED = "LOADED"
    READ_READY = "READ_READY"
    WRITE_PENDING = "WRITE_PENDING"
    VALIDATING = "VALIDATING"
    COMMITTING = "COMMITTING"
    READY = "READY"
    ERROR = "ERROR"

    @classmethod
    def from_value(cls, val: object) -> MemoryState:
        if isinstance(val, MemoryState):
            return val
        if isinstance(val, str):
            try:
                return cls(val.strip().upper())
            except ValueError as exc:
                raise ValidationError(f"Unknown memory state: {val}") from exc
        raise ValidationError(f"Cannot convert {type(val).__name__} to MemoryState")

    @property
    def can_read(self) -> bool:
        """Indicates whether neural read queries can proceed in this state."""
        return self in (
            MemoryState.LOADED,
            MemoryState.READ_READY,
            MemoryState.READY,
            MemoryState.WRITE_PENDING,
        )

    @property
    def can_propose_write(self) -> bool:
        """Indicates whether write proposal generation can proceed."""
        return self in (
            MemoryState.EMPTY,
            MemoryState.LOADED,
            MemoryState.READ_READY,
            MemoryState.READY,
        )
