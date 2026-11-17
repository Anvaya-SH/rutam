"""Schema definition and serialization for native memory training dataset examples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from rutam.config import nonempty_string
from rutam.exceptions import ValidationError

DATASET_SCHEMA_VERSION = "1.0.0"

VALID_OPERATIONS = {
    "NO_OP",
    "CREATE_NODE",
    "UPDATE_NODE",
    "PROPOSE_LINK",
    "PROPOSE_MERGE",
    "SUPERSEDE_NODE",
    "REQUEST_VALIDATION",
}

VALID_VALIDATION_DECISIONS = {"accept", "reject", "defer"}


@dataclass(frozen=True, slots=True)
class MemoryExampleRecord:
    """A single structured training example for RUTAM's native memory architecture."""

    example_id: str
    scenario_id: str
    scenario_family: str
    instruction: str
    context: str
    graph_before: dict[str, Any]
    expected_write_decision: bool
    expected_operation: str
    expected_target_nodes: list[int] = field(default_factory=list)
    expected_relationships: list[dict[str, Any]] = field(default_factory=list)
    expected_validation_decision: str = "accept"
    authorization_context: dict[str, Any] = field(default_factory=dict)
    provenance: str = "synthetic:rutam_dataset_builder"
    graph_after_expected: dict[str, Any] = field(default_factory=dict)
    difficulty: str = "easy"
    schema_version: str = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        nonempty_string(self.example_id, "example_id")
        nonempty_string(self.scenario_id, "scenario_id")
        nonempty_string(self.scenario_family, "scenario_family")
        nonempty_string(self.instruction, "instruction")
        if not isinstance(self.context, str):
            raise ValidationError("context must be a string")
        if not isinstance(self.expected_write_decision, bool):
            raise ValidationError("expected_write_decision must be a boolean")
        if self.expected_operation not in VALID_OPERATIONS:
            raise ValidationError(
                f"expected_operation '{self.expected_operation}' not in {VALID_OPERATIONS}"
            )
        if self.expected_validation_decision not in VALID_VALIDATION_DECISIONS:
            raise ValidationError(
                f"expected_validation_decision '{self.expected_validation_decision}' "
                f"not in {VALID_VALIDATION_DECISIONS}"
            )
        if not isinstance(self.graph_before, dict):
            raise ValidationError("graph_before must be a dictionary")
        if not isinstance(self.graph_after_expected, dict):
            raise ValidationError("graph_after_expected must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryExampleRecord:
        if not isinstance(data, dict):
            raise ValidationError("Record data must be a dictionary")
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> MemoryExampleRecord:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON record: {exc}") from exc
        return cls.from_dict(data)
