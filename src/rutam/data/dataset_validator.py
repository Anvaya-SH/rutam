"""Strict validation of memory training dataset records, graph transitions, and split integrity."""

from __future__ import annotations

from typing import Any

from rutam.data.example_schema import MemoryExampleRecord


class DatasetValidator:
    """Validates records, ground truth labels, graph states, and split integrity."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_records(self, records: list[MemoryExampleRecord]) -> bool:
        """Validate an in-memory list of records."""
        self.errors.clear()
        self.warnings.clear()

        seen_ids: set[str] = set()

        for _idx, r in enumerate(records):
            if r.example_id in seen_ids:
                self.errors.append(f"Duplicate example_id: {r.example_id}")
            seen_ids.add(r.example_id)

            # Validate graph before and after
            self._validate_graph_structure(r.graph_before, f"{r.example_id}:graph_before")
            self._validate_graph_structure(
                r.graph_after_expected, f"{r.example_id}:graph_after_expected"
            )

            # Check logical consistency
            if r.expected_operation == "NO_OP" and r.expected_write_decision is True:
                self.errors.append(
                    f"{r.example_id}: NO_OP cannot have expected_write_decision=True"
                )

            if r.expected_validation_decision == "reject" and r.expected_write_decision is True:
                self.errors.append(
                    f"{r.example_id}: rejected proposal cannot have expected_write_decision=True"
                )

        return len(self.errors) == 0

    def validate_splits(
        self,
        train_records: list[MemoryExampleRecord],
        val_records: list[MemoryExampleRecord],
        test_records: list[MemoryExampleRecord],
    ) -> bool:
        """Verify leakage prevention and distinct example IDs across splits."""
        all_ok = (
            self.validate_records(train_records)
            and self.validate_records(val_records)
            and self.validate_records(test_records)
        )

        train_ids = {r.example_id for r in train_records}
        val_ids = {r.example_id for r in val_records}
        test_ids = {r.example_id for r in test_records}

        if train_ids & val_ids:
            overlap = len(train_ids & val_ids)
            self.errors.append(f"Example ID overlap between train and val: {overlap}")
        if train_ids & test_ids:
            overlap = len(train_ids & test_ids)
            self.errors.append(f"Example ID overlap between train and test: {overlap}")
        if val_ids & test_ids:
            overlap = len(val_ids & test_ids)
            self.errors.append(f"Example ID overlap between val and test: {overlap}")

        train_fams = {r.scenario_family for r in train_records}
        val_fams = {r.scenario_family for r in val_records}
        test_fams = {r.scenario_family for r in test_records}

        if train_fams & val_fams:
            self.errors.append("Scenario family overlap between train and val")
        if train_fams & test_fams:
            self.errors.append("Scenario family overlap between train and test")
        if val_fams & test_fams:
            self.errors.append("Scenario family overlap between val and test")

        return len(self.errors) == 0 and all_ok

    def _validate_graph_structure(self, g: dict[str, Any], context: str) -> None:
        if not isinstance(g, dict):
            self.errors.append(f"{context} is not a dictionary")
            return
        if "nodes" not in g or "edges" not in g:
            self.errors.append(f"{context} missing 'nodes' or 'edges'")
            return

        node_ids = {n.get("id") for n in g["nodes"] if isinstance(n, dict)}
        for e in g["edges"]:
            if not isinstance(e, dict):
                self.errors.append(f"{context}: edge is not a dict")
                continue
            src = e.get("source_id")
            tgt = e.get("target_id")
            if src not in node_ids:
                self.errors.append(f"{context}: edge source {src} not in node IDs")
            if tgt not in node_ids:
                self.errors.append(f"{context}: edge target {tgt} not in node IDs")
