"""Statistical reporting and distribution summaries for memory training datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any

from rutam.data.example_schema import MemoryExampleRecord


def compute_dataset_statistics(
    records: list[MemoryExampleRecord],
    *,
    split_name: str = "all",
) -> dict[str, Any]:
    """Compute exact summary statistics across records."""
    total = len(records)
    if total == 0:
        return {"total_records": 0, "split": split_name}

    ops = Counter(r.expected_operation for r in records)
    decisions = Counter(r.expected_write_decision for r in records)
    val_decisions = Counter(r.expected_validation_decision for r in records)
    diffs = Counter(r.difficulty for r in records)
    families = {r.scenario_family for r in records}

    relations: Counter[str] = Counter()
    total_nodes_before = 0
    total_edges_before = 0

    for r in records:
        gb = r.graph_before
        total_nodes_before += len(gb.get("nodes", []))
        total_edges_before += len(gb.get("edges", []))
        for e in gb.get("edges", []):
            rel = e.get("relationship", "unknown")
            relations[rel] += 1
        for rel_dict in r.expected_relationships:
            if isinstance(rel_dict, dict) and "relationship" in rel_dict:
                relations[rel_dict["relationship"]] += 1

    return {
        "split": split_name,
        "total_records": total,
        "unique_scenario_families": len(families),
        "operations_distribution": dict(ops),
        "write_decision_distribution": {
            "write_true": decisions.get(True, 0),
            "write_false": decisions.get(False, 0),
        },
        "validation_decision_distribution": dict(val_decisions),
        "difficulty_distribution": dict(diffs),
        "relationship_frequencies": dict(relations),
        "avg_nodes_before": total_nodes_before / max(1, total),
        "avg_edges_before": total_edges_before / max(1, total),
    }
