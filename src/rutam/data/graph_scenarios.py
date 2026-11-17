"""Deterministic synthetic knowledge graph fixtures for scenario generation."""

from __future__ import annotations

from typing import Any


def make_empty_graph_dict() -> dict[str, Any]:
    return {"nodes": [], "edges": [], "relationship_types": {}}


def make_graph_dict(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    relationship_types: dict[str, int] | None = None,
) -> dict[str, Any]:
    if relationship_types is None:
        rel_set = {e["relationship"] for e in edges}
        rel_map = {r: idx for idx, r in enumerate(sorted(rel_set))}
    else:
        rel_map = relationship_types
    return {
        "nodes": list(nodes),
        "edges": list(edges),
        "relationship_types": rel_map,
    }


def make_node(node_id: int, node_type: str, content: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "node_type": node_type,
        "content": content,
    }


def make_edge(
    source_id: int, target_id: int, relationship: str, weight: float = 1.0
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship": relationship,
        "weight": weight,
    }
