"""Contradiction, structural conflict, and optimistic concurrency version checking."""

from __future__ import annotations

from typing import Any

from rutam.graph.operations import Graph
from rutam.graph.state import GraphState


def detect_conflicts(
    proposal: Any,
    graph_snapshot: Graph | GraphState,
    *,
    conflicting_relations: tuple[tuple[str, str], ...] = (),
    current_graph_version: int | None = None,
) -> tuple[bool, str | None, tuple[str, ...]]:
    """Detect mutually exclusive relationships, concurrent modifications, and stale versions."""
    conflicts: list[str] = []

    # 1. Version concurrency check
    curr_ver = current_graph_version
    if curr_ver is None:
        if isinstance(graph_snapshot, Graph):
            curr_ver = graph_snapshot.version
        elif hasattr(graph_snapshot, "graph_version"):
            curr_ver = getattr(graph_snapshot, "graph_version", None)

    prop_ver = getattr(proposal, "source_graph_version", None)
    if prop_ver is None:
        prop_ver = getattr(proposal, "graph_version", None)
    if prop_ver is not None and curr_ver is not None:
        if prop_ver < curr_ver:
            return (
                True,
                "STALE_GRAPH_VERSION",
                (f"Proposal source_graph_version {prop_ver} is older than active {curr_ver}",),
            )

    # 2. Mutually exclusive / conflicting relationship check
    if conflicting_relations:
        conflict_set = {
            frozenset((p[0].upper(), p[1].upper())) for p in conflicting_relations if len(p) == 2
        }
        src = getattr(proposal, "source_node_id", None) or getattr(proposal, "source_id", None)
        dst = getattr(proposal, "destination_node_id", None) or getattr(proposal, "target_id", None)
        rel = getattr(proposal, "relation_name", None) or getattr(proposal, "relationship", None)

        if src is not None and dst is not None and rel is not None:
            rel_upper = str(rel).upper()
            if isinstance(graph_snapshot, Graph):
                directed = graph_snapshot.config.graph.directed
                for s, d, r in graph_snapshot._edges:
                    matches = (s == src and d == dst) or (not directed and s == dst and d == src)
                    if matches and frozenset((rel_upper, r.upper())) in conflict_set:
                        msg = (
                            f"Relation {rel} conflicts with existing relation {r} "
                            f"between {src} and {dst}"
                        )
                        conflicts.append(msg)

    if conflicts:
        return True, "CONFLICTING_RELATIONSHIP", tuple(conflicts)

    return False, None, ()
