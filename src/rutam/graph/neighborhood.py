"""Sparse CPU topology preparation, separate from differentiable neural computation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from rutam.config import positive_int
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


def validate_topology(
    node_ids: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor
) -> None:
    for tensor, name in (
        (node_ids, "node_ids"),
        (edge_index, "edge_index"),
        (edge_types, "edge_types"),
    ):
        validate_tensor(tensor, name, torch.int64)
    if (
        node_ids.ndim != 1
        or (node_ids < 0).any().item()
        or torch.unique(node_ids).numel() != node_ids.numel()
    ):
        raise ValidationError("node_ids must be unique nonnegative IDs [N]")
    if edge_types.ndim != 1 or edge_index.shape != (2, edge_types.numel()):
        raise ValidationError("edge_index/edge_types must have shapes [2,E]/[E]")
    if (edge_types < 0).any().item():
        raise ValidationError("Relation IDs cannot be negative")
    if edge_index.numel() and (
        edge_index.min().item() < 0 or edge_index.max().item() >= node_ids.numel()
    ):
        raise ValidationError("Dangling edge row references")


@dataclass(frozen=True, slots=True)
class Neighbor:
    node: int
    edge: int
    relation: int
    direction: int  # +1 along stored source->target; -1 against it.


@dataclass(frozen=True, slots=True)
class SparseNeighborhood:
    node_ids: tuple[int, ...]
    outgoing: tuple[tuple[Neighbor, ...], ...]
    incoming: tuple[tuple[Neighbor, ...], ...]
    typed_edges: frozenset[tuple[int, int, int]]
    fingerprint: str
    graph_version: int | None

    def neighbors(
        self, row: int, direction: str = "outgoing", limit: int | None = None
    ) -> tuple[Neighbor, ...]:
        if limit is not None:
            positive_int(limit, "neighbor limit", minimum=0)
        if row < 0 or row >= len(self.node_ids):
            raise ValidationError("Neighbor row is outside graph")
        if direction == "outgoing":
            values = self.outgoing[row]
        elif direction == "incoming":
            values = self.incoming[row]
        elif direction == "both":
            values = tuple(
                sorted(
                    (*self.outgoing[row], *(x for x in self.incoming[row] if x.node != row)),
                    key=lambda x: (self.node_ids[x.node], x.relation, x.direction, x.edge),
                )
            )
        else:
            raise ValidationError("direction must be outgoing, incoming or both")
        return values if limit is None else values[:limit]


class NeighborhoodCache:
    """One immutable index, keyed by version AND full integer topology digest.

    Hashing is O(N+E) each call and catches in-place topology edits, even when a
    caller incorrectly reuses a version. Without a version, indexes are rebuilt.
    No feature/weight tensor is detached or cached by this preparation layer.
    """

    def __init__(self) -> None:
        self._index: SparseNeighborhood | None = None
        self.build_count = 0

    def clear(self) -> None:
        self._index = None

    def get(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        graph_version: int | None = None,
    ) -> SparseNeighborhood:
        validate_topology(node_ids, edge_index, edge_types)
        if graph_version is not None:
            positive_int(graph_version, "graph_version", minimum=0)
        digest = hashlib.sha256()
        for tensor in (node_ids, edge_index, edge_types):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        fingerprint = digest.hexdigest()
        if (
            self._index is not None
            and graph_version is not None
            and (
                self._index.graph_version == graph_version
                and self._index.fingerprint == fingerprint
            )
        ):
            return self._index
        ids = tuple(node_ids.tolist())
        out: list[list[Neighbor]] = [[] for _ in ids]
        inc: list[list[Neighbor]] = [[] for _ in ids]
        triples = set()
        for edge, (source, target, relation) in enumerate(
            zip(*edge_index.tolist(), edge_types.tolist(), strict=True)
        ):
            triple = (source, target, relation)
            if triple in triples:
                raise ValidationError("Duplicate directed typed edge")
            triples.add(triple)
            out[source].append(Neighbor(target, edge, relation, 1))
            inc[target].append(Neighbor(source, edge, relation, -1))

        def key(x: Neighbor) -> tuple[str, int, int]:
            return (ids[x.node], x.relation, x.edge)

        index = SparseNeighborhood(
            ids,
            tuple(tuple(sorted(x, key=key)) for x in out),
            tuple(tuple(sorted(x, key=key)) for x in inc),
            frozenset(triples),
            fingerprint,
            graph_version,
        )
        self.build_count += 1
        self._index = index if graph_version is not None else None
        return index
