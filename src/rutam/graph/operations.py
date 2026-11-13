"""Deterministic graph CRUD, bounded traversal and tensor conversion."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import torch

from rutam.config import RUTAMConfig, nonempty_string, positive_int
from rutam.exceptions import DuplicateIDError, GraphReferenceError, ValidationError
from rutam.graph.schema import Edge, Node, identifier, utc_now
from rutam.graph.state import GraphState, json_copy

if TYPE_CHECKING:
    from rutam.graph.storage import GraphStorage

Direction = Literal["outgoing", "incoming", "both"]


class Graph:
    """In-memory graph with stable IDs and optional transactional auto-save.

    This object is single-threaded. Node/edge records are immutable. All returned
    metadata and tensor snapshots are detached from live graph data.
    """

    def __init__(
        self,
        config: RUTAMConfig | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        storage: GraphStorage | None = None,
    ) -> None:
        self.config = RUTAMConfig() if config is None else config
        if not isinstance(self.config, RUTAMConfig):
            raise ValidationError("config must be a RUTAMConfig")
        self._nodes: dict[int, Node] = {}
        self._edges: dict[tuple[int, int, str], Edge] = {}
        self._relationship_types: dict[str, int] = {}
        self._retired_ids: set[int] = set()
        self._next_node_id = 0
        self._version = 0
        self._metadata = json_copy({} if metadata is None else metadata)
        self._storage = storage

    @property
    def version(self) -> int:
        return self._version

    @property
    def metadata(self) -> dict[str, Any]:
        return json_copy(self._metadata)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges[key] for key in sorted(self._edges))

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        value = json_copy(metadata)
        with self._mutation():
            self._metadata = value

    def _check_provenance(self, provenance: str | None) -> None:
        if self.config.memory.require_provenance and provenance is None:
            raise ValidationError("Provenance is required for every node and relationship")

    def _normalize_node(self, node: Node) -> Node:
        self._check_provenance(node.provenance)
        if node.embedding is not None:
            if len(node.embedding) != self.config.graph.embedding_dim:
                raise ValidationError("Node embedding length does not match graph.embedding_dim")
            values = torch.tensor(node.embedding, dtype=torch.float32).tolist()
            node = replace(node, embedding=tuple(values))
        return node

    def create_node(
        self, node_type: str, content: str, *, node_id: int | None = None, **fields: Any
    ) -> Node:
        return self.add_node(
            Node(
                id=self._next_node_id if node_id is None else node_id,
                node_type=node_type,
                content=content,
                **fields,
            )
        )

    def add_node(self, node: Node) -> Node:
        if not isinstance(node, Node):
            raise ValidationError("add_node requires a Node")
        node = self._normalize_node(node)
        if node.id in self._nodes or node.id in self._retired_ids:
            raise DuplicateIDError(f"Node ID {node.id} is already live or retired")
        if len(self._nodes) >= self.config.graph.max_nodes:
            raise ValidationError("Graph max_nodes capacity reached")
        with self._mutation():
            self._nodes[node.id] = node
            self._next_node_id = max(self._next_node_id, node.id + 1)
        return node

    def get_node(self, node_id: int) -> Node:
        identifier(node_id)
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise GraphReferenceError(f"Node {node_id} does not exist") from exc

    def update_node(self, node_id: int, **changes: Any) -> Node:
        node = self.get_node(node_id)
        allowed = {"node_type", "content", "provenance", "confidence", "embedding"}
        if not changes.keys() <= allowed:
            raise ValidationError(
                "Only node content, type, provenance, confidence and embedding change"
            )
        node = self._normalize_node(replace(node, **changes, updated_at=utc_now()))
        with self._mutation():
            self._nodes[node_id] = node
        return node

    def delete_node(self, node_id: int) -> None:
        self.get_node(node_id)
        with self._mutation():
            del self._nodes[node_id]
            self._retired_ids.add(node_id)
            self._edges = {
                key: edge
                for key, edge in self._edges.items()
                if node_id not in (edge.source_id, edge.target_id)
            }

    def _key(self, source_id: int, target_id: int, relationship: str) -> tuple[int, int, str]:
        identifier(source_id, "source_id")
        identifier(target_id, "target_id")
        nonempty_string(relationship, "relationship")
        if not self.config.graph.directed and source_id > target_id:
            source_id, target_id = target_id, source_id
        return source_id, target_id, relationship

    def add_relationship(
        self, source_id: int, target_id: int, relationship: str, **fields: Any
    ) -> Edge:
        return self.add_edge(Edge(source_id, target_id, relationship, **fields))

    def add_edge(self, edge: Edge) -> Edge:
        if not isinstance(edge, Edge):
            raise ValidationError("add_edge requires an Edge")
        self.get_node(edge.source_id)
        self.get_node(edge.target_id)
        self._check_provenance(edge.provenance)
        key = self._key(*edge.key)
        edge = replace(
            edge,
            source_id=key[0],
            target_id=key[1],
            weight=torch.tensor(edge.weight, dtype=torch.float32).item(),
        )
        if key in self._edges:
            raise ValidationError(f"Relationship {key} already exists")
        if len(self._edges) >= self.config.graph.max_edges:
            raise ValidationError("Graph max_edges capacity reached")
        with self._mutation():
            if edge.relationship not in self._relationship_types:
                self._relationship_types[edge.relationship] = (
                    max(self._relationship_types.values(), default=-1) + 1
                )
            self._edges[key] = edge
        return edge

    def get_relationship(self, source_id: int, target_id: int, relationship: str) -> Edge:
        key = self._key(source_id, target_id, relationship)
        try:
            return self._edges[key]
        except KeyError as exc:
            raise GraphReferenceError(f"Relationship {key} does not exist") from exc

    def update_relationship(
        self, source_id: int, target_id: int, relationship: str, **changes: Any
    ) -> Edge:
        edge = self.get_relationship(source_id, target_id, relationship)
        if not changes.keys() <= {"weight", "provenance"}:
            raise ValidationError("Only relationship weight and provenance can change")
        edge = replace(edge, **changes, timestamp=utc_now())
        self._check_provenance(edge.provenance)
        edge = replace(edge, weight=torch.tensor(edge.weight, dtype=torch.float32).item())
        with self._mutation():
            self._edges[edge.key] = edge
        return edge

    def remove_relationship(self, source_id: int, target_id: int, relationship: str) -> None:
        edge = self.get_relationship(source_id, target_id, relationship)
        with self._mutation():
            del self._edges[edge.key]

    def _adjacency(self, direction: Direction, relationship: str | None) -> dict[int, set[int]]:
        if direction not in ("outgoing", "incoming", "both"):
            raise ValidationError("direction must be outgoing, incoming or both")
        if relationship is not None:
            nonempty_string(relationship, "relationship")
        adjacency: dict[int, set[int]] = {}
        for edge in self._edges.values():
            if relationship is not None and edge.relationship != relationship:
                continue
            if not self.config.graph.directed or direction in ("outgoing", "both"):
                adjacency.setdefault(edge.source_id, set()).add(edge.target_id)
            if not self.config.graph.directed or direction in ("incoming", "both"):
                adjacency.setdefault(edge.target_id, set()).add(edge.source_id)
        return adjacency

    def neighbors(
        self, node_id: int, *, direction: Direction = "outgoing", relationship: str | None = None
    ) -> tuple[Node, ...]:
        self.get_node(node_id)
        adjacency = self._adjacency(direction, relationship)
        return tuple(self._nodes[key] for key in sorted(adjacency.get(node_id, ())))

    def traverse(
        self,
        node_id: int,
        max_hops: int,
        *,
        direction: Direction = "outgoing",
        relationship: str | None = None,
    ) -> tuple[Node, ...]:
        """Breadth-first traversal with sorted neighbors; excludes the start node."""
        self.get_node(node_id)
        positive_int(max_hops, "max_hops", minimum=0)
        adjacency = self._adjacency(direction, relationship)
        seen, result = {node_id}, []
        queue = deque([(node_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth == max_hops:
                continue
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    result.append(self._nodes[neighbor])
                    queue.append((neighbor, depth + 1))
        return tuple(result)

    def to_state(self) -> GraphState:
        nodes, edges = self.nodes, self.edges
        rows = {node.id: row for row, node in enumerate(nodes)}
        embeddings = torch.zeros((len(nodes), self.config.graph.embedding_dim), dtype=torch.float32)
        node_metadata = []
        for row, node in enumerate(nodes):
            if node.embedding is not None:
                embeddings[row] = torch.tensor(node.embedding, dtype=torch.float32)
            record = asdict(node)
            del record["embedding"]
            record.update(
                created_at=node.created_at.isoformat(),
                updated_at=node.updated_at.isoformat(),
                has_embedding=node.embedding is not None,
            )
            node_metadata.append(record)
        edge_metadata = []
        for edge in edges:
            record = asdict(edge)
            del record["weight"]
            record["timestamp"] = edge.timestamp.isoformat()
            edge_metadata.append(record)
        state = GraphState(
            node_ids=torch.tensor([n.id for n in nodes], dtype=torch.int64),
            node_embeddings=embeddings,
            edge_index=torch.tensor(
                [[rows[e.source_id] for e in edges], [rows[e.target_id] for e in edges]],
                dtype=torch.int64,
            ).reshape(2, len(edges)),
            edge_types=torch.tensor(
                [self._relationship_types[e.relationship] for e in edges], dtype=torch.int64
            ),
            edge_weights=torch.tensor([e.weight for e in edges], dtype=torch.float32),
            node_metadata=node_metadata,
            edge_metadata=edge_metadata,
            relationship_types=dict(self._relationship_types),
            graph_version=self.version,
            config=self.config,
            next_node_id=self._next_node_id,
            retired_node_ids=tuple(sorted(self._retired_ids)),
            metadata=self.metadata,
        )
        state.validate()
        return state

    @classmethod
    def from_state(cls, state: GraphState, *, storage: GraphStorage | None = None) -> Graph:
        state.validate()
        graph = cls(state.config, metadata=state.metadata, storage=storage)
        try:
            for row, record in enumerate(state.node_metadata):
                values = dict(record)
                has_embedding = values.pop("has_embedding")
                if type(has_embedding) is not bool:
                    raise ValidationError("has_embedding must be a boolean")
                if not has_embedding and state.node_embeddings[row].count_nonzero().item():
                    raise ValidationError("Missing embeddings must have all-zero tensor rows")
                for field in ("created_at", "updated_at"):
                    values[field] = datetime.fromisoformat(values[field])
                values["embedding"] = (
                    tuple(state.node_embeddings[row].tolist()) if has_embedding else None
                )
                node = graph._normalize_node(Node(**values))
                if node.id != state.node_ids[row].item():
                    raise ValidationError("Node metadata and node_ids disagree")
                graph._nodes[node.id] = node
            labels = {value: key for key, value in state.relationship_types.items()}
            for column, record in enumerate(state.edge_metadata):
                values = dict(record)
                values["timestamp"] = datetime.fromisoformat(values["timestamp"])
                values["weight"] = state.edge_weights[column].item()
                edge = Edge(**values)
                graph._check_provenance(edge.provenance)
                source, target = state.edge_index[:, column].tolist()
                if (
                    (edge.source_id, edge.target_id)
                    != (state.node_ids[source].item(), state.node_ids[target].item())
                    or edge.relationship != labels[state.edge_types[column].item()]
                    or edge.key != graph._key(*edge.key)
                    or edge.key in graph._edges
                ):
                    raise ValidationError("Duplicate edge or inconsistent relationship metadata")
                graph._edges[edge.key] = edge
            if list(graph._edges) != sorted(graph._edges):
                raise ValidationError("Relationships must be in canonical sorted order")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"Invalid graph record metadata: {exc}") from exc
        graph._relationship_types = dict(state.relationship_types)
        graph._retired_ids = set(state.retired_node_ids)
        graph._next_node_id = state.next_node_id
        graph._version = state.graph_version
        return graph

    def save(self, storage: GraphStorage | None = None) -> None:
        from rutam.graph.storage import GraphStorage

        (storage or self._storage or GraphStorage(self.config.graph.path)).save(self)

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        backup = None
        if self.config.graph.auto_save:
            backup = (
                self._nodes.copy(),
                self._edges.copy(),
                self._relationship_types.copy(),
                self._retired_ids.copy(),
                self._next_node_id,
                self._version,
                self._metadata,
            )
        try:
            yield
            self._version += 1
            if self.config.graph.auto_save:
                self.save()
        except Exception:
            if backup is not None:
                (
                    self._nodes,
                    self._edges,
                    self._relationship_types,
                    self._retired_ids,
                    self._next_node_id,
                    self._version,
                    self._metadata,
                ) = backup
            raise
