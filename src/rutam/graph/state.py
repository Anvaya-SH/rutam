"""Tensor boundary for storage and future graph encoders (no learned components)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from rutam.config import RUTAMConfig, nonempty_string, positive_int
from rutam.exceptions import ValidationError
from rutam.graph.schema import MAX_ID, identifier


def json_copy(value: dict[str, Any]) -> dict[str, Any]:
    """Detach JSON metadata and reject non-JSON values, non-string keys and NaNs."""

    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValidationError("JSON metadata keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValidationError("Metadata must contain only JSON values")

    if not isinstance(value, dict):
        raise ValidationError("metadata must be a JSON object")
    try:
        check(value)
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError(f"Invalid JSON metadata: {exc}") from exc


@dataclass(slots=True)
class GraphState:
    """Detached snapshot. edge_index contains row indices, never persistent node IDs.

    Optional embeddings use zero rows plus a per-node `has_embedding` marker.
    Labels retain their IDs after the final edge of that type is removed.
    """

    node_ids: torch.Tensor
    node_embeddings: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    edge_weights: torch.Tensor
    node_metadata: list[dict[str, Any]]
    edge_metadata: list[dict[str, Any]]
    relationship_types: dict[str, int]
    graph_version: int
    config: RUTAMConfig
    next_node_id: int
    retired_node_ids: tuple[int, ...]
    metadata: dict[str, Any]
    retired_edge_ids: tuple[str, ...] = ()

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in ("node_ids", "node_embeddings", "edge_index", "edge_types", "edge_weights")
        }

    def validate(self) -> None:
        if not isinstance(self.config, RUTAMConfig):
            raise ValidationError("state.config must be a RUTAMConfig")
        for name, tensor in self.tensors().items():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValidationError(f"{name} must be a CPU tensor")
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                raise ValidationError(f"{name} must be a dense contiguous tensor")
        if self.node_ids.ndim != 1 or self.edge_types.ndim != 1:
            raise ValidationError("node_ids and edge_types must have rank 1")
        n, e = self.node_ids.numel(), self.edge_types.numel()
        expected = {
            "node_ids": (torch.int64, (n,)),
            "node_embeddings": (torch.float32, (n, self.config.graph.embedding_dim)),
            "edge_index": (torch.int64, (2, e)),
            "edge_types": (torch.int64, (e,)),
            "edge_weights": (torch.float32, (e,)),
        }
        for name, (dtype, shape) in expected.items():
            tensor = getattr(self, name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValidationError(f"{name} must have dtype {dtype} and shape {shape}")
        if n > self.config.graph.max_nodes or e > self.config.graph.max_edges:
            raise ValidationError("Graph exceeds configured capacity")
        ids = self.node_ids.tolist()
        if ids != sorted(set(ids)) or any(value < 0 for value in ids):
            raise ValidationError("node_ids must be sorted, unique and nonnegative")
        if e and (self.edge_index.min().item() < 0 or self.edge_index.max().item() >= n):
            raise ValidationError("edge_index contains dangling row references")
        for name in ("node_embeddings", "edge_weights"):
            if not torch.isfinite(getattr(self, name)).all().item():
                raise ValidationError(f"{name} contains nonfinite values")
        if (self.edge_weights < 0).any().item():
            raise ValidationError("edge_weights must be nonnegative")
        if not isinstance(self.relationship_types, dict):
            raise ValidationError("relationship_types must be an object")
        for label, type_id in self.relationship_types.items():
            nonempty_string(label, "relationship label")
            identifier(type_id, "relationship type ID")
        type_ids = list(self.relationship_types.values())
        if len(type_ids) != len(set(type_ids)):
            raise ValidationError("Relationship type IDs must be unique")
        if not set(self.edge_types.tolist()).issubset(type_ids):
            raise ValidationError("Unknown edge type ID")
        positive_int(self.graph_version, "graph_version", minimum=0)
        positive_int(self.next_node_id, "next_node_id", minimum=0)
        if self.next_node_id > MAX_ID + 1:
            raise ValidationError("next_node_id exceeds int64 allocation range")
        if not isinstance(self.retired_node_ids, tuple):
            raise ValidationError("retired_node_ids must be a tuple")
        for value in self.retired_node_ids:
            identifier(value, "retired node ID")
        retired = set(self.retired_node_ids)
        if len(retired) != len(self.retired_node_ids) or retired.intersection(ids):
            raise ValidationError("Retired IDs must be unique and disjoint from live IDs")
        if any(value >= self.next_node_id for value in (*ids, *retired)):
            raise ValidationError("next_node_id must exceed all allocated IDs")
        if not isinstance(self.node_metadata, list) or len(self.node_metadata) != n:
            raise ValidationError("node_metadata length must equal N")
        if not isinstance(self.edge_metadata, list) or len(self.edge_metadata) != e:
            raise ValidationError("edge_metadata length must equal E")
        for record in self.node_metadata + self.edge_metadata:
            json_copy(record)
        edge_ids = [record.get("edge_id") for record in self.edge_metadata]
        if any(edge_id is not None and (not isinstance(edge_id, str) or not edge_id) for edge_id in edge_ids):
            raise ValidationError("edge_id values must be nonempty strings")
        present_edge_ids = [edge_id for edge_id in edge_ids if edge_id is not None]
        if len(present_edge_ids) != len(set(present_edge_ids)):
            raise ValidationError("Live edge IDs must be unique")
        if not isinstance(self.retired_edge_ids, tuple):
            raise ValidationError("retired_edge_ids must be a tuple")
        for edge_id in self.retired_edge_ids:
            nonempty_string(edge_id, "retired edge ID")
        retired_edges = set(self.retired_edge_ids)
        if len(retired_edges) != len(self.retired_edge_ids) or retired_edges.intersection(present_edge_ids):
            raise ValidationError("Retired edge IDs must be unique and disjoint from live IDs")
        json_copy(self.metadata)

    def extract_subgraph(
        self,
        node_indices: torch.Tensor | list[int] | tuple[int, ...],
        *,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[GraphState, torch.Tensor]:
        """Extract a self-contained induced subgraph for bounded sparse execution.

        Parameters
        ----------
        node_indices : torch.Tensor | Sequence[int]
            Indices of nodes to retain from this state's node_ids tensor.
        edge_mask : torch.Tensor | None
            Optional boolean mask over edges (e.g. for validity / visibility filtering).

        Returns
        -------
        sub_state : GraphState
            A valid GraphState containing only the selected nodes and induced edges.
        retained_indices : torch.Tensor
            Sorted 1D int64 tensor of original node indices included in sub_state.
        """
        if isinstance(node_indices, torch.Tensor):
            valid = (node_indices >= 0) & (node_indices < self.node_ids.shape[0])
            retained = torch.unique(node_indices[valid])
        else:
            n = self.node_ids.shape[0]
            retained = torch.tensor(
                sorted(set(i for i in node_indices if 0 <= i < n)),
                dtype=torch.int64,
                device="cpu",
            )
        k = retained.shape[0]
        if k == 0:
            sub_state = GraphState(
                node_ids=torch.empty(0, dtype=torch.int64),
                node_embeddings=torch.empty(0, self.config.graph.embedding_dim, dtype=torch.float32),
                edge_index=torch.empty(2, 0, dtype=torch.int64),
                edge_types=torch.empty(0, dtype=torch.int64),
                edge_weights=torch.empty(0, dtype=torch.float32),
                node_metadata=[],
                edge_metadata=[],
                relationship_types=dict(self.relationship_types),
                graph_version=self.graph_version,
                config=self.config,
                next_node_id=self.next_node_id,
                retired_node_ids=self.retired_node_ids,
                metadata=self.metadata,
                retired_edge_ids=self.retired_edge_ids,
            )
            return sub_state, retained

        sub_node_ids = self.node_ids[retained]
        sub_node_embeddings = self.node_embeddings[retained]
        sub_node_metadata = [self.node_metadata[i.item()] for i in retained]

        inverse_map = torch.full((self.node_ids.shape[0],), -1, dtype=torch.int64, device="cpu")
        inverse_map[retained] = torch.arange(k, dtype=torch.int64, device="cpu")

        if self.edge_index.shape[1] > 0:
            src_new = inverse_map[self.edge_index[0]]
            dst_new = inverse_map[self.edge_index[1]]
            induced_edge_mask = (src_new >= 0) & (dst_new >= 0)
            if edge_mask is not None:
                induced_edge_mask = induced_edge_mask & edge_mask
            sub_edge_index = torch.stack([src_new[induced_edge_mask], dst_new[induced_edge_mask]], dim=0)
            sub_edge_types = self.edge_types[induced_edge_mask]
            sub_edge_weights = self.edge_weights[induced_edge_mask]
            mask_list = induced_edge_mask.tolist()
            sub_edge_metadata = [
                self.edge_metadata[i] for i, is_inc in enumerate(mask_list) if is_inc
            ]
        else:
            sub_edge_index = torch.empty(2, 0, dtype=torch.int64)
            sub_edge_types = torch.empty(0, dtype=torch.int64)
            sub_edge_weights = torch.empty(0, dtype=torch.float32)
            sub_edge_metadata = []

        sub_state = GraphState(
            node_ids=sub_node_ids,
            node_embeddings=sub_node_embeddings,
            edge_index=sub_edge_index,
            edge_types=sub_edge_types,
            edge_weights=sub_edge_weights,
            node_metadata=sub_node_metadata,
            edge_metadata=sub_edge_metadata,
            relationship_types=dict(self.relationship_types),
            graph_version=self.graph_version,
            config=self.config,
            next_node_id=self.next_node_id,
            retired_node_ids=self.retired_node_ids,
            metadata=self.metadata,
            retired_edge_ids=self.retired_edge_ids,
        )
        return sub_state, retained

