"""Integration bridge coordinating live graph state, storage snapshots, and neural cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rutam.exceptions import ValidationError
from rutam.graph.encoder import GraphEncoderOutput, RUTAMGraphEncoder
from rutam.graph.operations import Graph
from rutam.graph.state import GraphState

if TYPE_CHECKING:
    from rutam.graph.storage import GraphStorage


class GraphMemoryBridge:
    """Manages active graph snapshots, storage synchronization, and neural encoding cache."""

    def __init__(
        self,
        graph: Graph,
        *,
        storage: GraphStorage | None = None,
        use_cache: bool = True,
    ) -> None:
        if not isinstance(graph, Graph):
            raise ValidationError("GraphMemoryBridge requires a Graph instance")
        self.graph = graph
        self.storage = storage
        self.use_cache = use_cache

        self._cached_state: GraphState | None = None
        self._cached_encoding: GraphEncoderOutput | None = None
        self._cached_version: int = -1
        self._cached_encoding_key: tuple[object, ...] | None = None

        if self.use_cache:
            self.refresh()

    @property
    def version(self) -> int:
        return self.graph.version

    def invalidate_caches(self) -> None:
        """Explicitly clear cached graph snapshot and neural encodings."""
        self._cached_state = None
        self._cached_encoding = None
        self._cached_version = -1
        self._cached_encoding_key = None

    def refresh(self) -> GraphState:
        """Rebuild GraphState from active graph and invalidate encoding cache."""
        with self.graph._mutation_lock:
            self._cached_state = self.graph.to_state()
            self._cached_encoding = None
            # The cache version belongs to the snapshot, not a subsequent read
            # of the live graph, which may already have advanced.
            self._cached_version = self._cached_state.graph_version
            self._cached_encoding_key = None
            return self._cached_state

    def get_active_state(self) -> GraphState:
        """Retrieve active GraphState, refreshing automatically on version mutation."""
        with self.graph._mutation_lock:
            needs_refresh = (
                not self.use_cache
                or self._cached_state is None
                or self.graph.version != self._cached_version
            )
            if needs_refresh:
                return self.refresh()
            return self._cached_state

    def get_active_encoding(
        self,
        encoder: RUTAMGraphEncoder,
        *,
        state: GraphState | None = None,
        node_mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> GraphEncoderOutput:
        """Encode one coherent snapshot, caching only stable inference results.

        Training encodings are always recomputed so an autograd graph is never
        reused across backward passes. Explicit snapshots and node/edge-masked
        views are keyed independently from the active unfiltered graph.
        """
        if not isinstance(encoder, RUTAMGraphEncoder):
            raise ValidationError("encoder must be a RUTAMGraphEncoder")
        active_state = self.get_active_state() if state is None else state
        if not isinstance(active_state, GraphState):
            raise ValidationError("state must be a GraphState")
        active_state.validate()
        if node_mask is not None and (
            not isinstance(node_mask, torch.Tensor)
            or node_mask.dtype != torch.bool
            or node_mask.device.type != "cpu"
            or node_mask.shape != active_state.node_ids.shape
        ):
            raise ValidationError("node_mask must be a CPU bool tensor with shape [N]")
        if edge_mask is not None and (
            not isinstance(edge_mask, torch.Tensor)
            or edge_mask.dtype != torch.bool
            or edge_mask.device.type != "cpu"
            or edge_mask.shape != active_state.edge_types.shape
        ):
            raise ValidationError("edge_mask must be a CPU bool tensor with shape [E]")

        cacheable = (
            self.use_cache
            and active_state is self.get_active_state()
            and not encoder.training
            and not torch.is_grad_enabled()
        )
        mask_key = None if node_mask is None else tuple(bool(v) for v in node_mask.tolist())
        edge_mask_key = None if edge_mask is None else tuple(bool(v) for v in edge_mask.tolist())
        versions = tuple(
            (name, id(tensor), tensor._version)
            for name, tensor in (*encoder.named_parameters(), *encoder.named_buffers())
        )
        cache_key: tuple[object, ...] = (
            id(encoder),
            active_state.graph_version,
            tuple(
                (name, id(tensor), tensor._version)
                for name, tensor in active_state.tensors().items()
            ),
            active_state.config.graph.directed,
            versions,
            mask_key,
            edge_mask_key,
        )
        if cacheable and self._cached_encoding is not None:
            if cache_key == self._cached_encoding_key:
                return self._cached_encoding

        encoded = encoder(active_state, node_mask=node_mask, edge_mask=edge_mask)
        if cacheable:
            self._cached_encoding = encoded
            self._cached_encoding_key = cache_key
        return encoded

    def reload_from_storage(self, storage: GraphStorage | None = None) -> None:
        """Reload canonical persistent snapshot from storage into active graph."""
        active_storage = storage if storage is not None else self.storage
        if active_storage is None:
            raise ValidationError("Cannot reload: no GraphStorage configured")
        # Serialize the entire reload against writers, including the storage
        # read: otherwise a completed write between load and replacement is lost.
        with self.graph._mutation_lock:
            loaded_graph = active_storage.load()
            if loaded_graph.config != self.graph.config:
                raise ValidationError("Stored graph config does not match active graph config")
            self.graph._nodes = dict(loaded_graph._nodes)
            self.graph._edges = dict(loaded_graph._edges)
            self.graph._edge_ids = dict(loaded_graph._edge_ids)
            self.graph._relationship_types = dict(loaded_graph._relationship_types)
            self.graph._retired_ids = set(loaded_graph._retired_ids)
            self.graph._retired_edge_ids = set(loaded_graph._retired_edge_ids)
            self.graph._next_node_id = loaded_graph._next_node_id
            self.graph._version = loaded_graph._version
            self.graph._metadata = dict(loaded_graph._metadata)
            self.invalidate_caches()
            self.refresh()
