"""Bounded simple-path BFS followed by differentiable relation-aware path reasoning."""

from __future__ import annotations

from collections import deque

import torch
from torch import nn

from rutam.config import TraversalConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import (
    RelationEmbedding,
    require_finite,
    validate_features,
    validate_relation_ids,
    validate_tensor,
)
from rutam.graph.encoder import GraphEncoderOutput
from rutam.graph.neighborhood import NeighborhoodCache, SparseNeighborhood
from rutam.graph.path_encoder import RUTAMPathEncoder
from rutam.graph.path_scoring import RUTAMPathScorer
from rutam.graph.state import GraphState
from rutam.graph.traversal_output import PathCandidates, TraversalOutput


def enumerate_paths(
    index: SparseNeighborhood,
    seeds: torch.Tensor,
    seed_mask: torch.Tensor,
    config: TraversalConfig,
    *,
    node_mask: torch.Tensor | None = None,
    direction: str | None = None,
) -> PathCandidates:
    """Discrete bounded BFS prefixes. Simple paths avoid cycles; global visits deduplicate nodes."""
    validate_tensor(seeds, "seed_indices", torch.int64)
    validate_tensor(seed_mask, "seed_mask", torch.bool)
    if seeds.ndim != 2 or seed_mask.shape != seeds.shape:
        raise ValidationError("Seeds and seed_mask must be [B,K]")
    n = len(index.node_ids)
    direction = config.direction if direction is None else direction
    if direction not in ("outgoing", "incoming", "both"):
        raise ValidationError("Traversal direction must be outgoing, incoming or both")
    if ((seeds[seed_mask] < 0) | (seeds[seed_mask] >= n)).any().item():
        raise ValidationError("Active seed row is outside graph")
    if node_mask is not None:
        validate_tensor(node_mask, "node_mask", torch.bool)
        if node_mask.shape != (n,):
            raise ValidationError("node_mask must be [N]")
    allowed = [True] * n if node_mask is None else node_mask.tolist()
    b, p, length, v = (
        seeds.shape[0],
        config.max_candidate_paths,
        config.max_hops + 1,
        config.max_visited_nodes,
    )
    if (
        max(b * p * length * max(config.hidden_dim, config.path_hidden_dim), b * v)
        > config.max_path_elements
    ):
        raise ValidationError("Path workspace exceeds max_path_elements")
    nodes = torch.full((b, p, length), -1, dtype=torch.int64)
    edges = torch.full((b, p, length - 1), -1, dtype=torch.int64)
    relations = torch.full_like(edges, -1)
    directions = torch.zeros_like(edges)
    node_valid = torch.zeros_like(nodes, dtype=torch.bool)
    edge_valid = torch.zeros_like(edges, dtype=torch.bool)
    path_valid = torch.zeros((b, p), dtype=torch.bool)
    visited = torch.full((b, v), -1, dtype=torch.int64)
    visited_valid = torch.zeros_like(visited, dtype=torch.bool)
    endpoint_slots = torch.full((b, p), -1, dtype=torch.int64)
    for batch in range(b):
        roots = list(
            dict.fromkeys(
                row
                for row, active in zip(
                    seeds[batch].tolist(), seed_mask[batch].tolist(), strict=True
                )
                if active and allowed[row]
            )
        )
        if config.deterministic:
            roots.sort(key=lambda row: index.node_ids[row])
        roots = roots[:v]
        touched = set(roots)
        reported = list(roots) if config.include_seed_nodes else []
        queue = deque(((row,), (), (), ()) for row in roots)
        paths = list(queue)[:p] if config.include_seed_nodes else []
        while queue and len(paths) < p:
            rows, edge_rows, rels, signs = queue.popleft()
            if len(edge_rows) >= config.max_hops:
                continue
            for neighbor in index.neighbors(
                rows[-1], direction or config.direction, config.max_neighbors_per_node
            ):
                target = neighbor.node
                if not allowed[target] or target in rows:
                    continue
                if target not in touched:
                    if len(touched) == v:
                        continue
                    touched.add(target)
                    reported.append(target)
                elif target not in reported and target not in roots:
                    reported.append(target)
                path = (
                    rows + (target,),
                    edge_rows + (neighbor.edge,),
                    rels + (neighbor.relation,),
                    signs + (neighbor.direction,),
                )
                paths.append(path)
                queue.append(path)
                if len(paths) == p:
                    break
        slots = {row: slot for slot, row in enumerate(reported)}
        if reported:
            visited[batch, : len(reported)] = torch.tensor(reported)
            visited_valid[batch, : len(reported)] = True
        for candidate, (rows, edge_rows, rels, signs) in enumerate(paths):
            size = len(rows)
            nodes[batch, candidate, :size] = torch.tensor(rows)
            node_valid[batch, candidate, :size] = True
            if edge_rows:
                edges[batch, candidate, : size - 1] = torch.tensor(edge_rows)
                relations[batch, candidate, : size - 1] = torch.tensor(rels)
                directions[batch, candidate, : size - 1] = torch.tensor(signs)
                edge_valid[batch, candidate, : size - 1] = True
            path_valid[batch, candidate] = True
            endpoint_slots[batch, candidate] = slots.get(rows[-1], -1)
    return PathCandidates(
        nodes,
        edges,
        relations,
        directions,
        node_valid,
        edge_valid,
        path_valid,
        visited,
        visited_valid,
        endpoint_slots,
    )


class RUTAMGraphTraversal(nn.Module):
    def __init__(
        self,
        config: TraversalConfig | None = None,
        *,
        relation_embedding: RelationEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.config = TraversalConfig() if config is None else config
        if not isinstance(self.config, TraversalConfig):
            raise ValidationError("Traversal requires TraversalConfig")
        self.path_encoder = RUTAMPathEncoder(self.config, relation_embedding=relation_embedding)
        self.path_scorer = RUTAMPathScorer(self.config)
        self.index_cache = NeighborhoodCache()

    def forward(
        self,
        query_vectors: torch.Tensor,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        seed_indices: torch.Tensor,
        seed_mask: torch.Tensor,
        node_ids: torch.Tensor,
        *,
        graph_version: int | None = None,
        node_mask: torch.Tensor | None = None,
        direction: str | None = None,
    ) -> TraversalOutput:
        validate_features(query_vectors, self.config.query_dim, "query_vectors")
        validate_features(node_states, self.config.hidden_dim, "node_states")
        validate_tensor(node_ids, "node_ids", torch.int64)
        validate_tensor(seed_indices, "seed_indices", torch.int64)
        if seed_indices.ndim != 2:
            raise ValidationError("seed_indices must be [B,K]")
        validate_relation_ids(edge_types, self.config.num_relation_types)
        validate_tensor(edge_weights, "edge_weights", torch.float32)
        if edge_weights.shape != edge_types.shape:
            raise ValidationError("edge_weights must have shape [E]")
        if (
            node_ids.numel() != node_states.shape[0]
            or seed_indices.shape[0] != query_vectors.shape[0]
        ):
            raise ValidationError("Traversal batch/node dimensions disagree")
        index = self.index_cache.get(node_ids, edge_index, edge_types, graph_version)
        candidates = enumerate_paths(
            index, seed_indices, seed_mask, self.config, node_mask=node_mask, direction=direction
        )
        representations = self.path_encoder(node_states, candidates, edge_weights)
        scores = self.path_scorer(query_vectors, representations, candidates.path_mask)
        context = (scores.attention[..., None] * representations).sum(dim=1)
        require_finite(context, "reasoning context")
        visited_ids = _ids(node_ids, candidates.visited_indices, candidates.visited_mask)
        path_node_ids = _ids(node_ids, candidates.node_indices, candidates.node_mask)
        ends = candidates.node_mask.sum(2).clamp_min(1) - 1
        endpoint_ids = path_node_ids.gather(2, ends[..., None]).squeeze(-1)
        relevance = node_states.new_zeros(candidates.visited_indices.shape)
        relevance.scatter_add_(
            1,
            candidates.endpoint_slots.clamp_min(0),
            scores.attention * (candidates.endpoint_slots >= 0),
        )
        path_ids = torch.arange(self.config.max_candidate_paths).expand(query_vectors.shape[0], -1)
        path_ids = torch.where(candidates.path_mask, path_ids, -1)
        return TraversalOutput(
            candidates.visited_indices,
            visited_ids,
            candidates.visited_mask,
            relevance,
            representations,
            scores.logits,
            candidates.path_mask,
            context,
            scores.attention,
            scores.selected_indices,
            scores.selected_mask,
            candidates.node_indices,
            path_node_ids,
            candidates.relation_types,
            candidates.edge_indices,
            candidates.directions,
            candidates.node_mask,
            candidates.edge_mask,
            path_ids,
            endpoint_ids,
            graph_version,
            index.fingerprint,
        )

    def from_state(
        self,
        query_vectors: torch.Tensor,
        encoded: GraphEncoderOutput | torch.Tensor,
        state: GraphState,
        seed_indices: torch.Tensor,
        seed_mask: torch.Tensor,
        *,
        node_mask: torch.Tensor | None = None,
    ) -> TraversalOutput:
        state.validate()
        nodes = encoded.node_states if isinstance(encoded, GraphEncoderOutput) else encoded
        return self(
            query_vectors,
            nodes,
            state.edge_index,
            state.edge_types,
            state.edge_weights,
            seed_indices,
            seed_mask,
            state.node_ids,
            graph_version=state.graph_version,
            node_mask=node_mask,
            direction="both" if not state.config.graph.directed else self.config.direction,
        )


def _ids(ids: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if ids.numel() == 0:
        return torch.full_like(rows, -1)
    return torch.where(mask, ids[rows.clamp_min(0)], -1)
