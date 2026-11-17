"""Padded traversal structures. All IDs/indices use -1 padding; masks are authoritative."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PathCandidates:
    node_indices: torch.Tensor
    edge_indices: torch.Tensor
    relation_types: torch.Tensor
    directions: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    path_mask: torch.Tensor
    visited_indices: torch.Tensor
    visited_mask: torch.Tensor
    endpoint_slots: torch.Tensor


@dataclass(frozen=True, slots=True)
class PathScoringOutput:
    logits: torch.Tensor
    attention: torch.Tensor
    selected_indices: torch.Tensor
    selected_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class TraversalOutput:
    visited_indices: torch.Tensor
    visited_node_ids: torch.Tensor
    visited_mask: torch.Tensor
    node_relevance_scores: torch.Tensor
    path_representations: torch.Tensor
    path_scores: torch.Tensor
    path_mask: torch.Tensor
    reasoning_context: torch.Tensor
    path_attention: torch.Tensor
    selected_path_indices: torch.Tensor
    selected_path_mask: torch.Tensor
    path_node_indices: torch.Tensor
    path_node_ids: torch.Tensor
    path_relation_ids: torch.Tensor
    path_edge_indices: torch.Tensor
    path_directions: torch.Tensor
    path_node_mask: torch.Tensor
    path_edge_mask: torch.Tensor
    path_ids: torch.Tensor
    endpoint_node_ids: torch.Tensor
    graph_version: int | None
    topology_fingerprint: str
