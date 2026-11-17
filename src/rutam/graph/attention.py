"""RUTAM relation-aware native graph attention subsystem."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from rutam.config import GraphAttentionConfig, boolean
from rutam.exceptions import ValidationError
from rutam.graph.attention_output import GraphAttentionOutput
from rutam.graph.embeddings import require_finite, validate_tensor
from rutam.graph.relation_attention import RelationAttentionLayer


class RUTAMGraphAttention(nn.Module):
    """Relation-aware multi-head graph attention architecture for RUTAM.

    Processes graph nodes and typed edges with learned structural and relation-aware
    attention, optionally conditioned on incoming neural queries from language models.
    """

    def __init__(self, config: GraphAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = GraphAttentionConfig() if config is None else config
        if not isinstance(self.config, GraphAttentionConfig):
            raise ValidationError("RUTAMGraphAttention requires a GraphAttentionConfig")

        self.layers = nn.ModuleList(
            RelationAttentionLayer(self.config) for _ in range(self.config.num_layers)
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        *,
        query: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        subgraph_nodes: Sequence[int] | Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
        pool_graph: bool = True,
        snapshot_id: str | None = None,
    ) -> GraphAttentionOutput:
        """Execute relation-aware graph attention over full graph or bounded subgraph.

        Args:
            node_states: [N, H] float32 node embeddings.
            edge_index: [2, E] int64 edge tensor (source, destination).
            edge_types: [E] int64 relation types.
            edge_weights: [E] float32 edge weights.
            query: Optional [Q] or [B, Q] neural query vectors.
            node_mask: Optional [N] or [B, N] boolean mask.
            edge_mask: Optional [E] boolean mask.
            subgraph_nodes: Optional subset of node indices or IDs to restrict attention to.
            node_ids: Optional canonical persistent IDs for the N nodes.
            pool_graph: Whether to compute a pooled graph summary vector.
            snapshot_id: Optional snapshot identifier for provenance.

        Returns:
            GraphAttentionOutput with contextualized states, diagnostics, and attention.
        """
        self._validate_inputs(node_states, edge_index, edge_types, edge_weights)
        boolean(pool_graph, "pool_graph")

        n = node_states.shape[0]
        e = edge_index.shape[1]

        # Handle empty graph
        if n == 0:
            is_batched = query is not None and query.dim() == 2
            b = query.shape[0] if is_batched else 1
            empty_states = node_states.new_zeros(
                (b, 0, self.config.hidden_dim) if is_batched else (0, self.config.hidden_dim)
            )
            summary = (
                node_states.new_zeros(
                    (b, self.config.hidden_dim) if is_batched else (self.config.hidden_dim,)
                )
                if pool_graph
                else None
            )
            return GraphAttentionOutput(
                node_states=empty_states,
                edge_attention=None,
                graph_summary=summary,
                diagnostics={"empty_graph": True, "num_nodes": 0, "num_edges": 0},
                node_id_map=[] if node_ids is not None else None,
            )

        # Optional subgraph extraction
        subgraph_extracted = False
        node_id_map = list(node_ids) if node_ids is not None else None

        if subgraph_nodes is not None and len(subgraph_nodes) > 0:
            node_states, edge_index, edge_types, edge_weights, edge_mask, node_id_map = (
                self._extract_subgraph(
                    node_states,
                    edge_index,
                    edge_types,
                    edge_weights,
                    subgraph_nodes,
                    node_ids,
                    edge_mask,
                )
            )
            subgraph_extracted = True
            n = node_states.shape[0]
            e = edge_index.shape[1]

        # Enforce batch/graph size budget
        is_batched = query is not None and query.dim() == 2
        if is_batched:
            b = query.shape[0]
            total_elements = b * n * self.config.hidden_dim
            max_budget = self.config.max_subgraph_nodes * self.config.hidden_dim * 64
            if total_elements > max_budget:
                raise ValidationError(
                    f"Batched query attention exceeds memory budget: "
                    f"{total_elements} > {max_budget}"
                )

        current_states = node_states
        last_attention: torch.Tensor | None = None
        layer_diagnostics: list[dict[str, Any]] = []

        # Iterate attention layers
        for layer in self.layers:
            current_states, alpha, diag = layer(
                current_states,
                edge_index,
                edge_types,
                edge_weights,
                query=query,
                edge_mask=edge_mask,
            )
            last_attention = alpha
            layer_diagnostics.append(diag)

        # Graph summary pooling
        summary: torch.Tensor | None = None
        if pool_graph:
            if is_batched:
                summary = current_states.mean(dim=1) if n > 0 else current_states.sum(dim=1)
            else:
                summary = current_states.mean(dim=0) if n > 0 else current_states.sum(dim=0)
            require_finite(summary, "graph summary")

        edge_attn_to_return = last_attention if self.config.return_attention else None

        agg_diagnostics = {
            "num_nodes": n,
            "num_edges": e,
            "subgraph_extracted": subgraph_extracted,
            "layers": layer_diagnostics,
            "snapshot_id": snapshot_id,
        }
        if layer_diagnostics:
            agg_diagnostics["final_attention_entropy"] = layer_diagnostics[-1].get(
                "attention_entropy", 0.0
            )
            agg_diagnostics["mean_per_relation"] = layer_diagnostics[-1].get(
                "mean_per_relation", {}
            )
            agg_diagnostics["avg_active_neighbors"] = layer_diagnostics[-1].get(
                "avg_active_neighbors", 0.0
            )

        return GraphAttentionOutput(
            node_states=current_states,
            edge_attention=edge_attn_to_return,
            graph_summary=summary,
            diagnostics=agg_diagnostics,
            node_id_map=node_id_map,
            subgraph_extracted=subgraph_extracted,
        )

    def _extract_subgraph(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        subgraph_nodes: Sequence[int] | Sequence[str],
        node_ids: Sequence[str] | None,
        edge_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        list[str] | None,
    ]:
        """Extract induced subgraph given node indices or persistent IDs."""
        n = node_states.shape[0]

        # Resolve indices
        if isinstance(subgraph_nodes[0], str):
            if node_ids is None:
                raise ValidationError("subgraph_nodes has strings but node_ids was not provided")
            id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
            selected_indices = [id_to_idx[nid] for nid in subgraph_nodes if nid in id_to_idx]
        else:
            selected_indices = [int(idx) for idx in subgraph_nodes if 0 <= int(idx) < n]

        # Bounding to max_subgraph_nodes
        selected_indices = selected_indices[: self.config.max_subgraph_nodes]
        if not selected_indices:
            empty_node_states = node_states.new_zeros((0, self.config.hidden_dim))
            empty_edge_index = edge_index.new_zeros((2, 0))
            empty_edge_types = edge_types.new_zeros((0,))
            empty_edge_weights = edge_weights.new_zeros((0,))
            return (
                empty_node_states,
                empty_edge_index,
                empty_edge_types,
                empty_edge_weights,
                None,
                [],
            )

        selected_idx_tensor = torch.tensor(
            selected_indices, dtype=torch.int64, device=node_states.device
        )
        sub_nodes = node_states.index_select(0, selected_idx_tensor)

        # Mapping old index -> new index
        old_to_new = {old: new for new, old in enumerate(selected_indices)}

        # Find induced edges
        source, target = edge_index.unbind(0)
        kept_indices = []
        new_source = []
        new_target = []

        for i in range(edge_index.shape[1]):
            s_val = int(source[i].item())
            t_val = int(target[i].item())
            if s_val in old_to_new and t_val in old_to_new:
                if edge_mask is None or bool(edge_mask[i].item()):
                    kept_indices.append(i)
                    new_source.append(old_to_new[s_val])
                    new_target.append(old_to_new[t_val])

        kept_indices = kept_indices[: self.config.max_subgraph_edges]
        new_source = new_source[: self.config.max_subgraph_edges]
        new_target = new_target[: self.config.max_subgraph_edges]

        if kept_indices:
            idx_tensor = torch.tensor(kept_indices, dtype=torch.int64, device=edge_index.device)
            sub_edge_types = edge_types.index_select(0, idx_tensor)
            sub_edge_weights = edge_weights.index_select(0, idx_tensor)
            sub_edge_index = torch.tensor(
                [new_source, new_target], dtype=torch.int64, device=edge_index.device
            )
        else:
            sub_edge_types = edge_types.new_zeros((0,))
            sub_edge_weights = edge_weights.new_zeros((0,))
            sub_edge_index = edge_index.new_zeros((2, 0))

        sub_node_ids = [node_ids[idx] for idx in selected_indices] if node_ids is not None else None
        return sub_nodes, sub_edge_index, sub_edge_types, sub_edge_weights, None, sub_node_ids

    def _validate_inputs(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> None:
        validate_tensor(node_states, "node_states", torch.float32)
        validate_tensor(edge_index, "edge_index", torch.int64)
        validate_tensor(edge_types, "edge_types", torch.int64)
        validate_tensor(edge_weights, "edge_weights", torch.float32)
        if node_states.ndim != 2:
            raise ValidationError("node_states must have shape [N, H]")
        if edge_index.ndim != 2:
            raise ValidationError("edge_index must have shape [2, E]")
        if edge_types.ndim != 1 or edge_weights.ndim != 1:
            raise ValidationError("edge_types and edge_weights must be 1D tensors")
        if node_states.shape[1] != self.config.hidden_dim:
            raise ValidationError(
                f"node_states hidden_dim {node_states.shape[1]} != config {self.config.hidden_dim}"
            )
        if edge_index.shape[0] != 2:
            raise ValidationError("edge_index must have shape [2, E]")
