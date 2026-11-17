"""Sparse relation-aware multi-head graph edge attention using native PyTorch operations."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from rutam.config import GraphAttentionConfig
from rutam.exceptions import ValidationError
from rutam.graph.attention_bias import RelationAttentionBias
from rutam.graph.embeddings import require_finite, validate_tensor


class SparseRelationEdgeAttention(nn.Module):
    """Multi-head sparse attention over graph edges with relation-aware keys and values.

    Computes:
        score(u, v, r) = (W_q h_v)^T (W_k h_u + R_r) / sqrt(d_head) + biases
    Normalized strictly per destination node:
        alpha_{u, v} = softmax_{u' in In(v)}(score(u', v, r))
    Aggregated message:
        m_v = sum_{u in In(v)} alpha_{u, v} (W_v h_u + V_r)
    """

    def __init__(self, config: GraphAttentionConfig) -> None:
        super().__init__()
        if not isinstance(config, GraphAttentionConfig):
            raise ValidationError("SparseRelationEdgeAttention requires a GraphAttentionConfig")
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Node projections
        self.q_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False, device="cpu", dtype=torch.float32
        )
        self.k_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False, device="cpu", dtype=torch.float32
        )
        self.v_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False, device="cpu", dtype=torch.float32
        )

        # Relation projections: key contribution and value contribution
        self.relation_k_embed = nn.Embedding(
            config.num_relation_types, self.hidden_dim, device="cpu", dtype=torch.float32
        )
        self.relation_v_embed = nn.Embedding(
            config.num_relation_types, self.hidden_dim, device="cpu", dtype=torch.float32
        )
        nn.init.normal_(self.relation_k_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.relation_v_embed.weight, mean=0.0, std=0.02)

        # Bias and conditioning
        self.bias_module = RelationAttentionBias(config)

        # Output projection and dropout
        self.out_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, device="cpu", dtype=torch.float32
        )
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        query: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Compute sparse relation-aware attention.

        Args:
            node_states: [N, H] float32 tensor of node features.
            edge_index: [2, E] int64 tensor (source, destination).
            edge_types: [E] int64 tensor of relation IDs.
            edge_weights: [E] float32 tensor of edge weights.
            query: Optional [Q] or [B, Q] float32 tensor of query vectors.
            edge_mask: Optional [E] bool tensor (True = keep, False = mask out).

        Returns:
            aggregated: [N, H] (or [B, N, H] if query is [B, Q] with B > 1).
            attention_weights: [E, num_heads] or [B, E, num_heads] or None.
            diagnostics: dictionary of attention metrics.
        """
        self._validate_inputs(node_states, edge_index, edge_types, edge_weights)
        if node_states.dim() == 3:
            batch_size, n, _ = node_states.shape
            is_batched = True
        else:
            n = node_states.shape[0]
            is_batched = query is not None and query.dim() == 2
            batch_size = query.shape[0] if is_batched else 1

        e = edge_types.shape[0]

        if n == 0:
            shape = (batch_size, 0, self.hidden_dim) if is_batched else (0, self.hidden_dim)
            return node_states.new_zeros(shape), None, {"num_nodes": 0, "num_edges": 0}

        if e == 0:
            shape = (batch_size, n, self.hidden_dim) if is_batched else (n, self.hidden_dim)
            return node_states.new_zeros(shape), None, {"num_nodes": n, "num_edges": 0}

        source, target = edge_index.unbind(0)
        has_in_edges = node_states.new_zeros(n, dtype=torch.bool)
        has_in_edges.index_fill_(0, target, True)

        if node_states.dim() == 3:
            q_nodes = self.q_proj(node_states).view(batch_size, n, self.num_heads, self.head_dim)
            k_nodes = self.k_proj(node_states).view(batch_size, n, self.num_heads, self.head_dim)
            v_nodes = self.v_proj(node_states).view(batch_size, n, self.num_heads, self.head_dim)

            q_dst = q_nodes[:, target, :, :]
            k_src = k_nodes[:, source, :, :]
            v_src = v_nodes[:, source, :, :]

            r_k = self.relation_k_embed(edge_types).view(1, e, self.num_heads, self.head_dim)
            r_v = self.relation_v_embed(edge_types).view(1, e, self.num_heads, self.head_dim)

            scores = ((q_dst * (k_src + r_k)).sum(dim=-1)) * self.scale
            bias = self.bias_module.compute_bias(edge_types, edge_weights, query)
            if bias.dim() == 2:
                bias = bias.unsqueeze(0)
            scores = scores + bias

            alpha = self._batched_sparse_softmax(scores, target, n, edge_mask)
            alpha_dropped = self.attn_dropout(alpha)
            val = v_src + r_v
            msg = (alpha_dropped.unsqueeze(-1) * val).view(batch_size, e, self.hidden_dim)
            aggregated = node_states.new_zeros((batch_size, n, self.hidden_dim))
            aggregated.index_add_(1, target, msg)
            out = self.out_proj(aggregated)
            out = torch.where(has_in_edges.view(1, n, 1), out, torch.zeros_like(out))

        elif is_batched:
            q_nodes = self.q_proj(node_states).view(n, self.num_heads, self.head_dim)
            k_nodes = self.k_proj(node_states).view(n, self.num_heads, self.head_dim)
            v_nodes = self.v_proj(node_states).view(n, self.num_heads, self.head_dim)

            q_dst = q_nodes.index_select(0, target)
            k_src = k_nodes.index_select(0, source)
            v_src = v_nodes.index_select(0, source)

            r_k = self.relation_k_embed(edge_types).view(e, self.num_heads, self.head_dim)
            r_v = self.relation_v_embed(edge_types).view(e, self.num_heads, self.head_dim)

            scores = ((q_dst * (k_src + r_k)).sum(dim=-1)) * self.scale
            bias = self.bias_module.compute_bias(edge_types, edge_weights, query)
            scores = scores.unsqueeze(0) + bias

            alpha = self._batched_sparse_softmax(scores, target, n, edge_mask)
            alpha_dropped = self.attn_dropout(alpha)
            val = (v_src + r_v).unsqueeze(0)
            msg = (alpha_dropped.unsqueeze(-1) * val).view(batch_size, e, self.hidden_dim)
            aggregated = node_states.new_zeros((batch_size, n, self.hidden_dim))
            aggregated.index_add_(1, target, msg)
            out = self.out_proj(aggregated)
            out = torch.where(has_in_edges.view(1, n, 1), out, torch.zeros_like(out))

        else:
            q_nodes = self.q_proj(node_states).view(n, self.num_heads, self.head_dim)
            k_nodes = self.k_proj(node_states).view(n, self.num_heads, self.head_dim)
            v_nodes = self.v_proj(node_states).view(n, self.num_heads, self.head_dim)

            q_dst = q_nodes.index_select(0, target)
            k_src = k_nodes.index_select(0, source)
            v_src = v_nodes.index_select(0, source)

            r_k = self.relation_k_embed(edge_types).view(e, self.num_heads, self.head_dim)
            r_v = self.relation_v_embed(edge_types).view(e, self.num_heads, self.head_dim)

            scores = ((q_dst * (k_src + r_k)).sum(dim=-1)) * self.scale
            bias = self.bias_module.compute_bias(edge_types, edge_weights, query)
            scores = scores + bias

            alpha = self._unbatched_sparse_softmax(scores, target, n, edge_mask)
            alpha_dropped = self.attn_dropout(alpha)
            val = v_src + r_v
            msg = (alpha_dropped.unsqueeze(-1) * val).view(e, self.hidden_dim)
            aggregated = node_states.new_zeros((n, self.hidden_dim))
            aggregated.index_add_(0, target, msg)
            out = self.out_proj(aggregated)
            out = torch.where(has_in_edges.unsqueeze(-1), out, torch.zeros_like(out))

        require_finite(out, "edge attention output")
        diagnostics = self._compute_diagnostics(alpha, edge_types, edge_mask, n, e, is_batched)
        return out, alpha, diagnostics

    def _unbatched_sparse_softmax(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        num_nodes: int,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute stable per-destination softmax for unbatched [E, heads] scores."""
        # Find per-destination maximum score
        max_score = torch.full(
            (num_nodes, self.num_heads), -1e9, dtype=scores.dtype, device=scores.device
        )
        target_expanded = target.unsqueeze(-1).expand(-1, self.num_heads)
        max_score.scatter_reduce_(0, target_expanded, scores, reduce="amax", include_self=False)

        # Exponentials normalized by max
        exp_scores = torch.exp(scores - max_score.index_select(0, target))
        if edge_mask is not None:
            exp_scores = torch.where(
                edge_mask.unsqueeze(-1), exp_scores, torch.zeros_like(exp_scores)
            )

        sum_exp = torch.zeros((num_nodes, self.num_heads), dtype=scores.dtype, device=scores.device)
        sum_exp.index_add_(0, target, exp_scores)
        denom = sum_exp.index_select(0, target).clamp_min(1e-9)
        alpha = exp_scores / denom
        return alpha

    def _batched_sparse_softmax(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        num_nodes: int,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute stable per-destination softmax for batched [B, E, heads] scores."""
        b, e, _ = scores.shape
        max_score = torch.full(
            (b, num_nodes, self.num_heads), -1e9, dtype=scores.dtype, device=scores.device
        )
        target_expanded = target.view(1, e, 1).expand(b, -1, self.num_heads)
        max_score.scatter_reduce_(1, target_expanded, scores, reduce="amax", include_self=False)

        exp_scores = torch.exp(scores - max_score[:, target, :])
        if edge_mask is not None:
            mask_exp = edge_mask.view(1, e, 1).expand(b, -1, self.num_heads)
            exp_scores = torch.where(mask_exp, exp_scores, torch.zeros_like(exp_scores))

        sum_exp = torch.zeros(
            (b, num_nodes, self.num_heads), dtype=scores.dtype, device=scores.device
        )
        sum_exp.index_add_(1, target, exp_scores)
        denom = sum_exp[:, target, :].clamp_min(1e-9)
        alpha = exp_scores / denom
        return alpha

    def _compute_diagnostics(
        self,
        alpha: torch.Tensor,
        edge_types: torch.Tensor,
        edge_mask: torch.Tensor | None,
        num_nodes: int,
        num_edges: int,
        is_batched: bool,
    ) -> dict[str, Any]:
        """Calculate attention diagnostics without storing sensitive memory content."""
        with torch.no_grad():
            flat_alpha = alpha.mean(dim=0) if is_batched else alpha  # [E, heads]
            mean_heads = flat_alpha.mean(dim=-1)  # [E]

            # Attention entropy: - sum alpha * log(alpha + 1e-9)
            entropy = float(
                (-(flat_alpha * torch.log(flat_alpha.clamp_min(1e-9))).sum(dim=-1)).mean().item()
            )

            # Mean attention per relation category
            mean_per_relation: dict[int, float] = {}
            for r_id in torch.unique(edge_types):
                r_int = int(r_id.item())
                r_mask = edge_types == r_id
                mean_per_relation[r_int] = float(mean_heads[r_mask].mean().item())

            # Average active neighbors (alpha > 0.01)
            active_edges = float((mean_heads > 0.01).sum().item())
            avg_active_neighbors = active_edges / max(1, num_nodes)

            # Masked edges percentage
            if edge_mask is not None:
                masked_pct = float((~edge_mask).float().mean().item() * 100.0)
            else:
                masked_pct = 0.0

            return {
                "attention_entropy": entropy,
                "mean_per_relation": mean_per_relation,
                "avg_active_neighbors": avg_active_neighbors,
                "masked_edge_pct": masked_pct,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
            }

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
        if node_states.ndim not in (2, 3):
            raise ValidationError("node_states must have shape [N, H] or [B, N, H]")
        if edge_index.ndim != 2:
            raise ValidationError("edge_index must have shape [2, E]")
        if edge_types.ndim != 1 or edge_weights.ndim != 1:
            raise ValidationError("edge_types and edge_weights must be 1D tensors")
        if node_states.shape[-1] != self.hidden_dim:
            raise ValidationError(
                f"node_states dim {node_states.shape[-1]} != config {self.hidden_dim}"
            )
        if edge_index.shape[0] != 2:
            raise ValidationError("edge_index must have shape [2, E]")
        e = edge_index.shape[1]
        if edge_types.shape[0] != e or edge_weights.shape[0] != e:
            raise ValidationError(
                "edge_index, edge_types, and edge_weights must have matching length E"
            )
        num_nodes = node_states.shape[1] if node_states.dim() == 3 else node_states.shape[0]
        if e > 0 and num_nodes > 0:
            if edge_index.max() >= num_nodes or edge_index.min() < 0:
                raise ValidationError("edge_index references invalid node indices")
