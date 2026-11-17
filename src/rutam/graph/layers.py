"""Relation-aware sparse message passing using native differentiable PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import EncoderConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import (
    activation_module,
    require_finite,
    validate_features,
    validate_relation_ids,
    validate_tensor,
)


class RelationAwareMessagePassing(nn.Module):
    """Send weighted messages from source rows to destination rows.

    m_e = activation(W_s h_source + W_r relation_type)
    a_i = sum_{e: target=i} w_e m_e / max(1, in_degree(i))  [mean]
    u_i = dropout(activation(W_self h_i + b_self + a_i))
    h'_i = optional_LayerNorm(optional_residual(h_i) + u_i)

    Sum aggregation omits the degree division. Degrees count edges, including
    zero-weight edges; normalization does not cancel single-edge weight changes.
    Node and relation transforms are computed before edge gathering: no N x N
    tensor or per-edge H x H transform is constructed.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, EncoderConfig):
            raise ValidationError("RelationAwareMessagePassing requires an EncoderConfig")
        self.config = config
        self.source_linear = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False, device="cpu", dtype=torch.float32
        )
        self.relation_linear = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False, device="cpu", dtype=torch.float32
        )
        self.self_linear = nn.Linear(
            config.hidden_dim, config.hidden_dim, device="cpu", dtype=torch.float32
        )
        self.activation = activation_module(config.activation)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = (
            nn.LayerNorm(config.hidden_dim, device="cpu", dtype=torch.float32)
            if config.layer_norm
            else nn.Identity()
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> torch.Tensor:
        """Consume [N,H], [2,E], [E], [E], [R,H]; return [N,H]."""
        self._validate_inputs(node_states, edge_index, edge_types, edge_weights, relation_states)
        n = node_states.shape[0]
        source, target = edge_index.unbind(0)
        aggregated = node_states.new_zeros((n, self.config.hidden_dim))
        if source.numel():
            source_values = self.source_linear(node_states)
            relation_values = self.relation_linear(relation_states)
            messages = self.activation(
                source_values.index_select(0, source) + relation_values.index_select(0, edge_types)
            )
            require_finite(messages, "edge messages")
            factors = edge_weights
            if self.config.aggregation == "mean":
                # Divide before accumulating to reduce high-degree overflow risk.
                degree = node_states.new_zeros(n)
                degree.index_add_(0, target, torch.ones_like(edge_weights))
                factors = factors / degree.index_select(0, target).clamp_min(1)
            aggregated.index_add_(0, target, messages * factors.unsqueeze(-1))
        require_finite(aggregated, "aggregated messages")
        update = self.self_linear(node_states) + aggregated
        require_finite(update, "node update")
        result = self.dropout(self.activation(update))
        if self.config.residual:
            result = result + node_states
        require_finite(result, "residual update")
        result = self.norm(result)
        require_finite(result, "message-passing output")
        return result

    def _validate_inputs(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> None:
        validate_features(node_states, self.config.hidden_dim, "node_states")
        validate_features(relation_states, self.config.hidden_dim, "relation_states")
        if relation_states.shape[0] != self.config.num_relation_types:
            raise ValidationError("relation_states must contain the configured vocabulary")
        validate_relation_ids(edge_types, self.config.num_relation_types)
        validate_tensor(edge_index, "edge_index", torch.int64)
        validate_tensor(edge_weights, "edge_weights", torch.float32)
        e = edge_types.numel()
        if edge_index.shape != (2, e) or edge_weights.shape != (e,):
            raise ValidationError("edge_index and edge_weights must have shapes [2, E] and [E]")
        if e and (edge_index.min().item() < 0 or edge_index.max().item() >= node_states.shape[0]):
            raise ValidationError("edge_index contains out-of-range node rows")
        require_finite(edge_weights, "edge_weights")
        if (edge_weights < 0).any().item():
            raise ValidationError("edge_weights must be nonnegative")
        validate_tensor(self.self_linear.weight, "message-passing parameters", torch.float32)
