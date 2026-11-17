"""RUTAM's trainable graph encoder, independent of graph persistence/model fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rutam.config import EncoderConfig, GraphAttentionConfig, boolean
from rutam.exceptions import ValidationError
from rutam.graph.attention import RUTAMGraphAttention
from rutam.graph.embeddings import NodeProjection, RelationEmbedding, require_finite
from rutam.graph.layers import RelationAwareMessagePassing
from rutam.graph.state import GraphState


@dataclass(frozen=True, slots=True)
class GraphEncoderOutput:
    """Rows align with input node_ids; summary is [H], or None if pooling is disabled."""

    node_states: torch.Tensor
    graph_summary: torch.Tensor | None
    has_nodes: bool


class RUTAMGraphEncoder(nn.Module):
    """CPU float32 projection, shared relation vocabulary and L message-passing layers.

    forward reads a GraphState without modifying it, touching disk or detaching
    its feature/weight tensors. Parameters belong in a separate neural checkpoint,
    never graph.safetensors. This module contains no text encoder or LFM3 hooks.
    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
        *,
        graph_attention_config: GraphAttentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = EncoderConfig() if config is None else config
        if not isinstance(self.config, EncoderConfig):
            raise ValidationError("RUTAMGraphEncoder requires an EncoderConfig")
        self.node_projection = NodeProjection(self.config)
        self.relation_embeddings = RelationEmbedding(
            self.config.num_relation_types, self.config.hidden_dim
        )
        self.layers = nn.ModuleList(
            RelationAwareMessagePassing(self.config) for _ in range(self.config.num_layers)
        )
        if self.config.encoder_mode in ("graph_attention", "hybrid"):
            attn_cfg = (
                GraphAttentionConfig(
                    hidden_dim=self.config.hidden_dim,
                    num_relation_types=self.config.num_relation_types,
                    dropout=self.config.dropout,
                )
                if graph_attention_config is None
                else graph_attention_config
            )
            self.graph_attention = RUTAMGraphAttention(attn_cfg)
        else:
            self.graph_attention = None

    def forward(
        self,
        state: GraphState,
        *,
        pool_graph: bool = True,
        node_mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> GraphEncoderOutput:
        """Encode one graph or a visible node subset.

        ``node_mask`` and ``edge_mask`` are request-level visibility boundaries.
        Hidden nodes are excluded from message passing, including edges into or
        out of them; false edge-mask entries are never passed to message passing
        or graph attention. Callers with different visibility masks must encode
        each view separately. Empty visible subsets yield zero states and summary.
        """
        if not isinstance(state, GraphState):
            raise ValidationError("The encoder input must be a GraphState")
        boolean(pool_graph, "pool_graph")
        state.validate()
        if node_mask is not None and (
            not isinstance(node_mask, torch.Tensor)
            or node_mask.dtype != torch.bool
            or node_mask.device.type != "cpu"
            or node_mask.shape != state.node_ids.shape
        ):
            raise ValidationError("node_mask must be a CPU bool tensor with shape [N]")
        if edge_mask is not None and (
            not isinstance(edge_mask, torch.Tensor)
            or edge_mask.dtype != torch.bool
            or edge_mask.device.type != "cpu"
            or edge_mask.shape != state.edge_types.shape
        ):
            raise ValidationError("edge_mask must be a CPU bool tensor with shape [E]")
        if state.node_embeddings.shape[1] != self.config.input_dim:
            raise ValidationError("Graph embedding dimension does not match encoder.input_dim")
        if any(
            value >= self.config.num_relation_types for value in state.relationship_types.values()
        ):
            raise ValidationError(
                "Graph relationship vocabulary exceeds encoder.num_relation_types"
            )
        node_states = self.node_projection(state.node_embeddings)
        visible = (
            torch.ones(state.node_ids.shape, dtype=torch.bool)
            if node_mask is None
            else node_mask
        )
        node_states = torch.where(visible[:, None], node_states, 0)
        has_nodes = bool(visible.any().item())
        if has_nodes:
            edge_index, edge_types, edge_weights = self._message_edges(state, edge_mask=edge_mask)
            if node_mask is not None and edge_index.shape[1] > 0:
                edge_visible = visible[edge_index[0]] & visible[edge_index[1]]
                edge_index = edge_index[:, edge_visible]
                edge_types = edge_types[edge_visible]
                edge_weights = edge_weights[edge_visible]
            if self.config.encoder_mode in ("message_passing", "hybrid"):
                relations = self.relation_embeddings(
                    torch.arange(self.config.num_relation_types, dtype=torch.int64, device="cpu")
                )
                for layer in self.layers:
                    node_states = layer(
                        node_states, edge_index, edge_types, edge_weights, relations
                    )
                    node_states = torch.where(visible[:, None], node_states, 0)
            if (
                self.config.encoder_mode in ("graph_attention", "hybrid")
                and self.graph_attention is not None
            ):
                attn_out = self.graph_attention(
                    node_states, edge_index, edge_types, edge_weights, pool_graph=False
                )
                node_states = torch.where(visible[:, None], attn_out.node_states, 0)
        summary = None
        if pool_graph:
            # Empty sum is exactly zero, remains differentiable, and avoids mean([]).
            summary = (
                node_states.sum(dim=0)
                if not has_nodes
                else node_states.sum(dim=0) / visible.sum().clamp_min(1)
            )
            require_finite(summary, "graph summary")
        return GraphEncoderOutput(node_states, summary, has_nodes)

    @staticmethod
    def _message_edges(
        state: GraphState, *, edge_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index, edge_types, edge_weights = (
            state.edge_index,
            state.edge_types,
            state.edge_weights,
        )
        if edge_mask is not None:
            edge_index = edge_index[:, edge_mask]
            edge_types = edge_types[edge_mask]
            edge_weights = edge_weights[edge_mask]
        if state.config.graph.directed:
            return edge_index, edge_types, edge_weights
        # Stored undirected edges occur once; self-loops must not be doubled.
        non_self = edge_index[0] != edge_index[1]
        return (
            torch.cat((edge_index, edge_index.flip(0)[:, non_self]), dim=1),
            torch.cat((edge_types, edge_types[non_self])),
            torch.cat((edge_weights, edge_weights[non_self])),
        )
