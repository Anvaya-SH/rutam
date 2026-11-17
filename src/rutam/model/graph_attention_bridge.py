"""Bridge connecting graph attention contextualization, neural queries, and graph selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.attention import RUTAMGraphAttention
from rutam.graph.attention_output import GraphAttentionOutput
from rutam.graph.selector import GraphSelectorOutput, RUTAMGraphSelector
from rutam.graph.state import GraphState

if TYPE_CHECKING:
    from rutam.model.query_output import QueryHeadOutput


class GraphAttentionBridge(nn.Module):
    """Coordinates relation-aware graph attention and neural selection.

    Pipeline:
        1. Graph Nodes + Edges -> Relation-Aware Attention -> Contextualized States
        2. LM Query + Contextualized States -> Selection -> Selected States
    """

    def __init__(
        self,
        config: RUTAMConfig | None = None,
        *,
        graph_attention: RUTAMGraphAttention | None = None,
        selector: RUTAMGraphSelector | None = None,
    ) -> None:
        super().__init__()
        self.config = RUTAMConfig() if config is None else config
        self.graph_attention = (
            RUTAMGraphAttention(self.config.graph_attention)
            if graph_attention is None
            else graph_attention
        )
        self.selector = RUTAMGraphSelector(self.config.selector) if selector is None else selector

    def forward(
        self,
        graph_state: GraphState,
        query_input: torch.Tensor | QueryHeadOutput,
        *,
        node_states: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[GraphAttentionOutput, GraphSelectorOutput]:
        """Contextualize graph state with relation-aware attention and select relevant memories.

        Args:
            graph_state: Valid GraphState instance.
            query_input: Neural query tensor [B, Q] or QueryHeadOutput.
            node_states: Optional pre-projected states [N, H]. If None, uses graph embeddings.
            return_attention: Whether to retain attention coefficients.

        Returns:
            Tuple of (GraphAttentionOutput, GraphSelectorOutput).
        """
        if not isinstance(graph_state, GraphState):
            raise ValidationError("graph_state must be a GraphState instance")

        query_tensor = (
            query_input.query_vectors if hasattr(query_input, "query_vectors") else query_input
        )
        if not isinstance(query_tensor, torch.Tensor):
            raise ValidationError("query_input must be a Tensor or QueryHeadOutput")

        h_nodes = node_states if node_states is not None else graph_state.node_embeddings

        edge_index = graph_state.edge_index
        edge_types = graph_state.edge_types
        edge_weights = graph_state.edge_weights

        # 1. Relation-aware graph attention
        attn_output = self.graph_attention(
            h_nodes,
            edge_index,
            edge_types,
            edge_weights,
            query=query_tensor if self.config.graph_attention.query_conditioning else None,
            node_ids=graph_state.node_ids,
            pool_graph=True,
        )

        # 2. Query-conditioned memory selection using contextualized graph states
        # When batched query-conditioning is enabled, attn_output.node_states is [B, N, H] or [N, H]
        contextualized_nodes = attn_output.node_states
        if contextualized_nodes.dim() == 3:
            # Average across batch or select per-batch
            selection_nodes = contextualized_nodes.mean(dim=0)
        else:
            selection_nodes = contextualized_nodes

        node_id_tensor = torch.arange(
            len(graph_state.node_ids), dtype=torch.int64, device=selection_nodes.device
        )

        selector_output = self.selector(
            query_tensor,
            selection_nodes,
            node_id_tensor,
            return_attention=return_attention,
        )

        return attn_output, selector_output
