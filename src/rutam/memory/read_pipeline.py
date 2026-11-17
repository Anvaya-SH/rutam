"""Differentiable neural read pipeline: Query -> Selection -> Traversal -> Memory Context."""

from __future__ import annotations

from datetime import UTC, datetime

import torch
from torch import nn

from rutam.exceptions import ValidationError
from rutam.graph.encoder import GraphEncoderOutput, RUTAMGraphEncoder
from rutam.graph.operations import Graph
from rutam.graph.selector import RUTAMGraphSelector
from rutam.graph.state import GraphState
from rutam.graph.traversal import RUTAMGraphTraversal
from rutam.memory.controller_output import MemoryReadOutput
from rutam.memory.provenance import AuthorizationContext
from rutam.model.query_head import RUTAMNeuralQueryHead


class MemoryReadPipeline(nn.Module):
    """Executes a differentiable neural read over the active graph state."""

    def __init__(
        self,
        query_head: RUTAMNeuralQueryHead,
        selector: RUTAMGraphSelector,
        encoder: RUTAMGraphEncoder,
        traversal: RUTAMGraphTraversal | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(query_head, RUTAMNeuralQueryHead):
            raise ValidationError("query_head must be a RUTAMNeuralQueryHead")
        if not isinstance(selector, RUTAMGraphSelector):
            raise ValidationError("selector must be a RUTAMGraphSelector")
        if not isinstance(encoder, RUTAMGraphEncoder):
            raise ValidationError("encoder must be a RUTAMGraphEncoder")
        self.query_head = query_head
        self.selector = selector
        self.encoder = encoder
        self.traversal = traversal

    def forward(
        self,
        hidden_states: torch.Tensor,
        graph_snapshot: Graph | GraphState,
        *,
        attention_mask: torch.Tensor | None = None,
        memory_available: torch.Tensor | None = None,
        encoded_cache: GraphEncoderOutput | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> MemoryReadOutput:
        state = graph_snapshot.to_state() if isinstance(graph_snapshot, Graph) else graph_snapshot
        if not isinstance(state, GraphState):
            raise ValidationError("graph_snapshot must be a Graph or GraphState")

        if authorization_context is not None and not isinstance(
            authorization_context, AuthorizationContext
        ):
            raise ValidationError("authorization_context must be an AuthorizationContext")
        auth = authorization_context or AuthorizationContext()
        now = datetime.now(UTC)

        def currently_valid(record: dict[str, object]) -> bool:
            try:
                valid_from = record.get("valid_from")
                valid_until = record.get("valid_until")
                return (
                    (valid_from is None or datetime.fromisoformat(str(valid_from)) <= now)
                    and (valid_until is None or datetime.fromisoformat(str(valid_until)) > now)
                )
            except (TypeError, ValueError):
                # Malformed validity metadata must fail closed at the read boundary.
                return False

        can_read = "read" in auth.permissions
        visible_nodes = torch.tensor(
            [
                can_read
                and record.get("scope", "default") in auth.authorized_scopes
                and "read" in record.get("permissions", ("read", "write"))
                and record.get("verification_status", "unverified")
                not in ("retracted", "superseded", "expired")
                and currently_valid(record)
                for record in state.node_metadata
            ],
            dtype=torch.bool,
        )
        visible_edges = []
        for edge_row, record in enumerate(state.edge_metadata):
            source_row, target_row = state.edge_index[:, edge_row].tolist()
            visible_edges.append(
                record.get("scope", "default") in auth.authorized_scopes
                and record.get("status", "active")
                not in ("retracted", "superseded", "inactive", "expired")
                and currently_valid(record)
                and bool(visible_nodes[source_row])
                and bool(visible_nodes[target_row])
            )
        # Keep node row offsets stable for selection while removing unauthorized
        # edges before either message passing or neighborhood traversal.
        state, _ = state.extract_subgraph(
            torch.arange(state.node_ids.shape[0], dtype=torch.int64),
            edge_mask=torch.tensor(visible_edges, dtype=torch.bool),
        )

        # A cached encoding carries no authorization fingerprint, so it cannot
        # safely be reused for a scoped read.
        encoded = self.encoder(state, node_mask=visible_nodes)

        # 2. Query and graph selection
        selection_out = self.query_head.select_graph(
            hidden_states=hidden_states,
            selector=self.selector,
            node_states=encoded,
            node_ids=state.node_ids,
            attention_mask=attention_mask,
            graph_summary=encoded.graph_summary,
            memory_available=memory_available,
            node_mask=visible_nodes,
        )

        query = selection_out.query
        selection = selection_out.selection
        gated_ctx = selection_out.gated_memory_context
        traversal_out = None

        # 3. Optional multi-hop traversal reasoning
        if (
            self.traversal is not None
            and state.edge_index.shape[1] > 0
            and selection.selection_mask.any()
        ):
            traversal_out = self.traversal.from_state(
                query_vectors=query.query_vectors,
                encoded=encoded,
                state=state,
                seed_indices=selection.selected_indices,
                seed_mask=selection.selection_mask,
                node_mask=visible_nodes,
            )
            raw_context = traversal_out.reasoning_context
            gated_ctx = raw_context * query.query_gate
        else:
            raw_context = (
                selection.memory_context
                if selection.memory_context is not None
                else selection.selected_states[:, 0, :]
                if selection.selected_states.shape[1] > 0
                else torch.zeros(
                    (hidden_states.shape[0], encoded.node_states.shape[1]),
                    dtype=torch.float32,
                    device="cpu",
                )
            )
            if gated_ctx is None:
                gated_ctx = raw_context * query.query_gate

        return MemoryReadOutput(
            memory_context=raw_context,
            selected_node_ids=selection.selected_node_ids,
            selected_node_states=selection.selected_states,
            selected_scores=selection.selected_scores,
            selection_mask=selection.selection_mask,
            query_output=query,
            traversal_output=traversal_out,
            graph_version=state.graph_version,
            gated_context=gated_ctx,
        )
