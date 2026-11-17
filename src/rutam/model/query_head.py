"""Hidden-state-to-query projection and minimal native graph-read composition."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import QueryHeadConfig, SelectorConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor
from rutam.graph.encoder import GraphEncoderOutput
from rutam.graph.selector import RUTAMGraphSelector
from rutam.model.query_gating import QueryDecisionGate, availability_mask
from rutam.model.query_output import QueryHeadOutput, QuerySelectionOutput
from rutam.model.query_pooling import QueryPooling


def query_norm_loss(
    query_vectors: torch.Tensor,
    *,
    weight: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optional unit-norm penalty, not a raw L2 penalty that favors zero queries.

    This constrains magnitude only; it does not guarantee diverse or useful queries.
    Retrieval supervision remains necessary to learn input-dependent relevance.
    """
    validate_tensor(query_vectors, "query_vectors", torch.float32)
    if query_vectors.ndim != 2 or query_vectors.shape[1] == 0:
        raise ValidationError("query_vectors must have shape [B,Q] with Q>0")
    require_finite(query_vectors, "query_vectors")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not (0 <= weight <= 1.7976931348623157e308)
    ):
        raise ValidationError("Regularization weight must be finite and nonnegative")
    active = availability_mask(mask, query_vectors.shape[0])
    if weight == 0 or not active.any().item():
        return (query_vectors * 0).sum()
    squared_norm = query_vectors[active].double().square().sum(dim=1)
    loss = ((squared_norm - 1).square().mean() * weight).float()
    require_finite(loss, "query norm loss")
    return loss


class RUTAMNeuralQueryHead(nn.Module):
    """CPU float32 neural query head; no assumptions about LFM3 layer APIs/sizes.

    Pass selector_config for construction-time compatibility validation. Graph
    composition always verifies the actual selector at call time. Standalone
    forward assumes memory is available when no availability mask is provided;
    a zero graph summary is not evidence that a graph is empty.
    """

    def __init__(
        self,
        config: QueryHeadConfig | None = None,
        *,
        selector_config: SelectorConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = QueryHeadConfig() if config is None else config
        if not isinstance(self.config, QueryHeadConfig):
            raise ValidationError("RUTAMNeuralQueryHead requires a QueryHeadConfig")
        if selector_config is not None:
            self.config.validate_selector(selector_config)
        c = self.config
        self.pooling = QueryPooling(c)
        self.context_norm = nn.LayerNorm(c.backbone_hidden_dim, device="cpu", dtype=torch.float32)
        self.input_projection = nn.Linear(
            c.backbone_hidden_dim, c.intermediate_dim, device="cpu", dtype=torch.float32
        )
        self.output_projection = nn.Linear(
            c.intermediate_dim, c.query_dim, device="cpu", dtype=torch.float32
        )
        self.dropout = nn.Dropout(c.dropout)
        self.residual_projection: nn.Module | None = None
        if c.residual:
            self.residual_projection = (
                nn.Linear(
                    c.backbone_hidden_dim,
                    c.query_dim,
                    bias=False,
                    device="cpu",
                    dtype=torch.float32,
                )
                if c.residual_projection
                else nn.Identity()
            )
        self.graph_projection = (
            nn.Linear(
                c.graph_hidden_dim, c.query_dim, bias=False, device="cpu", dtype=torch.float32
            )
            if c.graph_conditioning
            else None
        )
        self.conditioning_gate = (
            nn.Linear(2 * c.query_dim, 1, device="cpu", dtype=torch.float32)
            if c.graph_conditioning and c.conditioning_mode == "gated_add"
            else None
        )
        self.decision_gate = QueryDecisionGate(c)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        graph_summary: torch.Tensor | None = None,
        memory_available: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> QueryHeadOutput:
        pooled = self.pooling(hidden_states, attention_mask, return_attention=return_attention)
        available = (
            availability_mask(memory_available, hidden_states.shape[0]) & pooled.valid_context
        )
        summary = self._graph_summary(graph_summary, hidden_states.shape[0])
        validate_tensor(self.input_projection.weight, "query head parameters", torch.float32)
        normalized = self.context_norm(pooled.context)
        require_finite(normalized, "normalized query context")
        intermediate = self.input_projection(normalized)
        require_finite(intermediate, "query intermediate projection")
        query = self.output_projection(self.dropout(F.gelu(intermediate)))
        if self.residual_projection is not None:
            query = query + self.residual_projection(pooled.context)
        require_finite(query, "projected query")
        if self.graph_projection is not None and summary is not None:
            graph = self.graph_projection(torch.where(available[:, None], summary, 0))
            require_finite(graph, "projected graph summary")
            scale = 1
            if self.conditioning_gate is not None:
                logits = self.conditioning_gate(torch.cat((query, graph), dim=1))
                require_finite(logits, "graph conditioning logits")
                scale = torch.sigmoid(logits)
            query = query + scale * graph
        require_finite(query, "conditioned query")
        if self.config.normalize_queries:
            query = F.normalize(query.double(), dim=1, eps=1e-12).float()
        query = torch.where(pooled.valid_context[:, None], query, 0)
        gate = self.decision_gate(pooled.context, available)
        return QueryHeadOutput(
            query,
            gate.gate,
            gate.logits,
            gate.should_query,
            pooled.attention_weights,
            pooled.valid_context,
        )

    def _graph_summary(self, summary: torch.Tensor | None, b: int) -> torch.Tensor | None:
        if summary is None:
            return None
        validate_tensor(summary, "graph_summary", torch.float32)
        if summary.shape == (self.config.graph_hidden_dim,):
            summary = summary.unsqueeze(0).expand(b, -1)
        if summary.shape != (b, self.config.graph_hidden_dim):
            raise ValidationError("graph_summary must have shape [H] or [B,H]")
        require_finite(summary, "graph_summary")
        return summary

    def regularization_loss(self, output: QueryHeadOutput) -> torch.Tensor:
        return query_norm_loss(
            output.query_vectors,
            weight=self.config.norm_regularization_weight,
            mask=output.valid_context,
        )

    def select_graph(
        self,
        hidden_states: torch.Tensor,
        selector: RUTAMGraphSelector,
        node_states: torch.Tensor | GraphEncoderOutput,
        node_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        graph_summary: torch.Tensor | None = None,
        memory_available: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        eligibility_mask: torch.Tensor | None = None,
        return_token_attention: bool | None = None,
        return_graph_attention: bool | None = None,
        candidate_projection: nn.Module | None = None,
    ) -> QuerySelectionOutput:
        """Compose existing APIs; training uses continuous gating after a soft read.

        In eval mode should_query additionally masks selector eligibility. This
        minimal adapter performs logical masking, not sparse batch routing: it
        may still score masked rows and allocates a [B,N] boolean eligibility mask.
        """
        if not isinstance(selector, RUTAMGraphSelector):
            raise ValidationError("selector must be a RUTAMGraphSelector")
        self.config.validate_selector(selector.config)
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        if hidden_states.ndim != 3:
            raise ValidationError("hidden_states must have shape [B,T,D]")
        nodes = (
            node_states.node_states if isinstance(node_states, GraphEncoderOutput) else node_states
        )
        if candidate_projection is None:
            validate_features(nodes, selector.config.hidden_dim, "node_states")
        else:
            projection_config = getattr(candidate_projection, "config", None)
            input_dim = getattr(projection_config, "input_dim", None)
            output_dim = getattr(projection_config, "hidden_dim", None)
            if input_dim is None or output_dim != selector.config.hidden_dim:
                raise ValidationError(
                    "candidate_projection must expose input_dim and selector-compatible hidden_dim"
                )
            validate_features(nodes, input_dim, "candidate node features")
        b, n = hidden_states.shape[0], nodes.shape[0]
        eligible = torch.ones((b, n), dtype=torch.bool, device="cpu")
        for mask, name, shape in (
            (node_mask, "node_mask", (n,)),
            (eligibility_mask, "eligibility_mask", (b, n)),
        ):
            if mask is not None:
                validate_tensor(mask, name, torch.bool)
                if mask.shape != shape:
                    raise ValidationError(f"{name} must have shape {shape}")
                eligible = eligible & mask
        available = eligible.any(dim=1) & availability_mask(memory_available, b)
        if graph_summary is None and isinstance(node_states, GraphEncoderOutput):
            graph_summary = node_states.graph_summary
        query = self(
            hidden_states,
            attention_mask,
            graph_summary=graph_summary,
            memory_available=available,
            return_attention=return_token_attention,
        )
        allowed = available & query.valid_context
        if query.should_query is not None:
            allowed = allowed & query.should_query
        if candidate_projection is None:
            selection = selector(
                query.query_vectors,
                node_states,
                node_ids,
                eligibility_mask=eligible & allowed[:, None],
                return_attention=return_graph_attention,
            )
        else:
            if return_graph_attention:
                raise ValidationError("Sparse candidate selection does not expose dense graph attention")
            selection = selector.select_projected(
                query.query_vectors,
                nodes,
                node_ids,
                candidate_projection,
                node_mask=torch.ones(n, dtype=torch.bool) if node_mask is None else node_mask,
                active=allowed,
                eligibility_mask=eligible & allowed[:, None],
            )
        context = selection.memory_context
        if context is not None:
            context = context * query.query_gate
        return QuerySelectionOutput(query, selection, context)
