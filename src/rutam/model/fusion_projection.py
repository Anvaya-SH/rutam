"""Linear projection layers mapping between LM hidden states and graph representations."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import GraphFusionConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


class FusionProjections(nn.Module):
    """Linear projections for multi-head cross-attention and reasoning context fusion."""

    def __init__(self, config: GraphFusionConfig) -> None:
        super().__init__()
        if not isinstance(config, GraphFusionConfig):
            raise ValidationError("FusionProjections requires a GraphFusionConfig")
        self.config = config

        d = config.hidden_dim
        h = config.graph_dim

        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(h, d, bias=True)
        self.v_proj = nn.Linear(h, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

        self.dropout = nn.Dropout(config.projection_dropout)

        if config.use_reasoning_context:
            self.reasoning_proj = nn.Linear(h, d, bias=True)
            self.reasoning_norm = nn.LayerNorm(d)
        else:
            self.reasoning_proj = None
            self.reasoning_norm = None

    def project_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project LM hidden states [B, T, D] to query representations [B, T, D]."""
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        dim = hidden_states.shape[-1]
        if dim != self.config.hidden_dim:
            raise ValidationError(
                f"hidden_states dim {dim} != config.hidden_dim {self.config.hidden_dim}"
            )
        return self.q_proj(hidden_states)

    def project_keys(self, graph_states: torch.Tensor) -> torch.Tensor:
        """Project graph states [B, K, H] to key representations [B, K, D]."""
        validate_tensor(graph_states, "graph_states", torch.float32)
        dim = graph_states.shape[-1]
        if dim != self.config.graph_dim:
            raise ValidationError(
                f"graph_states dim {dim} != config.graph_dim {self.config.graph_dim}"
            )
        return self.k_proj(graph_states)

    def project_values(self, graph_states: torch.Tensor) -> torch.Tensor:
        """Project graph states [B, K, H] to value representations [B, K, D]."""
        validate_tensor(graph_states, "graph_states", torch.float32)
        dim = graph_states.shape[-1]
        if dim != self.config.graph_dim:
            raise ValidationError(
                f"graph_states dim {dim} != config.graph_dim {self.config.graph_dim}"
            )
        return self.v_proj(graph_states)

    def project_output(self, context: torch.Tensor) -> torch.Tensor:
        """Project multi-head context [B, T, D] to output representations [B, T, D]."""
        validate_tensor(context, "context", torch.float32)
        return self.dropout(self.out_proj(context))

    def project_reasoning(self, reasoning_context: torch.Tensor) -> torch.Tensor:
        """Project reasoning context [B, H] or [B, 1, H] to LM dimension [B, 1, D]."""
        if self.reasoning_proj is None or self.reasoning_norm is None:
            raise ValidationError("Reasoning context projection is not configured")
        validate_tensor(reasoning_context, "reasoning_context", torch.float32)
        if reasoning_context.ndim == 2:
            reasoning_context = reasoning_context.unsqueeze(1)
        if reasoning_context.ndim != 3:
            raise ValidationError("reasoning_context must have shape [B, H] or [B, 1, H]")
        dim = reasoning_context.shape[-1]
        if dim != self.config.graph_dim:
            raise ValidationError(
                f"reasoning_context dim {dim} != graph_dim {self.config.graph_dim}"
            )
        projected = self.reasoning_proj(reasoning_context)
        return self.dropout(self.reasoning_norm(projected))
