"""Complete native graph fusion integrating cross-attention, gating, and residual scaling."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import GraphFusionConfig, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor
from rutam.model.fusion_gate import RUTAMFusionGate
from rutam.model.fusion_output import FusionOutput
from rutam.model.fusion_projection import FusionProjections
from rutam.model.graph_attention import RUTAMGraphAttention


class RUTAMGraphFusion(nn.Module):
    """Integrates graph representations into language model backbone hidden states.

    Combines multi-head cross-attention over graph memory, optional path reasoning
    context projection, dynamic gating, and identity-preserving residual scaling.
    """

    def __init__(self, config: GraphFusionConfig | RUTAMConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GraphFusionConfig()
        elif isinstance(config, RUTAMConfig):
            config = config.graph_fusion
        if not isinstance(config, GraphFusionConfig):
            raise ValidationError("RUTAMGraphFusion requires a GraphFusionConfig or RUTAMConfig")
        self.config = config

        self.projections = FusionProjections(config)
        self.attention = RUTAMGraphAttention(config, projections=self.projections)
        self.gate = RUTAMFusionGate(config)

        # Residual scale initialized near zero (e.g. 0.001) for identity preservation
        self.residual_scale = nn.Parameter(
            torch.tensor(config.residual_scale_init, dtype=torch.float32)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        graph_states: torch.Tensor,
        *,
        graph_mask: torch.Tensor | None = None,
        reasoning_context: torch.Tensor | None = None,
        query_gate: torch.Tensor | None = None,
        token_attention_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> FusionOutput:
        """Fuse graph memory into LM hidden states with multi-head attention and learned gating."""
        if not self.config.enabled:
            b, t, d = hidden_states.shape
            gate = torch.zeros((b, t, 1), dtype=torch.float32, device="cpu")
            return FusionOutput(
                fused_hidden_states=hidden_states,
                fusion_gate=gate,
                attention_weights=None,
                utilization_metrics={"mean_gate": 0.0, "max_gate": 0.0, "active_fraction": 0.0},
            )

        validate_tensor(hidden_states, "hidden_states", torch.float32)
        validate_tensor(graph_states, "graph_states", torch.float32)

        b, t, d = hidden_states.shape
        _, k, h = graph_states.shape

        if d != self.config.hidden_dim:
            raise ValidationError(
                f"hidden_states dim {d} != config.hidden_dim {self.config.hidden_dim}"
            )
        if h != self.config.graph_dim:
            raise ValidationError(
                f"graph_states dim {h} != config.graph_dim {self.config.graph_dim}"
            )

        is_empty_graph = k == 0 or (graph_mask is not None and not graph_mask.any())

        # Empty graph identity preservation
        if is_empty_graph and self.config.preserve_empty_identity:
            gate_shape = (b, 1, 1) if self.config.gate_mode == "sequence" else (b, t, 1)
            gate = torch.zeros(gate_shape, dtype=torch.float32, device="cpu")
            return FusionOutput(
                fused_hidden_states=hidden_states,
                fusion_gate=gate,
                attention_weights=None,
                utilization_metrics={"mean_gate": 0.0, "max_gate": 0.0, "active_fraction": 0.0},
            )

        # 1. Multi-head cross-attention over graph memory
        attn_context, attn_weights = self.attention(
            hidden_states,
            graph_states,
            graph_mask=graph_mask,
            return_attention=return_attention,
        )

        # 2. Optional path reasoning context projection
        if reasoning_context is not None and self.config.use_reasoning_context:
            projected_reasoning = self.projections.project_reasoning(reasoning_context)
            combined_context = attn_context + projected_reasoning
        else:
            combined_context = attn_context

        # 3. Dynamic gating
        gate = self.gate(
            hidden_states,
            combined_context,
            query_gate=query_gate,
            token_attention_mask=token_attention_mask,
        )

        # Mask gate for batch items without active graph nodes
        if graph_mask is not None:
            active_batch = graph_mask.any(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            gate = gate * active_batch.float()

        # 4. Fused hidden states with residual scaling
        fused_hidden_states = hidden_states + self.residual_scale * gate * combined_context

        # 5. Metrics
        mean_gate = float(gate.mean().item())
        max_gate = float(gate.max().item())
        active_fraction = float((gate > 0.1).float().mean().item())
        metrics = {
            "mean_gate": mean_gate,
            "max_gate": max_gate,
            "active_fraction": active_fraction,
        }

        return FusionOutput(
            fused_hidden_states=fused_hidden_states,
            fusion_gate=gate,
            attention_weights=attn_weights,
            utilization_metrics=metrics,
        )
