"""Synthetic testing double and layer integration mechanisms for LFM3 adaptation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from rutam.config import positive_int
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor
from rutam.model.backbone_protocol import BackboneProtocol
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.fusion_output import FusionOutput


class SyntheticDecoderLayer(nn.Module):
    """Lightweight test double representing an individual transformer/hybrid decoder layer."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        positive_int(hidden_dim, "hidden_dim")
        self.hidden_dim = hidden_dim
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
        )
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Self-attention block with residual
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed)
        hidden_states = hidden_states + attn_out

        # MLP block with residual
        normed_post = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_post)
        return hidden_states + mlp_out


class SyntheticBackbone(nn.Module):
    """Tiny CPU test double implementing BackboneProtocol for testing without 1.7B weights."""

    def __init__(
        self,
        hidden_dim: int = 16,
        vocab_size: int = 100,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        positive_int(hidden_dim, "hidden_dim")
        positive_int(vocab_size, "vocab_size")
        positive_int(num_layers, "num_layers")

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([SyntheticDecoderLayer(hidden_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_num_layers(self) -> int:
        return self.num_layers

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_layers(self) -> Sequence[nn.Module]:
        return tuple(self.layers)

    def get_output_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        return self.lm_head(self.final_norm(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long:
            raise ValidationError("input_ids must be a torch.long tensor")

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, **kwargs)
        normed = self.final_norm(x)
        logits = self.lm_head(normed)
        return {"last_hidden_state": normed, "logits": logits}


assert issubclass(SyntheticBackbone, BackboneProtocol)


class LayerIntegrationHook:
    """Manages reversible registration and cleanup of forward hooks on backbone layers."""

    def __init__(
        self,
        layer: nn.Module,
        hook_fn: Callable[..., Any],
    ) -> None:
        self.layer = layer
        self.handle = layer.register_forward_hook(hook_fn)

    def remove(self) -> None:
        """Remove forward hook cleanly."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class IntegratedLayerWrapper(nn.Module):
    """Explicitly wraps a backbone decoder layer with optional post-layer graph fusion."""

    def __init__(
        self,
        layer: nn.Module,
        fusion: RUTAMGraphFusion | None = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        # Keep fusion as a non-submodule reference to preserve backbone parameter boundaries
        object.__setattr__(self, "fusion", fusion)
        self.last_fusion_output: FusionOutput | None = None
        self._active_inputs: dict[str, Any] = {}

    def set_active_fusion_inputs(
        self,
        graph_states: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        reasoning_context: torch.Tensor | None = None,
        query_gate: torch.Tensor | None = None,
    ) -> None:
        self._active_inputs = {
            "graph_states": graph_states,
            "graph_mask": graph_mask,
            "reasoning_context": reasoning_context,
            "query_gate": query_gate,
        }

    def clear_active_fusion_inputs(self) -> None:
        self._active_inputs.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        graph_states: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        reasoning_context: torch.Tensor | None = None,
        query_gate: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.last_fusion_output = None
        out = self.layer(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            h = out[0]
            rest = out[1:]
        else:
            h = out
            rest = None

        g_states = (
            graph_states if graph_states is not None else self._active_inputs.get("graph_states")
        )
        g_mask = graph_mask if graph_mask is not None else self._active_inputs.get("graph_mask")
        r_ctx = (
            reasoning_context
            if reasoning_context is not None
            else self._active_inputs.get("reasoning_context")
        )
        q_gate = query_gate if query_gate is not None else self._active_inputs.get("query_gate")

        if self.fusion is not None and g_states is not None:
            fusion_res = self.fusion(
                h,
                g_states,
                graph_mask=g_mask,
                reasoning_context=r_ctx,
                query_gate=q_gate,
            )
            self.last_fusion_output = fusion_res
            h = fusion_res.fused_hidden_states

        return (h, *rest) if rest is not None else h
