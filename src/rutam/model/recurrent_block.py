"""Parameter-shared recurrent reasoning block with causal attention and graph conditioning."""

from __future__ import annotations

import math

import torch
from torch import nn

from rutam.config import RecurrentConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import activation_module, validate_tensor


class RUTAMRecurrentBlock(nn.Module):
    """Parameter-shared recurrent reasoning block.

    Reused across all recurrence loops t=0..N-1 with constant parameter count.
    Combines:
    1. Pre-LN causal multi-head self-attention with loop position conditioning.
    2. Optional graph cross-attention and path reasoning context injection.
    3. Pre-LN feed-forward network with residual connections.
    4. Causal safety ensuring no future token representations leak backward.
    """

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        if not isinstance(config, RecurrentConfig):
            raise ValidationError("RUTAMRecurrentBlock requires a RecurrentConfig")
        self.config = config

        d = config.hidden_dim
        n_heads = config.num_heads
        head_dim = d // n_heads

        self.hidden_dim = d
        self.num_heads = n_heads
        self.head_dim = head_dim

        # 1. Self-Attention components
        self.attn_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)

        # 2. Graph Conditioning components
        if config.use_graph_conditioning:
            h = config.graph_dim
            self.graph_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
            self.graph_q_proj = nn.Linear(d, d, bias=True)
            self.graph_k_proj = nn.Linear(h, d, bias=True)
            self.graph_v_proj = nn.Linear(h, d, bias=True)
            self.graph_out_proj = nn.Linear(d, d, bias=True)
            self.graph_dropout = nn.Dropout(config.dropout)

            self.reasoning_proj = nn.Linear(h, d, bias=True)
            self.reasoning_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

            gate_out_dim = d if config.gate_type == "elementwise" else 1
            self.graph_gate = nn.Sequential(
                nn.Linear(2 * d, d, bias=True),
                nn.GELU(),
                nn.Linear(d, gate_out_dim, bias=True),
            )
            self.graph_scale = nn.Parameter(
                torch.tensor(config.residual_scale_init, dtype=torch.float32)
            )
        else:
            self.graph_norm = None
            self.graph_q_proj = None
            self.graph_k_proj = None
            self.graph_v_proj = None
            self.graph_out_proj = None
            self.graph_dropout = None
            self.reasoning_proj = None
            self.reasoning_norm = None
            self.graph_gate = None
            self.graph_scale = None

        # 3. Feed-Forward Network components
        self.ffn_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ffn_up = nn.Linear(d, config.intermediate_dim, bias=True)
        self.ffn_act = activation_module(config.activation)
        self.ffn_dropout = nn.Dropout(config.dropout)
        self.ffn_down = nn.Linear(config.intermediate_dim, d, bias=True)

    def _causal_self_attention(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute causal multi-head self-attention over sequence tokens."""
        b, t, d = x.shape
        norm_x = self.attn_norm(x)

        q = self.q_proj(norm_x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, T, T]

        # Causal mask: upper triangular entries set to -inf
        if self.config.causal and t > 1:
            causal_mask = torch.triu(
                torch.full((t, t), float("-inf"), dtype=torch.float32, device="cpu"),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding / attention mask
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                scores = scores.masked_fill(~pad_mask, float("-inf"))
            elif attention_mask.dtype == torch.float32:
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores = scores + pad_mask

        attn_weights = torch.softmax(scores, dim=-1)
        # Handle all-masked positions if any
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]
        context = context.transpose(1, 2).contiguous().view(b, t, d)
        return self.attn_dropout(self.out_proj(context))

    def _graph_cross_attention(
        self,
        x: torch.Tensor,
        graph_states: torch.Tensor,
        *,
        graph_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attend from token representations to graph memory representations."""
        if self.graph_norm is None or self.graph_q_proj is None or self.graph_k_proj is None:
            return torch.zeros_like(x)

        b, t, d = x.shape
        _, k, _ = graph_states.shape

        if k == 0 or (graph_mask is not None and not graph_mask.any()):
            return torch.zeros_like(x)

        norm_x = self.graph_norm(x)
        q = self.graph_q_proj(norm_x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.graph_k_proj(graph_states)
        v_proj = self.graph_v_proj(graph_states)
        k_mat = k_proj.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)
        v_mat = v_proj.view(b, k, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_mat.transpose(-2, -1)) * scale  # [B, n_heads, T, K]

        if graph_mask is not None:
            mask = graph_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.graph_dropout(attn_weights)

        context = torch.matmul(attn_weights, v_mat)  # [B, n_heads, T, head_dim]
        context = context.transpose(1, 2).contiguous().view(b, t, d)
        return self.graph_dropout(self.graph_out_proj(context))

    def _project_reasoning(self, reasoning_context: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Project multi-hop reasoning path context into recurrent hidden space."""
        if self.reasoning_proj is None or self.reasoning_norm is None:
            return torch.zeros((reasoning_context.shape[0], seq_len, self.hidden_dim))

        if reasoning_context.ndim == 2:
            r = reasoning_context.unsqueeze(1)  # [B, 1, H]
        elif reasoning_context.ndim == 3:
            r = reasoning_context
        else:
            raise ValidationError("reasoning_context must have 2 or 3 dimensions")

        projected = self.reasoning_norm(self.reasoning_proj(r))  # [B, 1 or T, D]
        if projected.shape[1] == 1 and seq_len > 1:
            projected = projected.expand(-1, seq_len, -1)
        return projected

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        loop_embedding: torch.Tensor | None = None,
        graph_states: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        reasoning_context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute one recurrent reasoning block step.

        Args:
            hidden_states: Current representations [B, T, D].
            loop_embedding: Positional embedding for current iteration [1, 1, D] or [B, T, D].
            graph_states: Optional retrieved node representations [B, K, H].
            graph_mask: Optional boolean mask for valid graph nodes [B, K].
            reasoning_context: Optional multi-hop path reasoning summary [B, H] or [B, T, H].
            attention_mask: Optional causal/padding mask for sequence tokens [B, T].

        Returns:
            Candidate update tensor [B, T, D] representing proposed refinement delta.
        """
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        if hidden_states.ndim != 3:
            raise ValidationError("hidden_states must have shape [B, T, D]")

        b, t, d = hidden_states.shape
        if d != self.hidden_dim:
            raise ValidationError(f"hidden_states dim {d} != config.hidden_dim {self.hidden_dim}")

        # 1. Condition on loop position
        if loop_embedding is not None:
            validate_tensor(loop_embedding, "loop_embedding", torch.float32)
            attn_input = hidden_states + loop_embedding
        else:
            attn_input = hidden_states

        # 2. Causal Multi-Head Self-Attention
        attn_out = self._causal_self_attention(attn_input, attention_mask=attention_mask)
        cur = hidden_states + attn_out

        # 3. Optional Graph Conditioning
        if (
            self.config.use_graph_conditioning
            and self.graph_gate is not None
            and self.graph_scale is not None
        ):
            combined_graph = torch.zeros_like(cur)
            has_graph_input = False

            if graph_states is not None:
                validate_tensor(graph_states, "graph_states", torch.float32)
                graph_attn = self._graph_cross_attention(
                    cur, graph_states, graph_mask=graph_mask
                )
                combined_graph = combined_graph + graph_attn
                has_graph_input = True

            if reasoning_context is not None:
                validate_tensor(reasoning_context, "reasoning_context", torch.float32)
                reasoning_proj = self._project_reasoning(reasoning_context, seq_len=t)
                combined_graph = combined_graph + reasoning_proj
                has_graph_input = True

            if has_graph_input:
                gate_cat = torch.cat([cur, combined_graph], dim=-1)
                gate = torch.sigmoid(self.graph_gate(gate_cat))
                cur = cur + self.graph_scale * gate * combined_graph

        # 4. Feed-Forward Network
        ffn_in = self.ffn_norm(cur)
        ffn_out = self.ffn_down(self.ffn_dropout(self.ffn_act(self.ffn_up(ffn_in))))
        cur = cur + self.ffn_dropout(ffn_out)

        # Return refinement candidate delta
        candidate_delta = cur - hidden_states
        return candidate_delta
