"""Inference engine managing inputs, attention masks, diagnostics, and autoregressive generation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.model.rutam_outputs import RUTAMCausalLMOutput

if TYPE_CHECKING:
    from rutam.memory.provenance import AuthorizationContext


class RUTAMInferenceEngine:
    """Coordinates robust inference execution, attention masking, diagnostics, and generation."""

    def __init__(
        self,
        model: RUTAMForCausalLM,
        config: RUTAMConfig | None = None,
    ) -> None:
        if not isinstance(model, RUTAMForCausalLM):
            raise ValidationError(f"model must be a RUTAMForCausalLM, got {type(model).__name__}")
        self.model = model
        self.config = model.config if config is None else config
        self._last_diagnostics: dict[str, Any] = {}

    def validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate input tensors and create default attention mask if absent.

        Args:
            input_ids: Tensor of shape [B, T] with torch.long dtype.
            attention_mask: Optional attention mask of shape [B, T].

        Returns:
            Tuple of validated (input_ids, attention_mask).
        """
        if not isinstance(input_ids, torch.Tensor):
            raise ValidationError("input_ids must be a torch.Tensor")
        if input_ids.dtype != torch.long:
            raise ValidationError(f"input_ids must have dtype torch.long, got {input_ids.dtype}")
        if input_ids.ndim != 2:
            raise ValidationError(f"input_ids must have 2 dimensions [B, T], got {input_ids.ndim}")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValidationError("input_ids batch and sequence lengths must be greater than zero")

        b, t = input_ids.shape
        if attention_mask is None:
            mask = torch.ones((b, t), dtype=torch.bool, device=input_ids.device)
        else:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValidationError("attention_mask must be a torch.Tensor")
            if attention_mask.shape != (b, t):
                raise ValidationError(
                    f"attention_mask shape {attention_mask.shape} must match "
                    f"input_ids shape {(b, t)}"
                )
            mask = attention_mask

        return input_ids, mask

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        memory_available: torch.Tensor | None = None,
        generate_proposals: bool = False,
        authorization_context: AuthorizationContext | None = None,
        source_references: Any = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> RUTAMCausalLMOutput:
        """Execute validated forward pass with end-to-end telemetry and diagnostics collection."""
        start_time = time.perf_counter()
        clean_input_ids, clean_mask = self.validate_inputs(input_ids, attention_mask)

        diag = dict(diagnostics or {})
        diag["model_mode"] = self.model.mode
        diag["is_graph_enabled"] = self.model.is_graph_enabled
        diag["is_recurrent_enabled"] = self.model.is_recurrent_enabled

        out = self.model(
            input_ids=clean_input_ids,
            labels=labels,
            attention_mask=clean_mask,
            memory_available=memory_available,
            generate_proposals=generate_proposals,
            authorization_context=authorization_context,
            source_references=source_references,
            diagnostics=diag,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        out.diagnostics["engine_latency_ms"] = latency_ms
        out.diagnostics["total_tokens"] = clean_input_ids.numel()
        self._last_diagnostics = dict(out.diagnostics)

        return out

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        attention_mask: torch.Tensor | None = None,
        stop_tokens: Sequence[int] | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate subsequent token representations.

        Args:
            input_ids: Initial prompt token IDs [B, T].
            max_new_tokens: Maximum number of new tokens to synthesize.
            temperature: Sampling temperature (<= 1e-5 executes greedy argmax).
            top_k: Optional top-k filtering threshold.
            eos_token_id: Optional end-of-sequence token ID halting generation early.
            attention_mask: Optional prompt attention mask.
            stop_tokens: Optional set of token IDs that halt generation if sampled.
            authorization_context: Trusted read permissions and scopes for every decoding step.

        Returns:
            Concatenated tensor of [B, T + N] generated token IDs.
        """
        curr_input_ids, curr_mask = self.validate_inputs(input_ids, attention_mask)

        stop_set = set(stop_tokens or [])
        if eos_token_id is not None:
            stop_set.add(eos_token_id)

        snapshot = self.model.bridge.get_active_state()
        for _ in range(max_new_tokens):
            out = self.model(
                input_ids=curr_input_ids,
                attention_mask=curr_mask,
                generate_proposals=False,
                graph_state_snapshot=snapshot,
                authorization_context=authorization_context,
            )
            next_token_logits = out.logits[:, -1, :]  # [B, V]

            if temperature <= 1e-5:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]
            else:
                scaled_logits = next_token_logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    k = min(top_k, scaled_logits.shape[-1])
                    v, _ = torch.topk(scaled_logits, k)
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
            new_mask_col = torch.ones(
                (curr_mask.shape[0], 1), dtype=curr_mask.dtype, device=curr_mask.device
            )
            curr_mask = torch.cat([curr_mask, new_mask_col], dim=1)

            # Check halting condition across batch
            if stop_set and any(token.item() in stop_set for token in next_token.view(-1)):
                break

        return curr_input_ids

    def get_last_diagnostics(self) -> dict[str, Any]:
        """Return telemetric diagnostics from the most recent forward execution."""
        return dict(self._last_diagnostics)
