"""Integration layer connecting recurrent reasoning after graph fusion."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from rutam.config import RecurrentConfig, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor
from rutam.model.recurrent import RUTAMRecurrentEngine
from rutam.model.recurrent_output import RecurrentOutput


class RUTAMRecurrentIntegration(nn.Module):
    """Bridges graph fusion output and parameter-shared recurrent reasoning.

    Positioned immediately following graph memory fusion in RUTAM decoder flow.
    When disabled (default: enabled=False), functions as a clean identity pass-through
    with zero computational overhead or parameter allocation.
    When enabled, executes parameter-shared causal recurrent reasoning with graph
    conditioning and dynamic halting.
    """

    def __init__(
        self,
        config: RecurrentConfig | RUTAMConfig | None = None,
        *,
        engine: RUTAMRecurrentEngine | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            self.config = RecurrentConfig()
        elif isinstance(config, RUTAMConfig):
            rec = config.recurrent
            rec_dict = asdict(rec)
            modified = False
            # Harmonize dimensions with graph_fusion if not custom configured
            if rec.hidden_dim != config.graph_fusion.hidden_dim:
                d = config.graph_fusion.hidden_dim
                rec_dict["hidden_dim"] = d
                rec_dict["loop_embedding_dim"] = d
                # Ensure num_heads divides hidden_dim
                if d % rec.num_heads != 0:
                    rec_dict["num_heads"] = 4 if d % 4 == 0 else (2 if d % 2 == 0 else 1)
                modified = True
            if rec.graph_dim != config.graph_fusion.graph_dim:
                rec_dict["graph_dim"] = config.graph_fusion.graph_dim
                modified = True
            self.config = RecurrentConfig(**rec_dict) if modified else rec
        elif isinstance(config, RecurrentConfig):
            self.config = config
        else:
            raise ValidationError(
                "RUTAMRecurrentIntegration requires a RecurrentConfig or RUTAMConfig"
            )

        if engine is not None:
            if not isinstance(engine, RUTAMRecurrentEngine):
                raise ValidationError("engine must be a RUTAMRecurrentEngine")
            self.engine = engine
        elif self.config.enabled:
            self.engine = RUTAMRecurrentEngine(self.config)
        else:
            self.engine = None

    @property
    def is_enabled(self) -> bool:
        """Whether recurrent reasoning is active."""
        return self.config.enabled and self.engine is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        graph_states: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        reasoning_context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        num_loops: int | None = None,
    ) -> tuple[torch.Tensor, RecurrentOutput | None]:
        """Execute recurrent reasoning if enabled, or pass through unchanged.

        Args:
            hidden_states: LM representations after graph fusion [B, T, D].
            graph_states: Optional retrieved node states [B, K, H].
            graph_mask: Optional boolean node mask [B, K].
            reasoning_context: Optional multi-hop reasoning path summary [B, H] or [B, T, H].
            attention_mask: Optional token attention mask [B, T].
            num_loops: Optional loop count override.

        Returns:
            Tuple of (output_hidden_states [B, T, D], recurrent_output or None).
        """
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        if hidden_states.ndim != 3:
            raise ValidationError("hidden_states must have shape [B, T, D]")

        if not self.is_enabled:
            return hidden_states, None

        assert self.engine is not None
        recurrent_out = self.engine(
            hidden_states,
            num_loops=num_loops,
            graph_states=graph_states,
            graph_mask=graph_mask,
            reasoning_context=reasoning_context,
            attention_mask=attention_mask,
        )

        return recurrent_out.output_states, recurrent_out
