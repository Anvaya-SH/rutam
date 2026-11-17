"""Typed output structure for recurrent reasoning loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rutam.config import positive_int
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


@dataclass(frozen=True, slots=True)
class RecurrentOutput:
    """Output container for RUTAM parameter-shared recurrent reasoning.

    Attributes:
        output_states: Final refined hidden states [B, T, D].
        loop_states: Intermediate representations after each loop t=1..N [B, T, D].
        num_loops: Number of recurrent loops executed.
        halting_probabilities: Halting probabilities across loops [B, N] or None.
        continue_logits: Raw continue logits across loops [B, N] or None.
        cumulative_ponder_cost: Ponder cost / step count tensor [B] or float or None.
        residual_gates: Residual gate values across loops (each [B, T, 1] or [B, T, D]).
        diagnostics: Additional execution metrics and metadata.
    """

    output_states: torch.Tensor
    loop_states: tuple[torch.Tensor, ...] = ()
    num_loops: int = 1
    halting_probabilities: torch.Tensor | None = None
    continue_logits: torch.Tensor | None = None
    cumulative_ponder_cost: torch.Tensor | None = None
    residual_gates: tuple[torch.Tensor, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_tensor(self.output_states, "output_states", torch.float32)
        if self.output_states.ndim != 3:
            raise ValidationError(
                f"output_states must have shape [B, T, D], got {tuple(self.output_states.shape)}"
            )

        positive_int(self.num_loops, "num_loops", minimum=1)

        if not isinstance(self.loop_states, (tuple, list)):
            raise ValidationError("loop_states must be a sequence of tensors")
        for i, state in enumerate(self.loop_states):
            validate_tensor(state, f"loop_states[{i}]", torch.float32)
            if state.shape != self.output_states.shape:
                raise ValidationError(
                    f"loop_states[{i}] shape {tuple(state.shape)} != "
                    f"output_states shape {tuple(self.output_states.shape)}"
                )
        object.__setattr__(self, "loop_states", tuple(self.loop_states))

        if self.halting_probabilities is not None:
            validate_tensor(self.halting_probabilities, "halting_probabilities", torch.float32)

        if self.continue_logits is not None:
            validate_tensor(self.continue_logits, "continue_logits", torch.float32)

        if self.cumulative_ponder_cost is not None:
            validate_tensor(self.cumulative_ponder_cost, "cumulative_ponder_cost", torch.float32)

        if not isinstance(self.residual_gates, (tuple, list)):
            raise ValidationError("residual_gates must be a sequence of tensors")
        for i, gate in enumerate(self.residual_gates):
            validate_tensor(gate, f"residual_gates[{i}]", torch.float32)
        object.__setattr__(self, "residual_gates", tuple(self.residual_gates))

        if not isinstance(self.diagnostics, dict):
            raise ValidationError("diagnostics must be a dictionary")
