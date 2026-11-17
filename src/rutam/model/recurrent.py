"""Engine coordinating parameter-shared recurrent reasoning, loop embeddings, and halting."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import RecurrentConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor
from rutam.model.loop_controller import DynamicLoopController
from rutam.model.loop_embeddings import LoopPositionEmbedding
from rutam.model.recurrent_block import RUTAMRecurrentBlock
from rutam.model.recurrent_gate import RecurrentResidualGate
from rutam.model.recurrent_output import RecurrentOutput


class RUTAMRecurrentEngine(nn.Module):
    """Engine executing parameter-shared recurrent reasoning loops.

    Reuses a single recurrent reasoning block across multiple iterations, modulating
    hidden representations with loop position embeddings, optional graph conditioning,
    residual gating with identity-preserving scaling, and dynamic loop control.
    """

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        if not isinstance(config, RecurrentConfig):
            raise ValidationError("RUTAMRecurrentEngine requires a RecurrentConfig")
        self.config = config

        self.loop_embeddings = LoopPositionEmbedding(config)
        self.block = RUTAMRecurrentBlock(config)
        self.residual_gate = RecurrentResidualGate(config)
        self.controller = DynamicLoopController(config) if config.dynamic_halting else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        num_loops: int | None = None,
        graph_states: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        reasoning_context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> RecurrentOutput:
        """Execute recurrent reasoning over hidden representations.

        Args:
            hidden_states: Input representations [B, T, D].
            num_loops: Explicit loop count override (must be between min_loops and max_loops).
            graph_states: Optional retrieved graph memory representations [B, K, H].
            graph_mask: Optional boolean mask for active graph nodes [B, K].
            reasoning_context: Optional multi-hop reasoning path summary [B, H] or [B, T, H].
            attention_mask: Optional sequence attention mask [B, T].

        Returns:
            RecurrentOutput containing refined output states and recurrence intermediates.
        """
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        if hidden_states.ndim != 3:
            raise ValidationError("hidden_states must have shape [B, T, D]")

        if num_loops is not None:
            if type(num_loops) is not int:
                raise ValidationError(f"num_loops must be int, got {type(num_loops).__name__}")
            if not (self.config.min_loops <= num_loops <= self.config.max_loops):
                raise ValidationError(
                    f"num_loops {num_loops} must be in [{self.config.min_loops}, "
                    f"{self.config.max_loops}]"
                )

        max_iterations = (
            num_loops
            if num_loops is not None
            else (
                self.config.max_loops
                if self.config.dynamic_halting and self.controller is not None
                else self.config.default_loops
            )
        )

        cur_state = hidden_states
        loop_states: list[torch.Tensor] = []
        residual_gates: list[torch.Tensor] = []
        continue_logits_list: list[torch.Tensor] = []
        halt_probs_list: list[torch.Tensor] = []

        dynamic_halted = False

        for t in range(max_iterations):
            loop_emb = self.loop_embeddings.get_embedding(t)
            candidate = self.block(
                cur_state,
                loop_embedding=loop_emb,
                graph_states=graph_states,
                graph_mask=graph_mask,
                reasoning_context=reasoning_context,
                attention_mask=attention_mask,
            )

            next_state, gate = self.residual_gate(cur_state, candidate)
            loop_states.append(next_state)
            residual_gates.append(gate)

            # Check dynamic halting if enabled and loop count not fixed by user
            if num_loops is None and self.config.dynamic_halting and self.controller is not None:
                continue_logit, halt_prob, should_continue = self.controller(
                    next_state, loop_idx=t, attention_mask=attention_mask
                )
                continue_logits_list.append(continue_logit)
                halt_probs_list.append(halt_prob)
                cur_state = next_state
                if not should_continue:
                    dynamic_halted = True
                    break
            else:
                cur_state = next_state

        executed_loops = len(loop_states)
        all_continue_logits = (
            torch.cat(continue_logits_list, dim=-1) if continue_logits_list else None
        )
        all_halt_probs = torch.cat(halt_probs_list, dim=-1) if halt_probs_list else None

        # Ponder cost: step count per sequence [B] or scalar
        b = hidden_states.shape[0]
        ponder_cost = torch.full((b,), float(executed_loops), dtype=torch.float32, device="cpu")

        mean_gate = (
            float(torch.stack([g.mean() for g in residual_gates]).mean().item())
            if residual_gates
            else 0.0
        )

        diagnostics = {
            "num_loops": executed_loops,
            "dynamic_halted": dynamic_halted,
            "mean_residual_gate": mean_gate,
            "residual_scale": float(self.residual_gate.residual_scale.item()),
        }

        return RecurrentOutput(
            output_states=cur_state,
            loop_states=tuple(loop_states),
            num_loops=executed_loops,
            halting_probabilities=all_halt_probs,
            continue_logits=all_continue_logits,
            cumulative_ponder_cost=ponder_cost,
            residual_gates=tuple(residual_gates),
            diagnostics=diagnostics,
        )
