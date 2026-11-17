"""Session runtime managing model, memory runtime, and multi-turn stateful interaction."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.memory.controller_output import CommitResult
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.write_proposal import MemoryProposal
from rutam.model.model_factory import RUTAMModelFactory
from rutam.model.rutam_model import ModelMode, RUTAMForCausalLM
from rutam.model.rutam_outputs import RUTAMCausalLMOutput
from rutam.runtime.capabilities import RuntimeCapabilities
from rutam.runtime.inference import RUTAMInferenceEngine
from rutam.runtime.memory_runtime import MemoryRuntime


@dataclass
class SessionStepResult:
    """Result from a single multi-turn execution step within a RUTAMSession."""

    turn_index: int
    output: RUTAMCausalLMOutput
    generated_ids: torch.Tensor | None = None
    proposals: tuple[MemoryProposal, ...] = field(default_factory=tuple)
    commit_result: CommitResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RUTAMSession:
    """Interactive multi-turn session orchestrating inference, memory proposals, and commits.

    Maintains session-level history, telemetry, and security authorization context
    across sequential inference steps.
    """

    def __init__(
        self,
        model: RUTAMForCausalLM,
        config: RUTAMConfig | None = None,
        *,
        session_id: str | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> None:
        if not isinstance(model, RUTAMForCausalLM):
            raise ValidationError(f"model must be a RUTAMForCausalLM, got {type(model).__name__}")

        self.model = model
        self.config = model.config if config is None else config
        self.session_id = session_id or str(uuid.uuid4())
        self.authorization_context = authorization_context or AuthorizationContext(
            user_id="default_user",
            role="user",
        )

        # Coordinate runtime subsystems
        self.inference_engine = RUTAMInferenceEngine(self.model, self.config)
        self.memory_runtime = MemoryRuntime(self.model)
        self.capabilities = RuntimeCapabilities.from_model(self.model)

        # Multi-turn state tracking
        self.turn_index: int = 0
        self.history: list[dict[str, Any]] = []

    @property
    def mode(self) -> str:
        """Active architecture mode."""
        return self.model.mode

    def step(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        generate_proposals: bool = False,
        auto_commit: bool = False,
        max_new_tokens: int = 0,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        stop_tokens: Sequence[int] | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> SessionStepResult:
        """Execute a stateful multi-turn session step.

        Args:
            input_ids: Prompt token IDs [B, T].
            attention_mask: Optional attention mask.
            generate_proposals: Whether to execute write head to propose memory updates.
            auto_commit: If True and proposals generated, automatically validates and commits them.
            max_new_tokens: If > 0, executes autoregressive generation after forward pass.
            temperature: Sampling temperature for autoregressive generation.
            top_k: Top-k filtering threshold.
            eos_token_id: Early stopping token ID for generation.
            stop_tokens: Additional halting token IDs.
            authorization_context: Override authorization context for this step.

        Returns:
            SessionStepResult encapsulating logits, generated tokens, proposals, and commit results.
        """
        auth_ctx = authorization_context or self.authorization_context
        self.turn_index += 1

        # 1. Forward inference execution
        diag: dict[str, Any] = {"session_id": self.session_id, "turn_index": self.turn_index}
        out = self.inference_engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generate_proposals=generate_proposals,
            authorization_context=auth_ctx,
            diagnostics=diag,
        )

        proposals = out.write_proposals
        commit_res: CommitResult | None = None

        # 2. Optional automatic commit
        if auto_commit and proposals and self.memory_runtime.is_available:
            commit_res = self.memory_runtime.commit(
                proposals,
                require_valid=True,
                authorization_context=auth_ctx,
            )

        # 3. Optional autoregressive generation
        generated: torch.Tensor | None = None
        if max_new_tokens > 0:
            generated = self.inference_engine.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
                attention_mask=attention_mask,
                stop_tokens=stop_tokens,
                authorization_context=auth_ctx,
            )

        step_result = SessionStepResult(
            turn_index=self.turn_index,
            output=out,
            generated_ids=generated,
            proposals=proposals,
            commit_result=commit_res,
            diagnostics=out.diagnostics,
        )

        # 4. Record history
        self.history.append(
            {
                "turn_index": self.turn_index,
                "input_tokens": input_ids.shape[1],
                "generated_tokens": (
                    (generated.shape[1] - input_ids.shape[1]) if generated is not None else 0
                ),
                "proposals_count": len(proposals),
                "committed_count": commit_res.applied_mutations if commit_res else 0,
                "graph_version": self.model.graph.version,
                "latency_ms": out.diagnostics.get("engine_latency_ms", 0.0),
            }
        )

        return step_result

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
        """Autoregressively generate tokens under the session or supplied authorization."""
        return self.inference_engine.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            attention_mask=attention_mask,
            stop_tokens=stop_tokens,
            authorization_context=(
                self.authorization_context
                if authorization_context is None
                else authorization_context
            ),
        )

    def propose_memory(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> tuple[MemoryProposal, ...]:
        """Propose candidate memory updates without committing them."""
        return self.memory_runtime.propose(
            input_ids=input_ids,
            attention_mask=attention_mask,
            authorization_context=authorization_context or self.authorization_context,
        )

    def commit_memory(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        require_valid: bool = True,
        authorization_context: AuthorizationContext | None = None,
    ) -> CommitResult:
        """Explicitly commit candidate memory updates to graph memory."""
        return self.memory_runtime.commit(
            proposals,
            require_valid=require_valid,
            authorization_context=authorization_context or self.authorization_context,
        )

    def save_snapshot(self, path: str | Path) -> None:
        """Save graph memory snapshot."""
        self.memory_runtime.save_snapshot(path)

    def restore_snapshot(self, path: str | Path) -> None:
        """Restore graph memory snapshot."""
        self.memory_runtime.restore_snapshot(path)

    def reset_history(self) -> None:
        """Clear turn history and reset turn counter."""
        self.turn_index = 0
        self.history.clear()

    def get_history(self) -> list[dict[str, Any]]:
        """Return the multi-turn session audit trail."""
        return list(self.history)

    @classmethod
    def create(
        cls,
        config: RUTAMConfig | None = None,
        *,
        mode: str | ModelMode = ModelMode.FULL_RUTAM,
        session_id: str | None = None,
        authorization_context: AuthorizationContext | None = None,
        **kwargs: Any,
    ) -> RUTAMSession:
        """Factory method to assemble a RUTAM model and initialize a session in one step."""
        model = RUTAMModelFactory.create_model(config=config, mode=mode, **kwargs)
        return cls(
            model=model,
            config=model.config,
            session_id=session_id,
            authorization_context=authorization_context,
        )
