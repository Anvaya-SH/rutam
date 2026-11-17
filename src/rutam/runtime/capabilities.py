"""Runtime capability verification and architectural feature inspection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.rutam_model import ModelMode, RUTAMForCausalLM


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Verifies and exposes architectural capabilities of a RUTAM model or session.

    Maps high-level capabilities to concrete subsystem flags across the 4 architecture modes:
    - baseline: Standard causal LM inference only.
    - graph_native: Graph memory retrieval, multi-hop traversal, and cross-attention fusion.
    - recurrent: Parameter-shared recurrent reasoning, loop embeddings, dynamic halting.
    - full_rutam: All graph memory and recurrent capabilities active concurrently.
    """

    mode: str
    supports_graph_memory: bool
    supports_multi_hop_traversal: bool
    supports_recurrent_reasoning: bool
    supports_dynamic_halting: bool
    supports_memory_proposals: bool
    supports_memory_validation: bool
    supports_memory_commits: bool
    supports_state_snapshots: bool
    supports_generation: bool = True

    @classmethod
    def from_model(cls, model: RUTAMForCausalLM) -> RuntimeCapabilities:
        """Inspect a RUTAMForCausalLM instance and extract runtime capabilities."""
        if not isinstance(model, RUTAMForCausalLM):
            raise ValidationError(f"model must be a RUTAMForCausalLM, got {type(model).__name__}")

        cfg = model.config
        mode = model.mode

        graph_active = model.is_graph_enabled
        rec_active = model.is_recurrent_enabled

        supports_traversal = (
            graph_active
            and cfg.rutam_model.graph_traversal_enabled
            and model.traversal is not None
        )

        return cls(
            mode=mode,
            supports_graph_memory=graph_active,
            supports_multi_hop_traversal=supports_traversal,
            supports_recurrent_reasoning=rec_active,
            supports_dynamic_halting=rec_active and cfg.recurrent.dynamic_halting,
            supports_memory_proposals=graph_active and cfg.rutam_model.memory_write_enabled,
            supports_memory_validation=model.validator is not None,
            supports_memory_commits=graph_active and model.memory_controller is not None,
            supports_state_snapshots=graph_active,
            supports_generation=True,
        )

    @classmethod
    def from_config(cls, config: RUTAMConfig) -> RuntimeCapabilities:
        """Inspect a RUTAMConfig instance and derive capability flags."""
        if not isinstance(config, RUTAMConfig):
            raise ValidationError("config must be a RUTAMConfig instance")

        graph_on = config.rutam_model.graph_enabled and config.rutam_model.graph_fusion_enabled
        rec_on = config.recurrent.enabled or config.rutam_model.recurrent_loops_enabled

        if graph_on and rec_on:
            mode = ModelMode.FULL_RUTAM.value
        elif graph_on:
            mode = ModelMode.GRAPH_NATIVE.value
        elif rec_on:
            mode = ModelMode.RECURRENT.value
        else:
            mode = ModelMode.BASELINE.value

        return cls(
            mode=mode,
            supports_graph_memory=graph_on,
            supports_multi_hop_traversal=graph_on and config.rutam_model.graph_traversal_enabled,
            supports_recurrent_reasoning=rec_on,
            supports_dynamic_halting=rec_on and config.recurrent.dynamic_halting,
            supports_memory_proposals=graph_on and config.rutam_model.memory_write_enabled,
            supports_memory_validation=True,
            supports_memory_commits=graph_on,
            supports_state_snapshots=graph_on,
            supports_generation=True,
        )

    def has(self, capability: str) -> bool:
        """Check whether a capability is supported by name."""
        attr_name = capability if capability.startswith("supports_") else f"supports_{capability}"
        return bool(getattr(self, attr_name, False))

    def verify(self, required: Sequence[str] | str) -> bool:
        """Verify that all required capabilities are supported, raising ValidationError if missing.

        Args:
            required: A single capability name or sequence of capability names.

        Returns:
            True if all required capabilities are satisfied.

        Raises:
            ValidationError: If any requested capability is not active in the current mode.
        """
        req_list = [required] if isinstance(required, str) else list(required)
        missing: list[str] = []

        for req in req_list:
            attr_name = req if req.startswith("supports_") else f"supports_{req}"
            if not getattr(self, attr_name, False):
                missing.append(req)

        if missing:
            raise ValidationError(
                f"Missing required capabilities in mode '{self.mode}': {', '.join(missing)}"
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert capabilities to dictionary."""
        return asdict(self)
