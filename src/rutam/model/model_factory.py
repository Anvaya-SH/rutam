"""Factory for constructing unified RUTAM causal language models without duplicate instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.encoder import RUTAMGraphEncoder
from rutam.graph.operations import Graph
from rutam.graph.selector import RUTAMGraphSelector
from rutam.graph.storage import GraphStorage
from rutam.graph.traversal import RUTAMGraphTraversal
from rutam.model.backbone_protocol import BackboneProtocol
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.recurrent_integration import RUTAMRecurrentIntegration
from rutam.model.rutam_model import ModelMode, RUTAMForCausalLM

if TYPE_CHECKING:
    from rutam.memory.controller import RUTAMMemoryController
    from rutam.memory.validator import RUTAMMemoryValidator
    from rutam.memory.write_head import RUTAMMemoryWriteHead
    from rutam.model.lfm3_bridge import RUTAMLFM3Bridge


class RUTAMModelFactory:
    """Constructs fully integrated, non-duplicated RUTAM neural models.

    Ensures that shared subsystems (graph operations, neural encoders, query heads,
    selectors, traversals, write heads, and discrete memory validators) are instantiated
    once and shared identically between RUTAMForCausalLM and RUTAMMemoryController,
    preventing duplicate weights, dangling instances, or inconsistent graph snapshots.
    """

    @classmethod
    def create_lfm3_bridge(
        cls,
        backbone: nn.Module,
        config: RUTAMConfig | None = None,
        **kwargs: Any,
    ) -> RUTAMLFM3Bridge:
        """Attach graph fusion to a loaded LFM3 model without replacing its forward path."""
        from rutam.model.lfm3_bridge import RUTAMLFM3Bridge

        return RUTAMLFM3Bridge(backbone, config=config, **kwargs)

    @classmethod
    def create_model(
        cls,
        config: RUTAMConfig | None = None,
        *,
        mode: str | ModelMode | None = None,
        backbone: BackboneProtocol | nn.Module | None = None,
        graph: Graph | None = None,
        storage: GraphStorage | None = None,
        encoder: RUTAMGraphEncoder | None = None,
        selector: RUTAMGraphSelector | None = None,
        query_head: Any | None = None,
        traversal: RUTAMGraphTraversal | None = None,
        fusion: RUTAMGraphFusion | None = None,
        recurrent: RUTAMRecurrentIntegration | None = None,
        write_head: RUTAMMemoryWriteHead | None = None,
        validator: RUTAMMemoryValidator | None = None,
        memory_controller: RUTAMMemoryController | None = None,
        freeze_backbone: bool | None = None,
    ) -> RUTAMForCausalLM:
        """Create a complete RUTAMForCausalLM model with unified component binding.

        Args:
            config: Full RUTAM configuration. Defaults to default RUTAMConfig().
            mode: Desired architecture mode ('baseline', 'graph_native', 'recurrent', 'full_rutam').
            backbone: Pre-instantiated backbone adhering to BackboneProtocol.
            graph: Existing Graph instance to bind.
            storage: Persistent storage backend.
            encoder: Graph encoder neural submodule.
            selector: Graph selector neural submodule.
            query_head: Neural query head submodule.
            traversal: Multi-hop graph traversal engine.
            fusion: Cross-attention graph fusion submodule.
            recurrent: Parameter-shared recurrent integration module.
            write_head: Neural memory write head submodule.
            validator: Discrete memory transition validator.
            memory_controller: Pre-existing memory controller (if provided, submodules are bound
                from it).
            freeze_backbone: Optional explicit override for backbone parameter freezing.

        Returns:
            Fully assembled RUTAMForCausalLM model.
        """
        # Resolve configuration and operational mode
        base_cfg = RUTAMConfig() if config is None else config
        if not isinstance(base_cfg, RUTAMConfig):
            raise ValidationError(f"config must be a RUTAMConfig instance, got {type(config)}")

        if mode is not None:
            mode_val = mode.value if hasattr(mode, "value") else str(mode)
            resolved_config = base_cfg.with_mode(mode_val)
        else:
            resolved_config = base_cfg

        # Validate backbone if supplied
        if backbone is not None and not isinstance(backbone, BackboneProtocol):
            if getattr(getattr(backbone, "config", None), "model_type", None) == "lfm3":
                raise ValidationError(
                    "Hugging Face LFM3 uses a nested hybrid stack; call "
                    "RUTAMModelFactory.create_lfm3_bridge() instead of create_model()"
                )
            raise ValidationError(
                f"backbone must implement BackboneProtocol, got {type(backbone).__name__}"
            )

        # 1. Single unified graph instance
        resolved_graph = graph if graph is not None else Graph(resolved_config)

        from rutam.memory.controller import RUTAMMemoryController
        from rutam.memory.validator import RUTAMMemoryValidator
        from rutam.memory.write_head import RUTAMMemoryWriteHead

        # 2. Submodule instantiation without duplication
        if memory_controller is not None:
            if not isinstance(memory_controller, RUTAMMemoryController):
                raise ValidationError("memory_controller must be a RUTAMMemoryController")
            resolved_encoder = memory_controller.encoder if encoder is None else encoder
            resolved_selector = memory_controller.selector if selector is None else selector
            resolved_query_head = (
                memory_controller.query_head if query_head is None else query_head
            )
            resolved_traversal = memory_controller.traversal if traversal is None else traversal
            resolved_write_head = (
                memory_controller.write_head if write_head is None else write_head
            )
            resolved_validator = memory_controller.validator if validator is None else validator
            resolved_fusion = (
                fusion if fusion is not None else RUTAMGraphFusion(resolved_config.graph_fusion)
            )
            resolved_controller = memory_controller
        else:
            resolved_encoder = (
                encoder if encoder is not None else RUTAMGraphEncoder(resolved_config.encoder)
            )
            resolved_selector = (
                selector if selector is not None else RUTAMGraphSelector(resolved_config.selector)
            )
            from rutam.model.query_head import RUTAMNeuralQueryHead

            resolved_query_head = (
                query_head
                if query_head is not None
                else RUTAMNeuralQueryHead(resolved_config.query_head)
            )
            resolved_traversal = (
                traversal
                if traversal is not None
                else (
                    RUTAMGraphTraversal(resolved_config.traversal)
                    if resolved_config.rutam_model.graph_traversal_enabled
                    else None
                )
            )
            resolved_fusion = (
                fusion if fusion is not None else RUTAMGraphFusion(resolved_config.graph_fusion)
            )
            resolved_write_head = (
                write_head
                if write_head is not None
                else RUTAMMemoryWriteHead(resolved_config.memory_write)
            )
            resolved_validator = (
                validator if validator is not None else RUTAMMemoryValidator(resolved_config)
            )

            # Build canonical shared memory controller
            resolved_controller = RUTAMMemoryController(
                config=resolved_config,
                graph=resolved_graph,
                storage=storage,
                encoder=resolved_encoder,
                selector=resolved_selector,
                query_head=resolved_query_head,
                traversal=resolved_traversal,
                write_head=resolved_write_head,
                validator=resolved_validator,
                traversal_enabled=resolved_config.rutam_model.graph_traversal_enabled,
            )

        # 3. Recurrent integration
        resolved_recurrent = (
            recurrent
            if recurrent is not None
            else RUTAMRecurrentIntegration(resolved_config)
        )

        # 4. Construct unified model
        model = RUTAMForCausalLM(
            config=resolved_config,
            backbone=backbone,
            graph=resolved_graph,
            storage=storage,
            encoder=resolved_encoder,
            selector=resolved_selector,
            query_head=resolved_query_head,
            traversal=resolved_traversal,
            fusion=resolved_fusion,
            recurrent=resolved_recurrent,
            write_head=resolved_write_head,
            validator=resolved_validator,
            memory_controller=resolved_controller,
        )

        # Optional explicit backbone freezing override
        if freeze_backbone is not None:
            if freeze_backbone:
                model.freeze_backbone()
            else:
                model.unfreeze_backbone()

        return model

    @classmethod
    def attach_backbone(
        cls,
        backbone: BackboneProtocol | nn.Module,
        config: RUTAMConfig | None = None,
        *,
        mode: str | ModelMode | None = None,
        **kwargs: Any,
    ) -> RUTAMForCausalLM:
        """Attach a verified backbone conforming to BackboneProtocol to a new RUTAM model.

        Args:
            backbone: Model implementing BackboneProtocol.
            config: Optional RUTAMConfig.
            mode: Optional operational mode.
            **kwargs: Additional parameters forwarded to create_model.

        Returns:
            RUTAMForCausalLM with backbone attached.
        """
        if not isinstance(backbone, BackboneProtocol):
            raise ValidationError(
                f"backbone must implement BackboneProtocol, got {type(backbone).__name__}"
            )
        return cls.create_model(config=config, mode=mode, backbone=backbone, **kwargs)

    @classmethod
    def create_baseline(
        cls,
        config: RUTAMConfig | None = None,
        **kwargs: Any,
    ) -> RUTAMForCausalLM:
        """Create a model in baseline mode (no graph memory, no recurrent reasoning)."""
        return cls.create_model(config=config, mode=ModelMode.BASELINE, **kwargs)

    @classmethod
    def create_graph_native(
        cls,
        config: RUTAMConfig | None = None,
        **kwargs: Any,
    ) -> RUTAMForCausalLM:
        """Create a model in graph_native mode (graph memory active, recurrent off)."""
        return cls.create_model(config=config, mode=ModelMode.GRAPH_NATIVE, **kwargs)

    @classmethod
    def create_recurrent(
        cls,
        config: RUTAMConfig | None = None,
        **kwargs: Any,
    ) -> RUTAMForCausalLM:
        """Create a model in recurrent mode (graph memory disabled, recurrent reasoning active)."""
        return cls.create_model(config=config, mode=ModelMode.RECURRENT, **kwargs)

    @classmethod
    def create_full_rutam(
        cls,
        config: RUTAMConfig | None = None,
        **kwargs: Any,
    ) -> RUTAMForCausalLM:
        """Create a model in full_rutam mode (both graph memory and recurrent reasoning active)."""
        return cls.create_model(config=config, mode=ModelMode.FULL_RUTAM, **kwargs)
