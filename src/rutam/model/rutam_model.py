"""Complete native graph-aware causal language model combining neural backbone and graph memory."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.encoder import RUTAMGraphEncoder
from rutam.graph.operations import Graph
from rutam.graph.selector import RUTAMGraphSelector
from rutam.graph.traversal import RUTAMGraphTraversal
from rutam.model.backbone_protocol import BackboneProtocol
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.graph_integration import GraphMemoryBridge
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.parameter_groups import ParameterGroupManager
from rutam.model.query_head import RUTAMNeuralQueryHead
from rutam.model.recurrent_integration import RUTAMRecurrentIntegration
from rutam.model.rutam_forward import execute_rutam_forward
from rutam.model.rutam_outputs import RUTAMCausalLMOutput

if TYPE_CHECKING:
    from rutam.graph.storage import GraphStorage
    from rutam.memory.controller import RUTAMMemoryController
    from rutam.memory.controller_output import CommitResult
    from rutam.memory.provenance import AuthorizationContext
    from rutam.memory.validation_output import ValidationBatchResult
    from rutam.memory.validator import RUTAMMemoryValidator
    from rutam.memory.write_head import RUTAMMemoryWriteHead
    from rutam.memory.write_proposal import MemoryProposal


class ModelMode(StrEnum):
    """The four operational architectural modes supported by RUTAM."""

    BASELINE = "baseline"
    GRAPH_NATIVE = "graph_native"
    RECURRENT = "recurrent"
    FULL_RUTAM = "full_rutam"


class RUTAMForCausalLM(nn.Module):
    """Complete native graph-aware causal language model combining neural backbone and graph memory.

    Coordinates end-to-end inference and training across four distinct modes:
    - baseline: neural backbone only (no graph memory, no recurrent loops).
    - graph_native: neural backbone with graph memory querying, traversal, and fusion.
    - recurrent: neural backbone with parameter-shared recurrent reasoning.
    - full_rutam: unified architecture with both graph memory and recurrent refinement.
    """

    def __init__(
        self,
        config: RUTAMConfig | None = None,
        *,
        mode: str | ModelMode | None = None,
        backbone: BackboneProtocol | nn.Module | None = None,
        graph: Graph | None = None,
        storage: GraphStorage | None = None,
        encoder: RUTAMGraphEncoder | None = None,
        selector: RUTAMGraphSelector | None = None,
        query_head: RUTAMNeuralQueryHead | None = None,
        traversal: RUTAMGraphTraversal | None = None,
        fusion: RUTAMGraphFusion | None = None,
        recurrent: RUTAMRecurrentIntegration | None = None,
        write_head: RUTAMMemoryWriteHead | None = None,
        validator: RUTAMMemoryValidator | None = None,
        memory_controller: RUTAMMemoryController | None = None,
    ) -> None:
        super().__init__()
        if mode is not None:
            mode_val = mode.value if hasattr(mode, "value") else str(mode)
            base_cfg = RUTAMConfig() if config is None else config
            self.config = base_cfg.with_mode(mode_val)
        else:
            self.config = RUTAMConfig() if config is None else config
        if not isinstance(self.config, RUTAMConfig):
            raise ValidationError("RUTAMForCausalLM requires a RUTAMConfig")

        self.graph = graph if graph is not None else Graph(self.config)
        self.storage = storage

        # Bridge for active snapshot, caching, and storage reloads
        self.bridge = GraphMemoryBridge(
            self.graph,
            storage=self.storage,
            use_cache=self.config.rutam_model.use_cached_graph_states,
        )

        # Backbone initialization
        if backbone is not None:
            if not isinstance(backbone, BackboneProtocol):
                raise ValidationError("backbone must implement BackboneProtocol")
            self.backbone = backbone
        else:
            self.backbone = SyntheticBackbone(
                hidden_dim=self.config.graph_fusion.hidden_dim,
                vocab_size=1000,
                num_layers=4,
            )

        # Neural graph components
        from rutam.memory.controller import RUTAMMemoryController
        from rutam.memory.validator import RUTAMMemoryValidator
        from rutam.memory.write_head import RUTAMMemoryWriteHead

        if memory_controller is not None:
            if not isinstance(memory_controller, RUTAMMemoryController):
                raise ValidationError("memory_controller must be a RUTAMMemoryController")
            self.memory_controller = memory_controller
            self.encoder = memory_controller.encoder if encoder is None else encoder
            self.selector = memory_controller.selector if selector is None else selector
            self.query_head = memory_controller.query_head if query_head is None else query_head
            self.traversal = memory_controller.traversal if traversal is None else traversal
            self.write_head = memory_controller.write_head if write_head is None else write_head
            self.validator = memory_controller.validator if validator is None else validator
            self.fusion = (
                fusion if fusion is not None else RUTAMGraphFusion(self.config.graph_fusion)
            )
        else:
            self.encoder = (
                encoder if encoder is not None else RUTAMGraphEncoder(self.config.encoder)
            )
            self.selector = (
                selector if selector is not None else RUTAMGraphSelector(self.config.selector)
            )
            self.query_head = (
                query_head
                if query_head is not None
                else RUTAMNeuralQueryHead(self.config.query_head)
            )
            self.traversal = (
                traversal
                if traversal is not None
                else (
                    RUTAMGraphTraversal(self.config.traversal)
                    if self.config.rutam_model.graph_traversal_enabled
                    else None
                )
            )
            self.fusion = (
                fusion if fusion is not None else RUTAMGraphFusion(self.config.graph_fusion)
            )
            self.write_head = (
                write_head
                if write_head is not None
                else RUTAMMemoryWriteHead(self.config.memory_write)
            )
            self.validator = (
                validator if validator is not None else RUTAMMemoryValidator(self.config)
            )
            self.memory_controller = RUTAMMemoryController(
                config=self.config,
                graph=self.graph,
                storage=self.storage,
                encoder=self.encoder,
                selector=self.selector,
                query_head=self.query_head,
                traversal=self.traversal,
                write_head=self.write_head,
                validator=self.validator,
                traversal_enabled=self.config.rutam_model.graph_traversal_enabled,
            )

        if recurrent is not None:
            if not isinstance(recurrent, RUTAMRecurrentIntegration):
                raise ValidationError("recurrent must be a RUTAMRecurrentIntegration")
            self.recurrent = recurrent
        else:
            self.recurrent = RUTAMRecurrentIntegration(self.config)

        self.parameter_group_manager = ParameterGroupManager()

        # Parameter freezing
        if self.config.rutam_model.backbone_frozen:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the language model backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the language model backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def is_backbone_frozen(self) -> bool:
        """Check whether all backbone parameters are frozen."""
        params = list(self.backbone.parameters())
        return len(params) == 0 or all(not p.requires_grad for p in params)

    @property
    def is_graph_enabled(self) -> bool:
        """Whether graph memory retrieval and cross-attention fusion are active."""
        return bool(
            self.config.rutam_model.graph_enabled
            and self.config.rutam_model.graph_fusion_enabled
        )

    @property
    def is_recurrent_enabled(self) -> bool:
        """Whether recurrent reasoning refinement is active."""
        return bool(
            self.recurrent.is_enabled
            or (self.config.recurrent.enabled or self.config.rutam_model.recurrent_loops_enabled)
        )

    @property
    def mode(self) -> str:
        """Return the current operational architecture mode string."""
        if self.is_graph_enabled and self.is_recurrent_enabled:
            return ModelMode.FULL_RUTAM.value
        elif self.is_graph_enabled and not self.is_recurrent_enabled:
            return ModelMode.GRAPH_NATIVE.value
        elif not self.is_graph_enabled and self.is_recurrent_enabled:
            return ModelMode.RECURRENT.value
        else:
            return ModelMode.BASELINE.value

    @property
    def is_baseline(self) -> bool:
        """True if operating in baseline mode (no graph, no recurrent)."""
        return self.mode == ModelMode.BASELINE.value

    @property
    def is_graph_native(self) -> bool:
        """True if operating in graph-native mode (graph on, recurrent off)."""
        return self.mode == ModelMode.GRAPH_NATIVE.value

    @property
    def is_recurrent(self) -> bool:
        """True if operating in recurrent mode (graph off, recurrent on)."""
        return self.mode == ModelMode.RECURRENT.value

    @property
    def is_full_rutam(self) -> bool:
        """True if operating in full RUTAM mode (both graph on and recurrent on)."""
        return self.mode == ModelMode.FULL_RUTAM.value

    def set_mode(self, mode: str | ModelMode) -> None:
        """Dynamically switch model mode between baseline, graph_native, recurrent, full_rutam."""
        mode_val = mode.value if hasattr(mode, "value") else str(mode)
        self.config = self.config.with_mode(mode_val)
        if self.config.recurrent.enabled and not self.recurrent.is_enabled:
            self.recurrent = RUTAMRecurrentIntegration(self.config)
        elif not self.config.recurrent.enabled and self.recurrent.is_enabled:
            self.recurrent = RUTAMRecurrentIntegration(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        memory_available: torch.Tensor | None = None,
        generate_proposals: bool = False,
        authorization_context: AuthorizationContext | None = None,
        source_references: Any = None,
        diagnostics: dict[str, Any] | None = None,
        node_mask: torch.Tensor | None = None,
        graph_state_snapshot: Any | None = None,
    ) -> RUTAMCausalLMOutput:
        """Execute unified graph-aware forward computation.

        Args:
            input_ids: Dense CPU torch.long tensor of shape [B, T].
            labels: Optional dense CPU torch.long tensor of shape [B, T].
            attention_mask: Optional boolean or float attention mask tensor.
            memory_available: Optional boolean indicator tensor of shape [B].
            generate_proposals: Whether to execute write head to generate proposals.
            authorization_context: Provenance and authorization context for proposals.
            diagnostics: Optional diagnostic dictionary to populate.

        Returns:
            RUTAMCausalLMOutput containing logits, optional loss, graph intermediates,
            and memory write proposals.
        """
        return execute_rutam_forward(
            backbone=self.backbone,
            encoder=self.encoder if self.config.rutam_model.graph_encoder_enabled else None,
            selector=self.selector if self.config.rutam_model.graph_selector_enabled else None,
            query_head=self.query_head,
            traversal=self.traversal if self.config.rutam_model.graph_traversal_enabled else None,
            fusion=self.fusion if self.config.rutam_model.graph_fusion_enabled else None,
            recurrent=self.recurrent,
            write_head=self.write_head if self.config.rutam_model.memory_write_enabled else None,
            bridge=self.bridge,
            config=self.config,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            memory_available=memory_available,
            generate_proposals=generate_proposals,
            authorization_context=authorization_context,
            source_references=source_references,
            diagnostics=diagnostics,
            node_mask=node_mask,
            graph_state_snapshot=graph_state_snapshot,
        )

    def propose_memory_updates(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        memory_available: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
        source_references: Any = None,
    ) -> tuple[MemoryProposal, ...]:
        """Execute forward pass with memory write proposal generation enabled."""
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_available=memory_available,
            generate_proposals=True,
            authorization_context=authorization_context,
            source_references=source_references,
        )
        return output.write_proposals

    def validate_memory_updates(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        authorization_context: AuthorizationContext | None = None,
    ) -> ValidationBatchResult:
        """Validate staged memory update proposals against current graph state."""
        return self.memory_controller.validate_updates(
            proposals,
            authorization_context=authorization_context,
        )

    def commit_memory_updates(
        self,
        proposals: Sequence[MemoryProposal],
        *,
        authorization_context: AuthorizationContext | None = None,
    ) -> CommitResult:
        """Transactionally commit validated proposals to graph and invalidate neural caches."""
        result = self.memory_controller.commit_validated_updates(proposals, authorization_context=authorization_context)
        self.bridge.invalidate_caches()
        return result

    def reload_graph(self, storage: GraphStorage | None = None) -> None:
        """Reload graph state from persistent storage and invalidate neural caches."""
        self.bridge.reload_from_storage(storage)
        self.memory_controller.refresh_graph_state()

    def get_parameter_groups(
        self,
        include_frozen: bool = False,
        only_active: bool = False,
    ) -> list[dict[str, Any]]:
        """Partition model parameters into optimizer groups by submodule and weight decay."""
        if only_active:
            modules: dict[str, Any] = {"backbone": self.backbone}
            if self.is_graph_enabled:
                modules.update({
                    "encoder": self.encoder,
                    "selector": self.selector,
                    "query_head": self.query_head,
                    "traversal": self.traversal,
                    "write_head": self.write_head,
                    "fusion": self.fusion,
                })
            if self.is_recurrent_enabled and self.recurrent.is_enabled:
                modules["recurrent"] = self.recurrent
        else:
            modules = {
                "backbone": self.backbone,
                "encoder": self.encoder,
                "selector": self.selector,
                "query_head": self.query_head,
                "traversal": self.traversal,
                "write_head": self.write_head,
                "fusion": self.fusion,
            }
            if self.recurrent.is_enabled:
                modules["recurrent"] = self.recurrent
        return self.parameter_group_manager.create_parameter_groups(
            modules, include_frozen=include_frozen
        )

    def count_parameters(self, only_active: bool = False) -> dict[str, dict[str, int]]:
        """Count total, trainable, and frozen parameters across submodules."""
        if only_active:
            modules: dict[str, Any] = {"backbone": self.backbone}
            if self.is_graph_enabled:
                modules.update({
                    "encoder": self.encoder,
                    "selector": self.selector,
                    "query_head": self.query_head,
                    "traversal": self.traversal,
                    "write_head": self.write_head,
                    "fusion": self.fusion,
                })
            if self.is_recurrent_enabled and self.recurrent.is_enabled:
                modules["recurrent"] = self.recurrent
        else:
            modules = {
                "backbone": self.backbone,
                "encoder": self.encoder,
                "selector": self.selector,
                "query_head": self.query_head,
                "traversal": self.traversal,
                "write_head": self.write_head,
                "fusion": self.fusion,
            }
            if self.recurrent.is_enabled:
                modules["recurrent"] = self.recurrent
        return self.parameter_group_manager.count_parameters(modules)
