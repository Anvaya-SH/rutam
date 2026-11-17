"""LFM3 backbone adapter coordinating layer fusion, parameter freezing, and hook management."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from rutam.config import LFM3AdapterConfig, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor
from rutam.model.adapter_output import AdapterOutput, BackboneInspectionReport
from rutam.model.architecture_inspection import inspect_backbone
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.fusion_output import FusionOutput
from rutam.model.layer_integration import (
    IntegratedLayerWrapper,
    LayerIntegrationHook,
    SyntheticBackbone,
)


class RUTAMLFM3Adapter(nn.Module):
    """Orchestrates seamless integration between LFM3 language model backbones and graph fusion."""

    def __init__(
        self,
        config: LFM3AdapterConfig | RUTAMConfig | None = None,
        *,
        backbone: Any = None,
        fusion_modules: Mapping[int, RUTAMGraphFusion] | RUTAMGraphFusion | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = LFM3AdapterConfig()
        elif isinstance(config, RUTAMConfig):
            config = config.lfm3_adapter
        if not isinstance(config, LFM3AdapterConfig):
            raise ValidationError("RUTAMLFM3Adapter requires an LFM3AdapterConfig or RUTAMConfig")
        self.config = config

        self.backbone = (
            backbone if backbone is not None else SyntheticBackbone(hidden_dim=256, num_layers=4)
        )
        self.inspection: BackboneInspectionReport = inspect_backbone(self.backbone, self.config)

        # Set up fusion modules
        self.fusion_modules = nn.ModuleDict()
        fusion_locs = self.config.fusion_locations
        if isinstance(fusion_modules, RUTAMGraphFusion):
            for loc in fusion_locs:
                self.fusion_modules[str(loc)] = fusion_modules
        elif isinstance(fusion_modules, Mapping):
            for loc, mod in fusion_modules.items():
                self.fusion_modules[str(loc)] = mod
        else:
            for loc in fusion_locs:
                self.fusion_modules[str(loc)] = RUTAMGraphFusion()

        self._hooks: list[LayerIntegrationHook] = []
        self._wrapped_layers: dict[int, nn.Module] = {}
        self._captured_fusion_outputs: list[FusionOutput] = []

        # Parameter freezing policies
        if self.config.backbone_frozen:
            self.freeze_backbone()
        if self.config.graph_modules_trainable:
            self.set_graph_modules_trainable(True)

        if self.config.integration_mode == "explicit":
            self._wrap_layers_explicitly()
        elif self.config.integration_mode == "hook":
            self.attach_hooks()

    def _wrap_layers_explicitly(self) -> None:
        if not hasattr(self.backbone, "layers"):
            return
        layers = self.backbone.layers
        for loc_str, fusion_mod in self.fusion_modules.items():
            loc = int(loc_str)
            if loc < len(layers):
                orig_layer = layers[loc]
                self._wrapped_layers[loc] = orig_layer
                layers[loc] = IntegratedLayerWrapper(orig_layer, fusion=fusion_mod)

    def attach_hooks(self) -> None:
        """Attach reversible forward hooks for post-layer graph fusion."""
        self.detach_hooks()
        if not hasattr(self.backbone, "layers"):
            return
        layers = self.backbone.layers
        for loc_str, fusion_mod in self.fusion_modules.items():
            loc = int(loc_str)
            if loc < len(layers):
                layer = layers[loc]

                def make_hook(f_mod: RUTAMGraphFusion) -> Any:
                    def hook_fn(
                        module: nn.Module,
                        layer_input: Any,
                        layer_output: Any,
                    ) -> Any:
                        h = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                        # Captured context from forward pass
                        ctx = getattr(self, "_active_fusion_inputs", {})
                        if ctx.get("graph_states") is not None:
                            f_out = f_mod(
                                h,
                                ctx["graph_states"],
                                graph_mask=ctx.get("graph_mask"),
                                reasoning_context=ctx.get("reasoning_context"),
                                query_gate=ctx.get("query_gate"),
                            )
                            self._captured_fusion_outputs.append(f_out)
                            h = f_out.fused_hidden_states
                        return (h, *layer_output[1:]) if isinstance(layer_output, tuple) else h

                    return hook_fn

                self._hooks.append(LayerIntegrationHook(layer, make_hook(fusion_mod)))

    def detach_hooks(self) -> None:
        """Cleanly remove registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def unwrap_layers(self) -> None:
        """Restore original layer modules in explicit mode."""
        if hasattr(self.backbone, "layers"):
            layers = self.backbone.layers
            for loc, orig in self._wrapped_layers.items():
                layers[loc] = orig
        self._wrapped_layers.clear()

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for parameter-efficient adaptation."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_layers(self, layer_indices: Sequence[int]) -> None:
        """Unfreeze specific backbone decoder layers while keeping other layers frozen."""
        if hasattr(self.backbone, "layers"):
            layers = self.backbone.layers
            for idx in layer_indices:
                if 0 <= idx < len(layers):
                    for param in layers[idx].parameters():
                        param.requires_grad = True

    def set_graph_modules_trainable(self, trainable: bool = True) -> None:
        """Enable or disable gradients across all graph fusion parameters."""
        for param in self.fusion_modules.parameters():
            param.requires_grad = trainable

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        fusion_lr: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """Return parameter groups with differentiated learning rates for optimizer setup."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        fusion_params = [p for p in self.fusion_modules.parameters() if p.requires_grad]

        groups: list[dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
        if fusion_params:
            groups.append({"params": fusion_params, "lr": fusion_lr, "name": "fusion"})
        return groups

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        graph_states: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
        reasoning_context: torch.Tensor | None = None,
        query_gate: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AdapterOutput:
        """Forward pass across backbone layers with integrated graph memory fusion."""
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long:
            raise ValidationError("input_ids must be a torch.long tensor")

        self._captured_fusion_outputs.clear()
        self._active_fusion_inputs = {
            "graph_states": graph_states,
            "graph_mask": graph_mask,
            "reasoning_context": reasoning_context,
            "query_gate": query_gate,
        }

        try:
            # Forward through backbone
            if self.config.integration_mode == "explicit":
                for loc in sorted(self._wrapped_layers.keys()):
                    wrapper = self.backbone.layers[loc]
                    if isinstance(wrapper, IntegratedLayerWrapper):
                        wrapper.set_active_fusion_inputs(
                            graph_states=graph_states,
                            graph_mask=graph_mask,
                            reasoning_context=reasoning_context,
                            query_gate=query_gate,
                        )
                res = self.backbone(
                    input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
                # Collect fusion outputs from wrappers
                for loc in sorted(self._wrapped_layers.keys()):
                    wrapper = self.backbone.layers[loc]
                    if isinstance(wrapper, IntegratedLayerWrapper):
                        if wrapper.last_fusion_output is not None:
                            self._captured_fusion_outputs.append(wrapper.last_fusion_output)
                        wrapper.clear_active_fusion_inputs()
            else:
                res = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)

            # Extract hidden states and logits
            if isinstance(res, dict):
                h = res.get("last_hidden_state", res.get("hidden_states"))
                logits = res.get("logits")
            elif isinstance(res, tuple):
                h = res[0]
                logits = res[1] if len(res) > 1 and isinstance(res[1], torch.Tensor) else None
            else:
                h = res
                logits = None

            if logits is None and hasattr(self.backbone, "get_output_logits"):
                logits = self.backbone.get_output_logits(h)

            validate_tensor(h, "hidden_states", torch.float32)
            fused_states = (
                self._captured_fusion_outputs[-1].fused_hidden_states
                if self._captured_fusion_outputs
                else None
            )

            return AdapterOutput(
                hidden_states=h,
                logits=logits,
                fused_hidden_states=fused_states,
                fusion_outputs=tuple(self._captured_fusion_outputs),
                metrics={
                    "fusion_count": len(self._captured_fusion_outputs),
                    "backbone_frozen": self.config.backbone_frozen,
                },
            )
        finally:
            self._active_fusion_inputs = {}
            for loc in self._wrapped_layers:
                wrapper = self.backbone.layers[loc]
                if isinstance(wrapper, IntegratedLayerWrapper):
                    wrapper.clear_active_fusion_inputs()
