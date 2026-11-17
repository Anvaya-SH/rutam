"""Architectural inspection utilities discovering backbone layers and integration points."""

from __future__ import annotations

from typing import Any

from rutam.config import LFM3AdapterConfig
from rutam.exceptions import ValidationError
from rutam.model.adapter_output import BackboneInspectionReport
from rutam.model.backbone_protocol import BackboneProtocol


def inspect_backbone(
    backbone: Any,
    config: LFM3AdapterConfig | None = None,
) -> BackboneInspectionReport:
    """Inspect and validate language model backbone architecture without modifying weights."""
    if backbone is None:
        raise ValidationError("Backbone cannot be None")

    backbone_id = config.backbone_id if config is not None else "custom_backbone"
    notes: list[str] = []

    # 1. Protocol inspection or structural fallback
    if isinstance(backbone, BackboneProtocol):
        hidden_dim = backbone.get_hidden_dim()
        vocab_size = backbone.get_vocab_size()
        num_layers = backbone.get_num_layers()
        layers = backbone.get_layers()
        layer_type = type(layers[0]).__name__ if layers else "unknown"
        discovered_layers = tuple(
            f"layer_{idx}_{type(layer).__name__}" for idx, layer in enumerate(layers)
        )
        verified = True
        notes.append("Backbone implements runtime BackboneProtocol directly")
    elif hasattr(backbone, "layers") and hasattr(backbone, "config"):
        cfg = backbone.config
        hidden_dim = getattr(cfg, "hidden_size", getattr(cfg, "hidden_dim", 256))
        vocab_size = getattr(cfg, "vocab_size", 32000)
        layers = backbone.layers
        num_layers = len(layers)
        layer_type = type(layers[0]).__name__ if layers else "unknown"
        discovered_layers = tuple(
            f"layer_{idx}_{type(layer).__name__}" for idx, layer in enumerate(layers)
        )
        verified = True
        notes.append("Backbone discovered via standard attribute introspection")
    else:
        if config is not None and config.require_verified_layers:
            raise ValidationError(
                f"Backbone of type {type(backbone).__name__} does not conform to "
                "BackboneProtocol and lacks standard layer attributes"
            )
        hidden_dim = getattr(backbone, "hidden_dim", 256)
        vocab_size = getattr(backbone, "vocab_size", 32000)
        num_layers = getattr(backbone, "num_layers", 0)
        layer_type = "unverified"
        discovered_layers = ()
        verified = False
        notes.append("Unverified generic backbone structure")

    # 2. Count parameters
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    if hasattr(backbone, "parameters"):
        for param in backbone.parameters():
            num = param.numel()
            total_params += num
            if param.requires_grad:
                trainable_params += num
            else:
                frozen_params += num

    param_counts = {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }

    # 3. Validate configured fusion locations against discovered layer count
    if config is not None and config.fusion_locations:
        for loc in config.fusion_locations:
            if loc >= num_layers:
                raise ValidationError(
                    f"Configured fusion location {loc} exceeds backbone layer count {num_layers}"
                )

    return BackboneInspectionReport(
        backbone_id=backbone_id,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        layer_type=layer_type,
        verified_integration=verified,
        discovered_layers=discovered_layers,
        parameter_counts=param_counts,
        validation_notes=tuple(notes),
    )
