"""Architecture manifest generation detailing model structure, parameters, and capabilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.rutam_model import RUTAMForCausalLM


@dataclass
class ArchitectureManifest:
    """Complete structured manifest describing an assembled RUTAM model architecture."""

    version: str
    project_name: str
    mode: str
    backbone: dict[str, Any]
    components: dict[str, dict[str, Any]]
    parameter_counts: dict[str, Any]
    capabilities: dict[str, bool]
    graph_config: dict[str, Any]
    recurrent_config: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to Python dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def generate_architecture_manifest(
    model: RUTAMForCausalLM,
    config: RUTAMConfig | None = None,
) -> ArchitectureManifest:
    """Generate comprehensive architecture manifest from an assembled RUTAM model instance.

    Args:
        model: Instantiated RUTAMForCausalLM model.
        config: Optional configuration override (defaults to model.config).

    Returns:
        Populated ArchitectureManifest describing the model assembly.
    """
    if not isinstance(model, RUTAMForCausalLM):
        raise ValidationError(f"model must be a RUTAMForCausalLM, got {type(model).__name__}")

    cfg = model.config if config is None else config
    if not isinstance(cfg, RUTAMConfig):
        raise ValidationError("config must be a RUTAMConfig instance")

    # 1. Inspect backbone
    backbone_info: dict[str, Any] = {
        "class_name": type(model.backbone).__name__,
        "is_frozen": model.is_backbone_frozen(),
    }
    if hasattr(model.backbone, "get_hidden_dim"):
        backbone_info["hidden_dim"] = model.backbone.get_hidden_dim()
    if hasattr(model.backbone, "get_num_layers"):
        backbone_info["num_layers"] = model.backbone.get_num_layers()
    if hasattr(model.backbone, "get_vocab_size"):
        backbone_info["vocab_size"] = model.backbone.get_vocab_size()

    # 2. Inspect components and submodule parameter counts
    counts = model.count_parameters(only_active=False)

    components: dict[str, dict[str, Any]] = {
        "encoder": {
            "enabled": cfg.rutam_model.graph_encoder_enabled and model.is_graph_enabled,
            "class_name": type(model.encoder).__name__,
            "params": counts.get("encoder", {}),
        },
        "selector": {
            "enabled": cfg.rutam_model.graph_selector_enabled and model.is_graph_enabled,
            "class_name": type(model.selector).__name__,
            "params": counts.get("selector", {}),
        },
        "query_head": {
            "enabled": model.query_head is not None,
            "class_name": type(model.query_head).__name__,
            "params": counts.get("query_head", {}),
        },
        "traversal": {
            "enabled": (
                cfg.rutam_model.graph_traversal_enabled
                and model.is_graph_enabled
                and model.traversal is not None
            ),
            "class_name": type(model.traversal).__name__ if model.traversal is not None else "None",
            "params": counts.get("traversal", {}),
        },
        "fusion": {
            "enabled": cfg.rutam_model.graph_fusion_enabled and model.is_graph_enabled,
            "class_name": type(model.fusion).__name__,
            "params": counts.get("fusion", {}),
        },
        "recurrent": {
            "enabled": model.is_recurrent_enabled,
            "class_name": type(model.recurrent).__name__,
            "params": counts.get("recurrent", {}),
        },
        "write_head": {
            "enabled": cfg.rutam_model.memory_write_enabled and model.is_graph_enabled,
            "class_name": type(model.write_head).__name__,
            "params": counts.get("write_head", {}),
        },
        "validator": {
            "enabled": model.validator is not None,
            "class_name": type(model.validator).__name__,
        },
        "controller": {
            "enabled": model.memory_controller is not None,
            "class_name": type(model.memory_controller).__name__,
        },
        "bridge": {
            "enabled": model.bridge is not None,
            "class_name": type(model.bridge).__name__,
        },
    }

    # Total parameter rollup
    total_trainable = sum(m.get("trainable", 0) for m in counts.values())
    total_frozen = sum(m.get("frozen", 0) for m in counts.values())
    parameter_counts = {
        "total_trainable": total_trainable,
        "total_frozen": total_frozen,
        "total": total_trainable + total_frozen,
        "by_module": counts,
    }

    # 3. Model capabilities
    capabilities = {
        "graph_memory": model.is_graph_enabled,
        "recurrent_reasoning": model.is_recurrent_enabled,
        "multi_hop_traversal": (
            model.is_graph_enabled
            and cfg.rutam_model.graph_traversal_enabled
            and model.traversal is not None
        ),
        "dynamic_halting": model.is_recurrent_enabled and cfg.recurrent.dynamic_halting,
        "memory_proposals": model.is_graph_enabled and cfg.rutam_model.memory_write_enabled,
        "discrete_validation": model.validator is not None,
        "transaction_safety": True,
        "cached_graph_states": cfg.rutam_model.use_cached_graph_states,
    }

    # 4. Graph & recurrent configuration snapshots
    graph_cfg = {
        "embedding_dim": cfg.graph.embedding_dim,
        "num_relation_types": cfg.encoder.num_relation_types,
        "max_nodes": cfg.graph.max_nodes,
        "max_edges": cfg.graph.max_edges,
    }

    rec_cfg = {
        "enabled": cfg.recurrent.enabled,
        "hidden_dim": cfg.recurrent.hidden_dim,
        "num_heads": cfg.recurrent.num_heads,
        "max_loops": cfg.recurrent.max_loops,
        "min_loops": cfg.recurrent.min_loops,
        "default_loops": cfg.recurrent.default_loops,
        "dynamic_halting": cfg.recurrent.dynamic_halting,
        "halting_threshold": cfg.recurrent.halting_threshold,
    }

    return ArchitectureManifest(
        version="0.1.0",
        project_name=cfg.project_name,
        mode=model.mode,
        backbone=backbone_info,
        components=components,
        parameter_counts=parameter_counts,
        capabilities=capabilities,
        graph_config=graph_cfg,
        recurrent_config=rec_cfg,
        created_at=datetime.now(UTC).isoformat(),
    )
