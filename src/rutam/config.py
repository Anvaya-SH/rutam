"""Strict, dependency-light configuration with deterministic YAML serialization."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rutam.exceptions import ValidationError


def positive_int(value: object, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")


def boolean(value: object, name: str) -> None:
    if type(value) is not bool:
        raise ValidationError(f"{name} must be a boolean")


def nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class GraphConfig:
    path: str = "data/graph"
    embedding_dim: int = 128
    directed: bool = True
    max_nodes: int = 100_000
    max_edges: int = 500_000
    initial_nodes: int = 0
    auto_save: bool = False

    def __post_init__(self) -> None:
        nonempty_string(self.path, "graph.path")
        for name in ("embedding_dim", "max_nodes", "max_edges"):
            positive_int(getattr(self, name), f"graph.{name}")
        positive_int(self.initial_nodes, "graph.initial_nodes", minimum=0)
        if self.initial_nodes != 0:
            raise ValidationError(
                "initial_nodes must be 0: nodes require explicit content/provenance"
            )
        boolean(self.directed, "graph.directed")
        boolean(self.auto_save, "graph.auto_save")


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    validation_enabled: bool = True
    require_provenance: bool = True

    def __post_init__(self) -> None:
        boolean(self.validation_enabled, "memory.validation_enabled")
        boolean(self.require_provenance, "memory.require_provenance")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.device != "cpu" or self.dtype != "float32":
            raise ValidationError("This foundation supports only device=cpu and dtype=float32")


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Neural dimensions are independent of any future backbone interface.

    input_dim is checked against graph features when encoding, rather than here,
    so existing graph-only configurations with different dimensions still load.
    """

    input_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_relation_types: int = 32
    dropout: float = 0.1
    activation: str = "gelu"
    aggregation: str = "mean"
    residual: bool = True
    layer_norm: bool = True
    encoder_mode: str = "message_passing"

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "num_layers", "num_relation_types"):
            positive_int(getattr(self, name), f"encoder.{name}")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0 <= self.dropout < 1
            or not math.isfinite(self.dropout)
        ):
            raise ValidationError("encoder.dropout must be a finite number in [0, 1)")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValidationError("encoder.activation must be gelu, relu or silu")
        if self.aggregation not in ("mean", "sum"):
            raise ValidationError("encoder.aggregation must be mean or sum")
        if self.encoder_mode not in ("message_passing", "graph_attention", "hybrid"):
            raise ValidationError(
                "encoder.encoder_mode must be message_passing, graph_attention or hybrid"
            )
        boolean(self.residual, "encoder.residual")
        boolean(self.layer_norm, "encoder.layer_norm")


@dataclass(frozen=True, slots=True)
class GraphAttentionConfig:
    """Settings for sparse relation-aware native graph attention."""

    enabled: bool = True
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    relation_bias: bool = True
    query_conditioning: bool = True
    residual: bool = True
    normalization: str = "layer_norm"
    sparse_attention: bool = True
    max_subgraph_nodes: int = 256
    max_subgraph_edges: int = 2048
    return_attention: bool = False
    num_relation_types: int = 32
    query_dim: int = 128

    def __post_init__(self) -> None:
        boolean(self.enabled, "graph_attention.enabled")
        for name in (
            "hidden_dim",
            "num_heads",
            "num_layers",
            "max_subgraph_nodes",
            "max_subgraph_edges",
            "num_relation_types",
            "query_dim",
        ):
            positive_int(getattr(self, name), f"graph_attention.{name}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValidationError("graph_attention.hidden_dim must be divisible by num_heads")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0 <= self.dropout < 1
            or not math.isfinite(self.dropout)
        ):
            raise ValidationError("graph_attention.dropout must be a finite number in [0, 1)")
        if self.normalization not in ("layer_norm", "none"):
            raise ValidationError("graph_attention.normalization must be layer_norm or none")
        for name in (
            "relation_bias",
            "query_conditioning",
            "residual",
            "sparse_attention",
            "return_attention",
        ):
            boolean(getattr(self, name), f"graph_attention.{name}")


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    """Neural graph selection settings; max_dense_elements bounds dense workspaces."""

    query_dim: int = 128
    hidden_dim: int = 128
    score_dim: int = 128
    scoring_method: str = "scaled_dot"
    top_k: int = 16
    selection_mode: str = "hard"
    temperature: float = 1.0
    normalize_vectors: bool = False
    chunk_size: int | None = 4096
    return_attention: bool = False
    max_dense_elements: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("query_dim", "hidden_dim", "score_dim", "top_k", "max_dense_elements"):
            positive_int(getattr(self, name), f"selector.{name}")
        if self.chunk_size is not None:
            positive_int(self.chunk_size, "selector.chunk_size")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 < self.temperature <= 1.7976931348623157e308
        ):
            raise ValidationError("selector.temperature must be a finite positive number")
        if self.scoring_method not in ("scaled_dot", "additive"):
            raise ValidationError("selector.scoring_method must be scaled_dot or additive")
        if self.selection_mode not in ("hard", "soft"):
            raise ValidationError("selector.selection_mode must be hard or soft")
        boolean(self.normalize_vectors, "selector.normalize_vectors")
        boolean(self.return_attention, "selector.return_attention")


@dataclass(frozen=True, slots=True)
class QueryHeadConfig:
    """Synthetic dimensions, not a claim about released LFM3 internals."""

    backbone_hidden_dim: int = 256
    graph_hidden_dim: int = 128
    query_dim: int = 128
    intermediate_dim: int = 256
    pooling: str = "attention"
    dropout: float = 0.1
    normalize_queries: bool = True
    graph_conditioning: bool = True
    conditioning_mode: str = "gated_add"
    query_gate_enabled: bool = True
    gate_threshold: float = 0.5
    return_attention: bool = False
    empty_sequence_policy: str = "error"
    gate_use_availability: bool = True
    residual: bool = False
    residual_projection: bool = False
    norm_regularization_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("backbone_hidden_dim", "graph_hidden_dim", "query_dim", "intermediate_dim"):
            positive_int(getattr(self, name), f"query_head.{name}")
        for name in (
            "normalize_queries",
            "graph_conditioning",
            "query_gate_enabled",
            "return_attention",
            "gate_use_availability",
            "residual",
            "residual_projection",
        ):
            boolean(getattr(self, name), f"query_head.{name}")
        for name, upper in (
            ("dropout", 1.0),
            ("gate_threshold", 1.0),
            ("norm_regularization_weight", 1.7976931348623157e308),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= upper
            ):
                raise ValidationError(f"query_head.{name} must be finite in [0,{upper}]")
        if self.dropout == 1:
            raise ValidationError("query_head.dropout must be less than 1")
        if self.pooling not in ("last_valid", "mean", "attention"):
            raise ValidationError("query_head.pooling must be last_valid, mean or attention")
        if self.conditioning_mode not in ("add", "gated_add"):
            raise ValidationError("query_head.conditioning_mode must be add or gated_add")
        if self.empty_sequence_policy not in ("error", "zeros"):
            raise ValidationError("query_head.empty_sequence_policy must be error or zeros")
        if self.residual_projection and not self.residual:
            raise ValidationError("residual_projection requires residual=True")
        if (
            self.residual
            and self.backbone_hidden_dim != self.query_dim
            and not self.residual_projection
        ):
            raise ValidationError("Unequal residual dimensions require residual_projection=True")

    def validate_selector(self, selector: SelectorConfig) -> None:
        if not isinstance(selector, SelectorConfig) or self.query_dim != selector.query_dim:
            raise ValidationError("query_head.query_dim must match selector.query_dim")


def bounded_float(
    value: object, name: str, *, minimum: float = 0, maximum: float = 1.7976931348623157e308
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= value <= maximum
    ):
        raise ValidationError(f"{name} must be finite in [{minimum}, {maximum}]")


@dataclass(frozen=True, slots=True)
class TraversalConfig:
    max_hops: int = 3
    max_neighbors_per_node: int = 16
    max_visited_nodes: int = 128
    max_candidate_paths: int = 64
    max_selected_paths: int = 8
    direction: str = "outgoing"
    deterministic: bool = True
    path_encoder: str = "gru"
    path_hidden_dim: int = 128
    path_scoring: str = "additive"
    temperature: float = 1.0
    include_seed_nodes: bool = True
    query_dim: int = 128
    hidden_dim: int = 128
    num_relation_types: int = 32
    score_hidden_dim: int = 128
    scoring_chunk_size: int = 128
    include_edge_weights: bool = True
    dropout: float = 0.0
    max_path_elements: int = 1_000_000

    def __post_init__(self) -> None:
        positive_int(self.max_hops, "traversal.max_hops", minimum=0)
        for name in (
            "max_neighbors_per_node",
            "max_visited_nodes",
            "max_candidate_paths",
            "max_selected_paths",
            "path_hidden_dim",
            "query_dim",
            "hidden_dim",
            "num_relation_types",
            "score_hidden_dim",
            "scoring_chunk_size",
            "max_path_elements",
        ):
            positive_int(getattr(self, name), f"traversal.{name}")
        if self.max_selected_paths > self.max_candidate_paths:
            raise ValidationError("max_selected_paths cannot exceed max_candidate_paths")
        if self.direction not in ("outgoing", "incoming", "both"):
            raise ValidationError("traversal.direction must be outgoing, incoming or both")
        if self.path_encoder != "gru" or self.path_scoring not in (
            "additive",
            "scaled_dot",
            "cosine",
        ):
            raise ValidationError(
                "Supported traversal encoders/scorers: gru; additive/scaled_dot/cosine"
            )
        for name in ("deterministic", "include_seed_nodes", "include_edge_weights"):
            boolean(getattr(self, name), f"traversal.{name}")
        bounded_float(self.dropout, "traversal.dropout", maximum=1)
        if self.dropout == 1:
            raise ValidationError("traversal.dropout must be less than 1")
        bounded_float(self.temperature, "traversal.temperature")
        if self.temperature == 0:
            raise ValidationError("traversal.temperature must be positive")
        if (
            self.max_hops > 8
            or self.max_candidate_paths > 4096
            or self.max_neighbors_per_node > 128
        ):
            warnings.warn(
                "Large traversal settings can grow path prefixes combinatorially; "
                "candidate/visited/tensor budgets still truncate or reject work.",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True, slots=True)
class LinkLossWeights:
    link: float = 1.0
    relation: float = 1.0
    ranking: float = 0.2

    def __post_init__(self) -> None:
        for name in ("link", "relation", "ranking"):
            bounded_float(getattr(self, name), f"linker.loss_weights.{name}")
        if self.link + self.relation + self.ranking == 0:
            raise ValidationError("At least one linker loss weight must be positive")


@dataclass(frozen=True, slots=True)
class LinkerConfig:
    hidden_dim: int = 128
    scoring_hidden_dim: int = 256
    dropout: float = 0.1
    directed: bool = True
    allow_self_links: bool = False
    max_candidate_pairs: int = 512
    candidate_strategy: str = "neighborhood"
    max_candidate_hops: int = 2
    link_threshold: float = 0.75
    relation_threshold: float = 0.70
    max_proposals_per_call: int = 32
    use_structural_features: bool = True
    query_conditioning: bool = False
    negative_samples_per_positive: int = 3
    loss_weights: LinkLossWeights = field(default_factory=LinkLossWeights)
    query_dim: int = 128
    structural_feature_dim: int = 16
    max_neighbors_per_node: int = 32
    max_candidate_expansions: int = 4096
    similarity_chunk_size: int = 4096
    max_similarity_anchors: int = 32
    negative_sampling_seed: int = 0
    negative_sampling_attempts: int = 10000
    scoring_chunk_size: int = 128
    max_link_elements: int = 1_000_000
    link_temperature: float = 1.0
    relation_temperature: float = 1.0
    positive_weight: float = 1.0
    ranking_margin: float = 1.0
    relation_schema_version: int = 1
    relation_labels: tuple[str, ...] = (
        "related_to",
        "depends_on",
        "part_of",
        "causes",
        "supports",
        "contradicts",
        "derived_from",
        "precedes",
    )
    conflicting_relations: tuple[tuple[str, str], ...] = ()
    compatible_node_types: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "scoring_hidden_dim",
            "max_candidate_pairs",
            "max_proposals_per_call",
            "query_dim",
            "structural_feature_dim",
            "max_neighbors_per_node",
            "max_candidate_expansions",
            "similarity_chunk_size",
            "max_similarity_anchors",
            "negative_sampling_attempts",
            "relation_schema_version",
            "scoring_chunk_size",
            "max_link_elements",
        ):
            positive_int(getattr(self, name), f"linker.{name}")
        positive_int(self.negative_samples_per_positive, "negative_samples_per_positive", minimum=0)
        positive_int(self.negative_sampling_seed, "negative_sampling_seed", minimum=0)
        if type(self.max_candidate_hops) is not int or self.max_candidate_hops not in (1, 2):
            raise ValidationError("linker.max_candidate_hops must be 1 or 2")
        if self.candidate_strategy == "two_hop" and self.max_candidate_hops != 2:
            raise ValidationError("two_hop strategy requires max_candidate_hops=2")
        for name in (
            "directed",
            "allow_self_links",
            "use_structural_features",
            "query_conditioning",
        ):
            boolean(getattr(self, name), f"linker.{name}")
        if self.candidate_strategy not in (
            "neighborhood",
            "two_hop",
            "selected",
            "similarity",
            "explicit",
        ):
            raise ValidationError("Unknown linker candidate_strategy")
        for name in ("dropout", "link_threshold", "relation_threshold"):
            bounded_float(getattr(self, name), f"linker.{name}", maximum=1)
        if self.dropout == 1:
            raise ValidationError("linker.dropout must be less than 1")
        for name in (
            "positive_weight",
            "ranking_margin",
            "link_temperature",
            "relation_temperature",
        ):
            bounded_float(getattr(self, name), f"linker.{name}")
        if (
            self.positive_weight == 0
            or self.link_temperature == 0
            or self.relation_temperature == 0
        ):
            raise ValidationError("Positive weight and calibration temperatures must be positive")
        if isinstance(self.loss_weights, Mapping):
            try:
                object.__setattr__(self, "loss_weights", LinkLossWeights(**self.loss_weights))
            except TypeError as exc:
                raise ValidationError(f"Invalid linker loss_weights: {exc}") from exc
        if not isinstance(self.loss_weights, LinkLossWeights):
            raise ValidationError("loss_weights must be LinkLossWeights or a mapping")
        if not isinstance(self.relation_labels, (tuple, list)):
            raise ValidationError("relation_labels must be a sequence of explicit labels")
        labels = tuple(self.relation_labels)
        for label in labels:
            nonempty_string(label, "relation label")
        if not labels or len(set(labels)) != len(labels) or "NO_RELATION" in labels:
            raise ValidationError(
                "Relation labels must be nonempty, unique, and exclude reserved NO_RELATION"
            )
        object.__setattr__(self, "relation_labels", labels)
        for name in ("conflicting_relations", "compatible_node_types"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValidationError(f"{name} must be a sequence of pairs")
            normalized = []
            for pair in values:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValidationError(f"{name} entries must contain two labels")
                for label in pair:
                    nonempty_string(label, name)
                    if name == "conflicting_relations" and label not in labels:
                        raise ValidationError("Conflict annotations require known relation labels")
                normalized.append(tuple(pair))
            object.__setattr__(self, name, tuple(normalized))


@dataclass(frozen=True, slots=True)
class MemoryWriteConfig:
    enabled: bool = True
    hidden_dim: int = 128
    projection_dim: int = 256
    max_proposals: int = 4
    write_threshold: float = 0.70
    operation_count: int = 7
    use_graph_context: bool = True
    target_selection: bool = True
    normalize_embeddings: bool = True
    dropout: float = 0.1
    commit_automatically: bool = False
    backbone_hidden_dim: int = 256
    pooling: str = "last_valid"
    empty_sequence_policy: str = "error"

    def __post_init__(self) -> None:
        boolean(self.enabled, "memory_write.enabled")
        for name in (
            "hidden_dim",
            "projection_dim",
            "max_proposals",
            "operation_count",
            "backbone_hidden_dim",
        ):
            positive_int(getattr(self, name), f"memory_write.{name}")
        for name in (
            "use_graph_context",
            "target_selection",
            "normalize_embeddings",
        ):
            boolean(getattr(self, name), f"memory_write.{name}")
        bounded_float(
            self.write_threshold, "memory_write.write_threshold", minimum=0.0, maximum=1.0
        )
        bounded_float(self.dropout, "memory_write.dropout", minimum=0.0, maximum=1.0)
        if self.dropout == 1.0:
            raise ValidationError("memory_write.dropout must be less than 1")
        boolean(self.commit_automatically, "memory_write.commit_automatically")
        if self.commit_automatically:
            raise ValidationError(
                "memory_write.commit_automatically must remain false: proposal generation only"
            )
        if self.pooling not in ("last_valid", "mean", "attention"):
            raise ValidationError("memory_write.pooling must be last_valid, mean or attention")
        if self.empty_sequence_policy not in ("error", "zeros"):
            raise ValidationError("memory_write.empty_sequence_policy must be error or zeros")


@dataclass(frozen=True, slots=True)
class MemoryValidationConfig:
    enabled: bool = True
    require_provenance: bool = True
    allow_unverified_hypotheses: bool = False
    detect_duplicates: bool = True
    detect_structural_conflicts: bool = True
    reject_stale_updates: bool = True
    require_authorization: bool = True
    max_proposals_per_batch: int = 64
    max_proposal_bytes: int = 65536
    semantic_verification_enabled: bool = False
    default_decision: str = "defer"
    content_similarity_threshold: float = 0.98
    min_confidence: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "require_provenance",
            "allow_unverified_hypotheses",
            "detect_duplicates",
            "detect_structural_conflicts",
            "reject_stale_updates",
            "require_authorization",
            "semantic_verification_enabled",
        ):
            boolean(getattr(self, name), f"memory_validation.{name}")
        for name in ("max_proposals_per_batch", "max_proposal_bytes"):
            positive_int(getattr(self, name), f"memory_validation.{name}")
        if self.default_decision not in ("defer", "reject", "accept"):
            raise ValidationError(
                "memory_validation.default_decision must be defer, reject, or accept"
            )
        bounded_float(
            self.content_similarity_threshold,
            "memory_validation.content_similarity_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        bounded_float(
            self.min_confidence,
            "memory_validation.min_confidence",
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True, slots=True)
class MemoryControllerConfig:
    enabled: bool = True
    read_enabled: bool = True
    write_enabled: bool = True
    auto_commit: bool = False
    require_validation: bool = True
    max_reads_per_request: int = 4
    max_pending_proposals: int = 128
    transaction_mode: str = "atomic"
    preserve_snapshots: int = 5
    audit_enabled: bool = True
    cache_graph_states: bool = True
    allow_destructive_updates: bool = False

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "read_enabled",
            "write_enabled",
            "auto_commit",
            "require_validation",
            "audit_enabled",
            "cache_graph_states",
            "allow_destructive_updates",
        ):
            boolean(getattr(self, name), f"memory_controller.{name}")
        for name in ("max_reads_per_request", "max_pending_proposals"):
            positive_int(getattr(self, name), f"memory_controller.{name}")
        positive_int(self.preserve_snapshots, "memory_controller.preserve_snapshots", minimum=0)
        if self.transaction_mode not in ("atomic", "isolated", "immediate"):
            raise ValidationError(
                "memory_controller.transaction_mode must be atomic, isolated, or immediate"
            )



@dataclass(frozen=True, slots=True)
class MemoryLearningConfig:
    """Configuration for joint neural memory learning objectives and loss balancing."""

    enabled: bool = True
    curriculum_enabled: bool = True
    use_fixed_loss_weights: bool = True
    ignore_missing_targets: bool = True
    write_loss_weight: float = 1.0
    operation_loss_weight: float = 1.0
    retrieval_loss_weight: float = 1.0
    path_loss_weight: float = 0.5
    link_loss_weight: float = 0.5
    embedding_loss_weight: float = 0.5
    query_gate_loss_weight: float = 0.5
    update_target_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "curriculum_enabled",
            "use_fixed_loss_weights",
            "ignore_missing_targets",
        ):
            boolean(getattr(self, name), f"memory_learning.{name}")
        for name in (
            "write_loss_weight",
            "operation_loss_weight",
            "retrieval_loss_weight",
            "path_loss_weight",
            "link_loss_weight",
            "embedding_loss_weight",
            "query_gate_loss_weight",
            "update_target_loss_weight",
        ):
            val = getattr(self, name)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or val < 0.0
                or not math.isfinite(val)
            ):
                raise ValidationError(
                    f"memory_learning.{name} must be a nonnegative finite number"
                )


@dataclass(frozen=True, slots=True)
class GraphFusionConfig:
    enabled: bool = True
    hidden_dim: int = 256
    graph_dim: int = 128
    num_attention_heads: int = 4
    attention_dropout: float = 0.0
    projection_dropout: float = 0.1
    fusion_mode: str = "cross_attention"
    gate_mode: str = "token"
    residual_scale_init: float = 0.001
    use_reasoning_context: bool = True
    preserve_empty_identity: bool = True
    return_attention: bool = False

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "use_reasoning_context",
            "preserve_empty_identity",
            "return_attention",
        ):
            boolean(getattr(self, name), f"graph_fusion.{name}")
        for name in ("hidden_dim", "graph_dim", "num_attention_heads"):
            positive_int(getattr(self, name), f"graph_fusion.{name}")
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValidationError(
                f"graph_fusion.hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        bounded_float(
            self.attention_dropout, "graph_fusion.attention_dropout", minimum=0.0, maximum=1.0
        )
        if self.attention_dropout == 1.0:
            raise ValidationError("graph_fusion.attention_dropout must be less than 1")
        bounded_float(
            self.projection_dropout, "graph_fusion.projection_dropout", minimum=0.0, maximum=1.0
        )
        if self.projection_dropout == 1.0:
            raise ValidationError("graph_fusion.projection_dropout must be less than 1")
        bounded_float(self.residual_scale_init, "graph_fusion.residual_scale_init", minimum=0.0)
        if self.fusion_mode not in ("cross_attention",):
            raise ValidationError(
                f"Unsupported graph_fusion.fusion_mode: {self.fusion_mode}; must be cross_attention"
            )
        if self.gate_mode not in ("token", "sequence"):
            raise ValidationError(
                f"Unsupported graph_fusion.gate_mode: {self.gate_mode}; must be token or sequence"
            )


@dataclass(frozen=True, slots=True)
class LFM3AdapterConfig:
    backbone_id: str = "LiquidAI/LFM3-1.7B-Base"
    enabled: bool = True
    fusion_locations: tuple[int, ...] = ()
    backbone_frozen: bool = True
    graph_modules_trainable: bool = True
    integration_mode: str = "explicit"
    require_verified_layers: bool = True
    preserve_output_interface: bool = True
    enable_cache: bool = False
    allow_remote_code: bool = False

    def __post_init__(self) -> None:
        nonempty_string(self.backbone_id, "lfm3_adapter.backbone_id")
        for name in (
            "enabled",
            "backbone_frozen",
            "graph_modules_trainable",
            "require_verified_layers",
            "preserve_output_interface",
            "enable_cache",
            "allow_remote_code",
        ):
            boolean(getattr(self, name), f"lfm3_adapter.{name}")
        if self.integration_mode not in ("explicit", "hook"):
            raise ValidationError(
                f"Unsupported lfm3_adapter.integration_mode: {self.integration_mode}; "
                "must be explicit or hook"
            )
        if not isinstance(self.fusion_locations, (tuple, list)):
            raise ValidationError("lfm3_adapter.fusion_locations must be a sequence of ints")
        for idx in self.fusion_locations:
            positive_int(idx, "fusion_location", minimum=0)
        object.__setattr__(self, "fusion_locations", tuple(self.fusion_locations))


@dataclass(frozen=True, slots=True)
class RutamModelConfig:
    backbone: str = "LiquidAI/LFM3-1.7B-Base"
    graph_enabled: bool = True
    graph_encoder_enabled: bool = True
    graph_selector_enabled: bool = True
    graph_traversal_enabled: bool = True
    graph_fusion_enabled: bool = True
    memory_write_enabled: bool = True
    memory_auto_commit: bool = False
    backbone_frozen: bool = True
    recurrent_loops_enabled: bool = False
    use_cached_graph_states: bool = True
    return_memory_diagnostics: bool = False
    query_layer_idx: int = 1
    fusion_layer_idx: int = 1
    mode: str = "graph_native"
    sparse_selection_enabled: bool = True
    max_sparse_nodes: int = 64
    sparse_max_hops: int = 2

    def __post_init__(self) -> None:
        nonempty_string(self.backbone, "rutam_model.backbone")
        for name in (
            "graph_enabled",
            "graph_encoder_enabled",
            "graph_selector_enabled",
            "graph_traversal_enabled",
            "graph_fusion_enabled",
            "memory_write_enabled",
            "memory_auto_commit",
            "backbone_frozen",
            "recurrent_loops_enabled",
            "use_cached_graph_states",
            "return_memory_diagnostics",
            "sparse_selection_enabled",
        ):
            boolean(getattr(self, name), f"rutam_model.{name}")
        positive_int(self.query_layer_idx, "rutam_model.query_layer_idx", minimum=0)
        positive_int(self.fusion_layer_idx, "rutam_model.fusion_layer_idx", minimum=0)
        positive_int(self.max_sparse_nodes, "rutam_model.max_sparse_nodes", minimum=1)
        positive_int(self.sparse_max_hops, "rutam_model.sparse_max_hops", minimum=0)
        if self.mode not in ("baseline", "graph_native", "recurrent", "full_rutam"):
            raise ValidationError(
                "rutam_model.mode must be one of baseline, graph_native, "
                f"recurrent, full_rutam, got {self.mode}"
            )


@dataclass(frozen=True, slots=True)
class RecurrentConfig:
    """Settings for Dory-inspired parameter-shared recurrent reasoning."""

    enabled: bool = False
    hidden_dim: int = 256
    num_heads: int = 4
    intermediate_dim: int = 1024
    max_loops: int = 8
    min_loops: int = 1
    default_loops: int = 3
    dynamic_halting: bool = False
    halting_threshold: float = 0.5
    halt_exploration_prob: float = 0.0
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    residual_scale_init: float = 0.001
    use_graph_conditioning: bool = True
    graph_dim: int = 128
    loop_embedding_dim: int = 256
    gate_type: str = "elementwise"
    activation: str = "gelu"
    causal: bool = True
    controller_hidden_dim: int = 64

    def __post_init__(self) -> None:
        boolean(self.enabled, "recurrent.enabled")
        for name in (
            "hidden_dim",
            "num_heads",
            "intermediate_dim",
            "max_loops",
            "min_loops",
            "default_loops",
            "graph_dim",
            "loop_embedding_dim",
            "controller_hidden_dim",
        ):
            positive_int(getattr(self, name), f"recurrent.{name}")
        if self.min_loops > self.max_loops:
            raise ValidationError("recurrent.min_loops cannot exceed max_loops")
        if not (self.min_loops <= self.default_loops <= self.max_loops):
            raise ValidationError("recurrent.default_loops must be in [min_loops, max_loops]")
        if self.hidden_dim % self.num_heads != 0:
            raise ValidationError("recurrent.hidden_dim must be divisible by num_heads")
        boolean(self.dynamic_halting, "recurrent.dynamic_halting")
        if (
            isinstance(self.halting_threshold, bool)
            or not isinstance(self.halting_threshold, (int, float))
            or not 0.0 < self.halting_threshold < 1.0
            or not math.isfinite(self.halting_threshold)
        ):
            raise ValidationError("recurrent.halting_threshold must be a finite number in (0, 1)")
        if (
            isinstance(self.halt_exploration_prob, bool)
            or not isinstance(self.halt_exploration_prob, (int, float))
            or not 0.0 <= self.halt_exploration_prob < 1.0
            or not math.isfinite(self.halt_exploration_prob)
        ):
            raise ValidationError("recurrent.halt_exploration_prob must be in [0, 1)")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= self.dropout < 1.0
            or not math.isfinite(self.dropout)
        ):
            raise ValidationError("recurrent.dropout must be in [0, 1)")
        if (
            isinstance(self.residual_scale_init, bool)
            or not isinstance(self.residual_scale_init, (int, float))
            or not math.isfinite(self.residual_scale_init)
            or self.residual_scale_init <= 0
        ):
            raise ValidationError("recurrent.residual_scale_init must be a positive finite float")
        if self.gate_type not in ("scalar", "elementwise"):
            raise ValidationError("recurrent.gate_type must be scalar or elementwise")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValidationError("recurrent.activation must be gelu, relu or silu")
        boolean(self.use_graph_conditioning, "recurrent.use_graph_conditioning")
        boolean(self.causal, "recurrent.causal")


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    base_dir: str = "checkpoints"
    run_id: str = "run_default"
    save_optimizer: bool = True
    save_graph: bool = True
    keep_last_n: int = 5
    verify_on_save: bool = True
    verify_on_load: bool = True

    def __post_init__(self) -> None:
        nonempty_string(self.base_dir, "checkpoint.base_dir")
        nonempty_string(self.run_id, "checkpoint.run_id")
        for name in ("save_optimizer", "save_graph", "verify_on_save", "verify_on_load"):
            boolean(getattr(self, name), f"checkpoint.{name}")
        positive_int(self.keep_last_n, "checkpoint.keep_last_n")


@dataclass(frozen=True, slots=True)
class RUTAMConfig:
    project_name: str = "RUTAM"
    backbone: str = "LiquidAI/LFM3-1.7B-Base"
    graph: GraphConfig = field(default_factory=GraphConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    graph_attention: GraphAttentionConfig = field(default_factory=GraphAttentionConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    query_head: QueryHeadConfig = field(default_factory=QueryHeadConfig)
    traversal: TraversalConfig = field(default_factory=TraversalConfig)
    linker: LinkerConfig = field(default_factory=LinkerConfig)
    memory_write: MemoryWriteConfig = field(default_factory=MemoryWriteConfig)
    memory_validation: MemoryValidationConfig = field(default_factory=MemoryValidationConfig)
    memory_controller: MemoryControllerConfig = field(default_factory=MemoryControllerConfig)
    memory_learning: MemoryLearningConfig = field(default_factory=MemoryLearningConfig)
    graph_fusion: GraphFusionConfig = field(default_factory=GraphFusionConfig)
    lfm3_adapter: LFM3AdapterConfig = field(default_factory=LFM3AdapterConfig)
    rutam_model: RutamModelConfig = field(default_factory=RutamModelConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        nonempty_string(self.project_name, "project_name")
        nonempty_string(self.backbone, "backbone")
        for name, kind in (
            ("graph", GraphConfig),
            ("memory", MemoryConfig),
            ("runtime", RuntimeConfig),
            ("encoder", EncoderConfig),
            ("graph_attention", GraphAttentionConfig),
            ("selector", SelectorConfig),
            ("query_head", QueryHeadConfig),
            ("traversal", TraversalConfig),
            ("linker", LinkerConfig),
            ("memory_write", MemoryWriteConfig),
            ("memory_validation", MemoryValidationConfig),
            ("memory_controller", MemoryControllerConfig),
            ("memory_learning", MemoryLearningConfig),
            ("graph_fusion", GraphFusionConfig),
            ("lfm3_adapter", LFM3AdapterConfig),
            ("rutam_model", RutamModelConfig),
            ("recurrent", RecurrentConfig),
            ("checkpoint", CheckpointConfig),
        ):
            if not isinstance(getattr(self, name), kind):
                raise ValidationError(f"{name} must be a {kind.__name__}")
        if (
            self.rutam_model.sparse_selection_enabled
            and self.selector.selection_mode == "hard"
            and not self.selector.return_attention
            and self.rutam_model.max_sparse_nodes < self.selector.top_k
        ):
            raise ValidationError(
                "rutam_model.max_sparse_nodes must be at least selector.top_k "
                "for sparse hard selection"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=True)

    def with_overrides(self, overrides: Mapping[str, Any]) -> RUTAMConfig:
        return self.from_dict(_merge(self.to_dict(), overrides))

    def with_mode(self, mode: str) -> RUTAMConfig:
        """Return a new configuration with flags configured for specified architecture mode.

        Supported modes:
        - baseline: No graph memory querying/fusion, no recurrent reasoning.
        - graph_native: Graph memory querying and fusion active, recurrent reasoning disabled.
        - recurrent: Graph memory disabled, recurrent reasoning active.
        - full_rutam: Both graph memory and recurrent reasoning active.
        """
        valid_modes = ("baseline", "graph_native", "recurrent", "full_rutam")
        if mode not in valid_modes:
            raise ValidationError(
                f"Invalid model mode: {mode!r}; must be one of {valid_modes}"
            )
        if mode == "baseline":
            return self.with_overrides({
                "rutam_model": {
                    "graph_enabled": False,
                    "graph_encoder_enabled": False,
                    "graph_selector_enabled": False,
                    "graph_traversal_enabled": False,
                    "graph_fusion_enabled": False,
                    "memory_write_enabled": False,
                    "recurrent_loops_enabled": False,
                    "mode": "baseline",
                },
                "recurrent": {"enabled": False},
            })
        elif mode == "graph_native":
            return self.with_overrides({
                "rutam_model": {
                    "graph_enabled": True,
                    "graph_encoder_enabled": True,
                    "graph_selector_enabled": True,
                    "graph_traversal_enabled": True,
                    "graph_fusion_enabled": True,
                    "memory_write_enabled": True,
                    "recurrent_loops_enabled": False,
                    "mode": "graph_native",
                },
                "recurrent": {"enabled": False},
            })
        elif mode == "recurrent":
            return self.with_overrides({
                "rutam_model": {
                    "graph_enabled": False,
                    "graph_encoder_enabled": False,
                    "graph_selector_enabled": False,
                    "graph_traversal_enabled": False,
                    "graph_fusion_enabled": False,
                    "memory_write_enabled": False,
                    "recurrent_loops_enabled": True,
                    "mode": "recurrent",
                },
                "recurrent": {"enabled": True},
            })
        elif mode == "full_rutam":
            return self.with_overrides({
                "rutam_model": {
                    "graph_enabled": True,
                    "graph_encoder_enabled": True,
                    "graph_selector_enabled": True,
                    "graph_traversal_enabled": True,
                    "graph_fusion_enabled": True,
                    "memory_write_enabled": True,
                    "recurrent_loops_enabled": True,
                    "mode": "full_rutam",
                },
                "recurrent": {"enabled": True},
            })
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RUTAMConfig:
        if not isinstance(data, Mapping):
            raise ValidationError("Configuration must be a mapping")
        values = dict(data)
        try:
            for name, kind in (
                ("graph", GraphConfig),
                ("memory", MemoryConfig),
                ("runtime", RuntimeConfig),
                ("encoder", EncoderConfig),
                ("graph_attention", GraphAttentionConfig),
                ("selector", SelectorConfig),
                ("query_head", QueryHeadConfig),
                ("traversal", TraversalConfig),
                ("linker", LinkerConfig),
                ("memory_write", MemoryWriteConfig),
                ("memory_validation", MemoryValidationConfig),
                ("memory_controller", MemoryControllerConfig),
                ("memory_learning", MemoryLearningConfig),
                ("graph_fusion", GraphFusionConfig),
                ("lfm3_adapter", LFM3AdapterConfig),
                ("rutam_model", RutamModelConfig),
                ("recurrent", RecurrentConfig),
                ("checkpoint", CheckpointConfig),
            ):
                if name in values:
                    if not isinstance(values[name], Mapping):
                        raise ValidationError(f"{name} must be a mapping")
                    values[name] = kind(**values[name])
            return cls(**values)
        except TypeError as exc:
            raise ValidationError(f"Unknown or invalid configuration field: {exc}") from exc

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, overrides: Mapping[str, Any] | None = None
    ) -> RUTAMConfig:
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValidationError(f"Cannot load configuration {path}: {exc}") from exc
        config = cls.from_dict({} if data is None else data)
        return config if overrides is None else config.with_overrides(overrides)


def _merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, Mapping):
        raise ValidationError("Overrides must be a mapping")
    for key, value in overrides.items():
        if key not in base:
            raise ValidationError(f"Unknown configuration field: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, Mapping):
                raise ValidationError(f"{key} override must be a mapping")
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base
