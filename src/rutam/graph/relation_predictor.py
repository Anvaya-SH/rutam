"""Versioned relation vocabulary and directed pair classifier, including NO_RELATION."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import LinkerConfig, boolean, nonempty_string, positive_int
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor
from rutam.graph.link_scoring import PairFeatureNetwork, _loss_mask, copy_expanded_parameters
from rutam.graph.state import GraphState


@dataclass(frozen=True, slots=True)
class RelationVocabulary:
    """Classifier class 0=NO_RELATION; real class r+1 corresponds to persistent graph ID r."""

    relation_labels: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        positive_int(self.schema_version, "relation schema version")
        if not isinstance(self.relation_labels, (tuple, list)):
            raise ValidationError("Explicit relation label sequence required")
        labels = tuple(self.relation_labels)
        for label in labels:
            nonempty_string(label, "relation label")
        if not labels or "NO_RELATION" in labels or len(set(labels)) != len(labels):
            raise ValidationError(
                "Real labels must be nonempty, unique and exclude reserved NO_RELATION"
            )
        object.__setattr__(self, "relation_labels", labels)

    @property
    def labels(self) -> tuple[str, ...]:
        return ("NO_RELATION", *self.relation_labels)

    def name(self, class_id: int) -> str:
        positive_int(class_id, "relation class ID", minimum=0)
        if class_id >= len(self.labels):
            raise ValidationError("Unknown relation class ID")
        return self.labels[class_id]

    def class_id(self, name: str) -> int:
        nonempty_string(name, "relation name")
        try:
            return self.labels.index(name)
        except ValueError as exc:
            raise ValidationError("Unknown relation name; use explicit schema migration") from exc

    def graph_relation_id(self, class_id: int) -> int | None:
        self.name(class_id)
        return None if class_id == 0 else class_id - 1

    def validate_graph_mapping(self, mapping: dict[str, int]) -> None:
        for label, relation in mapping.items():
            if (
                type(relation) is not int
                or relation < 0
                or relation >= len(self.relation_labels)
                or self.relation_labels[relation] != label
            ):
                raise ValidationError(
                    "Graph relation IDs/names do not match explicit linker vocabulary"
                )

    def extend(self, labels: tuple[str, ...]) -> RelationVocabulary:
        """Append only: old graph IDs AND class IDs remain unchanged."""
        if not isinstance(labels, (tuple, list)) or not labels:
            raise ValidationError("Schema migration requires new labels")
        return RelationVocabulary((*self.relation_labels, *labels), self.schema_version + 1)

    @classmethod
    def from_state(cls, state: GraphState) -> RelationVocabulary:
        state.validate()
        mapping = state.relationship_types
        if not mapping:
            raise ValidationError(
                "An empty relation schema needs explicitly configured real labels"
            )
        if set(mapping.values()) != set(range(len(mapping))):
            raise ValidationError(
                "Sparse graph relation IDs need an explicit schema mapping migration"
            )
        return cls(tuple(label for label, _ in sorted(mapping.items(), key=lambda x: x[1])))


class RUTAMRelationPredictor(nn.Module):
    def __init__(self, config: LinkerConfig | None = None) -> None:
        super().__init__()
        self.config = LinkerConfig() if config is None else config
        if not isinstance(self.config, LinkerConfig):
            raise ValidationError("Relation predictor requires LinkerConfig")
        self.vocabulary = RelationVocabulary(
            self.config.relation_labels, self.config.relation_schema_version
        )
        schema = json.dumps(
            {"labels": self.vocabulary.labels, "version": self.vocabulary.schema_version},
            sort_keys=True,
        ).encode()
        self.register_buffer(
            "relation_schema_signature",
            torch.tensor(list(hashlib.sha256(schema).digest()), dtype=torch.uint8, device="cpu"),
        )
        self.features = PairFeatureNetwork(self.config)
        self.output_projection = nn.Linear(
            self.config.scoring_hidden_dim,
            len(self.vocabulary.labels),
            device="cpu",
            dtype=torch.float32,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        pairs: torch.Tensor,
        structural_features: torch.Tensor | None = None,
        query_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.output_projection(
            self.features(nodes, pairs, structural_features, query_vectors)
        )
        require_finite(logits, "relation logits")
        return logits

    def migrate(self, additional_labels: tuple[str, ...]) -> RUTAMRelationPredictor:
        vocabulary = self.vocabulary.extend(additional_labels)
        result = RUTAMRelationPredictor(
            replace(
                self.config,
                relation_labels=vocabulary.relation_labels,
                relation_schema_version=vocabulary.schema_version,
            )
        )
        copy_expanded_parameters(self, result)
        result.train(self.training)
        return result

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        signature = state_dict.get(prefix + "relation_schema_signature")
        if signature is not None and not torch.equal(signature, self.relation_schema_signature):
            error_msgs.append(
                "Relation checkpoint schema differs; "
                "load matching configuration or migrate explicitly"
            )
            return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def relation_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    link_labels: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    allow_unlinked_relations: bool = False,
) -> torch.Tensor:
    validate_tensor(logits, "relation logits", torch.float32)
    if logits.ndim not in (2, 3) or logits.shape[-1] < 2:
        raise ValidationError("Relation logits must be [C,R] or [B,C,R], including NO_RELATION")
    require_finite(logits, "relation logits")
    validate_tensor(targets, "relation targets", torch.int64)
    if targets.shape != logits.shape[:-1]:
        raise ValidationError("Relation target shape differs")
    valid = _loss_mask(targets, mask)
    if ((targets < -1) | (targets >= logits.shape[-1])).any().item() or (
        targets[valid] < 0
    ).any().item():
        raise ValidationError("Unknown relation class target; -1 padding requires a false mask")
    boolean(allow_unlinked_relations, "allow_unlinked_relations")
    if link_labels is not None:
        if not isinstance(link_labels, torch.Tensor) or link_labels.dtype not in (
            torch.bool,
            torch.int64,
            torch.float32,
        ):
            raise ValidationError("Link labels must be binary CPU tensors")
        validate_tensor(link_labels, "link labels", link_labels.dtype)
        if (
            link_labels.shape != targets.shape
            or not ((link_labels == 0) | (link_labels == 1)).all().item()
        ):
            raise ValidationError("Binary link labels must align with relation labels")
        if (
            not allow_unlinked_relations
            and (
                ((link_labels == 0) & (targets > 0) & valid)
                | ((link_labels == 1) & (targets == 0) & valid)
            )
            .any()
            .item()
        ):
            raise ValidationError(
                "Nonexistent links require NO_RELATION; positive links require a real relation"
            )
    if not valid.any().item():
        return (logits * 0).sum()
    loss = F.cross_entropy(logits[valid].double(), targets[valid]).float()
    require_finite(loss, "relation classification loss")
    return loss
