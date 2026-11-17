"""Trainable node projection and bounded relationship vocabulary on CPU."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import EncoderConfig, positive_int
from rutam.exceptions import ValidationError


def activation_module(name: str) -> nn.Module:
    """Return a fresh stateless activation; configuration rejects unknown names."""
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValidationError("activation must be gelu, relu or silu")


def validate_tensor(tensor: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != dtype
        or tensor.layout != torch.strided
    ):
        raise ValidationError(f"{name} must be a dense CPU {dtype} tensor")


def require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValidationError(f"{name} contains nonfinite values; check feature/weight scale")


def validate_features(features: torch.Tensor, width: int, name: str) -> None:
    validate_tensor(features, name, torch.float32)
    if features.ndim != 2 or features.shape[1] != width:
        raise ValidationError(f"{name} must have shape [N, {width}]")
    require_finite(features, name)


def validate_relation_ids(relation_ids: torch.Tensor, num_types: int) -> None:
    validate_tensor(relation_ids, "relation IDs", torch.int64)
    if relation_ids.ndim != 1:
        raise ValidationError("relation IDs must have shape [E]")
    if relation_ids.numel() and (
        relation_ids.min().item() < 0 or relation_ids.max().item() >= num_types
    ):
        raise ValidationError(f"relation IDs must be in [0, {num_types})")


class NodeProjection(nn.Module):
    """Linear -> optional LayerNorm -> activation -> dropout, [N, D] to [N, H]."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, EncoderConfig):
            raise ValidationError("NodeProjection requires an EncoderConfig")
        self.config = config
        self.linear = nn.Linear(
            config.input_dim, config.hidden_dim, device="cpu", dtype=torch.float32
        )
        self.norm = (
            nn.LayerNorm(config.hidden_dim, device="cpu", dtype=torch.float32)
            if config.layer_norm
            else nn.Identity()
        )
        self.activation = activation_module(config.activation)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        validate_features(node_embeddings, self.config.input_dim, "node_embeddings")
        validate_tensor(self.linear.weight, "projection parameters", torch.float32)
        projected = self.linear(node_embeddings)
        require_finite(projected, "projected features")
        result = self.dropout(self.activation(self.norm(projected)))
        require_finite(result, "node projection output")
        return result

    def forward_for_selection(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Project candidate features without dropout before discrete top-k ranking.

        Candidate ranking and selected-score recomputation must see the same
        features. Dropout stays in the relational encoder after selection.
        """
        validate_features(node_embeddings, self.config.input_dim, "node_embeddings")
        validate_tensor(self.linear.weight, "projection parameters", torch.float32)
        projected = self.linear(node_embeddings)
        require_finite(projected, "projected candidate features")
        result = self.activation(self.norm(projected))
        require_finite(result, "candidate feature projection")
        return result


class RelationEmbedding(nn.Embedding):
    """A trainable [R, H] vocabulary with explicit CPU/int64 ID validation."""

    def __init__(self, num_relation_types: int, hidden_dim: int) -> None:
        positive_int(num_relation_types, "num_relation_types")
        positive_int(hidden_dim, "hidden_dim")
        super().__init__(num_relation_types, hidden_dim, device="cpu", dtype=torch.float32)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        validate_relation_ids(relation_ids, self.num_embeddings)
        validate_tensor(self.weight, "relation embedding parameters", torch.float32)
        result = super().forward(relation_ids)
        require_finite(result, "relation embeddings")
        return result
