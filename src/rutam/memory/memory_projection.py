"""Neural memory projection to candidate slot embeddings and target node compatibility scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rutam.config import MemoryWriteConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor


@dataclass(frozen=True, slots=True)
class MemoryProjectionOutput:
    """Output of candidate memory projection."""

    proposed_embeddings: torch.Tensor
    slot_logits: torch.Tensor
    slot_probabilities: torch.Tensor


class NeuralMemoryProjection(nn.Module):
    """Projects pooled representations to candidate memory slot embeddings [B, M, H].

    Architecture:
        Pooled Hidden State -> Linear(D, Intermediate) -> GELU -> Linear(Intermediate, M*H).
    Reshaped to [B, M, H] and optionally normalized. Slot gates estimate candidate slot validity.
    """

    def __init__(self, config: MemoryWriteConfig) -> None:
        super().__init__()
        if not isinstance(config, MemoryWriteConfig):
            raise ValidationError("NeuralMemoryProjection requires a MemoryWriteConfig")
        self.config = config
        d = config.backbone_hidden_dim
        inter = config.projection_dim
        m = config.max_proposals
        h = config.hidden_dim

        self.linear1 = nn.Linear(d, inter, device="cpu", dtype=torch.float32)
        self.linear2 = nn.Linear(inter, m * h, device="cpu", dtype=torch.float32)
        self.slot_gate = nn.Linear(inter, m, device="cpu", dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize slot 0 prior to high activation so primary proposal is active
        with torch.no_grad():
            self.slot_gate.bias.zero_()
            self.slot_gate.bias[0] = 2.0

    def forward(self, pooled_context: torch.Tensor) -> MemoryProjectionOutput:
        validate_features(pooled_context, self.config.backbone_hidden_dim, "pooled_context")
        require_finite(pooled_context, "pooled_context")
        b = pooled_context.shape[0]

        validate_tensor(self.linear1.weight, "memory projection parameters", torch.float32)
        hidden = self.dropout(F.gelu(self.linear1(pooled_context)))
        require_finite(hidden, "memory projection hidden states")

        flat_embeddings = self.linear2(hidden)
        require_finite(flat_embeddings, "flat proposed embeddings")

        embeddings = flat_embeddings.view(b, self.config.max_proposals, self.config.hidden_dim)
        if self.config.normalize_embeddings:
            embeddings = F.normalize(embeddings.double(), p=2, dim=-1, eps=1e-12).float()
        require_finite(embeddings, "proposed embeddings")

        slot_logits = self.slot_gate(hidden)
        require_finite(slot_logits, "slot logits")
        slot_probs = torch.sigmoid(slot_logits)

        return MemoryProjectionOutput(embeddings, slot_logits, slot_probs)


class TargetMemoryScorer(nn.Module):
    """Trainable compatibility scoring between proposed memory and selected nodes [B, K, H]."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValidationError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(
            hidden_dim, hidden_dim, bias=False, device="cpu", dtype=torch.float32
        )
        self.key_proj = nn.Linear(
            hidden_dim, hidden_dim, bias=False, device="cpu", dtype=torch.float32
        )
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(
        self,
        proposed_embedding: torch.Tensor,
        selected_memory_states: torch.Tensor,
    ) -> torch.Tensor:
        """Score compatibility [B, K] between proposed embedding [B, H] and selected nodes."""
        validate_tensor(proposed_embedding, "proposed_embedding", torch.float32)
        validate_tensor(selected_memory_states, "selected_memory_states", torch.float32)

        if proposed_embedding.ndim == 3:
            # If [B, M, H], take primary proposal [B, H]
            proposed_embedding = proposed_embedding[:, 0, :]
        if proposed_embedding.ndim != 2 or proposed_embedding.shape[1] != self.hidden_dim:
            raise ValidationError(f"proposed_embedding must have shape [B, {self.hidden_dim}]")
        if (
            selected_memory_states.ndim != 3
            or selected_memory_states.shape[0] != proposed_embedding.shape[0]
            or selected_memory_states.shape[2] != self.hidden_dim
        ):
            raise ValidationError(
                f"selected_memory_states must have shape [B, K, {self.hidden_dim}]"
            )
        require_finite(proposed_embedding, "proposed_embedding")
        require_finite(selected_memory_states, "selected_memory_states")

        validate_tensor(self.query_proj.weight, "target scorer parameters", torch.float32)
        q = self.query_proj(proposed_embedding)  # [B, H]
        k = self.key_proj(selected_memory_states)  # [B, K, H]

        scores = (q.unsqueeze(1) * k).sum(dim=-1) * self.scale  # [B, K]
        require_finite(scores, "target memory scores")
        return scores
