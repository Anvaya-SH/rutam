"""Runtime checkable protocol defining architectural expectations for LM backbones."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class BackboneProtocol(Protocol):
    """Protocol for language model backbones integrated with RUTAM graph memory.

    Enables inspection of hidden dimensions, layer counts, input embeddings,
    intermediate decoder layers, and output logit projection without hardcoded
    assumptions about private model internals.
    """

    def get_hidden_dim(self) -> int:
        """Return the backbone hidden representation dimension D."""
        ...

    def get_vocab_size(self) -> int:
        """Return the backbone vocabulary size."""
        ...

    def get_num_layers(self) -> int:
        """Return total count of transformer/hybrid decoder layers."""
        ...

    def get_input_embeddings(self) -> nn.Module:
        """Return token embedding module."""
        ...

    def get_layers(self) -> Sequence[nn.Module]:
        """Return ordered sequence of decoder layer modules."""
        ...

    def get_output_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project final hidden states [B, T, D] to unnormalized vocabulary logits [B, T, V]."""
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute forward pass over token IDs."""
        ...
