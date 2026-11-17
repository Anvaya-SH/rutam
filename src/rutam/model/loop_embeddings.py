"""Learned loop position embeddings for recurrent reasoning iterations."""

from __future__ import annotations

import torch
from torch import nn

from rutam.config import RecurrentConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_tensor


class LoopPositionEmbedding(nn.Module):
    """Learned positional embeddings distinguishing recurrence loop iterations t=0..max_loops-1.

    In parameter-shared recurrent architectures (such as Dory or Universal Transformers),
    the same block weights are reused across recurrent iterations. Loop embeddings provide
    an explicit inductive bias allowing the network to distinguish early pondering (t=0)
    from deep multi-hop reasoning (t=3 or higher).
    """

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        if not isinstance(config, RecurrentConfig):
            raise ValidationError("LoopPositionEmbedding requires a RecurrentConfig")
        self.config = config

        self.embeddings = nn.Embedding(config.max_loops, config.loop_embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

        if config.loop_embedding_dim != config.hidden_dim:
            self.proj = nn.Linear(config.loop_embedding_dim, config.hidden_dim, bias=False)
        else:
            self.proj = None

    def get_embedding(self, loop_idx: int) -> torch.Tensor:
        """Retrieve embedding for a specific loop index as shape [1, 1, D]."""
        if type(loop_idx) is not int:
            raise ValidationError(f"loop_idx must be an integer, got {type(loop_idx).__name__}")
        if not (0 <= loop_idx < self.config.max_loops):
            raise ValidationError(
                f"loop_idx {loop_idx} out of range [0, {self.config.max_loops})"
            )

        idx_tensor = torch.tensor([loop_idx], dtype=torch.long, device="cpu")
        emb = self.embeddings(idx_tensor)  # [1, E]
        if self.proj is not None:
            emb = self.proj(emb)  # [1, D]
        return emb.unsqueeze(0)  # [1, 1, D]

    def forward(
        self,
        loop_idx: int | torch.Tensor,
        *,
        batch_size: int | None = None,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Compute loop embedding tensor for the given loop index.

        Args:
            loop_idx: Integer loop index or 1D long tensor of loop indices.
            batch_size: Optional batch size to expand to [B, T, D].
            seq_len: Optional sequence length to expand to [B, T, D].

        Returns:
            Tensor of shape [1, 1, D] or [B, T, D].
        """
        if isinstance(loop_idx, int):
            emb = self.get_embedding(loop_idx)  # [1, 1, D]
            if batch_size is not None and seq_len is not None:
                return emb.expand(batch_size, seq_len, self.config.hidden_dim)
            return emb

        if isinstance(loop_idx, torch.Tensor):
            validate_tensor(loop_idx, "loop_idx", torch.long)
            if loop_idx.ndim == 0:
                idx = int(loop_idx.item())
                return self.forward(idx, batch_size=batch_size, seq_len=seq_len)
            if loop_idx.ndim == 1:
                if (loop_idx < 0).any() or (loop_idx >= self.config.max_loops).any():
                    raise ValidationError(
                        f"loop_idx tensor values must be in [0, {self.config.max_loops})"
                    )
                emb = self.embeddings(loop_idx)  # [B, E]
                if self.proj is not None:
                    emb = self.proj(emb)  # [B, D]
                emb = emb.unsqueeze(1)  # [B, 1, D]
                if seq_len is not None:
                    emb = emb.expand(loop_idx.shape[0], seq_len, self.config.hidden_dim)
                return emb
            raise ValidationError("loop_idx tensor must be scalar or 1D [B]")

        raise ValidationError("loop_idx must be an integer or torch.long tensor")
