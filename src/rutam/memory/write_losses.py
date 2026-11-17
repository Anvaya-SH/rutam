"""Supervised multi-objective loss calculation for memory write head training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rutam.config import bounded_float
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor
from rutam.memory.write_gate import availability_mask


@dataclass(frozen=True, slots=True)
class MemoryWriteLossOutput:
    """Supervised loss components for memory write head."""

    total: torch.Tensor
    write: torch.Tensor
    operation: torch.Tensor
    embedding: torch.Tensor
    target: torch.Tensor


def memory_write_loss(
    write_logits: torch.Tensor,
    write_targets: torch.Tensor,
    operation_logits: torch.Tensor,
    operation_targets: torch.Tensor | None = None,
    proposed_embeddings: torch.Tensor | None = None,
    target_embeddings: torch.Tensor | None = None,
    target_memory_scores: torch.Tensor | None = None,
    target_node_labels: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    mask: torch.Tensor | None = None,
    active_operations_only: bool = True,
) -> MemoryWriteLossOutput:
    """Calculate multi-task loss L = alpha*L_write + beta*L_op + gamma*L_emb + delta*L_target.

    Parameters:
        write_logits: [B, 1] raw write decision logits.
        write_targets: [B] or [B, 1] binary write labels (0 or 1).
        operation_logits: [B, O] raw logits over operation classes.
        operation_targets: optional [B] int64 class indices (0..O-1) or [B, O] probabilities.
        proposed_embeddings: optional [B, M, H] or [B, H] candidate embeddings.
        target_embeddings: optional [B, H] or [B, M, H] supervised target vectors.
        target_memory_scores: optional [B, K] compatibility scores over candidate nodes.
        target_node_labels: optional [B] int64 index or [B, K] float32 binary indicators.
        alpha, beta, gamma, delta: nonnegative loss component coefficients.
        mask: optional [B] sample inclusion mask.
        active_operations_only: if True, evaluate op and embedding losses on positive write samples.
    """
    for weight, name in (
        (alpha, "alpha"),
        (beta, "beta"),
        (gamma, "gamma"),
        (delta, "delta"),
    ):
        bounded_float(weight, f"memory_write_loss.{name}")

    validate_tensor(write_logits, "write_logits", torch.float32)
    if write_logits.ndim != 2 or write_logits.shape[1] != 1:
        raise ValidationError("write_logits must have shape [B, 1]")
    require_finite(write_logits, "write_logits")
    b = write_logits.shape[0]

    # Validate write_targets
    if not isinstance(write_targets, torch.Tensor) or write_targets.dtype not in (
        torch.bool,
        torch.int64,
        torch.float32,
    ):
        raise ValidationError("write_targets must be boolean, int64 or float32")
    validate_tensor(write_targets, "write_targets", write_targets.dtype)
    if write_targets.shape == (b,):
        write_targets = write_targets[:, None]
    if write_targets.shape != (b, 1):
        raise ValidationError("write_targets must have shape [B] or [B, 1]")
    if not ((write_targets == 0) | (write_targets == 1)).all().item():
        raise ValidationError("write_targets must be binary (0 or 1)")

    active = availability_mask(mask, b)

    # 1. L_write: BCEWithLogits
    if active.any().item():
        l_write = F.binary_cross_entropy_with_logits(
            write_logits[active], write_targets[active].float()
        )
    else:
        l_write = (write_logits * 0).sum()
    require_finite(l_write, "l_write")

    # 2. L_operation: CrossEntropy
    validate_tensor(operation_logits, "operation_logits", torch.float32)
    if operation_logits.ndim != 2 or operation_logits.shape[0] != b:
        raise ValidationError("operation_logits must have shape [B, O]")
    require_finite(operation_logits, "operation_logits")

    if operation_targets is not None:
        validate_tensor(operation_targets, "operation_targets", operation_targets.dtype)
        if operation_targets.ndim == 1:
            if operation_targets.shape != (b,):
                raise ValidationError("1D operation_targets must have shape [B]")
            if operation_targets.dtype not in (torch.int64, torch.int32):
                raise ValidationError("1D operation_targets must be integer class indices")
            if (
                operation_targets.min().item() < 0
                or operation_targets.max().item() >= operation_logits.shape[1]
            ):
                raise ValidationError(
                    f"operation_targets out of range [0, {operation_logits.shape[1] - 1}]"
                )
        elif operation_targets.ndim == 2:
            if operation_targets.shape != operation_logits.shape:
                raise ValidationError("2D operation_targets must have shape [B, O]")
        else:
            raise ValidationError("operation_targets must have 1 or 2 dimensions")

        if active_operations_only:
            op_active = active & (write_targets.squeeze(1) > 0)
        else:
            op_active = active

        if op_active.any().item():
            l_operation = F.cross_entropy(operation_logits[op_active], operation_targets[op_active])
        else:
            l_operation = (operation_logits * 0).sum()
    else:
        l_operation = (operation_logits * 0).sum()
    require_finite(l_operation, "l_operation")

    # 3. L_embedding: Cosine distance
    if proposed_embeddings is not None and target_embeddings is not None:
        validate_tensor(proposed_embeddings, "proposed_embeddings", torch.float32)
        validate_tensor(target_embeddings, "target_embeddings", torch.float32)
        p = proposed_embeddings[:, 0, :] if proposed_embeddings.ndim == 3 else proposed_embeddings
        t = target_embeddings[:, 0, :] if target_embeddings.ndim == 3 else target_embeddings
        if p.ndim != 2 or t.ndim != 2 or p.shape != t.shape or p.shape[0] != b:
            raise ValidationError("proposed and target embeddings must have matching shapes [B, H]")
        require_finite(p, "proposed_embeddings")
        require_finite(t, "target_embeddings")

        if active_operations_only:
            emb_active = active & (write_targets.squeeze(1) > 0)
        else:
            emb_active = active

        if emb_active.any().item():
            cos_sim = F.cosine_similarity(p[emb_active], t[emb_active], dim=-1)
            l_embedding = (1.0 - cos_sim).mean()
        else:
            l_embedding = (p * 0).sum()
    else:
        l_embedding = (write_logits * 0).sum()
    require_finite(l_embedding, "l_embedding")

    # 4. L_target: Target memory selection
    if target_memory_scores is not None and target_node_labels is not None:
        validate_tensor(target_memory_scores, "target_memory_scores", torch.float32)
        validate_tensor(target_node_labels, "target_node_labels", target_node_labels.dtype)
        if target_memory_scores.ndim != 2 or target_memory_scores.shape[0] != b:
            raise ValidationError("target_memory_scores must have shape [B, K]")
        require_finite(target_memory_scores, "target_memory_scores")

        if active_operations_only:
            tgt_active = active & (write_targets.squeeze(1) > 0)
        else:
            tgt_active = active

        if tgt_active.any().item():
            if target_node_labels.ndim == 1:
                if target_node_labels.shape != (b,):
                    raise ValidationError("1D target_node_labels must have shape [B]")
                if target_node_labels.dtype not in (torch.int64, torch.int32):
                    raise ValidationError("1D target_node_labels must be integer class indices")
                l_target = F.cross_entropy(
                    target_memory_scores[tgt_active], target_node_labels[tgt_active]
                )
            elif target_node_labels.ndim == 2:
                if target_node_labels.shape != target_memory_scores.shape:
                    raise ValidationError("2D target_node_labels must have shape [B, K]")
                l_target = F.binary_cross_entropy_with_logits(
                    target_memory_scores[tgt_active], target_node_labels[tgt_active].float()
                )
            else:
                raise ValidationError("target_node_labels must have 1 or 2 dimensions")
        else:
            l_target = (target_memory_scores * 0).sum()
    else:
        l_target = (write_logits * 0).sum()
    require_finite(l_target, "l_target")

    total = alpha * l_write + beta * l_operation + gamma * l_embedding + delta * l_target
    require_finite(total, "total memory write loss")

    return MemoryWriteLossOutput(
        total=total,
        write=l_write,
        operation=l_operation,
        embedding=l_embedding,
        target=l_target,
    )
