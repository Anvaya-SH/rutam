"""Loss functions for memory write decisions, operations, embeddings, retrieval, and links."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor


def write_decision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    pos_weight: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute binary cross-entropy loss for memory write gate decisions.

    Args:
        logits: [B] or [B, 1] raw gate logits.
        targets: [B] or [B, 1] float32 binary labels (1.0 = remember, 0.0 = skip).
        mask: Optional [B] or [B, 1] boolean mask of valid supervised examples.
        pos_weight: Optional positive class weighting for class imbalance.

    Returns:
        loss: scalar tensor.
        valid_count: number of supervised items contributing to the loss.
    """
    validate_tensor(logits, "write_decision logits", torch.float32)
    validate_tensor(targets, "write_decision targets", torch.float32)

    flat_logits = logits.view(-1)
    flat_targets = targets.view(-1)

    if mask is not None:
        flat_mask = mask.view(-1).bool()
        flat_logits = flat_logits[flat_mask]
        flat_targets = flat_targets[flat_mask]

    valid_count = flat_logits.numel()
    if valid_count == 0:
        return logits.new_tensor(0.0), 0

    pw = (
        torch.tensor(pos_weight, dtype=torch.float32, device=logits.device)
        if isinstance(pos_weight, (int, float))
        else pos_weight
    )

    loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, pos_weight=pw)
    require_finite(loss, "write_decision_loss")
    return loss, valid_count


def operation_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute cross-entropy loss for memory operation classification.

    Args:
        logits: [B, num_operations] float32 operation logits.
        targets: [B] int64 operation class indices.
        mask: Optional [B] boolean mask of supervised examples.
        class_weights: Optional [num_operations] float32 class weights.

    Returns:
        loss: scalar tensor.
        valid_count: number of supervised items.
    """
    validate_tensor(logits, "operation logits", torch.float32)
    validate_tensor(targets, "operation targets", torch.int64)

    if mask is not None:
        flat_mask = mask.view(-1).bool()
        logits = logits[flat_mask]
        targets = targets[flat_mask]

    valid_count = targets.numel()
    if valid_count == 0:
        return logits.new_tensor(0.0), 0

    loss = F.cross_entropy(logits, targets, weight=class_weights)
    require_finite(loss, "operation_classification_loss")
    return loss, valid_count


def embedding_alignment_loss(
    proposed_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    *,
    method: str = "cosine",
    negative_embeddings: torch.Tensor | None = None,
    margin: float = 0.2,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute embedding alignment loss between proposed candidate memory and reference memory.

    Args:
        proposed_embeddings: [B, H] or [B, M, H] proposed memory representations.
        reference_embeddings: [B, H] or [B, M, H] ground-truth reference representations.
        method: "cosine" or "contrastive" or "triplet".
        negative_embeddings: Optional [B, H] negative reference embeddings for contrastive/triplet.
        margin: Margin threshold for contrastive/triplet objectives.
        mask: Optional boolean mask of valid aligned slots.

    Returns:
        loss: scalar tensor.
        valid_count: number of aligned slots.
    """
    validate_tensor(proposed_embeddings, "proposed_embeddings", torch.float32)
    validate_tensor(reference_embeddings, "reference_embeddings", torch.float32)

    p_flat = proposed_embeddings.view(-1, proposed_embeddings.shape[-1])
    r_flat = reference_embeddings.view(-1, reference_embeddings.shape[-1])

    if p_flat.shape[0] != r_flat.shape[0]:
        raise ValidationError(
            f"Embedding slot mismatch: {p_flat.shape[0]} proposed vs {r_flat.shape[0]} reference"
        )

    if mask is not None:
        flat_mask = mask.view(-1).bool()
        p_flat = p_flat[flat_mask]
        r_flat = r_flat[flat_mask]

    valid_count = p_flat.shape[0]
    if valid_count == 0:
        return proposed_embeddings.new_tensor(0.0), 0

    p_norm = F.normalize(p_flat, p=2, dim=-1)
    r_norm = F.normalize(r_flat, p=2, dim=-1)

    cos_sim = (p_norm * r_norm).sum(dim=-1)  # [N]
    cos_loss = (1.0 - cos_sim).mean()

    if method == "cosine" or negative_embeddings is None:
        require_finite(cos_loss, "embedding_alignment_loss")
        return cos_loss, valid_count

    neg_flat = negative_embeddings.view(-1, negative_embeddings.shape[-1])
    if mask is not None:
        neg_flat = neg_flat[mask.view(-1).bool()]
    neg_norm = F.normalize(neg_flat, p=2, dim=-1)
    neg_sim = (p_norm * neg_norm).sum(dim=-1)

    triplet_loss = F.relu(neg_sim - cos_sim + margin).mean()
    total_loss = cos_loss + triplet_loss
    require_finite(total_loss, "embedding_triplet_loss")
    return total_loss, valid_count


def graph_retrieval_loss(
    relevance_logits: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    multi_positive_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute retrieval loss over graph nodes or candidates.

    Args:
        relevance_logits: [B, K] unnormalized relevance scores over K candidates.
        target_indices: [B] target index for single-positive retrieval.
        mask: Optional [B] boolean mask.
        multi_positive_mask: Optional [B, K] float32 binary multi-label targets.

    Returns:
        loss: scalar tensor.
        valid_count: number of supervised items.
    """
    validate_tensor(relevance_logits, "relevance_logits", torch.float32)

    if multi_positive_mask is not None:
        validate_tensor(multi_positive_mask, "multi_positive_mask", torch.float32)
        if mask is not None:
            flat_m = mask.view(-1).bool()
            relevance_logits = relevance_logits[flat_m]
            multi_positive_mask = multi_positive_mask[flat_m]
        valid_count = relevance_logits.shape[0]
        if valid_count == 0:
            return relevance_logits.new_tensor(0.0), 0
        loss = F.binary_cross_entropy_with_logits(relevance_logits, multi_positive_mask)
        require_finite(loss, "multi_positive_retrieval_loss")
        return loss, valid_count

    validate_tensor(target_indices, "target_indices", torch.int64)
    if mask is not None:
        flat_m = mask.view(-1).bool()
        relevance_logits = relevance_logits[flat_m]
        target_indices = target_indices[flat_m]

    valid_count = target_indices.numel()
    if valid_count == 0:
        return relevance_logits.new_tensor(0.0), 0

    loss = F.cross_entropy(relevance_logits, target_indices)
    require_finite(loss, "single_positive_retrieval_loss")
    return loss, valid_count


def query_gate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute binary cross-entropy loss for query gate decision (query vs do-not-query)."""
    return write_decision_loss(logits, targets, mask=mask)


def update_target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute cross-entropy loss identifying which existing node to update."""
    return operation_classification_loss(logits, targets, mask=mask)
