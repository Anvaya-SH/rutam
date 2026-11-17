"""Deterministic CPU evaluation metrics for memory decisions, operations, and retrieval."""

from __future__ import annotations

import torch


def binary_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision, recall, F1, accuracy, and false positive rate.

    Args:
        predictions: probabilities or binary 0/1 predictions.
        targets: binary 0/1 ground truth.
        threshold: classification threshold.
    """
    if predictions.numel() == 0 or targets.numel() == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0,
        }

    preds = (predictions.view(-1) >= threshold).long()
    trues = targets.view(-1).long()

    tp = float(((preds == 1) & (trues == 1)).sum().item())
    fp = float(((preds == 1) & (trues == 0)).sum().item())
    fn = float(((preds == 0) & (trues == 1)).sum().item())
    tn = float(((preds == 0) & (trues == 0)).sum().item())

    accuracy = (tp + tn) / max(1.0, tp + fp + fn + tn)
    precision = tp / max(1.0, tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / max(1.0, tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / max(1e-9, precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / max(1.0, fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def multi_class_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for multi-class classification."""
    if predictions.numel() == 0 or targets.numel() == 0:
        return 0.0
    if predictions.dim() > 1:
        preds = predictions.argmax(dim=-1)
    else:
        preds = predictions
    correct = (preds.view(-1) == targets.view(-1)).float().sum().item()
    return correct / max(1, targets.numel())


def recall_at_k(
    logits: torch.Tensor, target_indices: torch.Tensor, k: int = 1
) -> float:
    """Compute Recall@K for retrieval ranking."""
    if logits.numel() == 0 or target_indices.numel() == 0:
        return 0.0
    b, num_candidates = logits.shape
    k = min(k, num_candidates)
    top_k = logits.topk(k, dim=-1).indices  # [B, K]
    targets = target_indices.view(-1, 1)  # [B, 1]
    hits = (top_k == targets).any(dim=-1).float().sum().item()
    return hits / max(1, b)


def mean_reciprocal_rank(logits: torch.Tensor, target_indices: torch.Tensor) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    if logits.numel() == 0 or target_indices.numel() == 0:
        return 0.0
    b, _ = logits.shape
    sorted_indices = logits.argsort(dim=-1, descending=True)
    targets = target_indices.view(-1, 1)
    ranks = (sorted_indices == targets).nonzero()[:, 1] + 1  # 1-indexed
    rr = (1.0 / ranks.float()).sum().item()
    return rr / max(1, b)
