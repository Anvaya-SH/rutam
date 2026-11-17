"""Primary RUTAM memory learning objective coordinator and multi-task loss engine."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rutam.config import MemoryLearningConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite
from rutam.memory.write_proposal import MemoryOperation, MemoryWriteOutput
from rutam.model.query_output import QueryHeadOutput
from rutam.training.learning_output import MemoryLearningOutput
from rutam.training.loss_balancer import create_loss_balancer
from rutam.training.memory_losses import (
    embedding_alignment_loss,
    graph_retrieval_loss,
    operation_classification_loss,
    query_gate_loss,
    update_target_loss,
    write_decision_loss,
)
from rutam.training.memory_metrics import (
    binary_classification_metrics,
    multi_class_accuracy,
    recall_at_k,
)
from rutam.training.memory_targets import MemoryTrainingTarget


class RUTAMMemoryLearningObjective(nn.Module):
    """Joint supervised multi-task objective for native memory training.

    Coordinates loss calculation for write gating, operation selection, slot alignment,
    graph retrieval, relational linking, and query gating.
    """

    def __init__(self, config: MemoryLearningConfig | None = None) -> None:
        super().__init__()
        self.config = MemoryLearningConfig() if config is None else config
        if not isinstance(self.config, MemoryLearningConfig):
            raise ValidationError("RUTAMMemoryLearningObjective requires a MemoryLearningConfig")

        self.balancer = create_loss_balancer(self.config)

    def forward(
        self,
        *,
        write_output: MemoryWriteOutput | None = None,
        query_output: QueryHeadOutput | None = None,
        targets: Sequence[MemoryTrainingTarget] | None = None,
        retrieval_logits: torch.Tensor | None = None,
        retrieval_targets: torch.Tensor | None = None,
        link_loss: torch.Tensor | None = None,
        path_loss: torch.Tensor | None = None,
    ) -> MemoryLearningOutput:
        """Compute combined memory learning losses and diagnostics.

        Returns:
            MemoryLearningOutput containing total loss, loss breakdown, valid counts, and metrics.
        """
        loss_components: dict[str, torch.Tensor] = {}
        valid_counts: dict[str, int] = {}
        metrics: dict[str, float] = {}
        warnings: list[str] = []

        # 1. Write Decision Loss & Metrics
        if write_output is not None and targets is not None:
            write_targets = [t.write_decision for t in targets]
            valid_mask = [w is not None for w in write_targets]
            if any(valid_mask):
                t_tensor = torch.tensor(
                    [float(w) for w, v in zip(write_targets, valid_mask, strict=False) if v],
                    dtype=torch.float32,
                    device=write_output.write_logits.device,
                )
                m_tensor = torch.tensor(
                    valid_mask, dtype=torch.bool, device=write_output.write_logits.device
                )
                w_loss, count = write_decision_loss(
                    write_output.write_logits, t_tensor, mask=m_tensor
                )
                loss_components["write_decision"] = w_loss
                valid_counts["write_decision"] = count
                with torch.no_grad():
                    w_metrics = binary_classification_metrics(
                        write_output.write_probabilities[m_tensor], t_tensor
                    )
                    metrics["write_precision"] = w_metrics["precision"]
                    metrics["write_recall"] = w_metrics["recall"]
                    metrics["write_f1"] = w_metrics["f1"]
                    metrics["false_positive_rate"] = w_metrics["false_positive_rate"]
            else:
                warnings.append("No valid write_decision supervision in batch")

        # 2. Operation Classification Loss & Metrics
        if write_output is not None and targets is not None:
            op_labels = [
                t.operation_label.value
                if isinstance(t.operation_label, MemoryOperation)
                else None
                for t in targets
            ]
            valid_op_mask = [o is not None for o in op_labels]
            if any(valid_op_mask):
                op_tensor = torch.tensor(
                    [o for o, v in zip(op_labels, valid_op_mask, strict=False) if v],
                    dtype=torch.int64,
                    device=write_output.operation_logits.device,
                )
                m_tensor = torch.tensor(
                    valid_op_mask, dtype=torch.bool, device=write_output.operation_logits.device
                )
                op_loss, count = operation_classification_loss(
                    write_output.operation_logits, op_tensor, mask=m_tensor
                )
                loss_components["operation"] = op_loss
                valid_counts["operation"] = count
                with torch.no_grad():
                    metrics["operation_accuracy"] = multi_class_accuracy(
                        write_output.operation_logits[m_tensor], op_tensor
                    )
            else:
                warnings.append("No valid operation supervision in batch")

        # 3. Embedding Alignment Loss
        if write_output is not None and targets is not None:
            refs = [t.reference_embeddings for t in targets]
            if any(r is not None for r in refs):
                valid_ref_indices = [i for i, r in enumerate(refs) if r is not None]
                ref_tensors = torch.stack([refs[i] for i in valid_ref_indices]).to(
                    write_output.proposed_embeddings.device
                )
                prop_tensors = write_output.proposed_embeddings[valid_ref_indices]
                if prop_tensors.dim() == 3 and ref_tensors.dim() == 2:
                    prop_tensors = prop_tensors[:, 0, :]
                emb_loss, count = embedding_alignment_loss(prop_tensors, ref_tensors)
                loss_components["embedding"] = emb_loss
                valid_counts["embedding"] = count

        # 4. Graph Retrieval Loss
        if retrieval_logits is not None and retrieval_targets is not None:
            r_loss, count = graph_retrieval_loss(retrieval_logits, retrieval_targets)
            loss_components["retrieval"] = r_loss
            valid_counts["retrieval"] = count
            with torch.no_grad():
                metrics["retrieval_recall@1"] = recall_at_k(
                    retrieval_logits, retrieval_targets, k=1
                )
                metrics["retrieval_recall@3"] = recall_at_k(
                    retrieval_logits, retrieval_targets, k=3
                )

        # 5. Query Gate Loss
        has_gate_logits = query_output is not None and query_output.query_gate_logits is not None
        if has_gate_logits and targets is not None:
            q_targets = [float(len(t.relevant_node_ids) > 0) for t in targets]
            q_tensor = torch.tensor(
                q_targets, dtype=torch.float32, device=query_output.query_gate_logits.device
            )
            q_loss, count = query_gate_loss(query_output.query_gate_logits, q_tensor)
            loss_components["query_gate"] = q_loss
            valid_counts["query_gate"] = count

        # 6. Relational Linking Loss (External)
        if link_loss is not None:
            loss_components["link"] = link_loss
            valid_counts["link"] = 1

        # 7. Path Reasoning Loss (External)
        if path_loss is not None:
            loss_components["path"] = path_loss
            valid_counts["path"] = 1

        # 8. Update Target Selection Loss
        if (
            write_output is not None
            and write_output.target_memory_scores is not None
            and targets is not None
        ):
            target_node_labels = [
                t.target_node_ids[0] if t.target_node_ids else None
                for t in targets
            ]
            valid_tgt_mask = [tgt is not None for tgt in target_node_labels]
            if any(valid_tgt_mask):
                tgt_tensor = torch.tensor(
                    [tgt for tgt, v in zip(target_node_labels, valid_tgt_mask, strict=False) if v],
                    dtype=torch.int64,
                    device=write_output.target_memory_scores.device,
                )
                device = write_output.target_memory_scores.device
                m_tensor = torch.tensor(valid_tgt_mask, dtype=torch.bool, device=device)
                tgt_loss, count = update_target_loss(
                    write_output.target_memory_scores, tgt_tensor, mask=m_tensor
                )
                loss_components["update_target"] = tgt_loss
                valid_counts["update_target"] = count

        # Balance losses using configured strategy
        total_loss, scaled_components, effective_weights = self.balancer(
            loss_components, valid_counts
        )
        require_finite(total_loss, "final total loss")

        return MemoryLearningOutput(
            total_loss=total_loss,
            loss_components=loss_components,
            valid_counts=valid_counts,
            loss_weights=effective_weights,
            metrics=metrics,
            warnings=warnings,
        )
