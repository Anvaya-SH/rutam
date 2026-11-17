"""Native neural memory write head producing validated memory mutation proposals."""

from __future__ import annotations

import uuid
from typing import Any

import torch
from torch import nn

from rutam.config import MemoryWriteConfig, QueryHeadConfig
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_tensor
from rutam.memory.memory_projection import NeuralMemoryProjection, TargetMemoryScorer
from rutam.memory.operation_predictor import MemoryOperationPredictor
from rutam.memory.write_gate import MemoryWriteGate, availability_mask
from rutam.memory.write_losses import MemoryWriteLossOutput, memory_write_loss
from rutam.memory.write_proposal import (
    MemoryOperation,
    MemoryProposal,
    MemoryWriteOutput,
    ProvenanceRecord,
)
from rutam.model.query_pooling import QueryPooling


class RUTAMMemoryWriteHead(nn.Module):
    """Neural memory write head for candidate node/link proposals.

    Consumes backbone hidden states, conditions on graph summary and selected memory,
    predicts write utility, predicts discrete operation classes, projects candidate
    memory slot embeddings, and scores candidate target nodes without persistent mutation.
    """

    def __init__(self, config: MemoryWriteConfig | None = None) -> None:
        super().__init__()
        self.config = MemoryWriteConfig() if config is None else config
        if not isinstance(self.config, MemoryWriteConfig):
            raise ValidationError("RUTAMMemoryWriteHead requires a MemoryWriteConfig")
        c = self.config

        # Pooling mechanism reusing QueryPooling
        pooling_config = QueryHeadConfig(
            backbone_hidden_dim=c.backbone_hidden_dim,
            pooling=c.pooling,
            empty_sequence_policy=c.empty_sequence_policy,
        )
        self.pooling = QueryPooling(pooling_config)

        # Graph summary projection
        if c.use_graph_context:
            self.graph_summary_projection = nn.Linear(
                c.hidden_dim, c.backbone_hidden_dim, bias=False, device="cpu", dtype=torch.float32
            )
        else:
            self.graph_summary_projection = None

        # Core write modules
        self.write_gate = MemoryWriteGate(c)
        self.operation_predictor = MemoryOperationPredictor(c)
        self.memory_projection = NeuralMemoryProjection(c)

        # Target memory compatibility scorer
        if c.target_selection:
            self.target_scorer = TargetMemoryScorer(c.hidden_dim)
        else:
            self.target_scorer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        graph_summary: torch.Tensor | None = None,
        selected_memory_states: torch.Tensor | None = None,
        selected_memory_ids: torch.Tensor | None = None,
        source_references: Any = None,
        memory_available: torch.Tensor | None = None,
        source_graph_version: int | None = None,
        return_attention: bool | None = None,
    ) -> MemoryWriteOutput:
        validate_tensor(hidden_states, "hidden_states", torch.float32)
        if hidden_states.ndim != 3 or hidden_states.shape[2] != self.config.backbone_hidden_dim:
            raise ValidationError(
                f"hidden_states must have shape [B, T, {self.config.backbone_hidden_dim}]"
            )
        require_finite(hidden_states, "hidden_states")
        b = hidden_states.shape[0]

        available = availability_mask(memory_available, b)

        # 1. Pool token states
        pooled = self.pooling(hidden_states, attention_mask, return_attention=return_attention)
        context = pooled.context
        valid_context = pooled.valid_context

        # 2. Condition on graph summary if enabled and provided
        summary = self._validate_graph_summary(graph_summary, b)
        if self.graph_summary_projection is not None and summary is not None:
            validate_tensor(
                self.graph_summary_projection.weight, "graph summary parameters", torch.float32
            )
            masked_summary = torch.where(available[:, None], summary, 0.0)
            proj_summary = self.graph_summary_projection(masked_summary)
            require_finite(proj_summary, "projected graph summary")
            context = context + proj_summary

        # 3. Write Gate prediction
        gate_out = self.write_gate(context, available)

        # 4. Operation Predictor prediction
        op_out = self.operation_predictor(context)

        # 5. Neural Memory Projection
        proj_out = self.memory_projection(context)

        # 6. Target memory selection scoring
        target_scores = None
        if self.target_scorer is not None and selected_memory_states is not None:
            self._validate_selected_memory(selected_memory_states, selected_memory_ids, b)
            target_scores = self.target_scorer(proj_out.proposed_embeddings, selected_memory_states)

        # 7. Proposal mask calculation
        proposal_mask = self._compute_proposal_mask(
            b,
            valid_context,
            available,
            gate_out.decisions,
            op_out.predicted_operations,
            proj_out.slot_probabilities,
            gate_out.probabilities,
        )

        # 8. Materialize structured proposals
        proposals = self._build_proposals(
            b=b,
            proposal_mask=proposal_mask,
            predicted_ops=op_out.predicted_operations,
            op_probabilities=op_out.probabilities,
            write_probabilities=gate_out.probabilities,
            proposed_embeddings=proj_out.proposed_embeddings,
            target_scores=target_scores,
            selected_memory_ids=selected_memory_ids,
            source_references=source_references,
            source_graph_version=source_graph_version,
        )

        return MemoryWriteOutput(
            write_logits=gate_out.logits,
            write_probabilities=gate_out.probabilities,
            operation_logits=op_out.logits,
            operation_probabilities=op_out.probabilities,
            proposed_embeddings=proj_out.proposed_embeddings,
            target_memory_scores=target_scores,
            proposal_mask=proposal_mask,
            write_decisions=gate_out.decisions,
            proposals=proposals,
        )

    def _validate_graph_summary(self, summary: torch.Tensor | None, b: int) -> torch.Tensor | None:
        if summary is None:
            return None
        validate_tensor(summary, "graph_summary", torch.float32)
        if summary.shape == (self.config.hidden_dim,):
            summary = summary.unsqueeze(0).expand(b, -1)
        if summary.shape != (b, self.config.hidden_dim):
            h_dim = self.config.hidden_dim
            raise ValidationError(f"graph_summary must have shape [{h_dim}] or [B, {h_dim}]")
        require_finite(summary, "graph_summary")
        return summary

    def _validate_selected_memory(
        self,
        states: torch.Tensor,
        ids: torch.Tensor | None,
        b: int,
    ) -> None:
        validate_tensor(states, "selected_memory_states", torch.float32)
        if states.ndim != 3 or states.shape[0] != b or states.shape[2] != self.config.hidden_dim:
            raise ValidationError(
                f"selected_memory_states must have shape [B, K, {self.config.hidden_dim}]"
            )
        require_finite(states, "selected_memory_states")
        k = states.shape[1]
        if ids is not None:
            validate_tensor(ids, "selected_memory_ids", ids.dtype)
            if ids.ndim != 2 or ids.shape != (b, k):
                raise ValidationError(f"selected_memory_ids must have shape [B, {k}]")
            if ids.dtype not in (torch.int64, torch.int32):
                raise ValidationError("selected_memory_ids must have integer dtype")

    def _compute_proposal_mask(
        self,
        b: int,
        valid_context: torch.Tensor,
        available: torch.Tensor,
        write_decisions: torch.Tensor | None,
        predicted_ops: torch.Tensor,
        slot_probabilities: torch.Tensor,
        write_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        m = self.config.max_proposals
        mask = torch.zeros((b, m), dtype=torch.bool, device="cpu")

        for i in range(b):
            if not valid_context[i].item() or not available[i].item():
                continue
            is_no_op = predicted_ops[i].item() == MemoryOperation.NO_OP.value
            if is_no_op:
                continue

            # In inference mode, obey write_decisions
            if write_decisions is not None and not write_decisions[i].item():
                continue

            # Slot 0 is primary proposal
            mask[i, 0] = True
            # Additional slots activated if slot probability exceeds threshold
            for slot_idx in range(1, m):
                if slot_probabilities[i, slot_idx].item() >= 0.5:
                    mask[i, slot_idx] = True

        return mask

    def _build_proposals(
        self,
        b: int,
        proposal_mask: torch.Tensor,
        predicted_ops: torch.Tensor,
        op_probabilities: torch.Tensor,
        write_probabilities: torch.Tensor,
        proposed_embeddings: torch.Tensor,
        target_scores: torch.Tensor | None,
        selected_memory_ids: torch.Tensor | None,
        source_references: Any,
        source_graph_version: int | None,
    ) -> tuple[tuple[MemoryProposal, ...], ...]:
        batch_proposals: list[tuple[MemoryProposal, ...]] = []

        for i in range(b):
            proposals_for_sample: list[MemoryProposal] = []
            op_val = predicted_ops[i].item()
            op = MemoryOperation(op_val)
            w_prob = float(write_probabilities[i, 0].item())
            op_prob = float(op_probabilities[i, op_val].item())
            confidence = max(0.0, min(1.0, w_prob * op_prob))

            # Resolve target node id if compatible and scored
            target_node_ids: tuple[int, ...] = ()
            if (
                target_scores is not None
                and selected_memory_ids is not None
                and target_scores.shape[1] > 0
                and op
                in (
                    MemoryOperation.UPDATE_NODE,
                    MemoryOperation.SUPERSEDE_NODE,
                    MemoryOperation.PROPOSE_LINK,
                    MemoryOperation.PROPOSE_MERGE,
                )
            ):
                valid_targets = selected_memory_ids[i] >= 0
                if valid_targets.any():
                    scores = target_scores[i].masked_fill(~valid_targets, float("-inf"))
                    best_k = int(scores.argmax().item())
                    target_node_ids = (int(selected_memory_ids[i, best_k].item()),)

            # Determine provenance
            prov = self._extract_provenance(source_references, sample_idx=i)

            for m_idx in range(self.config.max_proposals):
                if proposal_mask[i, m_idx].item():
                    cand_emb = proposed_embeddings[i, m_idx].detach()
                    prop = MemoryProposal(
                        proposal_id=str(uuid.uuid4()),
                        operation=op,
                        candidate_embedding=cand_emb,
                        target_node_ids=target_node_ids,
                        proposed_link_ids=(),
                        provenance=prov,
                        confidence=confidence,
                        source_graph_version=source_graph_version,
                        origin_category="neural_write_head",
                        validation_status="pending",
                    )
                    proposals_for_sample.append(prop)

            batch_proposals.append(tuple(proposals_for_sample))

        return tuple(batch_proposals)

    def _extract_provenance(
        self, source_refs: Any, sample_idx: int
    ) -> ProvenanceRecord | dict[str, Any]:
        if isinstance(source_refs, ProvenanceRecord):
            return source_refs
        if isinstance(source_refs, dict):
            return source_refs
        if isinstance(source_refs, (list, tuple)) and sample_idx < len(source_refs):
            ref = source_refs[sample_idx]
            if isinstance(ref, (ProvenanceRecord, dict)):
                return ref
            if isinstance(ref, str):
                return ProvenanceRecord(source_id=ref, evidence=ref)
        if isinstance(source_refs, str):
            return ProvenanceRecord(source_id=source_refs, evidence=source_refs)
        return ProvenanceRecord(source_id=f"inference_batch_{sample_idx}")

    def loss(
        self,
        output: MemoryWriteOutput,
        write_targets: torch.Tensor,
        operation_targets: torch.Tensor | None = None,
        target_embeddings: torch.Tensor | None = None,
        target_node_labels: torch.Tensor | None = None,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        mask: torch.Tensor | None = None,
        active_operations_only: bool = True,
    ) -> MemoryWriteLossOutput:
        """Calculate supervised multi-task loss for this write head output."""
        return memory_write_loss(
            write_logits=output.write_logits,
            write_targets=write_targets,
            operation_logits=output.operation_logits,
            operation_targets=operation_targets,
            proposed_embeddings=output.proposed_embeddings,
            target_embeddings=target_embeddings,
            target_memory_scores=output.target_memory_scores,
            target_node_labels=target_node_labels,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            mask=mask,
            active_operations_only=active_operations_only,
        )
