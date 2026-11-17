"""Write pipeline coordinating proposal generation, validation, and pending update staging."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.graph.state import GraphState
from rutam.memory.controller_output import MemoryWritePipelineOutput
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.validation_output import ValidationDecision
from rutam.memory.validator import RUTAMMemoryValidator
from rutam.memory.write_head import RUTAMMemoryWriteHead
from rutam.memory.write_proposal import MemoryProposal


class MemoryWritePipeline(nn.Module):
    """Executes write proposal generation followed by discrete validation."""

    def __init__(
        self,
        write_head: RUTAMMemoryWriteHead,
        validator: RUTAMMemoryValidator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(write_head, RUTAMMemoryWriteHead):
            raise ValidationError("write_head must be a RUTAMMemoryWriteHead")
        self.write_head = write_head
        self.validator = validator

    def forward(
        self,
        hidden_states: torch.Tensor,
        graph_snapshot: Graph | GraphState,
        *,
        attention_mask: torch.Tensor | None = None,
        graph_summary: torch.Tensor | None = None,
        selected_memory_states: torch.Tensor | None = None,
        selected_memory_ids: torch.Tensor | None = None,
        source_references: Any = None,
        memory_available: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
        auto_validate: bool = True,
    ) -> MemoryWritePipelineOutput:
        curr_ver = (
            graph_snapshot.version
            if isinstance(graph_snapshot, Graph)
            else getattr(graph_snapshot, "graph_version", None)
        )

        # 1. Forward through neural write head
        write_out = self.write_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            graph_summary=graph_summary,
            selected_memory_states=selected_memory_states,
            selected_memory_ids=selected_memory_ids,
            source_references=source_references,
            memory_available=memory_available,
            source_graph_version=curr_ver,
        )

        flat_proposals: list[MemoryProposal] = [p for batch in write_out.proposals for p in batch]

        # 2. Discrete validation pass if enabled
        val_result = None
        accepted: list[MemoryProposal] = []
        rejected: list[MemoryProposal] = []
        deferred: list[MemoryProposal] = []

        if auto_validate and self.validator is not None and flat_proposals:
            val_result = self.validator.validate_batch(
                flat_proposals,
                graph_snapshot,
                authorization_context=authorization_context,
            )
            for prop, res in zip(flat_proposals, val_result.results, strict=True):
                if res.decision == ValidationDecision.ACCEPT:
                    accepted.append(prop)
                elif res.decision == ValidationDecision.DEFER:
                    deferred.append(prop)
                else:
                    rejected.append(prop)
        elif not auto_validate:
            accepted = list(flat_proposals)

        return MemoryWritePipelineOutput(
            proposals=write_out.proposals,
            write_output=write_out,
            validation_result=val_result,
            accepted_proposals=tuple(accepted),
            rejected_proposals=tuple(rejected),
            deferred_proposals=tuple(deferred),
        )
