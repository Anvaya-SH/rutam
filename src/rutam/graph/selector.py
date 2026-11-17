"""Native neural graph selection with exact stable top-k and differentiable attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rutam.config import SelectorConfig, boolean
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor
from rutam.graph.encoder import GraphEncoderOutput
from rutam.graph.scoring import NeuralRelevanceScorer, masked_softmax
from rutam.graph.selection import TopKSelection, stable_topk


@dataclass(frozen=True, slots=True)
class GraphSelectorOutput:
    selected_indices: torch.Tensor
    selected_node_ids: torch.Tensor
    selected_states: torch.Tensor
    selected_scores: torch.Tensor
    selection_mask: torch.Tensor
    attention_weights: torch.Tensor | None
    memory_context: torch.Tensor | None
    relevance_logits: torch.Tensor | None
    eligible_mask: torch.Tensor | None


class RUTAMGraphSelector(nn.Module):
    """Select graph memories using neural queries, never retrieved text.

    IDs must explicitly accompany node_states or the existing GraphEncoderOutput.
    Hard/no-attention mode ranks chunks under no_grad and recomputes winner
    scores with autograd. This preserves the exact mathematical selected-score
    gradient without retaining losing-chunk graphs; ranking is not differentiable.
    Soft mode uses dense attention, guarded by max_dense_elements, without detach.
    """

    def __init__(self, config: SelectorConfig | None = None) -> None:
        super().__init__()
        self.config = SelectorConfig() if config is None else config
        if not isinstance(self.config, SelectorConfig):
            raise ValidationError("RUTAMGraphSelector requires a SelectorConfig")
        self.scorer = NeuralRelevanceScorer(self.config)

    def forward(
        self,
        query_states: torch.Tensor,
        node_states: torch.Tensor | GraphEncoderOutput,
        node_ids: torch.Tensor,
        *,
        node_mask: torch.Tensor | None = None,
        eligibility_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> GraphSelectorOutput:
        encoded_output = node_states if isinstance(node_states, GraphEncoderOutput) else None
        if encoded_output is not None:
            node_states = encoded_output.node_states
        self._validate_inputs(query_states, node_states, node_ids, node_mask, eligibility_mask)
        if encoded_output is not None:
            if type(encoded_output.has_nodes) is not bool or encoded_output.has_nodes != (
                node_states.shape[0] > 0
            ):
                raise ValidationError("GraphEncoderOutput.has_nodes disagrees with node_states")
        requested = self.config.return_attention if return_attention is None else return_attention
        boolean(requested, "return_attention")
        b, n = query_states.shape[0], node_states.shape[0]
        logits, attention, context, eligible = None, None, None, None
        if self.config.selection_mode == "soft" or requested:
            # Explicit bound rather than an incorrect detached/chunk-normalized softmax.
            factor = self.config.score_dim if self.config.scoring_method == "additive" else 1
            if max(b * n * factor, n * self.config.score_dim) > self.config.max_dense_elements:
                raise ValidationError(
                    "Dense soft/attention workspace exceeds selector.max_dense_elements; "
                    "reduce graph/batch size, increase the explicit limit, or use hard mode "
                    "with return_attention=False"
                )
            logits = self._dense_logits(query_states, node_states)
            eligible = self._chunk_mask(b, 0, n, node_mask, eligibility_mask)
            attention = masked_softmax(logits, eligible)
            top = stable_topk(logits, node_ids, self.config.top_k, mask=eligible)
            if self.config.selection_mode == "soft":
                context = attention @ node_states
                require_finite(context, "memory context")
            # Export finite zeros for masked logits; eligible_mask must accompany losses.
            logits = torch.where(eligible, logits, 0)
            states = self._gather_states(node_states, top)
        else:
            if b * self.config.top_k * self.config.score_dim > self.config.max_dense_elements:
                raise ValidationError(
                    "Selected-score workspace exceeds selector.max_dense_elements"
                )
            with torch.no_grad():
                top = self._rank_chunks(
                    query_states, node_states, node_ids, node_mask, eligibility_mask
                )
            states = self._gather_states(node_states, top)
            scores = self.scorer.score_pairs(query_states, states)
            top = TopKSelection(
                top.indices, top.node_ids, torch.where(top.mask, scores, 0), top.mask
            )
        return GraphSelectorOutput(
            top.indices,
            top.node_ids,
            states,
            top.scores,
            top.mask,
            attention if requested else None,
            context,
            logits,
            eligible,
        )

    def select_projected(
        self,
        queries: torch.Tensor,
        features: torch.Tensor,
        node_ids: torch.Tensor,
        projection: nn.Module,
        *,
        node_mask: torch.Tensor,
        active: torch.Tensor,
        eligibility_mask: torch.Tensor | None = None,
    ) -> GraphSelectorOutput:
        """Select from existing features before relational encoding.

        Candidate projection and ranking run in bounded chunks without autograd.
        The winning feature rows are projected again with autograd, so selected
        scores still train the query, scorer, and projection. Top-k indices remain
        discrete. Full-graph soft attention cannot use this sparse contract.
        """
        if self.config.selection_mode != "hard" or self.config.return_attention:
            raise ValidationError("Sparse public graph reads require hard selection without full attention")
        input_dim = getattr(getattr(projection, "config", None), "input_dim", features.shape[-1])
        output_dim = getattr(getattr(projection, "config", None), "hidden_dim", self.config.hidden_dim)
        validate_features(features, input_dim, "candidate features")
        validate_tensor(node_ids, "node_ids", torch.int64)
        if node_ids.shape != (features.shape[0],):
            raise ValidationError("node_ids must have shape [N]")
        if output_dim != self.config.hidden_dim:
            raise ValidationError("candidate projection output must match selector.hidden_dim")
        b, n = queries.shape[0], features.shape[0]
        validate_tensor(active, "active", torch.bool)
        if active.shape != (b,):
            raise ValidationError("active must be a boolean [B] tensor")
        validate_tensor(node_mask, "node_mask", torch.bool)
        if node_mask.shape != (n,):
            raise ValidationError("node_mask must have shape [N]")
        if eligibility_mask is not None:
            validate_tensor(eligibility_mask, "eligibility_mask", torch.bool)
            if eligibility_mask.shape != (b, n):
                raise ValidationError("eligibility_mask must have shape [B,N]")
        if b * self.config.top_k * self.config.score_dim > self.config.max_dense_elements:
            raise ValidationError("Selected-score workspace exceeds selector.max_dense_elements")
        project = getattr(projection, "forward_for_selection", projection)
        size = self._chunk_size(b, n)
        with torch.no_grad():
            projected_queries = self.scorer.project_queries(queries)
            best = stable_topk(queries.new_empty((b, 0)), node_ids[:0], self.config.top_k)
            for start in range(0, n, size):
                end = min(start + size, n)
                rows = torch.arange(start, end, dtype=torch.int64)
                rows = rows[node_mask[start:end]]
                if rows.numel() == 0:
                    continue
                mask = active[:, None].expand(b, rows.numel())
                if eligibility_mask is not None:
                    mask = mask & eligibility_mask[:, rows]
                if not mask.any().item():
                    continue
                nodes = project(features[rows])
                scores = self.scorer.score_projected(projected_queries, self.scorer.project_nodes(nodes))
                rows = rows.unsqueeze(0).expand(b, -1)
                ids = node_ids[rows]
                best = stable_topk(torch.cat((best.scores, scores), 1),
                    torch.cat((best.node_ids, ids), 1), self.config.top_k,
                    mask=torch.cat((best.mask, mask), 1),
                    indices=torch.cat((best.indices, rows), 1))
        if n:
            selected = project(features[best.indices.clamp_min(0).reshape(-1)]).reshape(
                b, self.config.top_k, -1
            )
            selected = torch.where(best.mask[..., None], selected, 0)
        else:
            selected = queries.new_zeros((b, self.config.top_k, self.config.hidden_dim))
        scores = torch.where(best.mask, self.scorer.score_pairs(queries, selected), 0)
        eligible = node_mask[None, :].expand(b, -1) & active[:, None]
        if eligibility_mask is not None:
            eligible = eligible & eligibility_mask
        return GraphSelectorOutput(
            best.indices,
            best.node_ids,
            selected,
            scores,
            best.mask,
            None,
            None,
            None,
            eligible,
        )

    def _chunk_size(self, b: int, n: int) -> int:
        factor = (
            max(b, 1) * self.config.score_dim
            if self.config.scoring_method == "additive"
            else (max(b, self.config.score_dim, 1))
        )
        limit = self.config.max_dense_elements // factor
        if limit < 1:
            raise ValidationError("One scoring chunk exceeds selector.max_dense_elements")
        return max(1, min(self.config.chunk_size or max(n, 1), limit))

    def _dense_logits(self, queries: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        projected_queries = self.scorer.project_queries(queries)
        size = self._chunk_size(queries.shape[0], nodes.shape[0])
        if nodes.shape[0] == 0:
            return self.scorer.score_projected(projected_queries, self.scorer.project_nodes(nodes))
        return torch.cat(
            [
                self.scorer.score_projected(projected_queries, self.scorer.project_nodes(chunk))
                for chunk in nodes.split(size)
            ],
            dim=1,
        )

    def _rank_chunks(
        self,
        queries: torch.Tensor,
        nodes: torch.Tensor,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor | None,
        eligibility_mask: torch.Tensor | None,
    ) -> TopKSelection:
        b, n = queries.shape[0], nodes.shape[0]
        projected_queries = self.scorer.project_queries(queries)
        size = self._chunk_size(b, n)
        best = stable_topk(queries.new_empty((b, 0)), node_ids[:0], self.config.top_k)
        for start in range(0, n, size):
            end = min(start + size, n)
            scores = self.scorer.score_projected(
                projected_queries, self.scorer.project_nodes(nodes[start:end])
            )
            mask = self._chunk_mask(b, start, end, node_mask, eligibility_mask)
            rows = torch.arange(start, end, dtype=torch.int64).unsqueeze(0).expand(b, -1)
            ids = node_ids[start:end].unsqueeze(0).expand(b, -1)
            best = stable_topk(
                torch.cat((best.scores, scores), dim=1),
                torch.cat((best.node_ids, ids), dim=1),
                self.config.top_k,
                mask=torch.cat((best.mask, mask), dim=1),
                indices=torch.cat((best.indices, rows), dim=1),
            )
        return best

    @staticmethod
    def _gather_states(nodes: torch.Tensor, top: TopKSelection) -> torch.Tensor:
        n, h = nodes.shape
        if n == 0:
            return nodes.sum(dim=0).view(1, 1, h).expand(*top.indices.shape, h)
        # Clamp padding only for a safe gather; the exported states are explicitly zero.
        gathered = nodes[top.indices.clamp_min(0)]
        return torch.where(top.mask.unsqueeze(-1), gathered, 0)

    @staticmethod
    def _chunk_mask(
        b: int,
        start: int,
        end: int,
        node_mask: torch.Tensor | None,
        eligibility_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = torch.ones((b, end - start), dtype=torch.bool)
        if node_mask is not None:
            valid = valid & node_mask[None, start:end]
        if eligibility_mask is not None:
            valid = valid & eligibility_mask[:, start:end]
        return valid

    def _validate_inputs(
        self,
        queries: torch.Tensor,
        nodes: torch.Tensor,
        ids: torch.Tensor,
        node_mask: torch.Tensor | None,
        eligibility_mask: torch.Tensor | None,
    ) -> None:
        validate_features(queries, self.config.query_dim, "query_states")
        validate_features(nodes, self.config.hidden_dim, "node_states")
        validate_tensor(ids, "node_ids", torch.int64)
        b, n = queries.shape[0], nodes.shape[0]
        if ids.shape != (n,) or (ids < 0).any().item() or torch.unique(ids).numel() != n:
            raise ValidationError("node_ids must be unique nonnegative stable IDs with shape [N]")
        for mask, name, shape in (
            (node_mask, "node_mask", (n,)),
            (eligibility_mask, "eligibility_mask", (b, n)),
        ):
            if mask is not None:
                validate_tensor(mask, name, torch.bool)
                if mask.shape != shape:
                    raise ValidationError(f"{name} must have shape {shape}")
