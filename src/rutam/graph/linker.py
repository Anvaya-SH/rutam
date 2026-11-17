"""Bounded neural relationship proposals; this module has no graph mutation or storage API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import nn

from rutam.config import LinkerConfig, nonempty_string, positive_int
from rutam.exceptions import StaleGraphError, ValidationError
from rutam.graph.embeddings import require_finite, validate_features, validate_tensor
from rutam.graph.encoder import GraphEncoderOutput
from rutam.graph.link_candidates import (
    generate_link_candidates,
    sample_negative_pairs,
    structural_link_features,
)
from rutam.graph.link_output import LinkerOutput, LinkProposal, NegativeSamples
from rutam.graph.link_scoring import RUTAMLinkScorer, copy_expanded_parameters
from rutam.graph.neighborhood import NeighborhoodCache, SparseNeighborhood
from rutam.graph.relation_predictor import RelationVocabulary, RUTAMRelationPredictor
from rutam.graph.schema import utc_now
from rutam.graph.state import GraphState


class RUTAMGraphLinker(nn.Module):
    def __init__(self, config: LinkerConfig | None = None) -> None:
        super().__init__()
        self.config = LinkerConfig() if config is None else config
        if not isinstance(self.config, LinkerConfig):
            raise ValidationError("Linker requires LinkerConfig")
        self.link_scorer = RUTAMLinkScorer(self.config)
        self.relation_predictor = RUTAMRelationPredictor(self.config)
        self.index_cache = NeighborhoodCache()

    @property
    def vocabulary(self) -> RelationVocabulary:
        return self.relation_predictor.vocabulary

    def forward(
        self,
        node_states: torch.Tensor,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        query_vectors: torch.Tensor | None = None,
        candidate_pairs: torch.Tensor | None = None,
        *,
        graph_version: int = 0,
        expected_graph_version: int | None = None,
        node_mask: torch.Tensor | None = None,
        selected_indices: torch.Tensor | None = None,
        candidate_indices: torch.Tensor | None = None,
        node_types: tuple[str, ...] | None = None,
        query_mask: torch.Tensor | None = None,
        evidence_references: Mapping[tuple[int, int], tuple[str, ...]] | None = None,
    ) -> LinkerOutput:
        c = self.config
        positive_int(graph_version, "graph_version", minimum=0)
        if expected_graph_version is not None:
            positive_int(expected_graph_version, "expected_graph_version", minimum=0)
            if expected_graph_version != graph_version:
                raise StaleGraphError("Linker request has a stale graph version")
        validate_features(node_states, c.hidden_dim, "node_states")
        validate_tensor(node_ids, "node_ids", torch.int64)
        if node_ids.numel() != node_states.shape[0]:
            raise ValidationError("Node IDs and representations disagree")
        index = self.index_cache.get(node_ids, edge_index, edge_types, graph_version)
        if edge_types.numel() and edge_types.max().item() >= len(c.relation_labels):
            raise ValidationError("Unknown graph relation ID for linker vocabulary")
        candidates = generate_link_candidates(
            index,
            c,
            node_states=node_states,
            candidate_pairs=candidate_pairs,
            selected_indices=selected_indices,
            candidate_indices=candidate_indices,
            node_mask=node_mask,
        )
        pairs = candidates.pairs
        structure = structural_link_features(
            index,
            pairs,
            len(c.relation_labels),
            node_types=node_types,
            compatible_types=c.compatible_node_types,
            directed=c.directed,
        )
        if c.query_conditioning:
            if query_vectors is None:
                raise ValidationError("Query-conditioned linker requires [B,Q] queries")
            validate_features(query_vectors, c.query_dim, "query_vectors")
            b = query_vectors.shape[0]
        else:
            if query_vectors is not None or query_mask is not None:
                raise ValidationError("Queries/masks require query_conditioning=True")
            b = 1
        width = max(
            4 * c.hidden_dim
            + c.structural_feature_dim
            + (c.hidden_dim if c.query_conditioning else 0),
            c.scoring_hidden_dim,
            len(self.vocabulary.labels),
        )
        if b * pairs.shape[0] * width > c.max_link_elements:
            raise ValidationError("Link scoring workspace exceeds linker.max_link_elements")
        outputs, relations = [], []
        for start in range(0, max(1, pairs.shape[0]), c.scoring_chunk_size):
            chunk = pairs[start : start + c.scoring_chunk_size]
            features = structure[start : start + c.scoring_chunk_size]
            outputs.append(self.link_scorer(node_states, chunk, features, query_vectors))
            relations.append(self.relation_predictor(node_states, chunk, features, query_vectors))
        dimension = 1 if c.query_conditioning else 0
        logits = torch.cat(outputs, dim=dimension) / c.link_temperature
        relation_logits = torch.cat(relations, dim=dimension) / c.relation_temperature
        require_finite(logits, "link logits")
        require_finite(relation_logits, "relation logits")
        valid = torch.ones_like(logits, dtype=torch.bool)
        if query_mask is not None:
            validate_tensor(query_mask, "query_mask", torch.bool)
            if query_mask.shape != (b,):
                raise ValidationError("query_mask must be [B]")
            valid &= query_mask[:, None]
        logits = torch.where(valid, logits, 0)
        relation_logits = torch.where(valid[..., None], relation_logits, 0)
        probability = torch.sigmoid(logits) * valid
        relation_probability = torch.softmax(relation_logits, dim=-1) * valid[..., None]
        predicted = torch.where(valid, relation_probability.argmax(dim=-1), -1)
        relation_score = relation_probability.gather(-1, predicted.clamp_min(0)[..., None]).squeeze(
            -1
        )
        confidence = probability * relation_score * (predicted > 0)
        graph_relation_ids = torch.where(predicted > 0, predicted - 1, -1)
        proposal_mask, annotations, proposals = self._proposals(
            index,
            candidates.node_ids,
            pairs,
            probability,
            relation_score,
            confidence,
            predicted,
            valid,
            graph_version,
            evidence_references,
        )
        return LinkerOutput(
            pairs,
            candidates.node_ids,
            logits,
            probability,
            relation_logits,
            relation_probability,
            predicted,
            graph_relation_ids,
            confidence,
            proposal_mask,
            valid,
            structure,
            annotations,
            proposals,
            graph_version,
            index.fingerprint,
            candidates.duplicates_removed,
            candidates.truncated,
        )

    def _proposals(
        self,
        index: SparseNeighborhood,
        candidate_ids: torch.Tensor,
        pairs: torch.Tensor,
        link_probability: torch.Tensor,
        relation_probability: torch.Tensor,
        confidence: torch.Tensor,
        predicted: torch.Tensor,
        valid: torch.Tensor,
        graph_version: int,
        evidence: Mapping[tuple[int, int], tuple[str, ...]] | None,
    ) -> tuple[
        torch.Tensor, tuple[tuple[tuple[str, ...], ...], ...], tuple[tuple[LinkProposal, ...], ...]
    ]:
        if evidence is not None:
            if not isinstance(evidence, Mapping):
                raise ValidationError(
                    "Evidence references must be a mapping by persistent node ID pair"
                )
            ids = set(index.node_ids)
            for pair, refs in evidence.items():
                if (
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(type(x) is not int or x not in ids for x in pair)
                ):
                    raise ValidationError("Evidence keys must be valid persistent node ID pairs")
                if not isinstance(refs, (tuple, list)):
                    raise ValidationError("Evidence references must be a sequence of strings")
                for ref in refs:
                    nonempty_string(ref, "evidence reference")
        batched = self.config.query_conditioning
        probabilities = link_probability if batched else link_probability[None]
        relation_scores = relation_probability if batched else relation_probability[None]
        confidence_rows = confidence if batched else confidence[None]
        classes = predicted if batched else predicted[None]
        valid_rows = valid if batched else valid[None]
        b, count = classes.shape
        annotations: list[list[tuple[str, ...]]] = [[() for _ in range(count)] for _ in range(b)]
        eligible = []
        conflicts = {frozenset(pair) for pair in self.config.conflicting_relations}
        lookup: dict[tuple[int, int], set[int]] = {}
        for source, target, relation in index.typed_edges:
            lookup.setdefault((source, target), set()).add(relation)
            if not self.config.directed:
                lookup.setdefault((target, source), set()).add(relation)
        for batch in range(b):
            for row, (source, target) in enumerate(pairs.tolist()):
                tags = []
                class_id = int(classes[batch, row].item())
                relation_id = class_id - 1
                existing = lookup.get((source, target), set())
                reverse = lookup.get((target, source), set())
                duplicate = relation_id in existing
                if not valid_rows[batch, row]:
                    tags.append("query_masked")
                if class_id == 0:
                    tags.append("no_relation")
                if duplicate:
                    tags.append("identical_typed_edge")
                if reverse:
                    tags.append("reverse_edge_present")
                if relation_id in reverse:
                    tags.append("opposite_direction_same_relation")
                if existing - {relation_id}:
                    tags.append("other_relation_present")
                if class_id > 0 and any(
                    frozenset((self.vocabulary.name(class_id), self.config.relation_labels[x]))
                    in conflicts
                    for x in existing
                ):
                    tags.append("conflicting_relation_evidence")
                annotations[batch][row] = tuple(tags)
                if (
                    valid_rows[batch, row]
                    and class_id > 0
                    and not duplicate
                    and probabilities[batch, row] >= self.config.link_threshold
                    and relation_scores[batch, row] >= self.config.relation_threshold
                ):
                    source_id, target_id = candidate_ids[row].tolist()
                    eligible.append(
                        (
                            -float(confidence_rows[batch, row].item()),
                            batch,
                            source_id,
                            target_id,
                            row,
                            class_id,
                        )
                    )
        eligible.sort()
        mask = torch.zeros_like(valid_rows)
        proposals: list[list[LinkProposal]] = [[] for _ in range(b)]
        now = utc_now()
        for _, batch, source_id, target_id, row, class_id in eligible[
            : self.config.max_proposals_per_call
        ]:
            mask[batch, row] = True
            tags = annotations[batch][row]
            references = () if evidence is None else tuple(evidence.get((source_id, target_id), ()))
            proposals[batch].append(
                LinkProposal(
                    source_id,
                    target_id,
                    class_id - 1,
                    self.vocabulary.name(class_id),
                    float(probabilities[batch, row].item()),
                    float(relation_scores[batch, row].item()),
                    float(confidence_rows[batch, row].item()),
                    references,
                    graph_version,
                    index.fingerprint,
                    self.vocabulary.schema_version,
                    now,
                    "conflict_review" if "conflicting_relation_evidence" in tags else "unverified",
                    tags,
                    batch if batched else None,
                )
            )
        for _, batch, _, _, row, _ in eligible[self.config.max_proposals_per_call :]:
            annotations[batch][row] += ("proposal_budget_exceeded",)
        return (
            mask if batched else mask[0],
            tuple(tuple(x) for x in annotations),
            tuple(tuple(x) for x in proposals),
        )

    def from_state(
        self,
        encoded: GraphEncoderOutput | torch.Tensor,
        state: GraphState,
        query_vectors: torch.Tensor | None = None,
        **kwargs: object,
    ) -> LinkerOutput:
        state.validate()
        if state.config.graph.directed != self.config.directed:
            raise ValidationError("Linker directed policy must match graph state")
        self.vocabulary.validate_graph_mapping(state.relationship_types)
        nodes = encoded.node_states if isinstance(encoded, GraphEncoderOutput) else encoded
        types = tuple(record["node_type"] for record in state.node_metadata)
        # Evidence is attached only for actual candidates below;
        # use node provenance without a dense pair map.
        output = self(
            nodes,
            state.node_ids,
            state.edge_index,
            state.edge_types,
            query_vectors,
            graph_version=state.graph_version,
            node_types=types,
            **kwargs,
        )
        if "evidence_references" in kwargs:
            return output
        batches = []
        provenance = {
            int(node_id): record.get("provenance")
            for node_id, record in zip(state.node_ids.tolist(), state.node_metadata, strict=True)
        }
        for batch in output.proposal_batches:
            proposals = []
            for proposal in batch:
                references = tuple(
                    dict.fromkeys(
                        ref
                        for ref in (
                            provenance[proposal.source_node_id],
                            provenance[proposal.destination_node_id],
                        )
                        if ref
                    )
                )
                proposals.append(replace(proposal, evidence_references=references))
            batches.append(tuple(proposals))
        return replace(output, proposal_batches=tuple(batches))

    def migrate_relation_types(self, additional_labels: tuple[str, ...]) -> RUTAMGraphLinker:
        vocabulary = self.vocabulary.extend(additional_labels)
        result = RUTAMGraphLinker(
            replace(
                self.config,
                relation_labels=vocabulary.relation_labels,
                relation_schema_version=vocabulary.schema_version,
            )
        )
        copy_expanded_parameters(self.link_scorer, result.link_scorer)
        result.relation_predictor = self.relation_predictor.migrate(additional_labels)
        result.train(self.training)
        return result

    def sample_negatives(self, state: GraphState, positive_count: int) -> NegativeSamples:
        state.validate()
        positive_int(positive_count, "positive_count", minimum=0)
        index = self.index_cache.get(
            state.node_ids, state.edge_index, state.edge_types, state.graph_version
        )
        return sample_negative_pairs(
            index,
            min(
                self.config.max_candidate_pairs,
                positive_count * self.config.negative_samples_per_positive,
            ),
            seed=self.config.negative_sampling_seed,
            directed=self.config.directed,
            allow_self_links=self.config.allow_self_links,
            max_attempts=self.config.negative_sampling_attempts,
        )


def validate_proposal_snapshot(
    proposal: LinkProposal, state: GraphState, vocabulary: RelationVocabulary
) -> None:
    """Structural freshness check only, never factual confirmation or a graph write."""
    state.validate()
    vocabulary.validate_graph_mapping(state.relationship_types)
    index = NeighborhoodCache().get(
        state.node_ids, state.edge_index, state.edge_types, state.graph_version
    )
    if (
        proposal.graph_version != state.graph_version
        or proposal.topology_fingerprint != index.fingerprint
        or proposal.relation_schema_version != vocabulary.schema_version
    ):
        raise StaleGraphError("Proposal belongs to a different graph or relation schema version")
    if (
        proposal.source_node_id not in index.node_ids
        or proposal.destination_node_id not in index.node_ids
        or proposal.relation_id < 0
        or proposal.relation_id >= len(vocabulary.relation_labels)
        or vocabulary.relation_labels[proposal.relation_id] != proposal.relation_name
    ):
        raise ValidationError("Proposal node/relation references do not match snapshot")
