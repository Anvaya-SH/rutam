"""Composable memory validator enforcing structural integrity, provenance, and privacy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rutam.config import GraphConfig, MemoryValidationConfig, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.graph.state import GraphState
from rutam.memory.conflict_detection import detect_conflicts
from rutam.memory.duplicate_detection import detect_duplicates
from rutam.memory.provenance import (
    AuthorizationContext,
    ProvenanceVerifier,
    validate_provenance_record,
)
from rutam.memory.validation_output import (
    MemoryValidationResult,
    ValidationBatchResult,
    ValidationDecision,
)
from rutam.memory.validation_rules import (
    check_sensitive_data,
    check_structural_invariants,
)
from rutam.memory.write_proposal import MemoryOperation


class RUTAMMemoryValidator:
    """Validates memory and link proposals against structural, security, and version constraints.

    Invariants:
    1. Validator never mutates persistent storage or the active graph snapshot.
    2. Neural confidence never overrides missing authorization or structural violations.
    3. Stale proposals are rejected via optimistic concurrency version checks.
    """

    def __init__(
        self,
        config: RUTAMConfig | MemoryValidationConfig | None = None,
        *,
        graph_config: GraphConfig | None = None,
        provenance_verifier: ProvenanceVerifier | None = None,
    ) -> None:
        if provenance_verifier is not None and not callable(provenance_verifier):
            raise ValidationError("provenance_verifier must be callable or None")
        self.provenance_verifier = provenance_verifier
        if config is None:
            self.config = MemoryValidationConfig()
            self.graph_config = graph_config or GraphConfig()
            self.conflicting_relations: tuple[tuple[str, str], ...] = ()
        elif isinstance(config, RUTAMConfig):
            self.config = config.memory_validation
            self.graph_config = graph_config or config.graph
            self.conflicting_relations = config.linker.conflicting_relations
        elif isinstance(config, MemoryValidationConfig):
            self.config = config
            self.graph_config = graph_config or GraphConfig()
            self.conflicting_relations = ()
        else:
            raise ValidationError("Validator requires a RUTAMConfig or MemoryValidationConfig")

    def validate(
        self,
        proposal: Any,
        graph_snapshot: Graph | GraphState,
        *,
        authorization_context: AuthorizationContext | None = None,
        graph_config: GraphConfig | None = None,
    ) -> MemoryValidationResult:
        """Validate a single proposal against the active graph snapshot and security policies."""
        if not isinstance(graph_snapshot, (Graph, GraphState)):
            raise ValidationError("graph_snapshot must be a Graph or GraphState instance")

        g_cfg = graph_config or self.graph_config
        prop_id = getattr(proposal, "proposal_id", "anonymous_proposal")
        source_ver = getattr(proposal, "source_graph_version", None)

        # Authorization scopes are an independent commit boundary, not a
        # provenance claim. Reject a proposal whose declared write scope or
        # any existing target falls outside the caller's grants.
        if authorization_context is not None:
            if (
                getattr(proposal, "scope", "default")
                not in authorization_context.authorized_scopes
            ):
                return MemoryValidationResult(
                    proposal_id=prop_id,
                    decision=ValidationDecision.REJECT,
                    reason_codes=("MEMORY_SCOPE_NOT_AUTHORIZED",),
                    source_graph_version=source_ver,
                )
            owner_id = getattr(proposal, "owner_id", None)
            if (
                owner_id is not None
                and authorization_context.user_id is not None
                and owner_id != authorization_context.user_id
            ):
                return MemoryValidationResult(
                    proposal_id=prop_id,
                    decision=ValidationDecision.REJECT,
                    reason_codes=("MEMORY_OWNER_NOT_AUTHORIZED",),
                    source_graph_version=source_ver,
                )
            operation = getattr(proposal, "operation", MemoryOperation.NO_OP)
            if MemoryOperation.from_value(operation) != MemoryOperation.NO_OP:
                target_ids = tuple(getattr(proposal, "target_node_ids", ()))
                if isinstance(graph_snapshot, Graph):
                    target_records = []
                    for target_id in target_ids:
                        node = graph_snapshot._nodes.get(target_id)
                        if node is not None:
                            target_records.append((node.scope, node.permissions))
                else:
                    rows = {
                        int(node_id): row
                        for row, node_id in enumerate(graph_snapshot.node_ids.tolist())
                    }
                    target_records = []
                    for target_id in target_ids:
                        row = rows.get(target_id)
                        if row is not None:
                            record = graph_snapshot.node_metadata[row]
                            target_records.append(
                                (
                                    record.get("scope", "default"),
                                    record.get("permissions", ("read", "write")),
                                )
                            )
                if len(target_records) != len(target_ids):
                    return MemoryValidationResult(
                        proposal_id=prop_id,
                        decision=ValidationDecision.REJECT,
                        reason_codes=("MEMORY_TARGET_NOT_AUTHORIZED",),
                        source_graph_version=source_ver,
                    )
                for target_scope, target_permissions in target_records:
                    if (
                        target_scope not in authorization_context.authorized_scopes
                        or "write" not in target_permissions
                    ):
                        return MemoryValidationResult(
                            proposal_id=prop_id,
                            decision=ValidationDecision.REJECT,
                            reason_codes=("MEMORY_TARGET_NOT_AUTHORIZED",),
                            source_graph_version=source_ver,
                        )

        # 1. Structural Validation
        is_struct_valid, struct_reason = check_structural_invariants(
            proposal, graph_snapshot, g_cfg, self.config
        )
        if not is_struct_valid:
            return MemoryValidationResult(
                proposal_id=prop_id,
                decision=ValidationDecision.REJECT,
                reason_codes=(struct_reason or "STRUCTURAL_VALIDATION_FAILED",),
                source_graph_version=source_ver,
            )

        # 2. Sensitive Data & Privacy Boundary Check
        is_safe, sec_reason = check_sensitive_data(proposal, authorization_context)
        if not is_safe:
            return MemoryValidationResult(
                proposal_id=prop_id,
                decision=ValidationDecision.REJECT,
                reason_codes=(sec_reason or "SENSITIVE_DATA_DETECTED",),
                source_graph_version=source_ver,
            )

        # 3. Provenance Enforcement
        prov_obj = getattr(proposal, "provenance", None)
        is_prov_valid, prov_reason, _ = validate_provenance_record(
            prov_obj,
            authorization_context,
            require_provenance=self.config.require_provenance,
            allow_unverified_hypotheses=self.config.allow_unverified_hypotheses,
            provenance_verifier=self.provenance_verifier,
        )
        if not is_prov_valid:
            return MemoryValidationResult(
                proposal_id=prop_id,
                decision=ValidationDecision.REJECT,
                reason_codes=(prov_reason or "PROVENANCE_VALIDATION_FAILED",),
                source_graph_version=source_ver,
            )

        # 4. Version & Concurrency / Conflict Check
        curr_ver = (
            graph_snapshot.version
            if isinstance(graph_snapshot, Graph)
            else getattr(graph_snapshot, "graph_version", None)
        )
        has_conflict, conf_reason, detected_conflicts = detect_conflicts(
            proposal,
            graph_snapshot,
            conflicting_relations=self.conflicting_relations,
            current_graph_version=curr_ver,
        )
        if has_conflict:
            decision = (
                ValidationDecision.STALE
                if conf_reason == "STALE_GRAPH_VERSION"
                else ValidationDecision.CONFLICT
            )
            return MemoryValidationResult(
                proposal_id=prop_id,
                decision=decision,
                reason_codes=(conf_reason or "CONFLICT_DETECTED",),
                detected_conflicts=detected_conflicts,
                source_graph_version=source_ver,
            )

        # 5. Duplicate Detection
        if self.config.detect_duplicates:
            is_dup, dup_reason, match_ids = detect_duplicates(
                proposal,
                graph_snapshot,
                similarity_threshold=self.config.content_similarity_threshold,
            )
            if is_dup:
                return MemoryValidationResult(
                    proposal_id=prop_id,
                    decision=ValidationDecision.DUPLICATE,
                    reason_codes=(dup_reason or "DUPLICATE_DETECTED",),
                    existing_match_ids=match_ids,
                    source_graph_version=source_ver,
                )

        # 6. Confidence Calibration Check
        confidence = getattr(proposal, "confidence", 1.0)
        confidence_meta = {"confidence": float(confidence)}
        if confidence < self.config.min_confidence:
            return MemoryValidationResult(
                proposal_id=prop_id,
                decision=ValidationDecision.DEFER,
                reason_codes=("LOW_CONFIDENCE",),
                confidence_metadata=confidence_meta,
                source_graph_version=source_ver,
                requires_further_verification=True,
            )

        # All checks passed
        return MemoryValidationResult(
            proposal_id=prop_id,
            decision=ValidationDecision.ACCEPT,
            reason_codes=("PASSED_ALL_CHECKS",),
            confidence_metadata=confidence_meta,
            source_graph_version=source_ver,
            sanitized_proposal=proposal,
            requires_further_verification=False,
        )

    def validate_batch(
        self,
        proposals: Sequence[Any],
        graph_snapshot: Graph | GraphState,
        *,
        authorization_context: AuthorizationContext | None = None,
        graph_config: GraphConfig | None = None,
    ) -> ValidationBatchResult:
        """Validate a batch of proposals with batch size bounds."""
        if not isinstance(proposals, (list, tuple)):
            raise ValidationError("proposals must be a sequence")
        if len(proposals) > self.config.max_proposals_per_batch:
            limit = self.config.max_proposals_per_batch
            raise ValidationError(
                f"Batch exceeds max_proposals_per_batch ({len(proposals)} > {limit})"
            )

        results: list[MemoryValidationResult] = []
        for prop in proposals:
            res = self.validate(
                prop,
                graph_snapshot,
                authorization_context=authorization_context,
                graph_config=graph_config,
            )
            results.append(res)

        return ValidationBatchResult.from_results(tuple(results))
