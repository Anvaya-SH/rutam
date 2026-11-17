"""Structural and security validation rules for memory proposals."""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from rutam.config import GraphConfig, MemoryValidationConfig
from rutam.graph.operations import Graph
from rutam.graph.state import GraphState
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal

# Security patterns for credentials and secrets
SENSITIVE_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(
        r"(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?",
        re.IGNORECASE,
    ),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"password\s*(?::=|=|:)\s*['\"]?\S{6,}['\"]?", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),  # Google API key
]


def check_sensitive_data(
    proposal: Any,
    auth_context: AuthorizationContext | None = None,
) -> tuple[bool, str | None]:
    """Scan proposal text and metadata for leaked secrets or credentials."""
    if auth_context is not None and auth_context.allow_sensitive_data:
        return True, None

    # Inspect text representation and fields
    texts_to_check: list[str] = []
    if isinstance(proposal, MemoryProposal):
        if isinstance(proposal.provenance, str):
            texts_to_check.append(proposal.provenance)
        elif isinstance(proposal.provenance, dict):
            texts_to_check.append(json.dumps(proposal.provenance))
        elif hasattr(proposal.provenance, "to_dict"):
            texts_to_check.append(json.dumps(proposal.provenance.to_dict()))
        texts_to_check.append(proposal.origin_category)
        if proposal.content is not None:
            texts_to_check.append(proposal.content)
    elif hasattr(proposal, "content"):
        texts_to_check.append(str(proposal.content))

    for text in texts_to_check:
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                return False, "SENSITIVE_DATA_DETECTED"

    return True, None


def check_structural_invariants(
    proposal: Any,
    graph_snapshot: Graph | GraphState,
    graph_config: GraphConfig,
    validation_config: MemoryValidationConfig,
) -> tuple[bool, str | None]:
    """Validate structural graph integrity (valid IDs, embedding dims, capacity limits)."""
    # 1. Byte limit check
    if hasattr(proposal, "to_dict"):
        try:
            raw_bytes = len(json.dumps(proposal.to_dict()).encode("utf-8"))
            if raw_bytes > validation_config.max_proposal_bytes:
                return False, "PROPOSAL_BYTE_LIMIT_EXCEEDED"
        except (TypeError, ValueError):
            pass

    # 2. Embedding dimension check
    if hasattr(proposal, "candidate_embedding") and proposal.candidate_embedding is not None:
        emb = proposal.candidate_embedding
        if not isinstance(emb, torch.Tensor) or emb.dtype != torch.float32:
            return False, "INVALID_EMBEDDING_DTYPE"
        if emb.ndim != 1 or emb.shape[0] != graph_config.embedding_dim:
            return False, "EMBEDDING_DIMENSION_MISMATCH"
        if not torch.isfinite(emb).all():
            return False, "NON_FINITE_EMBEDDING"

    # 3. Operation-specific target existence checks
    live_ids: set[int] = set()
    node_count = 0
    edge_count = 0

    if isinstance(graph_snapshot, Graph):
        live_ids = set(graph_snapshot._nodes.keys())
        node_count = len(graph_snapshot._nodes)
        edge_count = len(graph_snapshot._edges)
    elif isinstance(graph_snapshot, GraphState):
        live_ids = set(graph_snapshot.node_ids.tolist())
        node_count = int(graph_snapshot.node_ids.shape[0])
        edge_count = (
            int(graph_snapshot.edge_index.shape[1]) if graph_snapshot.edge_index.ndim == 2 else 0
        )

    # Capacity check
    if hasattr(proposal, "operation"):
        op = MemoryOperation.from_value(proposal.operation)
        if op == MemoryOperation.CREATE_NODE and node_count >= graph_config.max_nodes:
            return False, "GRAPH_MAX_NODES_EXCEEDED"
        if op == MemoryOperation.PROPOSE_LINK and edge_count >= graph_config.max_edges:
            return False, "GRAPH_MAX_EDGES_EXCEEDED"

        if op in (
            MemoryOperation.UPDATE_NODE,
            MemoryOperation.SUPERSEDE_NODE,
            MemoryOperation.PROPOSE_MERGE,
        ):
            target_ids = getattr(proposal, "target_node_ids", ())
            if not target_ids:
                return False, "MISSING_TARGET_NODE"
            for tid in target_ids:
                if tid not in live_ids:
                    return False, "TARGET_NODE_NOT_FOUND"

        if op == MemoryOperation.PROPOSE_LINK:
            target_ids = getattr(proposal, "target_node_ids", ())
            if len(target_ids) >= 2:
                for tid in target_ids[:2]:
                    if tid not in live_ids:
                        return False, "DANGLING_EDGE_ENDPOINT"
    elif hasattr(proposal, "source_node_id") and hasattr(proposal, "destination_node_id"):
        # LinkProposal compatibility
        if proposal.source_node_id not in live_ids or proposal.destination_node_id not in live_ids:
            return False, "DANGLING_EDGE_ENDPOINT"

    return True, None
