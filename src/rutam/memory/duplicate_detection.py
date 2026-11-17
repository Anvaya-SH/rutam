"""Duplicate detection across stable IDs, normalized content, edges, and embeddings."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rutam.graph.operations import Graph
from rutam.graph.state import GraphState


def detect_duplicates(
    proposal: Any,
    graph_snapshot: Graph | GraphState,
    *,
    similarity_threshold: float = 0.98,
) -> tuple[bool, str | None, tuple[int, ...]]:
    """Detect duplicate IDs, exact normalized content, existing edges, and vector similarity."""
    # 1. Exact stable ID match
    proposed_id = getattr(proposal, "id", None) or getattr(proposal, "node_id", None)
    if proposed_id is not None:
        if isinstance(graph_snapshot, Graph):
            if proposed_id in graph_snapshot._nodes or proposed_id in graph_snapshot._retired_ids:
                return True, "DUPLICATE_ID", (proposed_id,)
        elif isinstance(graph_snapshot, GraphState):
            if proposed_id in graph_snapshot.node_ids.tolist():
                return True, "DUPLICATE_ID", (proposed_id,)

    # 2. Exact normalized content match
    content = getattr(proposal, "content", None)
    if content and isinstance(content, str):
        norm = " ".join(content.lower().strip().split())
        if isinstance(graph_snapshot, Graph):
            for node_id, node in graph_snapshot._nodes.items():
                if " ".join(node.content.lower().strip().split()) == norm:
                    return True, "DUPLICATE_CONTENT", (node_id,)
        elif isinstance(graph_snapshot, GraphState) and graph_snapshot.metadata:
            for node_id_str, meta in graph_snapshot.metadata.items():
                if isinstance(meta, dict) and "content" in meta:
                    if " ".join(str(meta["content"]).lower().strip().split()) == norm:
                        try:
                            return True, "DUPLICATE_CONTENT", (int(node_id_str),)
                        except ValueError:
                            pass

    # 3. Duplicate relationship match
    src = getattr(proposal, "source_node_id", None) or getattr(proposal, "source_id", None)
    dst = getattr(proposal, "destination_node_id", None) or getattr(proposal, "target_id", None)
    rel = getattr(proposal, "relation_name", None) or getattr(proposal, "relationship", None)

    if src is not None and dst is not None and rel is not None:
        if isinstance(graph_snapshot, Graph):
            if (src, dst, rel) in graph_snapshot._edges:
                return True, "DUPLICATE_RELATIONSHIP", (src, dst)
        elif isinstance(graph_snapshot, GraphState):
            pass

    # 4. Bounded vector similarity duplicate detection
    cand_emb = getattr(proposal, "candidate_embedding", None)
    if cand_emb is not None and isinstance(cand_emb, torch.Tensor) and cand_emb.ndim == 1:
        if isinstance(graph_snapshot, Graph):
            # Collect node embeddings
            match_id: int | None = None
            max_sim = -1.0
            cand_norm = F.normalize(cand_emb.unsqueeze(0).float(), p=2, dim=-1)
            for node_id, node in graph_snapshot._nodes.items():
                if node.embedding is not None:
                    node_t = torch.tensor(node.embedding, dtype=torch.float32).unsqueeze(0)
                    node_norm = F.normalize(node_t, p=2, dim=-1)
                    sim = float((cand_norm * node_norm).sum().item())
                    if sim > max_sim:
                        max_sim = sim
                        match_id = node_id
            if match_id is not None and max_sim >= similarity_threshold:
                return True, "DUPLICATE_SIMILARITY", (match_id,)

        elif isinstance(graph_snapshot, GraphState):
            if graph_snapshot.node_embeddings.shape[0] > 0:
                cand_norm = F.normalize(cand_emb.unsqueeze(0).float(), p=2, dim=-1)
                nodes_norm = F.normalize(graph_snapshot.node_embeddings.float(), p=2, dim=-1)
                sims = (cand_norm * nodes_norm).sum(dim=-1)
                best_val, best_idx = sims.max(dim=0)
                if float(best_val.item()) >= similarity_threshold:
                    match_id = int(graph_snapshot.node_ids[best_idx].item())
                    return True, "DUPLICATE_SIMILARITY", (match_id,)
                if float(best_val.item()) >= similarity_threshold:
                    match_id = int(graph_snapshot.node_ids[best_idx].item())
                    return True, "DUPLICATE_SIMILARITY", (match_id,)

    return False, None, ()
