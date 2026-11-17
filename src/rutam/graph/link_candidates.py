"""Bounded discrete candidate generation, structural features and negative sampling."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rutam.config import LinkerConfig, positive_int
from rutam.exceptions import ValidationError
from rutam.graph.embeddings import validate_features, validate_tensor
from rutam.graph.link_output import LinkCandidates, NegativeSamples
from rutam.graph.neighborhood import SparseNeighborhood
from rutam.graph.selection import stable_topk


def _rows(values: torch.Tensor | None, n: int, name: str) -> list[int]:
    if values is None:
        return list(range(n))
    validate_tensor(values, name, torch.int64)
    if values.ndim != 1 or ((values < 0) | (values >= n)).any().item():
        raise ValidationError(f"{name} must contain valid row offsets [K]")
    return list(dict.fromkeys(values.tolist()))


def generate_link_candidates(
    index: SparseNeighborhood,
    config: LinkerConfig,
    *,
    node_states: torch.Tensor | None = None,
    candidate_pairs: torch.Tensor | None = None,
    selected_indices: torch.Tensor | None = None,
    candidate_indices: torch.Tensor | None = None,
    node_mask: torch.Tensor | None = None,
) -> LinkCandidates:
    n = len(index.node_ids)
    allowed = [True] * n
    if node_mask is not None:
        validate_tensor(node_mask, "node_mask", torch.bool)
        if node_mask.shape != (n,):
            raise ValidationError("node_mask must be [N]")
        allowed = node_mask.tolist()
    sources = sorted(
        (x for x in _rows(selected_indices, n, "selected_indices") if allowed[x]),
        key=index.node_ids.__getitem__,
    )
    targets = sorted(
        (x for x in _rows(candidate_indices, n, "candidate_indices") if allowed[x]),
        key=index.node_ids.__getitem__,
    )
    target_set = set(targets)
    found: set[tuple[int, int]] = set()
    duplicates, attempts, truncated = 0, 0, False

    def add(source: int, target: int) -> bool:
        nonlocal duplicates, attempts, truncated
        attempts += 1
        if attempts > config.max_candidate_expansions:
            truncated = True
            return False
        if not allowed[source] or not allowed[target] or target not in target_set:
            return True
        if source == target and not config.allow_self_links:
            return True
        if not config.directed and index.node_ids[source] > index.node_ids[target]:
            source, target = target, source
        pair = (source, target)
        if pair in found:
            duplicates += 1
        else:
            found.add(pair)
        if len(found) >= config.max_candidate_pairs:
            truncated = True
            return False
        return True

    if candidate_pairs is not None:
        validate_tensor(candidate_pairs, "candidate_pairs", torch.int64)
        if candidate_pairs.ndim == 2 and candidate_pairs.shape[0] > config.max_candidate_expansions:
            raise ValidationError(
                "Explicit input exceeds max_candidate_expansions; supply bounded batches"
            )
        if (
            candidate_pairs.ndim != 2
            or candidate_pairs.shape[1] != 2
            or ((candidate_pairs < 0) | (candidate_pairs >= n)).any().item()
        ):
            raise ValidationError("Explicit candidate pairs must be valid row offsets [C,2]")
        if (
            not config.allow_self_links
            and (candidate_pairs[:, 0] == candidate_pairs[:, 1]).any().item()
        ):
            raise ValidationError("Self-link candidates are disabled")
        # Input-size validation is linear; expansion/scoring remains bounded.
        pairs = sorted(
            map(tuple, candidate_pairs.tolist()),
            key=lambda x: (index.node_ids[x[0]], index.node_ids[x[1]]),
        )
        for source, target in pairs:
            if not add(source, target):
                break
    elif config.candidate_strategy == "explicit":
        raise ValidationError("Explicit strategy requires candidate_pairs")
    elif config.candidate_strategy == "similarity":
        if selected_indices is None or len(sources) > config.max_similarity_anchors:
            raise ValidationError(
                "Similarity requires explicit anchors within max_similarity_anchors"
            )
        if node_states is None:
            raise ValidationError("Similarity candidates require node_states")
        validate_features(node_states, config.hidden_dim, "node_states")
        if node_states.shape[0] != n:
            raise ValidationError("node_states and topology node counts differ")
        with torch.no_grad():
            anchors = F.normalize(node_states[sources].double(), dim=1, eps=1e-12)
            ids = torch.tensor(index.node_ids, dtype=torch.int64)
            best = stable_topk(
                node_states.new_empty((len(sources), 0)), ids[:0], config.max_neighbors_per_node
            )
            for start in range(0, n, config.similarity_chunk_size):
                stop = min(start + config.similarity_chunk_size, n)
                values = F.normalize(node_states[start:stop].double(), dim=1, eps=1e-12)
                scores = (anchors @ values.T).float()
                valid = torch.tensor(allowed[start:stop]).expand(len(sources), -1).clone()
                valid &= torch.tensor([row in target_set for row in range(start, stop)])[None, :]
                if not config.allow_self_links:
                    valid &= torch.tensor(sources)[:, None] != torch.arange(start, stop)[None, :]
                rows = torch.arange(start, stop).expand(len(sources), -1)
                best = stable_topk(
                    torch.cat((best.scores, scores), 1),
                    torch.cat((best.node_ids, ids[start:stop].expand(len(sources), -1)), 1),
                    config.max_neighbors_per_node,
                    mask=torch.cat((best.mask, valid), 1),
                    indices=torch.cat((best.indices, rows), 1),
                )
            for anchor, source in enumerate(sources):
                for target in best.indices[anchor][best.mask[anchor]].tolist():
                    if not add(source, target):
                        break
                if truncated:
                    break
    elif config.candidate_strategy == "selected":
        if selected_indices is None:
            raise ValidationError("Selected pairing requires explicit selected_indices [K]")
        for source in sources:
            for target in targets:
                if not add(source, target):
                    break
            if truncated:
                break
    else:
        direction = "outgoing" if config.directed else "both"
        for source in sources:
            attempts += 1
            if attempts > config.max_candidate_expansions:
                truncated = True
                break
            neighbors = index.neighbors(source, direction, config.max_neighbors_per_node)
            for first in neighbors:
                if config.candidate_strategy != "two_hop" and not add(source, first.node):
                    break
                if config.max_candidate_hops == 2:
                    for second in index.neighbors(
                        first.node, direction, config.max_neighbors_per_node
                    ):
                        if not add(source, second.node):
                            break
                if truncated:
                    break
            if truncated:
                break
    ordered = sorted(found, key=lambda pair: (index.node_ids[pair[0]], index.node_ids[pair[1]]))
    pairs = torch.tensor(ordered, dtype=torch.int64).reshape(-1, 2)
    ids_tensor = torch.tensor(index.node_ids, dtype=torch.int64)
    return LinkCandidates(pairs, ids_tensor[pairs], duplicates, truncated)


def structural_link_features(
    index: SparseNeighborhood,
    pairs: torch.Tensor,
    num_relation_types: int,
    *,
    node_types: tuple[str, ...] | None = None,
    compatible_types: tuple[tuple[str, str], ...] = (),
    directed: bool = True,
) -> torch.Tensor:
    """[log1p(shared neighbors), two-hop flag, reverse flag, type compatibility, frequencies...]."""
    positive_int(num_relation_types, "num_relation_types")
    validate_tensor(pairs, "pairs", torch.int64)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or ((pairs < 0) | (pairs >= len(index.node_ids))).any().item()
    ):
        raise ValidationError("Structural feature pairs must be valid row offsets [C,2]")
    if node_types is not None and (
        len(node_types) != len(index.node_ids)
        or any(not isinstance(x, str) or not x for x in node_types)
    ):
        raise ValidationError("node_types must supply one nonempty string per node")
    direction = "outgoing" if directed else "both"
    outgoing = [
        set(x.node for x in index.neighbors(row, direction)) for row in range(len(index.node_ids))
    ]
    incoming = (
        [set(x.node for x in index.incoming[row]) for row in range(len(index.node_ids))]
        if directed
        else outgoing
    )
    frequencies = torch.zeros(num_relation_types, dtype=torch.float32)
    for _, _, relation in index.typed_edges:
        if relation >= num_relation_types:
            raise ValidationError("Unknown graph relation ID")
        frequencies[relation] += 1
    frequencies /= max(1, len(index.typed_edges))
    features = torch.zeros((pairs.shape[0], 4 + num_relation_types), dtype=torch.float32)
    compatibility = set(compatible_types)
    for row, (source, target) in enumerate(pairs.tolist()):
        features[row, 0] = torch.log1p(
            torch.tensor(float(len(outgoing[source] & outgoing[target])))
        )
        features[row, 1] = bool(outgoing[source] & incoming[target])
        features[row, 2] = source in outgoing[target]
        features[row, 3] = bool(
            node_types is not None and (node_types[source], node_types[target]) in compatibility
        )
    features[:, 4:] = frequencies
    return features


def sample_negative_pairs(
    index: SparseNeighborhood,
    num_samples: int,
    *,
    seed: int = 0,
    directed: bool = True,
    allow_self_links: bool = False,
    max_attempts: int = 10000,
) -> NegativeSamples:
    positive_int(num_samples, "num_samples", minimum=0)
    positive_int(seed, "seed", minimum=0)
    positive_int(max_attempts, "max_attempts")
    n = len(index.node_ids)
    known = {(s, t) for s, t, _ in index.typed_edges}
    if not directed:
        known |= {(t, s) for s, t in known}
    found = set()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if n:
        for _ in range(max_attempts):
            if len(found) == num_samples:
                break
            source, target = torch.randint(n, (2,), generator=generator).tolist()
            if (source == target and not allow_self_links) or (source, target) in known:
                continue
            if not directed and index.node_ids[source] > index.node_ids[target]:
                source, target = target, source
            found.add((source, target))
    ordered = sorted(found, key=lambda x: (index.node_ids[x[0]], index.node_ids[x[1]]))
    return NegativeSamples(
        torch.tensor(ordered, dtype=torch.int64).reshape(-1, 2),
        num_samples,
        len(found) < num_samples,
    )
