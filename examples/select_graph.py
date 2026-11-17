"""Ten-node/two-query CPU smoke test, with transparent identity scoring weights.

Run: uv run python examples/select_graph.py
The encoder is seeded/random; queries use two encoded node states as prototypes.
Identity selector projections make the demonstration interpretable, not trained.
"""

from __future__ import annotations

import torch

from rutam import Graph, RUTAMConfig, RUTAMGraphEncoder, RUTAMGraphSelector


def main() -> None:
    torch.manual_seed(7)
    config = RUTAMConfig().with_overrides(
        {"selector": {"top_k": 3, "normalize_vectors": True, "chunk_size": 4}}
    )
    graph = Graph(config)
    for row in range(10):
        features = torch.zeros(config.graph.embedding_dim)
        features[row] = 1
        graph.create_node(
            "synthetic",
            f"Synthetic node {row}",
            node_id=101 + 13 * row,
            provenance="example:synthetic",
            embedding=features.tolist(),
        )
    for row in range(9):
        graph.add_relationship(
            101 + 13 * row,
            114 + 13 * row,
            "supports" if row % 2 else "related",
            weight=0.5 + row * 0.05,
            provenance="example:synthetic",
        )
    state = graph.to_state()
    encoder = RUTAMGraphEncoder(config.encoder).eval()
    selector = RUTAMGraphSelector(config.selector).eval()
    with torch.no_grad():
        selector.scorer.query_projection.weight.copy_(torch.eye(config.selector.score_dim))
        selector.scorer.node_projection.weight.copy_(torch.eye(config.selector.score_dim))
        encoded = encoder(state)
        queries = torch.stack((encoded.node_states[0], encoded.node_states[-1]))
        selected = selector(queries, encoded, state.node_ids)
    print("Synthetic smoke test; seeded encoder and identity selector projections, no training.")
    print("persistent_node_ids:", state.node_ids.tolist())
    for batch, prototype in enumerate((101, 218)):
        print(
            f"query_{batch + 1} (prototype ID {prototype}): "
            f"selected IDs {selected.selected_node_ids[batch].tolist()}, "
            f"row indices {selected.selected_indices[batch].tolist()}"
        )
    print("selected_states:", tuple(selected.selected_states.shape))
    print("selector_parameters:", sum(p.numel() for p in selector.parameters() if p.requires_grad))


if __name__ == "__main__":
    main()
