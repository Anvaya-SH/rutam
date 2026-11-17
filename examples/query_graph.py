"""Honest, untrained hidden-state -> query-head -> graph-selector CPU demonstration.

Run: uv run python examples/query_graph.py
No weights are edited to force different rankings or gate decisions.
"""

from __future__ import annotations

import torch

from rutam import Graph, RUTAMConfig, RUTAMGraphEncoder, RUTAMGraphSelector, RUTAMNeuralQueryHead


def main() -> None:
    torch.manual_seed(7)
    config = RUTAMConfig().with_overrides({"selector": {"top_k": 3, "selection_mode": "soft"}})
    graph = Graph(config)
    for row in range(10):
        graph.create_node(
            "synthetic",
            f"Node {row}",
            node_id=101 + 13 * row,
            provenance="example:synthetic",
            embedding=torch.randn(config.graph.embedding_dim).tolist(),
        )
    for row in range(9):
        graph.add_relationship(
            101 + 13 * row, 114 + 13 * row, "links", provenance="example:synthetic"
        )
    state = graph.to_state()
    encoder = RUTAMGraphEncoder(config.encoder).eval()
    selector = RUTAMGraphSelector(config.selector).eval()
    head = RUTAMNeuralQueryHead(config.query_head, selector_config=config.selector).eval()
    hidden = torch.randn((2, 5, config.query_head.backbone_hidden_dim))
    mask = torch.tensor([[False, True, True, True, False], [True, True, True, False, False]])
    with torch.no_grad():
        result = head.select_graph(
            hidden, selector, encoder(state), state.node_ids, attention_mask=mask
        )
    print(
        "Untrained seeded modules; synthetic hidden states, no LFM3 checkpoint or edited weights."
    )
    print("hidden_states:", tuple(hidden.shape))
    print("query_vectors:", tuple(result.query.query_vectors.shape))
    print(
        "query_vectors_differ:",
        not torch.allclose(result.query.query_vectors[0], result.query.query_vectors[1]),
    )
    for batch in range(2):
        selected = result.selection.selected_node_ids[batch][
            result.selection.selection_mask[batch]
        ].tolist()
        print(
            f"input_{batch + 1}: gate={result.query.query_gate[batch, 0].item():.6f}, "
            f"should_query={result.query.should_query[batch].item()}, selected IDs={selected}"
        )
    if not result.query.should_query.all().item():
        print(
            "At least one random gate declined access. "
            "Retrieval relevance and graph-use utility still need training."
        )
    elif torch.equal(result.selection.selected_node_ids[0], result.selection.selected_node_ids[1]):
        print(
            "Selections match (or both reads were declined). "
            "Relevance and utility must be trained; "
            "different inputs do not guarantee different untrained rankings."
        )
    else:
        print("Different selections are a random-initialization smoke test, not learned retrieval.")
    print("query_head_parameters:", sum(p.numel() for p in head.parameters() if p.requires_grad))


if __name__ == "__main__":
    main()
