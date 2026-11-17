"""Run with: uv run python examples/encode_graph.py. No model or disk access."""

from __future__ import annotations

import torch

from rutam import Graph, RUTAMConfig, RUTAMGraphEncoder


def main() -> None:
    torch.manual_seed(7)
    config = RUTAMConfig()
    graph = Graph(config)
    for index in range(5):
        graph.create_node(
            "synthetic",
            f"Node {index}",
            node_id=10 * (index + 1),
            provenance="example:synthetic",
            embedding=torch.randn(config.graph.embedding_dim).tolist(),
        )
    for index in range(4):
        graph.add_relationship(
            10 * (index + 1),
            10 * (index + 2),
            "supports" if index % 2 else "links",
            weight=0.5 + index * 0.1,
            provenance="example:synthetic",
        )

    encoder = RUTAMGraphEncoder(config.encoder).eval()
    state = graph.to_state()  # Or GraphStorage(path).load().to_state(), outside forward.
    with torch.no_grad():
        output = encoder(state)
    print("node_ids:", state.node_ids.tolist())
    print("node_states:", tuple(output.node_states.shape), output.node_states.dtype)
    print("graph_summary:", tuple(output.graph_summary.shape))
    print("has_nodes:", output.has_nodes)
    print("trainable_parameters:", sum(p.numel() for p in encoder.parameters() if p.requires_grad))


if __name__ == "__main__":
    main()
