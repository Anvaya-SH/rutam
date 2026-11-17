"""8 nodes, 10 typed directed edges and two untrained neural queries on CPU."""

from __future__ import annotations

from time import perf_counter

import torch

from rutam import Graph, RUTAMConfig, RUTAMGraphEncoder, RUTAMGraphTraversal


def main() -> None:
    torch.manual_seed(7)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 8, "dropout": 0},
            "traversal": {
                "query_dim": 8,
                "hidden_dim": 8,
                "path_hidden_dim": 8,
                "score_hidden_dim": 8,
                "max_visited_nodes": 8,
                "max_candidate_paths": 24,
                "max_selected_paths": 3,
            },
        }
    )
    graph = Graph(config)
    for row in range(8):
        graph.create_node(
            "synthetic",
            str(row),
            node_id=(row + 1) * 10,
            provenance="example:synthetic",
            embedding=torch.randn(8).tolist(),
        )
    labels = ("links", "depends_on", "supports")
    for source, target, relation in (
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (0, 4, 0),
        (4, 3, 1),
        (3, 5, 2),
        (5, 6, 0),
        (6, 7, 1),
        (2, 0, 2),
        (4, 6, 0),
    ):
        graph.add_relationship(
            (source + 1) * 10, (target + 1) * 10, labels[relation], provenance="example:edge"
        )
    state = graph.to_state()
    encoder = RUTAMGraphEncoder(config.encoder).eval()
    traversal = RUTAMGraphTraversal(config.traversal).eval()
    queries = torch.randn((2, 8))
    seeds, seed_mask = torch.tensor([[0], [0]]), torch.ones((2, 1), dtype=torch.bool)
    with torch.no_grad():
        encoded = encoder(state)
        output = traversal.from_state(queries, encoded, state, seeds, seed_mask)
        start = perf_counter()
        for _ in range(10):
            traversal.from_state(queries, encoded, state, seeds, seed_mask)
        milliseconds = (perf_counter() - start) * 1000 / 10
    print("Untrained tensor reasoning smoke test; 8 nodes, 10 directed typed edges; seed ID 10.")
    print("visited IDs:", output.visited_node_ids[0][output.visited_mask[0]].tolist())
    print("candidate paths per query:", output.path_mask.sum(1).tolist())
    print("query logits differ:", not torch.allclose(output.path_scores[0], output.path_scores[1]))
    for batch in range(2):
        print(f"query {batch + 1} selected paths:")
        for candidate in output.selected_path_indices[batch][
            output.selected_path_mask[batch]
        ].tolist():
            nodes = output.path_node_ids[batch, candidate][
                output.path_node_mask[batch, candidate]
            ].tolist()
            relations = output.path_relation_ids[batch, candidate][
                output.path_edge_mask[batch, candidate]
            ].tolist()
            print(
                f"  nodes={nodes}, relation IDs={relations}, "
                f"logit={output.path_scores[batch, candidate].item():.6f}"
            )
    print("reasoning context shape:", tuple(output.reasoning_context.shape))
    print(f"mean warmed traversal CPU runtime, 10 calls: {milliseconds:.3f} ms/call")
    print("No trained semantic-reasoning quality or large-graph benchmark is claimed.")


if __name__ == "__main__":
    main()
