"""Small deterministic graphs for traversal and linking; no global RNG mutation."""

import pytest
import torch

from rutam import Graph, RUTAMConfig


@pytest.fixture
def reasoning_graph() -> Graph:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 4},
            "encoder": {"input_dim": 4, "hidden_dim": 8, "num_relation_types": 4, "dropout": 0},
            "selector": {
                "query_dim": 5,
                "hidden_dim": 8,
                "score_dim": 6,
                "top_k": 2,
                "selection_mode": "soft",
            },
            "query_head": {
                "backbone_hidden_dim": 6,
                "graph_hidden_dim": 8,
                "query_dim": 5,
                "intermediate_dim": 10,
                "dropout": 0,
            },
            "traversal": {
                "hidden_dim": 8,
                "path_hidden_dim": 8,
                "query_dim": 5,
                "num_relation_types": 4,
                "score_hidden_dim": 8,
                "max_visited_nodes": 8,
                "max_candidate_paths": 16,
                "max_selected_paths": 4,
                "max_neighbors_per_node": 4,
            },
        }
    )
    graph = Graph(config)
    generator = torch.Generator().manual_seed(123)
    for row in range(8):
        graph.create_node(
            "concept",
            str(row),
            node_id=(row + 1) * 10,
            provenance=f"test:node:{row}",
            embedding=torch.randn(4, generator=generator).tolist(),
        )
    labels = ("links", "supports", "depends_on")
    for source, target, kind in (
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
            (source + 1) * 10, (target + 1) * 10, labels[kind], provenance="test:edge", weight=0.7
        )
    return graph


@pytest.fixture
def link_graph() -> Graph:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 4},
            "encoder": {"input_dim": 4, "hidden_dim": 8, "dropout": 0},
            "linker": {
                "hidden_dim": 8,
                "scoring_hidden_dim": 16,
                "structural_feature_dim": 4,
                "query_dim": 5,
                "dropout": 0,
                "relation_labels": ["related_to", "depends_on", "supports"],
                "max_candidate_pairs": 32,
                "negative_sampling_attempts": 300,
            },
        }
    )
    graph = Graph(config)
    generator = torch.Generator().manual_seed(99)
    for row in range(6):
        graph.create_node(
            "concept" if row % 2 else "fact",
            str(row),
            node_id=(row + 1) * 10,
            provenance=f"test:source:{row}",
            embedding=torch.randn(4, generator=generator).tolist(),
        )
    labels = ("related_to", "depends_on", "supports")
    for source, target, kind in (
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (0, 2, 0),
        (3, 4, 1),
        (4, 5, 2),
        (5, 0, 0),
        (1, 4, 1),
    ):
        graph.add_relationship(
            (source + 1) * 10, (target + 1) * 10, labels[kind], provenance="test:edge"
        )
    return graph
