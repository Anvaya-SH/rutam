from dataclasses import replace
from datetime import UTC, datetime

import pytest
import torch

from rutam import Edge, Graph, Node, RUTAMConfig
from rutam.exceptions import DuplicateIDError, GraphReferenceError, ValidationError


def make_graph(**graph_overrides):
    return Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 3, **graph_overrides}}))


def node(graph, node_id=None, **fields):
    return graph.create_node(
        "fact", "RUTAM has graph memory", node_id=node_id, provenance="test:source", **fields
    )


def link(graph, source, target, relationship="relates_to", **fields):
    return graph.add_relationship(source, target, relationship, provenance="test:edge", **fields)


def test_empty_graph_tensor_contract():
    graph = make_graph()
    assert graph.nodes == graph.edges == ()
    assert graph.version == 0
    state = graph.to_state()
    assert state.node_ids.shape == (0,)
    assert state.node_embeddings.shape == (0, 3)
    assert state.edge_index.shape == (2, 0)
    assert state.edge_types.shape == state.edge_weights.shape == (0,)
    for name in ("node_ids", "edge_index", "edge_types"):
        assert getattr(state, name).dtype == torch.int64
    for name in ("node_embeddings", "edge_weights"):
        assert getattr(state, name).dtype == torch.float32
    state.validate()


@pytest.mark.parametrize("invalid", [{}, [], False])
def test_invalid_graph_configuration_objects(invalid):
    with pytest.raises(ValidationError):
        Graph(invalid)


def test_node_crud_and_stable_ids():
    graph = make_graph()
    first = node(graph, 10, embedding=[0.1, 0.2, 0.3], confidence=0.7)
    assert graph.get_node(10) == first
    updated = graph.update_node(10, content="Updated", confidence=0.8, embedding=None)
    assert updated.created_at == first.created_at
    assert updated.updated_at >= first.updated_at
    assert updated.content == "Updated" and updated.embedding is None
    with pytest.raises(DuplicateIDError):
        graph.add_node(first)
    graph.delete_node(10)
    with pytest.raises(GraphReferenceError):
        graph.get_node(10)
    with pytest.raises(DuplicateIDError):
        graph.add_node(first)
    assert node(graph).id == 11
    assert graph.version == 4


def test_relationship_crud_and_node_cleanup():
    graph = make_graph()
    for value in (9, 2, 7):
        node(graph, value)
    first = link(graph, 9, 2, weight=0.3)
    link(graph, 9, 2, "supports")
    link(graph, 7, 9)
    assert graph.get_relationship(9, 2, "relates_to") == first
    updated = graph.update_relationship(9, 2, "relates_to", weight=0.6)
    assert updated.weight == torch.tensor(0.6, dtype=torch.float32).item()
    assert updated.timestamp >= first.timestamp
    graph.remove_relationship(9, 2, "supports")
    with pytest.raises(GraphReferenceError):
        graph.get_relationship(9, 2, "supports")
    graph.delete_node(9)
    assert graph.edges == ()
    assert [n.id for n in graph.nodes] == [2, 7]
    assert graph.to_state().relationship_types == {"relates_to": 0, "supports": 1}


def test_deterministic_directed_neighbors_and_bounded_cycles():
    graph = make_graph()
    for value in (40, 10, 30, 20):
        node(graph, value)
    for source, target in ((10, 30), (10, 20), (20, 40), (40, 10)):
        link(graph, source, target)
    link(graph, 30, 40, "supports")
    assert [n.id for n in graph.neighbors(10)] == [20, 30]
    assert [n.id for n in graph.neighbors(10, direction="incoming")] == [40]
    assert [n.id for n in graph.neighbors(10, direction="both")] == [20, 30, 40]
    assert graph.traverse(10, 0) == ()
    assert [n.id for n in graph.traverse(10, 1)] == [20, 30]
    assert [n.id for n in graph.traverse(10, 20)] == [20, 30, 40]
    assert graph.neighbors(30, relationship="relates_to") == ()
    assert [n.id for n in graph.traverse(30, 2, relationship="supports")] == [40]


def test_undirected_canonical_edges_and_self_loops():
    graph = make_graph(directed=False)
    node(graph, 10)
    node(graph, 2)
    edge = link(graph, 10, 2)
    assert edge.key == (2, 10, "relates_to")
    assert graph.get_relationship(10, 2, "relates_to") == edge
    assert graph.get_relationship(2, 10, "relates_to") == edge
    assert [n.id for n in graph.neighbors(10, direction="incoming")] == [2]
    with pytest.raises(ValidationError):
        link(graph, 2, 10)
    link(graph, 2, 2)
    assert [n.id for n in graph.traverse(2, 5)] == [10]
    graph.update_relationship(10, 2, "relates_to", weight=2.0)
    graph.remove_relationship(10, 2, "relates_to")
    assert len(graph.edges) == 1


def test_row_indices_are_distinct_from_persistent_ids_and_state_is_detached():
    graph = make_graph()
    node(graph, 90, embedding=[1, 2, 3])
    node(graph, 7)
    link(graph, 90, 7)
    state = graph.to_state()
    assert state.node_ids.tolist() == [7, 90]
    assert state.edge_index.tolist() == [[1], [0]]
    assert state.node_embeddings.tolist() == [[0, 0, 0], [1, 2, 3]]
    restored = Graph.from_state(state)
    assert restored.nodes == graph.nodes and restored.edges == graph.edges
    state.node_embeddings[1, 0] = 99
    state.node_metadata[0]["content"] = "changed"
    assert graph.get_node(90).embedding[0] == 1
    assert restored.get_node(7).content == graph.get_node(7).content


@pytest.mark.parametrize(
    "action",
    [
        lambda g: g.get_node(42),
        lambda g: g.delete_node(42),
        lambda g: g.update_node(42, content="bad"),
        lambda g: link(g, 0, 42),
        lambda g: g.get_relationship(0, 0, "missing"),
        lambda g: g.remove_relationship(0, 0, "missing"),
        lambda g: g.update_relationship(0, 0, "missing", weight=2),
        lambda g: g.neighbors(42),
        lambda g: g.traverse(42, 2),
    ],
)
def test_invalid_references_do_not_mutate_graph(action):
    graph = make_graph()
    node(graph)
    with pytest.raises(GraphReferenceError):
        action(graph)
    assert graph.version == 1 and len(graph.nodes) == 1 and not graph.edges


@pytest.mark.parametrize(
    "fields",
    [
        {"id": True},
        {"id": -1},
        {"id": 2**63},
        {"node_type": ""},
        {"content": 1},
        {"confidence": 1.1},
        {"confidence": float("nan")},
        {"confidence": 10**400},
        {"provenance": ""},
        {"embedding": [float("inf")]},
        {"embedding": [1e39]},
        {"created_at": datetime(2020, 1, 1)},
        {
            "created_at": datetime(2025, 1, 1, tzinfo=UTC),
            "updated_at": datetime(2024, 1, 1, tzinfo=UTC),
        },
    ],
)
def test_invalid_node_schema(fields):
    with pytest.raises(ValidationError):
        Node(**({"id": 0, "node_type": "fact", "content": "text"} | fields))


@pytest.mark.parametrize(
    "fields",
    [
        {"source_id": True},
        {"target_id": -1},
        {"relationship": ""},
        {"weight": -0.1},
        {"weight": float("nan")},
        {"weight": 1e39},
        {"timestamp": datetime(2020, 1, 1)},
    ],
)
def test_invalid_edge_schema(fields):
    with pytest.raises(ValidationError):
        Edge(**({"source_id": 0, "target_id": 1, "relationship": "knows"} | fields))


def test_provenance_capacity_and_invalid_mutations():
    graph = make_graph(max_nodes=1, max_edges=1)
    with pytest.raises(ValidationError):
        graph.create_node("fact", "No source")
    node(graph)
    for operation in (
        lambda: node(graph),
        lambda: graph.update_node(0, id=2),
        lambda: graph.update_node(0, embedding=[1, 2]),
        lambda: graph.update_node(0, provenance=None),
        lambda: graph.add_relationship(0, 0, "unsourced"),
        lambda: graph.traverse(0, -1),
        lambda: graph.neighbors(0, direction="sideways"),
    ):
        with pytest.raises(ValidationError):
            operation()
    link(graph, 0, 0)
    with pytest.raises(ValidationError):
        link(graph, 0, 0, "another_type")
    with pytest.raises(ValidationError):
        graph.update_relationship(0, 0, "relates_to", unexpected="rename")
    assert graph.version == 2
    permissive = Graph(
        RUTAMConfig().with_overrides(
            {"memory": {"require_provenance": False, "validation_enabled": False}}
        )
    )
    permissive.create_node("fact", "No provenance required")
    permissive.add_relationship(0, 0, "self")
    with pytest.raises(GraphReferenceError):
        permissive.add_relationship(0, 99, "still invalid")


def test_metadata_is_detached_and_json_only():
    source = {"owner": {"name": "Anvaya.sh"}, "tags": ["foundation"]}
    graph = Graph(metadata=source)
    source["owner"]["name"] = "mutated"
    assert graph.metadata["owner"]["name"] == "Anvaya.sh"
    metadata = graph.metadata
    metadata["tags"].append("changed")
    assert graph.metadata["tags"] == ["foundation"]
    graph.set_metadata({"phase": 1})
    assert graph.version == 1
    for invalid in ({"bad": object()}, {1: "key"}, {"bad": float("nan")}):
        with pytest.raises(ValidationError):
            graph.set_metadata(invalid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_ids", torch.tensor([0], dtype=torch.int32)),
        ("node_embeddings", torch.zeros((1, 2), dtype=torch.float32)),
        ("node_embeddings", torch.zeros((1, 3), dtype=torch.float64)),
        ("node_embeddings", torch.tensor([[float("nan"), 0, 0]], dtype=torch.float32)),
        ("edge_index", torch.tensor([[0], [5]], dtype=torch.int64)),
        ("edge_index", torch.tensor([[0, 0]], dtype=torch.int64)),
        ("edge_types", torch.tensor([99], dtype=torch.int64)),
        ("edge_weights", torch.tensor([-1], dtype=torch.float32)),
        ("edge_weights", torch.tensor([1], dtype=torch.int64)),
        ("graph_version", True),
        ("next_node_id", 0),
        ("retired_node_ids", (0,)),
        ("relationship_types", {"relates_to": True}),
        ("node_metadata", []),
        ("edge_metadata", []),
    ],
)
def test_invalid_tensor_state(field, value):
    graph = make_graph()
    node(graph)
    link(graph, 0, 0)
    with pytest.raises(ValidationError):
        Graph.from_state(replace(graph.to_state(), **{field: value}))
