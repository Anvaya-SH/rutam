import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from rutam import Graph, GraphStorage, RUTAMConfig
from rutam.exceptions import (
    CorruptSnapshotError,
    DuplicateIDError,
    SnapshotNotFoundError,
    StorageError,
)
from rutam.graph.storage import MANIFEST_FILE, METADATA_FILE, TENSOR_FILE


def make_graph():
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 4}})
    graph = Graph(config, metadata={"project": "RUTAM", "owner": "Anvaya.sh"})
    graph.create_node("project", "RUTAM", node_id=20, provenance="test:a", embedding=[0.1] * 4)
    graph.create_node("organization", "Anvaya.sh", node_id=3, provenance="test:b")
    graph.create_node(
        "zero", "Explicit zero embedding", node_id=10, provenance="test:c", embedding=[0] * 4
    )
    graph.add_relationship(20, 3, "developed_by", provenance="test:edge", weight=0.7)
    graph.add_relationship(3, 20, "develops", provenance="test:edge", weight=0.4)
    return graph


def assert_equal(left, right):
    assert left.nodes == right.nodes
    assert left.edges == right.edges
    assert left.version == right.version
    assert left.metadata == right.metadata
    assert left.config == right.config
    a, b = left.to_state(), right.to_state()
    assert a.relationship_types == b.relationship_types
    assert a.next_node_id == b.next_node_id
    assert a.retired_node_ids == b.retired_node_ids
    for name in a.tensors():
        assert torch.equal(a.tensors()[name], b.tensors()[name])


def rewrite_metadata(path, mutate):
    """Recompute hashes to test semantic validation beyond checksum validation."""
    metadata_path = path / METADATA_FILE
    value = json.loads(metadata_path.read_text())
    mutate(value)
    payload = json.dumps(value).encode()
    metadata_path.write_bytes(payload)
    refresh_manifest(path, METADATA_FILE, payload)


def refresh_manifest(path, name, payload):
    manifest_path = path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("dimension", [1, 7, 128])
def test_empty_persistence(tmp_path, dimension):
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": dimension}}))
    storage = GraphStorage(tmp_path / "graph")
    storage.save(graph)
    loaded = storage.load()
    assert_equal(graph, loaded)
    assert loaded.to_state().node_embeddings.shape == (0, dimension)
    assert loaded.to_state().edge_index.shape == (2, 0)
    with safe_open(storage.path / TENSOR_FILE, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(graph.to_state().tensors())


def test_round_trip_and_optional_embedding_integrity(tmp_path):
    graph = make_graph()
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    loaded = storage.load()
    assert_equal(graph, loaded)
    assert loaded.get_node(3).embedding is None
    assert loaded.get_node(10).embedding == (0, 0, 0, 0)
    assert loaded.get_node(20).embedding == graph.get_node(20).embedding
    metadata = json.loads((tmp_path / METADATA_FILE).read_text())
    assert metadata["id_to_row"] == {"3": 0, "10": 1, "20": 2}
    assert metadata["schema_version"] == 1


def test_repeated_save_load_updates_and_retired_ids(tmp_path):
    graph = make_graph()
    storage = GraphStorage(tmp_path)
    graph.delete_node(20)
    for index in range(8):
        expected = graph.create_node("fact", f"Fact {index}", provenance=f"test:{index}")
        graph.add_relationship(3, expected.id, "links", provenance="test:edge")
        graph.update_node(3, content=f"Revision {index}")
        graph.set_metadata({"iteration": index})
        storage.save(graph)
        loaded = storage.load()
        assert_equal(graph, loaded)
        assert expected.id == 21 + index
        graph = loaded
    with pytest.raises(DuplicateIDError):
        graph.create_node("fact", "reused", node_id=20, provenance="test:source")
    assert len(graph.nodes) == 10
    assert len(graph.edges) == 8


def test_deleting_entire_graph_round_trips_tombstones_and_type_ids(tmp_path):
    graph = make_graph()
    for value in (3, 10, 20):
        graph.delete_node(value)
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    loaded = storage.load()
    assert_equal(graph, loaded)
    assert loaded.create_node("fact", "new", provenance="test").id == 21
    assert loaded.to_state().relationship_types == {"developed_by": 0, "develops": 1}


def test_missing_snapshots(tmp_path):
    with pytest.raises(SnapshotNotFoundError):
        GraphStorage(tmp_path / "missing").load()
    with pytest.raises(SnapshotNotFoundError):
        GraphStorage(tmp_path).load()


@pytest.mark.parametrize("name", [TENSOR_FILE, METADATA_FILE, MANIFEST_FILE])
def test_corrupt_or_missing_file(tmp_path, name):
    storage = GraphStorage(tmp_path)
    storage.save(make_graph())
    file = tmp_path / name
    original = file.read_bytes()
    file.write_bytes(original[: len(original) // 2])
    with pytest.raises(CorruptSnapshotError):
        storage.load()
    file.unlink()
    with pytest.raises(CorruptSnapshotError):
        storage.load()


def test_mixed_snapshots_are_rejected(tmp_path):
    graph = make_graph()
    first = GraphStorage(tmp_path / "first")
    second = GraphStorage(tmp_path / "second")
    first.save(graph)
    graph.update_node(3, content="New snapshot")
    second.save(graph)
    (first.path / METADATA_FILE).write_bytes((second.path / METADATA_FILE).read_bytes())
    with pytest.raises(CorruptSnapshotError, match="Checksum"):
        first.load()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(schema_version=2),
        lambda m: m.update(graph_version=-1),
        lambda m: m.update(snapshot_id="0" * 32),
        lambda m: m.update(next_node_id=1),
        lambda m: m.update(retired_node_ids=[3]),
        lambda m: m.update(id_to_row={"3": 1, "10": 0, "20": 2}),
        lambda m: m["nodes"][0].update(id=99),
        lambda m: m["nodes"][0].update(has_embedding="yes"),
        lambda m: m["nodes"][0].update(provenance=None),
        lambda m: m["edges"][0].update(source_id=999),
        lambda m: m["edges"][0].update(relationship="unknown"),
        lambda m: m.update(relationship_types={"a": 0, "b": 0}),
        lambda m: m["config"]["graph"].update(embedding_dim=99),
    ],
)
def test_semantically_invalid_metadata_with_valid_checksum(tmp_path, mutate):
    storage = GraphStorage(tmp_path)
    storage.save(make_graph())
    rewrite_metadata(tmp_path, mutate)
    with pytest.raises(CorruptSnapshotError):
        storage.load()


@pytest.mark.parametrize(
    "name,invalid",
    [
        ("node_ids", torch.tensor([3, 10, 20], dtype=torch.int32)),
        ("node_embeddings", torch.zeros((3, 2), dtype=torch.float32)),
        ("edge_index", torch.tensor([[0, 9], [2, 0]], dtype=torch.int64)),
        ("edge_weights", torch.tensor([float("nan"), 1], dtype=torch.float32)),
        ("edge_types", torch.tensor([42, 0], dtype=torch.int64)),
    ],
)
def test_corrupt_tensor_contract_with_valid_checksum(tmp_path, name, invalid):
    storage = GraphStorage(tmp_path)
    storage.save(make_graph())
    file = tmp_path / TENSOR_FILE
    tensors = load_tensors(file.read_bytes())
    with safe_open(file, framework="pt") as handle:
        header_metadata = handle.metadata()
    tensors[name] = invalid
    payload = save_tensors(tensors, metadata=header_metadata)
    file.write_bytes(payload)
    refresh_manifest(tmp_path, TENSOR_FILE, payload)
    with pytest.raises(CorruptSnapshotError):
        storage.load()


def test_interrupted_write_fails_closed_and_retry_recovers(tmp_path):
    graph = make_graph()
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    graph.update_node(3, content="Updated")
    import os

    original_replace = os.replace
    count = 0

    def interrupted(source, target):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated interruption between tensor and JSON replacement")
        original_replace(source, target)

    with patch("rutam.graph.storage.os.replace", side_effect=interrupted):
        with pytest.raises(StorageError, match="simulated interruption"):
            storage.save(graph)
    with pytest.raises(CorruptSnapshotError):
        storage.load()
    assert not list(tmp_path.glob("*.tmp"))
    storage.save(graph)
    assert_equal(graph, storage.load())


def test_failed_write_before_publication_preserves_previous_snapshot(tmp_path):
    graph = make_graph()
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    original = storage.load()
    graph.update_node(3, content="New")
    with patch("rutam.graph.storage.os.fsync", side_effect=OSError("disk full")):
        with pytest.raises(StorageError):
            storage.save(graph)
    assert_equal(original, storage.load())
    assert not list(tmp_path.glob("*.tmp"))


def test_auto_save_and_memory_rollback_on_failure(tmp_path):
    config = RUTAMConfig().with_overrides({"graph": {"auto_save": True, "embedding_dim": 2}})
    storage = GraphStorage(tmp_path)
    graph = Graph(config, storage=storage)
    graph.create_node("fact", "A", provenance="test")
    assert_equal(graph, storage.load())
    with patch.object(storage, "save", side_effect=StorageError("unavailable")):
        with pytest.raises(StorageError):
            graph.delete_node(0)
    assert graph.get_node(0).content == "A"
    assert graph.version == 1
    graph.create_node("fact", "B", provenance="test")
    assert_equal(graph, storage.load())
    loaded = storage.load()
    loaded.add_relationship(0, 1, "links", provenance="test")
    assert_equal(loaded, storage.load())


def test_concurrent_writers_produce_one_complete_snapshot(tmp_path):
    graphs = [make_graph() for _ in range(4)]
    for index, graph in enumerate(graphs):
        graph.set_metadata({"writer": index})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(GraphStorage(tmp_path).save, graphs))
    loaded = GraphStorage(tmp_path).load()
    assert_equal(graphs[loaded.metadata["writer"]], loaded)


def test_snapshot_preserves_undirected_policy(tmp_path):
    graph = make_graph()
    state = graph.to_state()
    # Build a separate undirected graph instead of changing a directed tensor state.
    undirected = Graph(replace(state.config, graph=replace(state.config.graph, directed=False)))
    for node in graph.nodes:
        undirected.add_node(node)
    undirected.add_relationship(20, 3, "links", provenance="test")
    storage = GraphStorage(tmp_path)
    storage.save(undirected)
    loaded = storage.load()
    assert_equal(undirected, loaded)
    assert loaded.get_relationship(20, 3, "links") == loaded.get_relationship(3, 20, "links")
