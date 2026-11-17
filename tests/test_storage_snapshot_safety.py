import threading
from unittest.mock import patch

import pytest

from rutam import Graph, GraphStorage, RUTAMConfig, SnapshotConflictError
from rutam.graph.storage import SnapshotPublicationError
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.transaction import GraphTransaction
from rutam.memory.write_proposal import MemoryProposal


def test_compatibility_export_write_cannot_corrupt_committed_generation(tmp_path):
    storage = GraphStorage(tmp_path)
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    graph.create_node("fact", "durable", provenance="test")
    storage.save(graph)
    (tmp_path / "graph.json").write_bytes(b"broken legacy export")
    (tmp_path / "graph.safetensors").write_bytes(b"broken legacy export")
    assert storage.load().nodes[0].content == "durable"


def test_same_graph_saves_cannot_publish_out_of_order(tmp_path):
    storage = GraphStorage(tmp_path)
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    graph.create_node("fact", "old", provenance="test")
    entered, resume = threading.Event(), threading.Event()
    real_to_state = graph.to_state
    calls = 0

    def delayed_state():
        nonlocal calls
        state = real_to_state()
        calls += 1
        if calls == 1:
            entered.set()
            assert resume.wait(5)
        return state

    with patch.object(graph, "to_state", side_effect=delayed_state):
        first = threading.Thread(target=storage.save, args=(graph,))
        first.start()
        assert entered.wait(5)
        second_done = threading.Event()

        def mutate_and_save():
            graph.update_node(0, content="new")
            storage.save(graph)
            second_done.set()

        second = threading.Thread(target=mutate_and_save)
        second.start()
        try:
            assert not second_done.wait(0.1)
        finally:
            resume.set()
            first.join(5)
            second.join(5)
        assert not first.is_alive() and not second.is_alive()
        assert second_done.is_set()
    assert storage.load().nodes[0].content == "new"


def test_post_publication_fsync_failure_is_ambiguous(tmp_path):
    import rutam.graph.storage as storage_module

    storage = GraphStorage(tmp_path)
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    graph.create_node("fact", "old", provenance="test")
    storage.save(graph)
    graph.update_node(0, content="new")
    real_sync = storage_module._sync_directory
    count = 0

    def fail_final_sync(path):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("injected post-publication fsync failure")
        return real_sync(path)

    with patch.object(storage_module, "_sync_directory", side_effect=fail_final_sync):
        with pytest.raises(SnapshotPublicationError, match="durability is uncertain"):
            storage.save(graph)
    # The error cannot imply an unchanged disk: the manifest was already replaced.
    assert storage.load().nodes[0].content == "new"


def test_autosave_retains_live_state_after_publication_error(tmp_path):
    import rutam.graph.storage as storage_module

    storage = GraphStorage(tmp_path)
    config = RUTAMConfig().with_overrides(
        {"graph": {"embedding_dim": 2, "auto_save": True}}
    )
    graph = Graph(config, storage=storage)
    graph.create_node("fact", "old", provenance="test")
    real_sync = storage_module._sync_directory
    count = 0

    def fail_final_sync(path):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("injected post-publication fsync failure")
        return real_sync(path)

    with patch.object(storage_module, "_sync_directory", side_effect=fail_final_sync):
        with pytest.raises(SnapshotPublicationError):
            graph.update_node(0, content="new")
    assert graph.nodes[0].content == storage.load().nodes[0].content == "new"


def test_independent_writers_reject_stale_manifest_generation(tmp_path):
    storage = GraphStorage(tmp_path)
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    graph.create_node("fact", "initial", provenance="test")
    storage.save(graph)

    writer_a = storage.load()
    writer_b = storage.load()

    writer_a.update_node(0, content="from_a")
    storage.save(writer_a)
    assert storage.load().nodes[0].content == "from_a"

    writer_b.update_node(0, content="from_b")
    with pytest.raises(SnapshotConflictError, match="STALE_SNAPSHOT_GENERATION"):
        storage.save(writer_b)
    assert storage.load().nodes[0].content == "from_a"


def test_explicit_lww_and_newer_generation_writer_are_supported(tmp_path):
    storage = GraphStorage(tmp_path)
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    graph.create_node("fact", "initial", provenance="test")
    storage.save(graph)
    writer_a = storage.load()
    stale_writer = storage.load()

    writer_a.update_node(0, content="accepted")
    storage.save(writer_a)
    writer_a.update_node(0, content="accepted_again")
    storage.save(writer_a)
    assert storage.load().nodes[0].content == "accepted_again"

    stale_writer.update_node(0, content="explicit_lww")
    storage.save(stale_writer, unconditional=True)
    assert storage.load().nodes[0].content == "explicit_lww"


def test_unbound_graph_does_not_adopt_latest_generation_implicitly(tmp_path):
    storage = GraphStorage(tmp_path)
    initial = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    initial.create_node("fact", "durable", provenance="test")
    storage.save(initial)

    replacement = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    replacement.create_node("fact", "replacement", provenance="test")
    with pytest.raises(SnapshotConflictError, match="STALE_SNAPSHOT_GENERATION"):
        storage.save(replacement)

    # Callers that deliberately want replacement can still opt into LWW.
    storage.save(replacement, unconditional=True)
    assert storage.load().nodes[0].content == "replacement"


def test_stale_rollback_cannot_replace_a_newer_writer(tmp_path, monkeypatch):
    storage = GraphStorage(tmp_path)
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    graph.create_node("fact", "first", provenance="test")
    storage.save(graph)
    first_generation = graph._storage_generation

    graph.update_node(0, content="second")
    storage.save(graph)
    real_save = storage.save

    def advance_then_rollback(rollback_graph, **kwargs):
        concurrent_writer = storage.load()
        concurrent_writer.update_node(0, content="concurrent")
        real_save(concurrent_writer)
        return real_save(rollback_graph, **kwargs)

    monkeypatch.setattr(storage, "save", advance_then_rollback)
    with pytest.raises(SnapshotConflictError, match="STALE_SNAPSHOT_GENERATION"):
        storage.rollback(first_generation)
    assert GraphStorage(tmp_path).load().nodes[0].content == "concurrent"


def test_concurrent_transactions_on_distinct_graphs_detect_storage_conflict(tmp_path):
    storage = GraphStorage(tmp_path)
    graph = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    graph.create_node("fact", "initial", provenance="test")
    storage.save(graph)

    g1 = storage.load()
    g2 = storage.load()

    auth = AuthorizationContext(permissions=frozenset({"commit"}))
    prov = ProvenanceRecord(source_id="test", authorization_category="user_input")

    with GraphTransaction(g1, storage=storage, authorization_context=auth) as tx1:
        tx1.apply_proposal(MemoryProposal(provenance=prov, content="tx1"))
        res1 = tx1.commit()
        assert res1.committed

    with GraphTransaction(g2, storage=storage, authorization_context=auth) as tx2:
        tx2.apply_proposal(MemoryProposal(provenance=prov, content="tx2"))
        res2 = tx2.commit()
        assert not res2.committed
        assert "STALE_SNAPSHOT_GENERATION" in res2.error_message

    loaded = storage.load()
    assert len(loaded.nodes) == 2
    assert loaded.nodes[1].content == "tx1"

