"""Regression tests for graph bridge snapshot and reload serialization."""
from threading import Event, Thread

from rutam.graph.operations import Graph
from rutam.model.graph_integration import GraphMemoryBridge


def test_refresh_tracks_snapshot_version_not_later_live_version(monkeypatch):
    graph = Graph()
    bridge = GraphMemoryBridge(graph)
    original = graph.to_state
    mutated = False

    def racing_snapshot():
        nonlocal mutated
        snapshot = original()
        if not mutated:
            mutated = True
            graph.set_metadata({"updated": True})
        return snapshot

    monkeypatch.setattr(graph, "to_state", racing_snapshot)
    old = bridge.refresh()
    assert old.graph_version != graph.version
    current = bridge.get_active_state()
    assert current.graph_version == graph.version
    assert current.metadata == {"updated": True}


def test_reload_serializes_storage_read_and_writes():
    graph = Graph()
    graph.set_metadata({"initial": True})
    stored = Graph()
    stored.set_metadata({"stored": True})
    bridge = GraphMemoryBridge(graph)
    load_started, finish_load = Event(), Event()
    write_started, write_done = Event(), Event()
    errors = []

    class Storage:
        def load(self):
            load_started.set()
            if not finish_load.wait(5):
                raise AssertionError("reload was not released")
            return stored

    def reload():
        try:
            bridge.reload_from_storage(Storage())
        except BaseException as exc:
            errors.append(exc)

    def write():
        try:
            write_started.set()
            graph.set_metadata({"concurrent_write": True})
            write_done.set()
        except BaseException as exc:
            errors.append(exc)

    loader = Thread(target=reload)
    writer = Thread(target=write)
    loader.start()
    assert load_started.wait(5)
    writer.start()
    assert write_started.wait(5)
    try:
        assert not write_done.wait(0.1), "writer passed reload during storage read"
    finally:
        finish_load.set()
        loader.join(5)
        writer.join(5)
    assert not loader.is_alive() and not writer.is_alive()
    assert not errors
    assert write_done.is_set()
    assert graph.metadata == {"concurrent_write": True}
    assert bridge.get_active_state().metadata == graph.metadata
