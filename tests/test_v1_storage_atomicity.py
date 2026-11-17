from unittest.mock import patch

import pytest

from rutam import Graph, GraphStorage, RUTAMConfig
from rutam.exceptions import CorruptSnapshotError, StorageError


def graph(text):
    value = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 2}}))
    value.create_node("fact", text, provenance="test")
    return value


def test_failed_manifest_publication_keeps_previous_snapshot(tmp_path):
    storage = GraphStorage(tmp_path)
    old = graph("durable")
    storage.save(old)
    new = graph("unpublished")
    real_replace = __import__("os").replace

    def fail_manifest(src, dst):
        if str(dst).endswith("graph.manifest.json"):
            raise OSError("injected manifest failure")
        return real_replace(src, dst)

    with patch("rutam.graph.storage.os.replace", side_effect=fail_manifest):
        with pytest.raises(StorageError):
            storage.save(new, expected_generation=old._storage_generation)
    assert storage.load().nodes[0].content == "durable"


def test_versioned_snapshot_empty_roundtrip_and_corruption(tmp_path):
    storage = GraphStorage(tmp_path)
    empty = Graph(RUTAMConfig().with_overrides({"graph": {"embedding_dim": 3}}))
    storage.save(empty)
    loaded = storage.load()
    assert loaded.to_state().node_embeddings.shape == (0, 3)
    snapshot = next(tmp_path.glob("snapshot-*"))
    tensor = snapshot / "graph.safetensors"
    tensor.write_bytes(tensor.read_bytes()[:-1])
    with pytest.raises(CorruptSnapshotError):
        storage.load()


def test_rollback_republishes_verified_generation(tmp_path):
    import json
    storage = GraphStorage(tmp_path)
    storage.save(graph('first'))
    first = json.loads((tmp_path / 'graph.manifest.json').read_text())['snapshot_id']
    storage.save(graph('second'), expected_generation=first)
    assert storage.rollback(first).nodes[0].content == 'first'
    assert storage.load().nodes[0].content == 'first'


def test_legacy_failure_preserves_previous_snapshot(tmp_path):
    import json
    import shutil
    storage = GraphStorage(tmp_path)
    storage.save(graph('legacy'))
    manifest_path = tmp_path / 'graph.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.pop('payload_directory', None)
    manifest_path.write_text(json.dumps(manifest))
    for path in tmp_path.glob('snapshot-*'):
        shutil.rmtree(path)
    real_replace = __import__('os').replace
    def fail_manifest(src, dst):
        if str(dst).endswith('graph.manifest.json'):
            raise OSError('injected failure')
        return real_replace(src, dst)
    with patch('rutam.graph.storage.os.replace', side_effect=fail_manifest):
        with pytest.raises(StorageError):
            storage.save(graph('new'), expected_generation=manifest['snapshot_id'])
    assert storage.load().nodes[0].content == 'legacy'


def test_failed_alias_replacement_restores_compatibility_exports(tmp_path):
    import json
    import os
    storage = GraphStorage(tmp_path)
    storage.save(graph('first'))
    committed = json.loads((tmp_path / 'graph.manifest.json').read_text())
    real_replace = os.replace
    def fail_metadata(src, dst):
        if str(dst).endswith('graph.json'):
            raise OSError('metadata export failure')
        return real_replace(src, dst)
    with patch('rutam.graph.storage.os.replace', side_effect=fail_metadata):
        with pytest.raises(StorageError):
            storage.save(graph('second'), expected_generation=committed['snapshot_id'])
    generation = tmp_path / ('snapshot-' + committed['snapshot_id'])
    for name in ('graph.json', 'graph.safetensors'):
        assert (tmp_path / name).read_bytes() == (generation / name).read_bytes()
    assert storage.load().nodes[0].content == 'first'
