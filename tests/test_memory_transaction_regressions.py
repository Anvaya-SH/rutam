"""Regression tests for failed memory transactions and mandatory validation."""
from unittest.mock import patch

import pytest

from rutam import Graph, GraphStorage, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.transaction import GraphTransaction
from rutam.memory.write_proposal import MemoryProposal

AUTH = AuthorizationContext(permissions=frozenset({'commit'}))
PROV = ProvenanceRecord(source_id='test', authorization_category='user_input')


def test_failed_apply_cannot_publish_mutation_after_ledger_exhaustion():
    graph = Graph(RUTAMConfig())
    graph._metadata['committed_proposals'] = [f'id-{i}' for i in range(10000)]
    tx = GraphTransaction(graph, authorization_context=AUTH)
    with pytest.raises(ValidationError, match='ledger capacity'):
        tx.apply_proposal(MemoryProposal(provenance=PROV, content='must not publish'))
    with pytest.raises(ValidationError):
        tx.commit()
    assert not graph.nodes


def test_injected_validator_cannot_override_builtin_security_validation():
    from types import SimpleNamespace

    from rutam.memory.validation_output import ValidationDecision

    graph = Graph(RUTAMConfig())
    validator = SimpleNamespace(
        validate=lambda *a, **k: SimpleNamespace(decision=ValidationDecision.ACCEPT)
    )
    tx = GraphTransaction(graph, authorization_context=AUTH, validator=validator)
    with pytest.raises(ValidationError, match='not approved'):
        tx.apply_proposal(MemoryProposal(provenance='unverified'))
    assert not graph.nodes


def test_post_commit_maintenance_exception_reports_success():
    graph = Graph(RUTAMConfig())
    tx = GraphTransaction(graph, authorization_context=AUTH)
    tx.apply_proposal(MemoryProposal(provenance=PROV, content='persisted'))
    tx._audit_logger = object()
    with patch('rutam.memory.transaction.AuditRecord', side_effect=RuntimeError('audit failed')):
        result = tx.commit()
    assert result.committed and 'audit failed' in result.error_message
    assert len(graph.nodes) == 1


def test_post_manifest_sync_failure_updates_live_and_warns(tmp_path):
    from rutam.graph import storage as storage_module

    graph = Graph(RUTAMConfig())
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    tx = GraphTransaction(graph, storage=storage, authorization_context=AUTH)
    tx.apply_proposal(MemoryProposal(provenance=PROV, content="published"))
    sync = storage_module._sync_directory
    calls = 0

    def fail_after_publication(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected final fsync failure")
        return sync(path)

    with patch.object(storage_module, "_sync_directory", side_effect=fail_after_publication):
        result = tx.commit()
    assert calls >= 3
    assert result.committed and "durability is uncertain" in result.error_message
    assert graph.version == storage.load().version
    assert graph.nodes[0].content == storage.load().nodes[0].content == "published"


def test_autosave_staging_never_publishes_before_commit(tmp_path):
    config = RUTAMConfig().with_overrides(
        {"graph": {"auto_save": True, "path": str(tmp_path)}}
    )
    graph = Graph(config)
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    tx = GraphTransaction(graph, storage=storage, authorization_context=AUTH)
    tx.apply_proposal(MemoryProposal(provenance=PROV, content="staged"))
    assert not graph.nodes
    assert not storage.load().nodes
    assert tx.commit().committed
    assert storage.load().config.graph.auto_save
    assert storage.load().nodes[0].content == "staged"


def test_autosave_transaction_uses_configured_storage_when_not_explicit(tmp_path):
    config = RUTAMConfig().with_overrides(
        {"graph": {"auto_save": True, "path": str(tmp_path)}}
    )
    graph = Graph(config)
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    tx = GraphTransaction(graph, authorization_context=AUTH)
    tx.apply_proposal(MemoryProposal(provenance=PROV, content="implicit"))
    assert not storage.load().nodes
    assert tx.commit().committed
    assert storage.load().nodes[0].content == "implicit"


def test_snapshot_archive_is_independently_loadable(tmp_path):
    graph = Graph(RUTAMConfig())
    storage = GraphStorage(tmp_path)
    tx = GraphTransaction(graph, storage=storage, authorization_context=AUTH)
    tx.apply_proposal(MemoryProposal(provenance=PROV, content="archived"))
    assert tx.commit().committed
    archived = GraphStorage(tmp_path / "snapshots" / "v1").load()
    assert archived.version == graph.version
    assert archived.nodes[0].content == "archived"


def test_post_commit_error_preserves_durability_warning(tmp_path):
    from rutam.graph.storage import SnapshotPublicationError

    graph = Graph(RUTAMConfig())
    storage = GraphStorage(tmp_path)
    tx = GraphTransaction(graph, storage=storage, authorization_context=AUTH)
    tx.apply_proposal(MemoryProposal(provenance=PROV, content="warning"))
    with patch.object(
        storage, "save", side_effect=SnapshotPublicationError("durability uncertain")
    ):
        with patch.object(
            tx, "_preserve_snapshot_archive", side_effect=RuntimeError("archive failed")
        ):
            result = tx.commit()
    assert result.committed
    assert "durability uncertain" in result.error_message
    assert "archive failed" in result.error_message
