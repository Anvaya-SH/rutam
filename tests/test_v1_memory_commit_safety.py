"""Public commit boundaries, failure isolation and persistent replay protection."""
from unittest.mock import patch
import pytest
import torch
from rutam import Graph, GraphStorage, RUTAMConfig
from rutam.exceptions import ValidationError, StorageError
from rutam.memory.transaction import GraphTransaction
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.write_proposal import MemoryProposal, MemoryOperation

AUTH = AuthorizationContext(permissions=frozenset({'commit'}))


def setup(tmp_path):
    cfg = RUTAMConfig().with_overrides({'graph': {'embedding_dim': 2}})
    graph = Graph(cfg)
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    return graph, storage


def proposal(**kwargs):
    return MemoryProposal(proposal_id='test-write', candidate_embedding=torch.tensor([1., 0.]),
                          provenance=ProvenanceRecord(source_id='test', authorization_category='user_input'),
                          **kwargs)


def test_failed_save_preserves_live_and_durable_state(tmp_path):
    graph, storage = setup(tmp_path)
    with GraphTransaction(graph, storage=storage, authorization_context=AUTH) as tx:
        tx.apply_proposal(proposal())
        with patch.object(storage, 'save', side_effect=StorageError('injected')):
            result = tx.commit()
    assert not result.committed and result.rollback_performed
    assert len(graph.nodes) == len(storage.load().nodes) == 0
    assert graph.version == 0


def test_public_transaction_requires_authorization(tmp_path):
    graph, storage = setup(tmp_path)
    for auth in (None, AuthorizationContext()):
        with GraphTransaction(graph, storage=storage, authorization_context=auth) as tx:
            with pytest.raises(ValidationError, match='COMMIT_NOT_AUTHORIZED'):
                tx.apply_proposal(proposal())
    assert not graph.nodes and not storage.load().nodes


def test_invalid_provenance_and_stale_transaction_rejected(tmp_path):
    graph, storage = setup(tmp_path)
    with GraphTransaction(graph, authorization_context=AUTH) as tx:
        with pytest.raises(ValidationError, match='not approved'):
            tx.apply_proposal(MemoryProposal(provenance='unverified'))
    tx = GraphTransaction(graph, authorization_context=AUTH)
    tx.apply_proposal(proposal())
    graph.create_node('fact', 'concurrent', provenance='test')
    assert not tx.commit().committed
    assert len(graph.nodes) == 1


def test_valid_commit_reload_and_replay_rejection(tmp_path):
    graph, storage = setup(tmp_path)
    with GraphTransaction(graph, storage=storage, authorization_context=AUTH) as tx:
        tx.apply_proposal(proposal(source_graph_version=0))
        assert tx.commit().committed
    loaded = storage.load()
    with GraphTransaction(loaded, authorization_context=AUTH) as tx:
        with pytest.raises(ValidationError, match='ALREADY_COMMITTED'):
            tx.apply_proposal(proposal())
    assert len(loaded.nodes) == 1


def test_noop_needs_no_authorization_and_does_not_mutate(tmp_path):
    graph, _ = setup(tmp_path)
    with GraphTransaction(graph) as tx:
        tx.apply_proposal(MemoryProposal(operation=MemoryOperation.NO_OP))
        assert tx.commit().applied_mutations == 0
    assert graph.version == 0


def test_content_and_delete_persist(tmp_path):
    graph, storage = setup(tmp_path)
    p = MemoryProposal(content='controlled record', node_type='fact', provenance=ProvenanceRecord(source_id='test', authorization_category='user_input'))
    with GraphTransaction(graph, storage=storage, authorization_context=AUTH) as tx:
        tx.apply_proposal(p)
        assert tx.commit().committed
    node_id = graph.nodes[0].id
    assert storage.load().nodes[0].content == 'controlled record'
    with GraphTransaction(graph, storage=storage, authorization_context=AUTH) as tx:
        tx.apply_proposal(MemoryProposal(operation=MemoryOperation.DELETE_NODE, target_node_ids=(node_id,), provenance=p.provenance))
        assert tx.commit().committed
    assert not storage.load().nodes


def test_direct_concurrent_commits_do_not_lose_updates(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    graph, _ = setup(tmp_path)
    txs = [GraphTransaction(graph, authorization_context=AUTH) for _ in range(2)]
    for tx in txs:
        tx.apply_proposal(proposal())
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda tx: tx.commit(), txs))
    assert sum(r.committed for r in results) == 1
    assert len(graph.nodes) == 1


def test_empty_runtime_validation_has_complete_result():
    from rutam.diagnostics.integration import create_verification_model
    from rutam.runtime.memory_runtime import MemoryRuntime
    model, _ = create_verification_model(num_nodes=0)
    result = MemoryRuntime(model).validate([])
    assert result.results == ()
    assert result.accepted_count == result.rejected_count == result.stale_count == 0
