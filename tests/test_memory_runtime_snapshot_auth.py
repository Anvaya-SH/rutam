import pytest

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.memory.provenance import AuthorizationContext
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.runtime.memory_runtime import MemoryRuntime


def _runtime_with_two_scopes() -> MemoryRuntime:
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 4}})
    graph = Graph(config)
    graph.create_node("fact", "alpha-secret", scope="alpha", provenance="test")
    graph.create_node("fact", "beta-secret", scope="beta", provenance="test")
    graph.add_relationship(0, 1, "cross_scope", scope="beta", provenance="test")
    return MemoryRuntime(RUTAMForCausalLM(config=config, graph=graph))


def test_raw_snapshot_and_global_counts_reject_scoped_clients():
    runtime = _runtime_with_two_scopes()
    scoped = AuthorizationContext(
        user_id="alpha-user",
        authorized_scopes=("alpha",),
        permissions=frozenset({"read"}),
    )

    with pytest.raises(ValidationError, match="read_all_scopes"):
        runtime.get_graph_state()
    with pytest.raises(ValidationError, match="read_all_scopes"):
        runtime.get_graph_state(authorization_context=scoped)

    partial_all_scopes = AuthorizationContext(
        user_id="alpha-admin",
        role="service",
        authorized_scopes=("alpha",),
        permissions=frozenset({"read", "read_all_scopes"}),
    )
    with pytest.raises(ValidationError, match="not bound to every active graph scope"):
        runtime.get_graph_state(authorization_context=partial_all_scopes)
    with pytest.raises(ValidationError, match="read_all_scopes"):
        runtime.get_node_count(authorization_context=scoped)
    with pytest.raises(ValidationError, match="read_all_scopes"):
        runtime.get_edge_count(authorization_context=scoped)


def test_explicit_full_snapshot_privilege_preserves_internal_access():
    runtime = _runtime_with_two_scopes()
    admin = AuthorizationContext(
        user_id="runtime-admin",
        role="admin",
        authorized_scopes=("alpha",),
        permissions=frozenset({"read"}),
    )
    service = AuthorizationContext(
        user_id="trusted-service",
        role="service",
        authorized_scopes=("alpha", "beta"),
        permissions=frozenset({"read", "read_all_scopes"}),
    )

    state = runtime.get_graph_state(authorization_context=admin)
    assert {record["content"] for record in state.node_metadata} == {
        "alpha-secret",
        "beta-secret",
    }
    assert runtime.get_node_count(authorization_context=service) == 2
    assert runtime.get_edge_count(authorization_context=service) == 1


def test_all_scopes_permission_does_not_override_node_read_denial():
    runtime = _runtime_with_two_scopes()
    runtime.model.graph.create_node(
        "fact",
        "write-only",
        scope="alpha",
        permissions=("write",),
        provenance="test",
    )
    service = AuthorizationContext(
        user_id="trusted-service",
        role="service",
        authorized_scopes=("alpha", "beta"),
        permissions=frozenset({"read", "read_all_scopes"}),
    )

    with pytest.raises(ValidationError, match="node-level read permission"):
        runtime.get_graph_state(authorization_context=service)


def test_commit_version_reporting_does_not_use_public_snapshot_getter(monkeypatch):
    runtime = _runtime_with_two_scopes()

    def forbidden_snapshot(**kwargs):
        raise AssertionError("internal commit must not request a raw public snapshot")

    monkeypatch.setattr(runtime, "get_graph_state", forbidden_snapshot)
    result = runtime.commit(())
    assert result.new_graph_version == runtime.model.graph.version
