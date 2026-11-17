"""Session APIs must retain authorization for every autoregressive forward pass."""

from unittest.mock import patch

import torch
from test_rutam_forward import _create_small_model_and_graph

from rutam.memory.provenance import AuthorizationContext
from rutam.runtime.session import RUTAMSession


def _scoped_model():
    model, graph = _create_small_model_and_graph(num_nodes=2)
    graph.update_node(10, scope="default")
    graph.update_node(20, scope="alpha")
    model.eval()
    return model


def _alpha_reader() -> AuthorizationContext:
    return AuthorizationContext(
        user_id="alpha-reader",
        authorized_scopes=("alpha",),
        permissions=frozenset({"read"}),
    )


def _capture_authorization_and_visible_ids(model):
    forward_contexts = []
    visible_ids: list[set[int]] = []
    original_forward = model.forward
    original_select_graph = model.query_head.select_graph

    def capture_forward(*args, **kwargs):
        forward_contexts.append(kwargs.get("authorization_context"))
        return original_forward(*args, **kwargs)

    def capture_select_graph(*args, **kwargs):
        node_ids = kwargs["node_ids"]
        node_mask = kwargs["node_mask"]
        visible_ids.append(set(node_ids[node_mask].tolist()))
        return original_select_graph(*args, **kwargs)

    return (
        patch.object(model, "forward", side_effect=capture_forward),
        patch.object(model.query_head, "select_graph", side_effect=capture_select_graph),
        forward_contexts,
        visible_ids,
    )


def test_step_keeps_override_scope_through_every_generated_token():
    model = _scoped_model()
    session = RUTAMSession(model)
    forward_spy, select_spy, forward_contexts, visible_ids = _capture_authorization_and_visible_ids(
        model
    )

    with forward_spy, select_spy:
        result = session.step(
            torch.tensor([[1]]),
            max_new_tokens=3,
            temperature=0,
            authorization_context=_alpha_reader(),
        )

    assert result.generated_ids is not None
    assert result.generated_ids.shape == (1, 4)
    assert len(forward_contexts) == 4  # Initial forward plus three generated-token forwards.
    assert len(visible_ids) == 7  # Prefix selection runs once for each token in each forward.
    assert visible_ids == [{20}] * 7
    assert forward_contexts == [_alpha_reader()] * 4


def test_generate_uses_session_scope_for_every_generated_token():
    model = _scoped_model()
    session = RUTAMSession(model, authorization_context=_alpha_reader())
    forward_spy, select_spy, forward_contexts, visible_ids = _capture_authorization_and_visible_ids(
        model
    )

    with forward_spy, select_spy:
        generated = session.generate(torch.tensor([[1]]), max_new_tokens=3, temperature=0)

    assert generated.shape == (1, 4)
    assert len(forward_contexts) == 3
    assert len(visible_ids) == 6
    assert visible_ids == [{20}] * 6
    assert forward_contexts == [_alpha_reader()] * 3


def test_generate_accepts_and_applies_per_call_scope_override():
    model = _scoped_model()
    session = RUTAMSession(model)
    forward_spy, select_spy, forward_contexts, visible_ids = _capture_authorization_and_visible_ids(
        model
    )

    with forward_spy, select_spy:
        generated = session.generate(
            torch.tensor([[1]]),
            max_new_tokens=2,
            temperature=0,
            authorization_context=_alpha_reader(),
        )

    assert generated.shape == (1, 3)
    assert len(forward_contexts) == 2
    assert len(visible_ids) == 3
    assert visible_ids == [{20}] * 3
    assert forward_contexts == [_alpha_reader()] * 2
