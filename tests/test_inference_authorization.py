"""Generation must retain its caller's authorization for the whole request."""
from unittest.mock import patch

import torch

from rutam.memory.provenance import AuthorizationContext
from rutam.runtime.inference import RUTAMInferenceEngine
from test_rutam_forward import _create_small_model_and_graph


def test_generate_passes_authorization_and_pins_graph_snapshot():
    model, _ = _create_small_model_and_graph()
    model.eval()
    engine = RUTAMInferenceEngine(model)
    auth = AuthorizationContext(permissions=frozenset(), authorized_scopes=("private",))
    with patch.object(model, "forward", wraps=model.forward) as forward:
        result = engine.generate(
            torch.tensor([[1, 2]]), max_new_tokens=3, temperature=0,
            authorization_context=auth,
        )
    assert result.shape == (1, 5)
    calls = forward.call_args_list
    assert len(calls) == 3
    assert all(call.kwargs["authorization_context"] is auth for call in calls)
    snapshots = [call.kwargs["graph_state_snapshot"] for call in calls]
    assert all(snapshot is snapshots[0] for snapshot in snapshots)
