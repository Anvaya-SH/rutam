"""Causality and repeated-autograd regressions on the actual RUTAM forward path.

SyntheticBackbone is deliberately a tiny test backbone, not an LFM3 checkpoint.
"""
import pytest
import torch
from test_rutam_forward import _create_small_model_and_graph


def test_graph_conditioning_is_prefix_causal():
    torch.manual_seed(13)
    model, graph = _create_small_model_and_graph()
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]])
    before = graph.version
    with torch.no_grad():
        full = model(ids).logits
        for end in range(1, 5):
            prefix = model(ids[:, :end]).logits
            torch.testing.assert_close(full[:, :end], prefix, rtol=2e-5, atol=2e-6)
    assert graph.version == before


def test_recurrent_graph_conditioning_is_prefix_causal():
    from rutam.model.recurrent_integration import RUTAMRecurrentIntegration

    torch.manual_seed(29)
    model, _ = _create_small_model_and_graph()
    model.config = model.config.with_overrides({"recurrent": {
        "enabled": True, "hidden_dim": 16, "graph_dim": 16,
        "loop_embedding_dim": 16, "num_heads": 2, "intermediate_dim": 32,
        "default_loops": 2, "dynamic_halting": True, "dropout": 0.0,
    }})
    model.recurrent = RUTAMRecurrentIntegration(model.config)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        full = model(ids).logits
        for end in range(1, ids.shape[1] + 1):
            torch.testing.assert_close(full[:, :end], model(ids[:, :end]).logits,
                                       rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("ids,labels", [
    ([[1]], [[1]]),
    ([[1, 2, 3]], [[-100, -100, -100]]),
])
def test_empty_shifted_supervision_is_finite_and_differentiable(ids, labels):
    model, _ = _create_small_model_and_graph(0)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(True)
    loss = model(torch.tensor(ids), labels=torch.tensor(labels)).loss
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()


def test_explicit_adapter_does_not_reuse_previous_fusion():
    from rutam.config import GraphFusionConfig, LFM3AdapterConfig
    from rutam.model.fusion import RUTAMGraphFusion
    from rutam.model.layer_integration import SyntheticBackbone
    from rutam.model.lfm3_adapter import RUTAMLFM3Adapter

    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=20, num_layers=2)
    adapter = RUTAMLFM3Adapter(
        LFM3AdapterConfig(fusion_locations=(0,), integration_mode="explicit"),
        backbone=backbone,
        fusion_modules=RUTAMGraphFusion(GraphFusionConfig(hidden_dim=16, graph_dim=8, num_attention_heads=2)),
    )
    ids = torch.tensor([[1, 2]])
    assert len(adapter(ids, graph_states=torch.randn(1, 2, 8)).fusion_outputs) == 1
    output = adapter(ids)
    assert output.fusion_outputs == () and output.fused_hidden_states is None


def test_encoding_cache_detects_inplace_snapshot_mutation():
    torch.manual_seed(37)
    model, _ = _create_small_model_and_graph()
    model.encoder.eval()
    state = model.bridge.get_active_state()
    with torch.no_grad():
        first = model.bridge.get_active_encoding(model.encoder, state=state)
        assert model.bridge.get_active_encoding(model.encoder, state=state) is first
        state.node_embeddings.add_(1.0)
        second = model.bridge.get_active_encoding(model.encoder, state=state)
        assert second is not first
        torch.testing.assert_close(second.node_states, model.encoder(state).node_states)


def test_graph_encoder_supports_two_backward_steps():
    torch.manual_seed(15)
    model, _ = _create_small_model_and_graph()
    model.train()
    ids = torch.tensor([[1, 2, 3]])
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        out.loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


def test_empty_graph_baseline_parity():
    model, _ = _create_small_model_and_graph(0)
    model.eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, model.backbone(ids)['logits'])


def test_layer_internal_type_error_is_not_swallowed():
    model, _ = _create_small_model_and_graph(0)
    class BadLayer(torch.nn.Module):
        def forward(self, hidden, attention_mask=None):
            if attention_mask is not None:
                raise TypeError('internal layer failure')
            return hidden
    model.backbone.layers[0] = BadLayer()
    with pytest.raises(TypeError, match='internal layer failure'):
        model(torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2, dtype=torch.bool))


def test_private_scope_is_excluded_before_message_passing():
    model, graph = _create_small_model_and_graph()
    for node in graph.nodes:
        graph.update_node(node.id, scope='private')
    model.eval()
    ids = torch.tensor([[1, 2]])
    with torch.no_grad():
        result = model(ids)
        baseline = model.backbone(ids)['logits']
    assert result.fusion_output is None
    torch.testing.assert_close(result.logits, baseline)
