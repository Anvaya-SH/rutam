"""Integration tests for RUTAM graph attention with encoder, bridge, and fusion."""

import torch

from rutam.config import EncoderConfig, RUTAMConfig
from rutam.graph.encoder import RUTAMGraphEncoder
from rutam.graph.operations import Graph
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.graph_attention_bridge import GraphAttentionBridge


def _build_test_graph() -> Graph:
    g = Graph()
    emb0 = tuple(torch.randn(128).tolist())
    emb1 = tuple(torch.randn(128).tolist())
    emb2 = tuple(torch.randn(128).tolist())
    n0 = g.create_node("fact", "Node 0", embedding=emb0, provenance="test:n0")
    n1 = g.create_node("fact", "Node 1", embedding=emb1, provenance="test:n1")
    n2 = g.create_node("fact", "Node 2", embedding=emb2, provenance="test:n2")
    g.add_relationship(n0.id, n1.id, "depends_on", weight=1.0, provenance="test:e0")
    g.add_relationship(n1.id, n2.id, "calls", weight=0.8, provenance="test:e1")
    return g


def test_encoder_modes():
    graph = _build_test_graph()
    state = graph.to_state()

    # 1. Message passing mode (baseline)
    cfg_mp = EncoderConfig(hidden_dim=128, encoder_mode="message_passing")
    enc_mp = RUTAMGraphEncoder(cfg_mp)
    out_mp = enc_mp(state)
    assert out_mp.node_states.shape == (3, 128)
    assert out_mp.graph_summary.shape == (128,)

    # 2. Graph attention mode
    cfg_ga = EncoderConfig(hidden_dim=128, encoder_mode="graph_attention")
    enc_ga = RUTAMGraphEncoder(cfg_ga)
    out_ga = enc_ga(state)
    assert out_ga.node_states.shape == (3, 128)
    assert out_ga.graph_summary.shape == (128,)

    # 3. Hybrid mode
    cfg_hyb = EncoderConfig(hidden_dim=128, encoder_mode="hybrid")
    enc_hyb = RUTAMGraphEncoder(cfg_hyb)
    out_hyb = enc_hyb(state)
    assert out_hyb.node_states.shape == (3, 128)
    assert out_hyb.graph_summary.shape == (128,)


def test_graph_attention_bridge_with_selector():
    graph = _build_test_graph()
    state = graph.to_state()

    config = RUTAMConfig()
    bridge = GraphAttentionBridge(config)

    # Synthetic query from language model: [1, 128]
    query = torch.randn(1, 128)

    attn_out, selector_out = bridge(state, query)

    assert attn_out.node_states.shape == (1, 3, 128)
    assert selector_out.selected_indices.shape[0] == 1  # Batch size 1
    assert selector_out.selected_states is not None


def test_compatibility_with_existing_fusion():
    graph = _build_test_graph()
    state = graph.to_state()

    # Contextualize with graph attention
    attn = RUTAMGraphEncoder(EncoderConfig(hidden_dim=128, encoder_mode="graph_attention"))
    encoded = attn(state)

    # Feed into existing RUTAMGraphFusion
    fusion = RUTAMGraphFusion(RUTAMConfig().graph_fusion)
    lm_hidden = torch.randn(2, 8, 256)  # [B, T, D_LM]
    fused_out = fusion(lm_hidden, encoded.node_states.unsqueeze(0).expand(2, -1, -1))

    assert fused_out.fused_hidden_states.shape == (2, 8, 256)
    assert not torch.isnan(fused_out.fused_hidden_states).any()
