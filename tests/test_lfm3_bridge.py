"""Contract tests for the LFM3 layer-2 bridge, using tiny Transformers-shaped doubles."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.graph.schema import Node
from rutam.memory.provenance import AuthorizationContext
from rutam.model.lfm3_bridge import RUTAMLFM3Bridge
from rutam.model.model_factory import RUTAMModelFactory
from rutam.model.query_head import RUTAMNeuralQueryHead


class FakeCache:
    def __init__(self) -> None:
        self.updated_layers: list[int] = []


class Lfm3Attention(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.last_position_embeddings = None
        self.last_seq_idx = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings,
        past_key_values: FakeCache | None,
        seq_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        self.last_position_embeddings = position_embeddings
        self.last_seq_idx = seq_idx
        if past_key_values is not None:
            past_key_values.updated_layers.append(self.layer_idx)
        return hidden_states, None


class Lfm3DecoderLayer(nn.Module):
    def __init__(self, index: int, *, full_attention: bool = False) -> None:
        super().__init__()
        self.layer_idx = index
        self.self_attn = Lfm3Attention(index) if full_attention else None
        self.last_inputs: dict[str, object] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        seq_idx=None,
        **kwargs,
    ) -> torch.Tensor:
        self.last_inputs = {
            "position_embeddings": position_embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "seq_idx": seq_idx,
            **kwargs,
        }
        if self.self_attn is not None:
            hidden_states, _ = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                seq_idx=seq_idx,
            )
        return hidden_states + (self.layer_idx + 1) * 0.001


class Lfm3Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            Lfm3DecoderLayer(index, full_attention=index in (2, 6, 10, 14, 18, 22))
            for index in range(24)
        )
        self.config = config


class Lfm3ForCausalLM(nn.Module):
    def __init__(self, *, layer_types=None) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="lfm3",
            hidden_size=8,
            num_hidden_layers=24,
            layer_types=layer_types
            or [
                "full_attention" if index in (2, 6, 10, 14, 18, 22) else "conv"
                for index in range(24)
            ],
        )
        self.model = Lfm3Model(self.config)
        self.embed_tokens = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.last_cache: FakeCache | None = None
        self.after_prefill = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        seq_idx=None,
        **kwargs,
    ):
        hidden_states = self.embed_tokens(input_ids).to(torch.bfloat16)
        positions = (torch.ones(1), torch.zeros(1))
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=positions,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                seq_idx=seq_idx,
                **kwargs,
            )
        return SimpleNamespace(
            logits=self.lm_head(hidden_states.float()),
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def generate(self, input_ids: torch.Tensor, *, attention_mask=None, **kwargs):
        cache = FakeCache()
        batch, prompt_length = input_ids.shape
        initial_mask = (
            torch.ones(batch, prompt_length, dtype=torch.long)
            if attention_mask is None
            else attention_mask
        )
        position_ids = torch.arange(prompt_length).expand(batch, -1)
        outputs = [
            self.forward(
                input_ids,
                attention_mask=initial_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                seq_idx=position_ids.to(torch.int32),
                **kwargs,
            )
        ]
        if self.after_prefill is not None:
            self.after_prefill()
        for step in range(2):
            token = input_ids[:, -1:] + step + 1
            prefix_length = prompt_length + step + 1
            full_mask = torch.ones(batch, prefix_length, dtype=torch.long)
            position = torch.full((batch, 1), prefix_length - 1, dtype=torch.long)
            outputs.append(
                self.forward(
                    token,
                    attention_mask=full_mask,
                    position_ids=position,
                    past_key_values=cache,
                    use_cache=True,
                    seq_idx=position.to(torch.int32),
                    **kwargs,
                )
            )
        self.last_cache = cache
        return torch.cat((input_ids, torch.ones(batch, 2, dtype=input_ids.dtype)), dim=1)


def tiny_config() -> RUTAMConfig:
    return RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 4},
            "encoder": {"input_dim": 4, "hidden_dim": 8, "num_layers": 1},
            "selector": {"query_dim": 4, "hidden_dim": 8, "score_dim": 8, "top_k": 1},
            "query_head": {
                "backbone_hidden_dim": 8,
                "graph_hidden_dim": 8,
                "query_dim": 4,
                "intermediate_dim": 8,
                "dropout": 0.0,
            },
            "graph_fusion": {
                "hidden_dim": 8,
                "graph_dim": 8,
                "num_attention_heads": 2,
                "projection_dropout": 0.0,
                "residual_scale_init": 0.25,
            },
            "rutam_model": {
                "max_sparse_nodes": 4,
                "graph_traversal_enabled": False,
                "backbone_frozen": False,
            },
        }
    )


def graph_with_one_node(config: RUTAMConfig) -> Graph:
    graph = Graph(config)
    graph.add_node(
        Node(
            id=0,
            node_type="fact",
            content="bridge test node",
            provenance="test fixture",
            embedding=(1.0, -0.5, 0.25, 0.75),
        )
    )
    return graph


def test_layer_2_bridge_preserves_bf16_shape_and_original_call_contract() -> None:
    torch.manual_seed(19)
    config = tiny_config()
    graph = graph_with_one_node(config)
    backbone = Lfm3ForCausalLM()
    bridge = RUTAMModelFactory.create_lfm3_bridge(
        backbone, config=config, graph=graph
    )
    bridge.eval()
    bridge.query_head.decision_gate.output_projection.weight.data.zero_()
    bridge.query_head.decision_gate.output_projection.bias.data.fill_(10.0)
    input_ids = torch.tensor([[2, 4, 7]])
    attention_mask = torch.tensor([[0, 1, 1]])
    position_ids = torch.tensor([[0, 1, 2]])
    seq_idx = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    cache = FakeCache()

    output = bridge(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        seq_idx=seq_idx,
        output_attentions=True,
    )

    layer = backbone.model.layers[2]
    assert output.hidden_states.shape == (1, 3, 8)
    assert output.hidden_states.dtype is torch.bfloat16
    assert layer.last_inputs["position_embeddings"] == (torch.ones(1), torch.zeros(1))
    assert layer.last_inputs["attention_mask"] is attention_mask
    assert layer.last_inputs["position_ids"] is position_ids
    assert layer.last_inputs["past_key_values"] is cache
    assert layer.last_inputs["seq_idx"] is seq_idx
    assert layer.last_inputs["output_attentions"] is True
    assert backbone.model.layers[2].self_attn.last_position_embeddings == layer.last_inputs[
        "position_embeddings"
    ]
    assert cache.updated_layers == [2, 6, 10, 14, 18, 22]
    assert len(bridge.last_fusion_outputs) == input_ids.shape[1]

    empty = Graph(config).to_state()
    baseline = bridge(input_ids, graph_state_snapshot=empty).hidden_states
    fused = bridge(input_ids, graph_state_snapshot=graph.to_state()).hidden_states
    assert not torch.equal(fused, baseline)
    assert fused.dtype is torch.bfloat16
    bridge.close()


def test_generate_keeps_graph_snapshot_across_prefill_and_cached_steps() -> None:
    torch.manual_seed(23)
    config = tiny_config()
    graph = graph_with_one_node(config)
    backbone = Lfm3ForCausalLM()
    bridge = RUTAMLFM3Bridge(backbone, config=config, graph=graph)
    bridge.eval()
    bridge.query_head.decision_gate.output_projection.weight.data.zero_()
    bridge.query_head.decision_gate.output_projection.bias.data.fill_(10.0)
    initial_version = graph.version

    def mutate_live_graph() -> None:
        graph.add_node(
            Node(
                id=1,
                node_type="fact",
                content="added after the prefill snapshot",
                provenance="test fixture",
                embedding=(-1.0, 0.5, -0.25, -0.75),
            )
        )

    backbone.after_prefill = mutate_live_graph
    input_ids = torch.tensor([[1, 3, 5]])
    generated = bridge.generate(input_ids, attention_mask=torch.ones_like(input_ids))

    assert generated.shape == (1, 5)
    assert bridge.last_graph_version == initial_version
    assert len(bridge.last_fusion_outputs) == input_ids.shape[1] + 2
    assert backbone.last_cache is not None
    assert len(backbone.last_cache.updated_layers) == 6 * 3
    assert backbone.last_cache.updated_layers.count(2) == 3
    assert all(
        output.fused_hidden_states.dtype is torch.float32
        for output in bridge.last_fusion_outputs
    )
    bridge.close()


def test_unsupported_layer_layout_fails_before_mutating_backbone() -> None:
    layer_types = ["conv"] * 24
    backbone = Lfm3ForCausalLM(layer_types=layer_types)
    target_layer = backbone.model.layers[2]
    with pytest.raises(ValidationError, match="first full_attention"):
        RUTAMLFM3Bridge(backbone, config=tiny_config())
    assert not target_layer._forward_hooks


def test_generation_rejects_beam_cache_reordering_until_supported() -> None:
    bridge = RUTAMLFM3Bridge(Lfm3ForCausalLM(), config=tiny_config())
    with pytest.raises(ValidationError, match="beam/cache reordering is not integrated"):
        bridge.generate(torch.tensor([[1, 2]]), num_beams=2)
    with pytest.raises(ValidationError, match="beam/cache reordering is not integrated"):
        bridge.generate(torch.tensor([[1, 2]]), generation_config={"num_beams": 2})
    bridge.close()


def test_custom_query_head_must_handle_empty_prefixes() -> None:
    config = tiny_config()
    backbone = Lfm3ForCausalLM()
    query_head = RUTAMNeuralQueryHead(config.query_head, selector_config=config.selector)
    with pytest.raises(ValidationError, match="empty_sequence_policy must be 'zeros'"):
        RUTAMLFM3Bridge(backbone, config=config, query_head=query_head)
    assert not backbone.model.layers[2]._forward_hooks


def test_request_filters_scope_permissions_and_malformed_validity() -> None:
    config = tiny_config()
    graph = graph_with_one_node(config)
    bridge = RUTAMLFM3Bridge(Lfm3ForCausalLM(), config=config, graph=graph)
    snapshot = graph.to_state()
    auth = AuthorizationContext()

    snapshot.node_metadata[0]["scope"] = "private"
    assert bridge._make_request(snapshot, None, auth, None).visible_state.node_ids.numel() == 0

    snapshot.node_metadata[0]["scope"] = "default"
    snapshot.node_metadata[0]["valid_until"] = "not-a-timestamp"
    assert bridge._make_request(snapshot, None, auth, None).visible_state.node_ids.numel() == 0

    assert (
        bridge._make_request(
            graph.to_state(),
            None,
            AuthorizationContext(permissions=frozenset()),
            None,
        ).visible_state.node_ids.numel()
        == 0
    )
    bridge.close()


def test_generic_factory_directs_lfm3_to_the_bridge_factory() -> None:
    with pytest.raises(ValidationError, match="create_lfm3_bridge"):
        RUTAMModelFactory.create_model(backbone=Lfm3ForCausalLM())

