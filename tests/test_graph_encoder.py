"""CPU integration tests: graph structure, storage compatibility and actual autograd."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.utils._python_dispatch import TorchDispatchMode

from rutam import EncoderConfig, Graph, GraphStorage, RUTAMConfig, RUTAMGraphEncoder
from rutam.exceptions import ValidationError


def make_graph(count: int = 5, *, directed: bool = True) -> Graph:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 4, "directed": directed},
            "encoder": {"input_dim": 4, "hidden_dim": 8, "num_relation_types": 4, "dropout": 0.2},
        }
    )
    graph = Graph(config)
    # Insert in reverse ID order; GraphState canonicalizes rows before encoding.
    for row in reversed(range(count)):
        graph.create_node(
            "synthetic",
            str(row),
            node_id=5 + row * 11,
            provenance="test:node",
            embedding=[(row + 1) / 10, row % 2 * 0.7, 1 - row * 0.15, -0.2 * row],
        )
    for row in range(count - 1):
        graph.add_relationship(
            5 + row * 11,
            5 + (row + 1) * 11,
            "supports" if row % 2 else "related",
            weight=0.4 + row * 0.1,
            provenance="test:edge",
        )
    return graph


def model_for(graph: Graph, **overrides: Any) -> RUTAMGraphEncoder:
    torch.manual_seed(13)
    return RUTAMGraphEncoder(replace(graph.config.encoder, **overrides))


@pytest.mark.parametrize("count", [0, 1, 5])
def test_empty_single_and_connected_shapes(count: int) -> None:
    graph = make_graph(count)
    output = model_for(graph).eval()(graph.to_state())
    assert output.node_states.shape == (count, 8)
    assert output.graph_summary.shape == (8,)
    assert output.node_states.dtype == output.graph_summary.dtype == torch.float32
    assert output.node_states.device.type == "cpu"
    assert output.has_nodes is (count > 0)
    assert torch.isfinite(output.node_states).all() and torch.isfinite(output.graph_summary).all()
    if count == 0:
        assert torch.equal(output.graph_summary, torch.zeros(8))
    else:
        torch.testing.assert_close(output.graph_summary, output.node_states.mean(dim=0))


def test_empty_output_supports_zero_loss_backward_and_optional_pooling() -> None:
    graph = make_graph(0)
    model = model_for(graph)
    output = model(graph.to_state())
    output.graph_summary.sum().backward()
    assert model.node_projection.linear.weight.grad is not None
    assert model(graph.to_state(), pool_graph=False).graph_summary is None


def test_edge_weights_affect_only_nodes_reachable_within_layer_count() -> None:
    graph = make_graph()
    state = graph.to_state()
    model = model_for(graph).eval()
    before = model(state).node_states
    weights = state.edge_weights.clone()
    weights[0] = 1.7
    after = model(replace(state, edge_weights=weights)).node_states
    assert not torch.allclose(before[1], after[1])
    assert not torch.allclose(before[2], after[2])
    torch.testing.assert_close(before[[0, 3, 4]], after[[0, 3, 4]], rtol=0, atol=0)


def test_relation_type_changes_representations() -> None:
    graph = make_graph()
    state = graph.to_state()
    model = model_for(graph).eval()
    # Both IDs already exist in the graph's persistent vocabulary.
    before = model(state).node_states
    after = model(replace(state, edge_types=1 - state.edge_types)).node_states
    assert not torch.allclose(before[1:], after[1:])
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)


def test_evaluation_is_deterministic_and_training_dropout_is_active() -> None:
    graph = make_graph()
    state = graph.to_state()
    model = model_for(graph)
    model.eval()
    first, second = model(state), model(state)
    torch.testing.assert_close(first.node_states, second.node_states, rtol=0, atol=0)
    torch.testing.assert_close(first.graph_summary, second.graph_summary, rtol=0, atol=0)
    model.train()
    assert not torch.equal(model(state).node_states, model(state).node_states)


def test_gradients_reach_projection_relations_layers_inputs_and_edge_weights() -> None:
    graph = make_graph()
    state = graph.to_state()
    state.node_embeddings.requires_grad_()
    state.edge_weights.requires_grad_()
    model = model_for(graph, dropout=0)
    output = model(state)
    target = torch.linspace(-1, 1, output.node_states.numel()).reshape_as(output.node_states)
    (output.node_states - target).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name
    for tensor in (state.node_embeddings, state.edge_weights):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad).item() > 0
    assert torch.count_nonzero(model.relation_embeddings.weight.grad[2:]).item() == 0


def test_real_optimizer_step_lowers_a_tiny_synthetic_objective() -> None:
    graph = make_graph()
    state = graph.to_state()
    model = model_for(graph, dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    target = torch.linspace(-0.7, 0.8, 40).reshape(5, 8)
    before = model(state).node_states
    loss = (before - target).square().mean()
    original_weights = model.node_projection.linear.weight.detach().clone()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = (model(state).node_states - target).square().mean()
    assert after.item() < loss.item()
    assert not torch.equal(original_weights, model.node_projection.linear.weight)


def test_undirected_edges_expand_both_ways_without_doubling_self_loops() -> None:
    graph = make_graph(2, directed=False)
    graph.add_relationship(5, 5, "self", provenance="test:self")
    state = graph.to_state()
    edges, types, weights = RUTAMGraphEncoder._message_edges(state)
    assert edges.shape == (2, 3) and types.shape == weights.shape == (3,)
    assert set(map(tuple, edges.T.tolist())) == {(0, 0), (0, 1), (1, 0)}
    model = model_for(graph).eval()
    before = model(state).node_states
    altered = state.node_embeddings.clone()
    altered[1] = torch.tensor([2.0, -1.0, 3.0, 1.0])
    after = model(replace(state, node_embeddings=altered)).node_states
    assert not torch.allclose(before[0], after[0])
    assert state.edge_index.shape == (2, 2)


def test_sparse_processing_never_allocates_dense_adjacency() -> None:
    graph = make_graph(257)
    state = graph.to_state()
    model = model_for(graph).eval()

    class SparseGuard(TorchDispatchMode):
        """Inspect actual operator outputs rather than relying on source text."""

        additions: int = 0

        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            result = func(*args, **(kwargs or {}))
            if func._schema.name == "aten::index_add_":
                self.additions += 1
            outputs = result if isinstance(result, (tuple, list)) else (result,)
            for value in outputs:
                if isinstance(value, torch.Tensor):
                    assert value.shape[-2:] != (257, 257), "dense N x N adjacency allocated"
            return result

    with SparseGuard() as guard:
        output = model(state)
    assert guard.additions >= 2
    assert output.node_states.shape == (257, 8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_embeddings", torch.zeros((5, 3))),
        ("node_embeddings", torch.zeros((5, 4), dtype=torch.float64)),
        ("node_embeddings", torch.full((5, 4), float("nan"))),
        ("edge_index", torch.tensor([[0], [5]])),
        ("edge_types", torch.tensor([99, 0, 1, 0])),
        ("edge_weights", torch.zeros(3)),
    ],
)
def test_invalid_graph_tensor_shapes_dtypes_and_references(field: str, value: torch.Tensor) -> None:
    graph = make_graph()
    with pytest.raises(ValidationError):
        model_for(graph)(replace(graph.to_state(), **{field: value}))


def test_encoder_dimension_and_vocabulary_validation() -> None:
    graph = make_graph()
    state = graph.to_state()
    with pytest.raises(ValidationError, match="input_dim"):
        RUTAMGraphEncoder()(state)
    with pytest.raises(ValidationError, match="vocabulary"):
        model_for(graph, num_relation_types=1)(state)
    # Even an unused persisted label must fit the encoder's vocabulary.
    state.relationship_types["retired_label"] = 4
    with pytest.raises(ValidationError, match="vocabulary"):
        model_for(graph)(state)
    with pytest.raises(ValidationError):
        RUTAMGraphEncoder({})
    with pytest.raises(ValidationError):
        model_for(graph)(None)
    with pytest.raises(ValidationError):
        model_for(graph)(graph.to_state(), pool_graph="yes")
    with pytest.raises(ValidationError, match="CPU"):
        model_for(graph).double()(graph.to_state())


def test_storage_round_trip_and_neural_checkpoint_are_separate(tmp_path: Path) -> None:
    graph = make_graph()
    storage = GraphStorage(tmp_path / "graph")
    storage.save(graph)
    original_files = {file.name: file.read_bytes() for file in storage.path.glob("graph.*")}
    model = model_for(graph).eval()
    before = model(graph.to_state())
    loaded_state = storage.load().to_state()
    with (
        patch.object(GraphStorage, "save", side_effect=AssertionError("forward wrote to disk")),
        patch.object(GraphStorage, "load", side_effect=AssertionError("forward read from disk")),
    ):
        after = model(loaded_state)
        after.node_states.square().sum().backward()
    torch.testing.assert_close(before.node_states, after.node_states, rtol=0, atol=0)
    assert loaded_state.node_ids.tolist() == [5, 16, 27, 38, 49]
    assert original_files == {file.name: file.read_bytes() for file in storage.path.glob("graph.*")}
    with torch.no_grad():
        save_file(model.state_dict(), tmp_path / "encoder.safetensors")
    restored = model_for(graph).eval()
    restored.load_state_dict(load_file(tmp_path / "encoder.safetensors"))
    torch.testing.assert_close(
        restored(loaded_state).node_states, before.node_states, rtol=0, atol=0
    )
    assert set(loaded_state.tensors()) == {
        "node_ids",
        "node_embeddings",
        "edge_index",
        "edge_types",
        "edge_weights",
    }


def test_prompt_01_snapshot_without_encoder_configuration_still_loads(tmp_path: Path) -> None:
    graph = make_graph()
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    manifest_path = tmp_path / "graph.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata_path = tmp_path / manifest["payload_directory"] / "graph.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["config"].pop("encoder")
    payload = json.dumps(metadata).encode()
    metadata_path.write_bytes(payload)
    manifest_path = tmp_path / "graph.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["graph.json"] = {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    restored = storage.load()
    assert restored.config.encoder == EncoderConfig()
    torch.testing.assert_close(
        model_for(graph).eval()(restored.to_state()).node_states,
        model_for(graph).eval()(graph.to_state()).node_states,
        rtol=0,
        atol=0,
    )


def test_encoder_configuration_yaml_and_overrides_are_deterministic(tmp_path: Path) -> None:
    config = RUTAMConfig().with_overrides({"encoder": {"hidden_dim": 17, "dropout": 0}})
    path = tmp_path / "config.yaml"
    path.write_text(config.to_yaml())
    assert RUTAMConfig.from_yaml(path) == config
    assert RUTAMConfig.from_yaml(path).to_yaml() == config.to_yaml()
    assert (
        RUTAMConfig.from_yaml(path, overrides={"encoder": {"num_layers": 3}}).encoder.num_layers
        == 3
    )
    with pytest.raises(ValidationError):
        config.with_overrides({"encoder": {"num_layers": -1}})
    with pytest.raises(ValidationError):
        config.with_overrides({"encoder": {"unknown": True}})


def test_default_trainable_parameter_count() -> None:
    model = RUTAMGraphEncoder()
    assert (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        == 119936
    )


def test_initialization_uses_cpu_float32_without_changing_global_defaults() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = RUTAMGraphEncoder()
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)
