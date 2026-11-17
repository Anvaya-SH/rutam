"""CPU selector integration, exact chunk merging, masks and honest gradient behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rutam import (
    Graph,
    GraphStorage,
    RUTAMConfig,
    RUTAMGraphEncoder,
    RUTAMGraphSelector,
    SelectorConfig,
    node_retrieval_loss,
)
from rutam.exceptions import ValidationError
from rutam.graph.encoder import GraphEncoderOutput


def selector(**settings: Any) -> RUTAMGraphSelector:
    model = RUTAMGraphSelector(
        SelectorConfig(query_dim=2, hidden_dim=2, score_dim=2, top_k=3, **settings)
    )
    with torch.no_grad():
        model.scorer.query_projection.weight.copy_(torch.eye(2))
        model.scorer.node_projection.weight.copy_(torch.eye(2))
    return model


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.eye(2),
        torch.tensor([[3.0, 0.0], [0.0, 4.0], [1.0, 2.0], [-1.0, -1.0]]),
        torch.tensor([80, 7, 50, 21]),
    )


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("n", [0, 1, 4])
def test_empty_single_topk_padding_and_shapes(mode: str, n: int) -> None:
    queries, nodes, ids = inputs()
    output = selector(selection_mode=mode, return_attention=True)(queries, nodes[:n], ids[:n])
    assert output.selected_indices.shape == output.selected_node_ids.shape == (2, 3)
    assert output.selected_states.shape == (2, 3, 2)
    assert output.selected_scores.shape == output.selection_mask.shape == (2, 3)
    assert output.attention_weights.shape == (2, n)
    assert output.selected_indices.dtype == output.selected_node_ids.dtype == torch.int64
    assert output.selection_mask.dtype == torch.bool
    assert output.selected_states.dtype == output.selected_scores.dtype == torch.float32
    assert (output.selected_indices[~output.selection_mask] == -1).all()
    assert (output.selected_node_ids[~output.selection_mask] == -1).all()
    assert output.selected_states[~output.selection_mask].count_nonzero() == 0
    assert output.selected_scores[~output.selection_mask].count_nonzero() == 0
    for value in vars_for_output(output):
        assert torch.isfinite(value).all()
    if n == 0:
        assert not output.selection_mask.any()
        if mode == "soft":
            assert output.memory_context.shape == (2, 2)
            assert output.memory_context.count_nonzero() == 0


def vars_for_output(output: Any) -> list[torch.Tensor]:
    return [
        getattr(output, name)
        for name in output.__dataclass_fields__
        if isinstance(getattr(output, name), torch.Tensor)
    ]


def test_queries_are_independent_and_ids_are_not_row_offsets() -> None:
    output = selector()(*inputs())
    assert output.selected_indices.tolist() == [[0, 2, 1], [1, 2, 0]]
    assert output.selected_node_ids.tolist() == [[80, 50, 7], [7, 50, 80]]
    assert output.attention_weights is None and output.memory_context is None
    assert output.relevance_logits is None


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_combined_validity_per_query_eligibility_and_all_masked_rows(mode: str) -> None:
    queries, nodes, ids = inputs()
    nodes.requires_grad_()
    model = selector(selection_mode=mode, return_attention=True)
    output = model(
        queries,
        nodes,
        ids,
        node_mask=torch.tensor([False, True, True, True]),
        eligibility_mask=torch.tensor([[True, True, False, True], [False] * 4]),
    )
    assert output.selected_node_ids.tolist() == [[7, 21, -1], [-1, -1, -1]]
    assert output.attention_weights[0, [0, 2]].count_nonzero() == 0
    assert output.attention_weights[1].count_nonzero() == 0
    assert output.attention_weights[0].sum().item() == pytest.approx(1.0)
    assert not output.selection_mask[1].any()
    if mode == "soft":
        assert output.memory_context[1].count_nonzero() == 0
        output.memory_context.square().sum().backward()
        assert torch.isfinite(nodes.grad).all()


@pytest.mark.parametrize("method", ["scaled_dot", "additive"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 20, None])
def test_chunked_topk_matches_full_scoring_across_boundaries(
    method: str, chunk_size: int | None
) -> None:
    torch.manual_seed(11)
    model = selector(scoring_method=method, chunk_size=chunk_size)
    queries, nodes = torch.randn((3, 2)), torch.randn((11, 2))
    ids = torch.tensor([90, 4, 80, 7, 22, 31, 12, 1, 65, 51, 9])
    eligibility = torch.rand((3, 11)) > 0.3
    eligibility[2] = False
    chunked = model(queries, nodes, ids, eligibility_mask=eligibility)
    full_model = RUTAMGraphSelector(replace(model.config, chunk_size=None, return_attention=True))
    full_model.load_state_dict(model.state_dict())
    full = full_model(queries, nodes, ids, eligibility_mask=eligibility)
    assert torch.equal(chunked.selected_indices, full.selected_indices)
    assert torch.equal(chunked.selected_node_ids, full.selected_node_ids)
    torch.testing.assert_close(chunked.selected_scores, full.selected_scores)
    torch.testing.assert_close(chunked.selected_states, full.selected_states)


def test_equal_score_id_order_is_stable_across_chunks_and_row_permutations() -> None:
    model = selector(chunk_size=2)
    queries, nodes, ids = (
        torch.zeros((2, 2)),
        torch.ones((7, 2)),
        torch.tensor([71, 9, 52, 1, 33, 4, 11]),
    )
    expected = [[1, 4, 9], [1, 4, 9]]
    assert model(queries, nodes, ids).selected_node_ids.tolist() == expected
    order = torch.tensor([6, 5, 4, 3, 2, 1, 0])
    assert model(queries, nodes[order], ids[order]).selected_node_ids.tolist() == expected


@pytest.mark.parametrize("method", ["scaled_dot", "additive"])
def test_soft_context_and_supervision_train_query_and_node_projections(method: str) -> None:
    torch.manual_seed(4)
    model = selector(selection_mode="soft", scoring_method=method)
    queries, nodes, ids = inputs()
    queries.requires_grad_()
    nodes.requires_grad_()
    output = model(queries, nodes, ids)
    assert output.attention_weights is None  # Context still uses attention internally.
    expected_weights = torch.softmax(model.scorer(queries, nodes), dim=1)
    torch.testing.assert_close(output.memory_context, expected_weights @ nodes)
    loss = output.memory_context.square().mean() + node_retrieval_loss(
        output.relevance_logits, torch.tensor([0, 1]), mask=output.eligible_mask
    )
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    assert queries.grad.count_nonzero() > 0 and nodes.grad.count_nonzero() > 0


def test_hard_gather_has_no_ranking_gradient_but_selected_scores_do() -> None:
    model = selector(chunk_size=1)
    queries, nodes, ids = inputs()
    queries.requires_grad_()
    nodes.requires_grad_()
    output = model(queries, nodes, ids)
    output.selected_states.sum().backward()
    assert queries.grad is None and model.scorer.query_projection.weight.grad is None
    assert nodes.grad.count_nonzero() > 0
    assert not output.selected_indices.requires_grad
    model.zero_grad()
    output = model(queries, nodes, ids)
    output.selected_scores.square().sum().backward()
    assert queries.grad.count_nonzero() > 0
    assert model.scorer.query_projection.weight.grad.count_nonzero() > 0
    assert model.scorer.node_projection.weight.grad.count_nonzero() > 0


@pytest.mark.parametrize("method", ["scaled_dot", "additive"])
def test_two_pass_hard_selected_score_gradients_equal_dense_gather(method: str) -> None:
    torch.manual_seed(28)
    model = selector(chunk_size=1, scoring_method=method)
    dense = RUTAMGraphSelector(replace(model.config, return_attention=True))
    dense.load_state_dict(model.state_dict())
    queries, nodes, ids = inputs()
    q1, q2 = queries.clone().requires_grad_(), queries.clone().requires_grad_()
    n1, n2 = nodes.clone().requires_grad_(), nodes.clone().requires_grad_()
    first, second = model(q1, n1, ids), dense(q2, n2, ids)
    assert torch.equal(first.selected_indices, second.selected_indices)
    first.selected_scores.square().sum().backward()
    second.selected_scores.square().sum().backward()
    torch.testing.assert_close(q1.grad, q2.grad)
    torch.testing.assert_close(n1.grad, n2.grad)
    for p, other in zip(model.parameters(), dense.parameters(), strict=True):
        torch.testing.assert_close(p.grad, other.grad)


def test_soft_context_optimizer_step_lowers_synthetic_loss() -> None:
    model = selector(selection_mode="soft")
    queries, nodes, ids = inputs()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    target = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    loss = (model(queries, nodes, ids).memory_context - target).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = (model(queries, nodes, ids).memory_context - target).square().mean()
    assert after.item() < loss.item()


def test_soft_and_attention_limits_fail_explicitly_and_hard_stays_chunked() -> None:
    queries, nodes, ids = inputs()
    model = selector(max_dense_elements=12, chunk_size=1)
    assert model(queries, nodes, ids).selected_indices.shape == (2, 3)
    # Additive workspace B*N*S = 16, while hard winner work is B*K*S = 12.
    model = selector(scoring_method="additive", max_dense_elements=12)
    assert model(queries, nodes, ids).selected_indices.shape == (2, 3)
    with pytest.raises(ValidationError, match="max_dense_elements"):
        model(queries, nodes, ids, return_attention=True)
    model = RUTAMGraphSelector(replace(model.config, selection_mode="soft"))
    with pytest.raises(ValidationError, match="max_dense_elements"):
        model(queries, nodes, ids)


def test_chunking_allocates_neither_full_attention_nor_node_adjacency() -> None:
    torch.manual_seed(8)
    model = selector(chunk_size=7)
    queries, nodes, ids = torch.randn((2, 2)), torch.randn((101, 2)), torch.arange(101)

    class AllocationGuard(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            result = func(*args, **(kwargs or {}))
            for value in result if isinstance(result, (tuple, list)) else (result,):
                if isinstance(value, torch.Tensor):
                    assert value.shape[-2:] not in ((101, 101), (2, 101))
            return result

    with AllocationGuard():
        result = model(queries, nodes, ids)
    assert result.attention_weights is None and result.relevance_logits is None


def test_encoder_to_selector_storage_compatibility_and_end_to_end_gradient(tmp_path: Path) -> None:
    torch.manual_seed(16)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 3},
            "encoder": {"input_dim": 3, "hidden_dim": 4, "dropout": 0},
            "selector": {"query_dim": 2, "hidden_dim": 4, "score_dim": 3, "selection_mode": "soft"},
        }
    )
    graph = Graph(config)
    for row in range(5):
        graph.create_node(
            "synthetic",
            str(row),
            node_id=9 + row * 10,
            provenance="test",
            embedding=torch.randn(3).tolist(),
        )
    for row in range(4):
        graph.add_relationship(9 + row * 10, 19 + row * 10, "links", provenance="test")
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    original = {file.name: file.read_bytes() for file in tmp_path.glob("graph.*")}
    state = storage.load().to_state()
    encoder = RUTAMGraphEncoder(config.encoder)
    model = RUTAMGraphSelector(config.selector)
    encoded = encoder(state)
    with (
        patch.object(GraphStorage, "save", side_effect=AssertionError("selector wrote graph")),
        patch.object(GraphStorage, "load", side_effect=AssertionError("selector read disk")),
    ):
        output = model(torch.randn((2, 2)), encoded, state.node_ids)
        node_retrieval_loss(
            output.relevance_logits, torch.tensor([0, 4]), mask=output.eligible_mask
        ).backward()
    assert encoder.node_projection.linear.weight.grad.count_nonzero() > 0
    assert encoder.relation_embeddings.weight.grad.count_nonzero() > 0
    assert original == {file.name: file.read_bytes() for file in tmp_path.glob("graph.*")}
    assert all(
        node_id in state.node_ids.tolist()
        for node_id in output.selected_node_ids[output.selection_mask].tolist()
    )


def test_invalid_inputs_and_encoder_output_consistency() -> None:
    queries, nodes, ids = inputs()
    model = selector()
    for q, n, identifier in [
        (queries[:, :1], nodes, ids),
        (queries, nodes[:, :1], ids),
        (queries, nodes, ids.float()),
        (queries, nodes, torch.tensor([1, 1, 2, 3])),
        (queries, nodes, torch.tensor([-1, 2, 3, 4])),
    ]:
        with pytest.raises(ValidationError):
            model(q, n, identifier)
    for kwargs in (
        {"node_mask": torch.ones(4)},
        {"node_mask": torch.ones(3, dtype=torch.bool)},
        {"eligibility_mask": torch.ones((1, 4), dtype=torch.bool)},
        {"return_attention": "yes"},
    ):
        with pytest.raises(ValidationError):
            model(queries, nodes, ids, **kwargs)
    with pytest.raises(ValidationError, match="has_nodes"):
        model(queries, GraphEncoderOutput(nodes, None, False), ids)


def test_selector_config_serialization_and_legacy_defaults(tmp_path: Path) -> None:
    config = RUTAMConfig().with_overrides({"selector": {"top_k": 5, "chunk_size": None}})
    path = tmp_path / "config.yaml"
    path.write_text(config.to_yaml())
    assert RUTAMConfig.from_yaml(path) == config
    legacy = config.to_dict()
    legacy.pop("selector")
    assert RUTAMConfig.from_dict(legacy).selector == SelectorConfig()
    with pytest.raises(ValidationError):
        RUTAMConfig.from_dict({"selector": {"unknown": 1}})


@pytest.mark.parametrize("method", ["scaled_dot", "additive"])
@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_empty_batch(method: str, mode: str) -> None:
    _, nodes, ids = inputs()
    output = selector(scoring_method=method, selection_mode=mode)(torch.empty((0, 2)), nodes, ids)
    assert output.selected_indices.shape == (0, 3)
    assert output.selected_states.shape == (0, 3, 2)


@pytest.mark.parametrize("method", ["scaled_dot", "additive"])
@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("n", [0, 4])
def test_all_masked_and_empty_backward_passes_are_finite(method: str, mode: str, n: int) -> None:
    queries, nodes, ids = inputs()
    queries.requires_grad_()
    nodes = nodes[:n].clone().requires_grad_()
    model = selector(scoring_method=method, selection_mode=mode)
    output = model(queries, nodes, ids[:n], node_mask=torch.zeros(n, dtype=torch.bool))
    loss = output.selected_scores.sum() + output.selected_states.sum()
    if mode == "soft":
        loss = loss + output.memory_context.sum()
    assert loss.item() == 0
    loss.backward()
    for tensor in (queries, nodes, *model.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.count_nonzero() == 0
