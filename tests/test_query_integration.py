"""Hidden states -> neural queries -> existing graph selector, with actual gradients."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from rutam import (
    Graph,
    GraphStorage,
    RUTAMConfig,
    RUTAMGraphEncoder,
    RUTAMGraphSelector,
    RUTAMNeuralQueryHead,
    node_retrieval_loss,
    query_decision_loss,
)
from rutam.exceptions import ValidationError


def setup(
    count: int = 5,
) -> tuple[Graph, RUTAMGraphEncoder, RUTAMGraphSelector, RUTAMNeuralQueryHead]:
    torch.manual_seed(31)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 4},
            "encoder": {"input_dim": 4, "hidden_dim": 6, "dropout": 0},
            "selector": {
                "query_dim": 5,
                "hidden_dim": 6,
                "score_dim": 4,
                "selection_mode": "soft",
                "top_k": 3,
            },
            "query_head": {
                "backbone_hidden_dim": 8,
                "graph_hidden_dim": 6,
                "query_dim": 5,
                "intermediate_dim": 12,
                "dropout": 0,
            },
        }
    )
    graph = Graph(config)
    for row in range(count):
        graph.create_node(
            "synthetic",
            str(row),
            node_id=10 + row * 10,
            provenance="test",
            embedding=torch.randn(4).tolist(),
        )
    for row in range(count - 1):
        graph.add_relationship(10 + row * 10, 20 + row * 10, "links", provenance="test")
    return (
        graph,
        RUTAMGraphEncoder(config.encoder),
        RUTAMGraphSelector(config.selector),
        RUTAMNeuralQueryHead(config.query_head, selector_config=config.selector),
    )


def test_retrieval_backpropagates_through_query_head_and_encoder() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    hidden = torch.randn((2, 4, 8), requires_grad=True)
    output = head.select_graph(hidden, selector, encoder(state), state.node_ids)
    loss = node_retrieval_loss(
        output.selection.relevance_logits, torch.tensor([0, 4]), mask=output.selection.eligible_mask
    )
    loss.backward()
    for parameter in (
        head.input_projection.weight,
        head.output_projection.weight,
        head.pooling.attention_score.weight,
        head.graph_projection.weight,
        head.conditioning_gate.weight,
        selector.scorer.query_projection.weight,
        selector.scorer.node_projection.weight,
        encoder.node_projection.linear.weight,
        encoder.relation_embeddings.weight,
    ):
        assert parameter.grad is not None and parameter.grad.count_nonzero() > 0
        assert torch.isfinite(parameter.grad).all()
    assert hidden.grad.count_nonzero() > 0
    # Retrieval logits alone do not train a utility gate; supervision/downstream reads do.
    assert head.decision_gate.output_projection.weight.grad is None


def test_downstream_gated_context_teaches_continuous_gate_without_hard_threshold() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    output = head.select_graph(torch.randn((2, 3, 8)), selector, encoder(state), state.node_ids)
    assert output.query.should_query is None
    torch.testing.assert_close(
        output.gated_memory_context, output.selection.memory_context * output.query.query_gate
    )
    output.gated_memory_context.square().mean().backward()
    assert head.decision_gate.output_projection.weight.grad.count_nonzero() > 0
    assert head.output_projection.weight.grad.count_nonzero() > 0


@pytest.mark.parametrize("training", [True, False])
def test_empty_graph_is_safe_and_never_queries(training: bool) -> None:
    graph, encoder, selector, head = setup(0)
    head.train(training)
    state = graph.to_state()
    output = head.select_graph(torch.randn((2, 3, 8)), selector, encoder(state), state.node_ids)
    assert output.query.query_gate.count_nonzero() == 0
    assert output.query.should_query is None if training else not output.query.should_query.any()
    assert not output.selection.selection_mask.any()
    assert (output.selection.selected_node_ids == -1).all()
    assert output.gated_memory_context.shape == (2, 6)
    assert output.gated_memory_context.count_nonzero() == 0
    assert torch.isfinite(output.query.query_vectors).all()
    loss = node_retrieval_loss(
        output.selection.relevance_logits,
        torch.tensor([-1, -1]),
        mask=output.selection.eligible_mask,
        ignore_unlabeled=True,
    )
    loss.backward()
    assert torch.isfinite(head.output_projection.weight.grad).all()


def test_inference_threshold_disables_reads_but_training_keeps_continuous_paths() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    hidden = torch.stack((torch.ones((3, 8)), -torch.ones((3, 8))))
    with torch.no_grad():
        gate = head.decision_gate
        gate.input_projection.weight.zero_()
        gate.input_projection.bias.zero_()
        gate.input_projection.weight[0, 0] = 1
        gate.output_projection.weight.zero_()
        gate.output_projection.bias.zero_()
        gate.output_projection.weight[0, 0] = 1
    head.eval()
    output = head.select_graph(hidden, selector, encoder(state), state.node_ids)
    assert output.query.should_query.tolist() == [True, False]
    assert output.selection.selection_mask[0].any()
    assert not output.selection.selection_mask[1].any()
    assert output.gated_memory_context[1].count_nonzero() == 0
    head.train()
    output = head.select_graph(hidden, selector, encoder(state), state.node_ids)
    assert output.query.should_query is None
    assert output.selection.selection_mask.all()
    assert 0 < output.query.query_gate[1] < 0.5


def test_masked_nodes_and_external_availability_disable_querying() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    head.eval()
    output = head.select_graph(
        torch.randn((2, 3, 8)),
        selector,
        encoder(state),
        state.node_ids,
        memory_available=torch.tensor([True, False]),
        node_mask=torch.zeros(5, dtype=torch.bool),
    )
    assert output.query.query_gate.count_nonzero() == 0
    assert not output.query.should_query.any()
    assert not output.selection.selection_mask.any()


def test_tiny_joint_optimizer_step_lowers_retrieval_and_gate_loss() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    encoded = encoder(state)
    hidden = torch.randn((2, 4, 8))
    optimizer = torch.optim.Adam([*head.parameters(), *selector.parameters()], lr=0.001)

    def objective() -> torch.Tensor:
        output = head.select_graph(hidden, selector, encoded, state.node_ids)
        return node_retrieval_loss(
            output.selection.relevance_logits,
            torch.tensor([0, 4]),
            mask=output.selection.eligible_mask,
        ) + query_decision_loss(output.query.query_gate_logits, torch.tensor([1, 0]))

    loss = objective()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = objective()
    assert after.item() < loss.item()


def test_storage_is_unchanged_and_no_io_occurs_in_query_selection(tmp_path: Path) -> None:
    graph, encoder, selector, head = setup()
    storage = GraphStorage(tmp_path)
    storage.save(graph)
    before = {file.name: file.read_bytes() for file in tmp_path.glob("graph.*")}
    state = storage.load().to_state()
    encoded = encoder(state)
    with (
        patch.object(GraphStorage, "load", side_effect=AssertionError("read in query head")),
        patch.object(GraphStorage, "save", side_effect=AssertionError("write in query head")),
    ):
        output = head.select_graph(torch.randn((2, 3, 8)), selector, encoded, state.node_ids)
    assert output.selection.selected_node_ids.shape == (2, 3)
    assert before == {file.name: file.read_bytes() for file in tmp_path.glob("graph.*")}


def test_actual_selector_dimension_mismatch_rejected_at_integration_time() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    wrong = RUTAMGraphSelector(replace(selector.config, query_dim=7))
    with pytest.raises(ValidationError, match="query_dim"):
        head.select_graph(torch.randn((2, 3, 8)), wrong, encoder(state), state.node_ids)


def test_hard_selector_and_independent_batched_eligibility() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    hard = RUTAMGraphSelector(replace(selector.config, selection_mode="hard"))
    eligible = torch.tensor(
        [[True, False, False, False, False], [False, False, False, False, True]]
    )
    output = head.select_graph(
        torch.randn((2, 3, 8)), hard, encoder(state), state.node_ids, eligibility_mask=eligible
    )
    assert output.selection.selected_node_ids[:, 0].tolist() == [10, 50]
    assert output.gated_memory_context is None


def test_zero_batch_end_to_end() -> None:
    graph, encoder, selector, head = setup()
    state = graph.to_state()
    output = head.select_graph(torch.empty((0, 3, 8)), selector, encoder(state), state.node_ids)
    assert output.query.query_vectors.shape == (0, 5)
    assert output.selection.selected_states.shape == (0, 3, 6)
    assert output.gated_memory_context.shape == (0, 6)
