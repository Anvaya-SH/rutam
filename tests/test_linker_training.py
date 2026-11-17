"""Small optimizer and workspace checks, without training-pipeline infrastructure."""

from dataclasses import replace

import pytest
import torch

from rutam import Graph, RUTAMGraphEncoder, RUTAMGraphLinker, combined_link_loss
from rutam.exceptions import ValidationError


def test_tiny_linker_optimizer_step(link_graph: Graph) -> None:
    torch.manual_seed(63)
    graph = link_graph
    state = graph.to_state()
    encoded = RUTAMGraphEncoder(graph.config.encoder)(state)
    model = RUTAMGraphLinker(graph.config.linker)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pairs = torch.tensor([[0, 1], [0, 3]])

    def objective() -> torch.Tensor:
        output = model.from_state(encoded, state, candidate_pairs=pairs)
        return combined_link_loss(
            output, torch.tensor([1, 0]), torch.tensor([1, 0]), config=model.config
        ).total

    loss = objective()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = objective()
    assert after.item() < loss.item()


def test_explicit_candidate_and_scoring_workspace_limits(link_graph: Graph) -> None:
    state = link_graph.to_state()
    model = RUTAMGraphLinker(replace(link_graph.config.linker, max_candidate_expansions=2))
    with pytest.raises(ValidationError, match="max_candidate_expansions"):
        model.from_state(
            torch.ones((6, 8)), state, candidate_pairs=torch.tensor([[0, 1], [0, 2], [0, 3]])
        )
    model = RUTAMGraphLinker(replace(link_graph.config.linker, max_link_elements=1))
    with pytest.raises(ValidationError, match="max_link_elements"):
        model.from_state(torch.ones((6, 8)), state)
