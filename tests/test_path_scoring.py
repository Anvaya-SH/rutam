"""Query-dependent neural path scores and analytical supervised losses."""

import math
from dataclasses import replace

import pytest
import torch

from rutam import (
    RUTAMPathScorer,
    TraversalConfig,
    endpoint_supervision_loss,
    pairwise_path_ranking_loss,
    path_retrieval_loss,
)
from rutam.exceptions import ValidationError


@pytest.mark.parametrize("method", ["additive", "scaled_dot", "cosine"])
def test_queries_attention_chunk_equivalence_and_gradients(method: str) -> None:
    torch.manual_seed(14)
    c = TraversalConfig(
        query_dim=3,
        hidden_dim=4,
        score_hidden_dim=5,
        path_scoring=method,
        max_candidate_paths=6,
        max_selected_paths=2,
    )
    scorer = RUTAMPathScorer(c)
    chunked = RUTAMPathScorer(replace(c, scoring_chunk_size=1))
    chunked.load_state_dict(scorer.state_dict())
    queries = torch.randn((2, 3), requires_grad=True)
    paths = torch.randn((2, 6, 4), requires_grad=True)
    mask = torch.tensor([[True, True, False, True, False, True], [False] * 6])
    output = scorer(queries, paths, mask)
    torch.testing.assert_close(output.logits, chunked(queries, paths, mask).logits)
    assert output.attention[0].sum().item() == pytest.approx(1)
    assert output.attention[1].count_nonzero() == 0
    assert output.attention[~mask].count_nonzero() == 0
    altered = scorer(-queries, paths, mask)
    assert not torch.allclose(output.logits[0], altered.logits[0])
    (output.attention[..., None] * paths).sum(1).square().sum().backward()
    assert queries.grad.count_nonzero() > 0 and paths.grad.count_nonzero() > 0
    assert scorer.scorer.query_projection.weight.grad.count_nonzero() > 0


def test_path_and_endpoint_targets_are_distinct_and_multi_positive() -> None:
    logits = torch.tensor([[0.0, math.log(2), math.log(3)]], requires_grad=True)
    loss = path_retrieval_loss(logits, torch.tensor([[8, 7]]), path_ids=torch.tensor([[9, 8, 7]]))
    assert loss.item() == pytest.approx(-math.log(5 / 6), abs=1e-7)
    endpoint = endpoint_supervision_loss(
        logits, torch.tensor([[100, 200, 200]]), torch.tensor([200])
    )
    torch.testing.assert_close(loss, endpoint)
    with pytest.raises(ValidationError):
        path_retrieval_loss(logits, torch.tensor([200]), path_ids=torch.tensor([[9, 8, 7]]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_pairwise_margin_and_padding() -> None:
    logits = torch.tensor([[0.0, 2.0]], requires_grad=True)
    loss = pairwise_path_ranking_loss(logits, torch.tensor([[0, -1]]), torch.tensor([[1, -1]]))
    assert loss.item() == 3
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1.0, 1.0]]))
    with pytest.raises(ValidationError):
        pairwise_path_ranking_loss(logits, torch.tensor([[0]]), torch.tensor([[-1]]))
