"""Unit tests for held-out evaluation and memory simulation."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.data.scenario_generator import ScenarioGenerator
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.training.dataloader import MemoryDataLoader
from rutam.training.evaluator import RUTAMEvaluator


def _make_small_model() -> RUTAMForCausalLM:
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 16, "num_relation_types": 4},
            "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
            "query_head": {
                "backbone_hidden_dim": 16,
                "graph_hidden_dim": 16,
                "query_dim": 8,
                "intermediate_dim": 16,
            },
            "graph_fusion": {
                "hidden_dim": 16,
                "graph_dim": 16,
                "num_attention_heads": 2,
            },
            "memory_write": {
                "backbone_hidden_dim": 16,
                "hidden_dim": 16,
                "projection_dim": 16,
            },
            "rutam_model": {"backbone_frozen": True, "return_memory_diagnostics": True},
        }
    )
    return RUTAMForCausalLM(config=config)


def test_evaluator_metrics_and_state_preservation() -> None:
    """Verify evaluator computes averages and preserves model.training state."""
    model = _make_small_model()
    model.train()

    gen = ScenarioGenerator(seed=42)
    records = gen.generate_creation_scenarios(6)
    loader = MemoryDataLoader(records, batch_size=2, shuffle=False)

    evaluator = RUTAMEvaluator(simulate_memory=True)
    res = evaluator.evaluate(model, loader)

    assert res.num_examples == 6
    assert res.mean_loss > 0.0
    assert torch.isfinite(torch.tensor(res.mean_loss))
    assert "write_precision" in res.metrics

    # Check model returned to training mode
    assert model.training is True
    assert "acceptance_rate" in res.simulation_metrics
