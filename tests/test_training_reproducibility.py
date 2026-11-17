"""Unit tests verifying reproducibility of training loss and weight trajectories."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.data.scenario_generator import ScenarioGenerator
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.training.reproducibility import set_seed
from rutam.training.train_config import TrainingConfig
from rutam.training.trainer import RUTAMTrainer


def _make_small_model(seed: int = 42) -> RUTAMForCausalLM:
    set_seed(seed)
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


def test_training_reproducibility() -> None:
    """Verify that identical seeds produce identical loss trajectory across steps."""
    gen1 = ScenarioGenerator(seed=42)
    records1 = gen1.generate_creation_scenarios(8)

    gen2 = ScenarioGenerator(seed=42)
    records2 = gen2.generate_creation_scenarios(8)

    model1 = _make_small_model(seed=123)
    cfg1 = TrainingConfig(mode="write_gate", learning_rate=0.01, max_steps=5, seed=42)
    trainer1 = RUTAMTrainer(model=model1, config=cfg1, train_dataset=records1)
    summary1 = trainer1.train()

    model2 = _make_small_model(seed=123)
    cfg2 = TrainingConfig(mode="write_gate", learning_rate=0.01, max_steps=5, seed=42)
    trainer2 = RUTAMTrainer(model=model2, config=cfg2, train_dataset=records2)
    summary2 = trainer2.train()

    # Exact equality of losses
    assert summary1.total_steps == summary2.total_steps
    assert abs(summary1.initial_loss - summary2.initial_loss) < 1e-6
    assert abs(summary1.final_loss - summary2.final_loss) < 1e-6

    # Verify identical weight tensors
    w1 = model1.write_head.write_gate.linear1.weight
    w2 = model2.write_head.write_gate.linear1.weight
    assert torch.equal(w1, w2)
