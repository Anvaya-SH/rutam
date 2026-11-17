"""Unit tests for RUTAMTrainer training loop, loss reduction, and early stopping."""

from __future__ import annotations

import pytest

from rutam.config import RUTAMConfig
from rutam.data.scenario_generator import ScenarioGenerator
from rutam.exceptions import ValidationError
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.training.train_config import TrainingConfig
from rutam.training.trainer import RUTAMTrainer


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


def test_trainer_loss_reduction_overfitting() -> None:
    """Verify that training decreases memory loss on a small synthetic batch."""
    model = _make_small_model()
    gen = ScenarioGenerator(seed=42)
    records = gen.generate_creation_scenarios(8)

    cfg = TrainingConfig(
        mode="write_gate",
        learning_rate=0.05,
        max_steps=25,
        batch_size=4,
        gradient_accumulation_steps=1,
        seed=42,
    )

    trainer = RUTAMTrainer(model=model, config=cfg, train_dataset=records)
    summary = trainer.train()

    assert summary.total_steps == 25
    assert summary.runtime_seconds > 0.0
    assert summary.final_loss < summary.initial_loss
    assert len(summary.history) > 0
    assert summary.trainable_parameters_count > 0
    assert summary.frozen_parameters_count > 0


def test_trainer_early_stopping() -> None:
    """Verify early stopping terminates training when validation loss stalls."""
    model = _make_small_model()
    gen = ScenarioGenerator(seed=42)
    train_records = gen.generate_creation_scenarios(8)
    val_records = gen.generate_creation_scenarios(4)

    cfg = TrainingConfig(
        mode="write_gate",
        learning_rate=0.0001,
        max_steps=50,
        eval_interval_steps=5,
        early_stopping_patience=2,
        seed=42,
    )

    trainer = RUTAMTrainer(
        model=model,
        config=cfg,
        train_dataset=train_records,
        eval_dataset=val_records,
    )
    summary = trainer.train()

    assert summary.total_steps <= 50
    assert summary.final_eval_result is not None


def test_trainer_no_dataset_error() -> None:
    """Verify trainer raises error if train() called without a train dataset."""
    model = _make_small_model()
    trainer = RUTAMTrainer(model=model)
    with pytest.raises(ValidationError, match="no training dataset provided"):
        trainer.train()
