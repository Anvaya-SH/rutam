"""Integration tests for neural optimization and training step dynamics in RUTAM."""

from __future__ import annotations

import torch

from rutam.data.scenario_generator import ScenarioGenerator
from rutam.diagnostics.integration import create_verification_model
from rutam.training.train_config import TrainingConfig
from rutam.training.trainer import RUTAMTrainer


def test_single_training_step_parameter_update() -> None:
    """Verify that an optimizer step updates trainable parameters while preserving backbone."""
    model, _ = create_verification_model(
        num_nodes=4, hidden_dim=16, vocab_size=128, backbone_frozen=True
    )

    # Record initial weights
    backbone_initial = [p.clone().detach() for p in model.backbone.parameters()]
    fusion_initial = [p.clone().detach() for p in model.fusion.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)

    input_ids = torch.randint(0, 60, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 60, (2, 8), dtype=torch.long)

    # Training step
    optimizer.zero_grad()
    model.bridge.invalidate_caches()
    out = model(input_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    optimizer.step()

    # Backbone parameters must be strictly identical
    for init_p, curr_p in zip(backbone_initial, model.backbone.parameters(), strict=True):
        assert torch.equal(init_p, curr_p)

    # Fusion parameters must have updated
    fusion_current = [p for p in model.fusion.parameters() if p.requires_grad]
    has_changed = any(
        not torch.equal(init_p, curr_p)
        for init_p, curr_p in zip(fusion_initial, fusion_current, strict=True)
    )
    assert has_changed is True


def test_multi_step_training_loss_decrease() -> None:
    """Verify that RUTAMTrainer decreases loss over multiple optimization steps."""
    model, _ = create_verification_model(
        num_nodes=4, hidden_dim=16, vocab_size=128, backbone_frozen=True
    )
    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate_creation_scenarios(8)

    config = TrainingConfig(
        mode="write_gate",
        learning_rate=0.05,
        max_steps=20,
        batch_size=4,
        seed=42,
    )

    trainer = RUTAMTrainer(model=model, config=config, train_dataset=scenarios)
    summary = trainer.train()

    assert summary.total_steps == 20
    assert summary.final_loss < summary.initial_loss
    assert summary.trainable_parameters_count > 0
    assert summary.frozen_parameters_count > 0


def test_parameter_group_partitioning() -> None:
    """Verify parameter group partitioning for weight decay differentiation."""
    model, _ = create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=60)
    groups = model.get_parameter_groups(include_frozen=False)

    assert len(groups) > 0
    for grp in groups:
        assert "params" in grp
        assert len(grp["params"]) > 0
        assert "weight_decay" in grp
