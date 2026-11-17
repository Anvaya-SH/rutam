"""Unit tests verifying gradient flow and parameter freezing during training."""

from __future__ import annotations

import torch

from rutam.config import RUTAMConfig
from rutam.data.scenario_generator import ScenarioGenerator
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.training.batch_collator import MemoryBatchCollator
from rutam.training.memory_objectives import RUTAMMemoryLearningObjective
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


def test_frozen_backbone_has_no_gradients() -> None:
    """Verify that language model backbone parameters remain frozen during backprop."""
    torch.manual_seed(42)
    model = _make_small_model()
    gen = ScenarioGenerator(seed=42)
    records = gen.generate_creation_scenarios(2)

    collator = MemoryBatchCollator(vocab_size=100, max_seq_len=16)
    batch = collator(records)

    out = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        generate_proposals=True,
    )

    objective = RUTAMMemoryLearningObjective()
    learn_out = objective(
        write_output=out.write_output,
        query_output=(
            out.graph_selection_output.query_output
            if out.graph_selection_output is not None
            else None
        ),
        targets=batch.targets,
    )

    learn_out.total_loss.backward()

    # Backbone parameters must have no gradients
    for p in model.backbone.parameters():
        assert p.grad is None

    # Memory write head parameters must have valid non-zero gradients
    write_gate_grads = [
        p.grad for p in model.write_head.write_gate.parameters() if p.grad is not None
    ]
    assert len(write_gate_grads) > 0
    assert all(torch.isfinite(g).all() for g in write_gate_grads)


def test_write_gate_mode_freezes_other_modules() -> None:
    """Verify mode 'write_gate' isolates gradient flow strictly to write gate."""
    model = _make_small_model()
    cfg = TrainingConfig(mode="write_gate")
    _ = RUTAMTrainer(model=model, config=cfg)

    # Only write_gate parameters have requires_grad=True
    for name, p in model.named_parameters():
        if "write_gate" in name:
            assert p.requires_grad is True
        else:
            assert p.requires_grad is False
