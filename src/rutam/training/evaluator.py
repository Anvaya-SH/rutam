"""Held-out validation evaluator and memory simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.model.rutam_outputs import RUTAMCausalLMOutput
from rutam.training.dataloader import MemoryDataLoader
from rutam.training.learning_output import MemoryLearningOutput
from rutam.training.memory_objectives import RUTAMMemoryLearningObjective


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and diagnostics produced by evaluating on held-out validation examples."""

    mean_loss: float
    loss_components: dict[str, float]
    metrics: dict[str, float]
    num_examples: int
    simulation_metrics: dict[str, Any] = field(default_factory=dict)


class RUTAMEvaluator:
    """Evaluates RUTAM models on memory datasets with optional isolated simulation."""

    def __init__(
        self,
        objective: RUTAMMemoryLearningObjective | None = None,
        simulate_memory: bool = False,
    ) -> None:
        self.objective = objective if objective is not None else RUTAMMemoryLearningObjective()
        self.simulate_memory = simulate_memory

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        eval_loader: MemoryDataLoader,
    ) -> EvaluationResult:
        """Run validation evaluation over eval_loader."""
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_examples = 0
        loss_components_acc: dict[str, float] = {}
        metrics_acc: dict[str, float] = {}
        batch_count = 0

        sim_accepted = 0
        sim_rejected = 0
        sim_deferred = 0
        sim_total = 0

        for batch in eval_loader:
            batch_size = batch.batch_size
            total_examples += batch_size
            batch_count += 1

            if isinstance(model, RUTAMForCausalLM):
                model_out: RUTAMCausalLMOutput = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    generate_proposals=True,
                )
                query_out = (
                    model_out.graph_selection_output.query_output
                    if model_out.graph_selection_output is not None
                    else None
                )
                learning_out: MemoryLearningOutput = self.objective(
                    write_output=model_out.write_output,
                    query_output=query_out,
                    targets=batch.targets,
                )

                if self.simulate_memory and model_out.write_proposals:
                    for prop in model_out.write_proposals:
                        sim_total += 1
                        decision = getattr(prop, "validation_decision", "accept")
                        if decision == "accept":
                            sim_accepted += 1
                        elif decision == "reject":
                            sim_rejected += 1
                        else:
                            sim_deferred += 1
            else:
                # Standalone write head or custom submodule
                write_out = model(batch.input_ids)
                learning_out = self.objective(write_output=write_out, targets=batch.targets)

            loss_val = learning_out.total_loss.item()
            total_loss += loss_val

            for k, v in learning_out.loss_components.items():
                loss_components_acc[k] = loss_components_acc.get(k, 0.0) + v.item()

            for k, v in learning_out.metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v

        if was_training:
            model.train()

        num_batches = max(1, batch_count)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components_acc.items()}
        avg_metrics = {k: v / num_batches for k, v in metrics_acc.items()}

        sim_metrics: dict[str, Any] = {}
        if self.simulate_memory:
            sim_metrics = {
                "proposals_simulated": sim_total,
                "acceptance_rate": (sim_accepted / sim_total) if sim_total > 0 else 0.0,
                "rejection_rate": (sim_rejected / sim_total) if sim_total > 0 else 0.0,
                "deferral_rate": (sim_deferred / sim_total) if sim_total > 0 else 0.0,
            }

        return EvaluationResult(
            mean_loss=avg_loss,
            loss_components=avg_components,
            metrics=avg_metrics,
            num_examples=total_examples,
            simulation_metrics=sim_metrics,
        )
