"""Local CPU neural training engine for RUTAM native memory modules."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rutam.data.example_schema import MemoryExampleRecord
from rutam.exceptions import NumericalInstabilityError, ValidationError
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.model.rutam_outputs import RUTAMCausalLMOutput
from rutam.training.dataloader import MemoryDataLoader
from rutam.training.evaluator import EvaluationResult, RUTAMEvaluator
from rutam.training.learning_output import MemoryLearningOutput
from rutam.training.memory_objectives import RUTAMMemoryLearningObjective
from rutam.training.optimizer import create_optimizer
from rutam.training.reproducibility import set_seed
from rutam.training.scheduler import create_lr_scheduler
from rutam.training.train_config import TrainingConfig


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Summary record reporting training outcomes, metrics, parameter counts, and duration."""

    initial_loss: float
    final_loss: float
    total_steps: int
    runtime_seconds: float
    history: list[dict[str, Any]]
    best_eval_loss: float | None
    final_eval_result: EvaluationResult | None
    trainable_parameters_count: int
    frozen_parameters_count: int
    warnings: list[str] = field(default_factory=list)


class RUTAMTrainer:
    """Orchestrates local training, evaluation, gradient accumulation, and parameter updates."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        train_dataset: str | Path | Sequence[MemoryExampleRecord] | None = None,
        eval_dataset: str | Path | Sequence[MemoryExampleRecord] | None = None,
        objective: RUTAMMemoryLearningObjective | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        evaluator: RUTAMEvaluator | None = None,
    ) -> None:
        self.model = model
        self.config = TrainingConfig() if config is None else config
        set_seed(self.config.seed)

        self._configure_trainable_modules()

        self.objective = objective if objective is not None else RUTAMMemoryLearningObjective()
        self.evaluator = (
            evaluator
            if evaluator is not None
            else RUTAMEvaluator(
                objective=self.objective,
                simulate_memory=self.config.memory_simulation_eval,
            )
        )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.train_loader: MemoryDataLoader | None = None
        if self.train_dataset is not None:
            self.train_loader = MemoryDataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                seed=self.config.seed,
            )

        self.eval_loader: MemoryDataLoader | None = None
        if self.eval_dataset is not None:
            self.eval_loader = MemoryDataLoader(
                dataset=self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                seed=self.config.seed,
            )

        self.optimizer = (
            optimizer if optimizer is not None else create_optimizer(self.model, self.config)
        )
        self.scheduler = create_lr_scheduler(self.optimizer, self.config, self.config.max_steps)

    def _configure_trainable_modules(self) -> None:
        """Freeze or unfreeze submodules according to the configured training mode."""
        if not isinstance(self.model, RUTAMForCausalLM):
            return

        # By default, keep backbone frozen on local CPU
        self.model.freeze_backbone()

        mode = self.config.mode
        if mode == "write_gate":
            # Freeze everything except write gate
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.write_head.write_gate.parameters():
                param.requires_grad = True

        elif mode == "memory_operation":
            # Freeze everything except operation classifier and candidate generator
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.write_head.operation_classifier.parameters():
                param.requires_grad = True
            for param in self.model.write_head.candidate_generator.parameters():
                param.requires_grad = True

        elif mode == "graph_retrieval":
            # Train query head and graph selector
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.query_head.parameters():
                param.requires_grad = True
            if self.model.selector is not None:
                for param in self.model.selector.parameters():
                    param.requires_grad = True

        elif mode == "relationship":
            # Train link prediction & traversal weights
            for param in self.model.parameters():
                param.requires_grad = False
            if self.model.traversal is not None:
                for param in self.model.traversal.parameters():
                    param.requires_grad = True

        elif mode == "joint_memory":
            # Train write head and query head; keep backbone frozen
            self.model.freeze_backbone()
            if self.model.write_head is not None:
                for param in self.model.write_head.parameters():
                    param.requires_grad = True
            if self.model.query_head is not None:
                for param in self.model.query_head.parameters():
                    param.requires_grad = True

    def train(self) -> TrainingSummary:
        """Execute training loop with gradient accumulation, evaluation, and early stopping."""
        if self.train_loader is None:
            raise ValidationError("Cannot train: no training dataset provided")

        start_time = time.perf_counter()
        self.model.train()

        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

        if trainable_count == 0:
            raise ValidationError("Cannot train: model has 0 trainable parameters")

        step = 0
        epoch = 0
        initial_loss: float | None = None
        current_loss: float = 0.0
        history: list[dict[str, Any]] = []
        best_eval_loss: float | None = None
        patience_counter = 0
        final_eval: EvaluationResult | None = None
        warnings: list[str] = []

        self.optimizer.zero_grad()
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        while step < self.config.max_steps:
            self.train_loader.set_epoch(epoch)
            for batch in self.train_loader:
                if step >= self.config.max_steps:
                    break

                # 1. Forward pass
                if isinstance(self.model, RUTAMForCausalLM):
                    out: RUTAMCausalLMOutput = self.model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        generate_proposals=True,
                    )
                    query_out = (
                        out.graph_selection_output.query_output
                        if out.graph_selection_output is not None
                        else None
                    )
                    learn_out: MemoryLearningOutput = self.objective(
                        write_output=out.write_output,
                        query_output=query_out,
                        targets=batch.targets,
                    )
                else:
                    write_out = self.model(batch.input_ids)
                    learn_out = self.objective(write_output=write_out, targets=batch.targets)

                loss = learn_out.total_loss
                loss_val = loss.item()

                if torch.isnan(loss) or torch.isinf(loss):
                    raise NumericalInstabilityError(
                        f"Encountered non-finite loss at step {step}: {loss_val}"
                    )

                if initial_loss is None:
                    initial_loss = loss_val
                current_loss = loss_val

                # 2. Scaled backward pass
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                # 3. Optimizer step on accumulation boundary
                if (step + 1) % accum_steps == 0 or (step + 1) == self.config.max_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                step += 1

                # 4. Periodic logging
                if step % self.config.log_interval_steps == 0 or step == 1:
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "loss": loss_val,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    history.append(entry)

                # 5. Periodic evaluation
                if self.eval_loader is not None and (
                    step % self.config.eval_interval_steps == 0 or step == self.config.max_steps
                ):
                    eval_res = self.evaluator.evaluate(self.model, self.eval_loader)
                    final_eval = eval_res
                    eval_loss = eval_res.mean_loss

                    if best_eval_loss is None or eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if (
                        self.config.early_stopping_patience is not None
                        and patience_counter >= self.config.early_stopping_patience
                    ):
                        warnings.append(f"Early stopping triggered at step {step}")
                        break

            epoch += 1

        elapsed = time.perf_counter() - start_time

        return TrainingSummary(
            initial_loss=initial_loss if initial_loss is not None else 0.0,
            final_loss=current_loss,
            total_steps=step,
            runtime_seconds=elapsed,
            history=history,
            best_eval_loss=best_eval_loss,
            final_eval_result=final_eval,
            trainable_parameters_count=trainable_count,
            frozen_parameters_count=frozen_count,
            warnings=warnings,
        )
