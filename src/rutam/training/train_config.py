"""Configuration dataclass for local CPU-compatible neural training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rutam.config import positive_int
from rutam.exceptions import ValidationError

TrainingMode = Literal[
    "joint_memory",
    "write_gate",
    "memory_operation",
    "graph_retrieval",
    "relationship",
    "graph_reasoning",
]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Settings controlling training loop, optimization, evaluation, and hardware budget."""

    mode: TrainingMode = "joint_memory"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    optimizer_type: Literal["adamw", "sgd"] = "adamw"
    momentum: float = 0.9  # Used when optimizer_type is "sgd"
    lr_scheduler_type: Literal["constant", "linear", "cosine"] = "constant"
    warmup_steps: int = 0
    max_steps: int = 100
    num_epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_interval_steps: int = 20
    early_stopping_patience: int | None = None
    seed: int = 42
    device: str = "cpu"
    log_interval_steps: int = 10
    memory_simulation_eval: bool = False
    vocab_size: int = 100
    max_seq_len: int = 64
    extra_options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_modes = {
            "joint_memory",
            "write_gate",
            "memory_operation",
            "graph_retrieval",
            "relationship",
            "graph_reasoning",
        }
        if self.mode not in valid_modes:
            raise ValidationError(
                f"Invalid training mode: {self.mode}. Must be one of {valid_modes}"
            )

        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0.0:
            raise ValidationError(f"learning_rate must be positive, got {self.learning_rate}")

        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0.0:
            raise ValidationError(f"weight_decay must be non-negative, got {self.weight_decay}")

        if self.optimizer_type not in {"adamw", "sgd"}:
            raise ValidationError(
                f"optimizer_type must be 'adamw' or 'sgd', got {self.optimizer_type}"
            )

        if self.lr_scheduler_type not in {"constant", "linear", "cosine"}:
            raise ValidationError(f"Invalid lr_scheduler_type: {self.lr_scheduler_type}")

        positive_int(self.max_steps, "max_steps")
        positive_int(self.num_epochs, "num_epochs")
        positive_int(self.batch_size, "batch_size")
        positive_int(self.gradient_accumulation_steps, "gradient_accumulation_steps")
        if not isinstance(self.max_grad_norm, (int, float)) or self.max_grad_norm <= 0.0:
            raise ValidationError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        positive_int(self.eval_interval_steps, "eval_interval_steps")
        positive_int(self.vocab_size, "vocab_size")
        positive_int(self.max_seq_len, "max_seq_len")
        if self.device != "cpu":
            raise ValidationError(f"Local training engine is CPU-only, got device {self.device}")
