"""Parameter group management and learning rate scheduling across model components."""

from __future__ import annotations

from typing import Any

from torch import nn

from rutam.exceptions import ValidationError


class ParameterGroupManager:
    """Partitions unified model parameters into distinct optimizer parameter groups."""

    DEFAULT_LEARNING_RATES = {
        "backbone": 1e-5,
        "encoder": 1e-4,
        "selector": 1e-4,
        "query_head": 1e-4,
        "traversal": 1e-4,
        "linker": 1e-4,
        "write_head": 1e-4,
        "fusion": 1e-4,
    }

    def __init__(
        self,
        learning_rates: dict[str, float] | None = None,
        weight_decay: float = 0.01,
        no_decay_keywords: tuple[str, ...] = ("bias", "LayerNorm.weight", "layernorm.weight"),
    ) -> None:
        self.learning_rates = dict(self.DEFAULT_LEARNING_RATES)
        if learning_rates:
            self.learning_rates.update(learning_rates)
        self.weight_decay = weight_decay
        self.no_decay_keywords = no_decay_keywords

    def create_parameter_groups(
        self,
        modules: dict[str, nn.Module | None],
        include_frozen: bool = False,
    ) -> list[dict[str, Any]]:
        """Partition trainable parameters by submodule role and weight decay policy."""
        groups: list[dict[str, Any]] = []

        for name, module in modules.items():
            if module is None:
                continue
            if not isinstance(module, nn.Module):
                raise ValidationError(f"Expected nn.Module for group '{name}', got {type(module)}")

            lr = self.learning_rates.get(name, 1e-4)
            decay_params = []
            no_decay_params = []

            for p_name, param in module.named_parameters():
                if not param.requires_grad and not include_frozen:
                    continue

                if any(nd in p_name for nd in self.no_decay_keywords):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            if decay_params:
                groups.append(
                    {
                        "params": decay_params,
                        "lr": lr,
                        "weight_decay": self.weight_decay,
                        "name": f"{name}_decay",
                    }
                )
            if no_decay_params:
                groups.append(
                    {
                        "params": no_decay_params,
                        "lr": lr,
                        "weight_decay": 0.0,
                        "name": f"{name}_no_decay",
                    }
                )

        return groups

    def count_parameters(self, modules: dict[str, nn.Module | None]) -> dict[str, dict[str, int]]:
        """Count total and trainable parameters across submodules."""
        summary: dict[str, dict[str, int]] = {}
        for name, module in modules.items():
            if module is None:
                continue
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
        return summary
