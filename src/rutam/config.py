"""Strict, dependency-light configuration with deterministic YAML serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rutam.exceptions import ValidationError


def positive_int(value: object, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")


def boolean(value: object, name: str) -> None:
    if type(value) is not bool:
        raise ValidationError(f"{name} must be a boolean")


def nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class GraphConfig:
    path: str = "data/graph"
    embedding_dim: int = 128
    directed: bool = True
    max_nodes: int = 100_000
    max_edges: int = 500_000
    initial_nodes: int = 0
    auto_save: bool = False

    def __post_init__(self) -> None:
        nonempty_string(self.path, "graph.path")
        for name in ("embedding_dim", "max_nodes", "max_edges"):
            positive_int(getattr(self, name), f"graph.{name}")
        positive_int(self.initial_nodes, "graph.initial_nodes", minimum=0)
        if self.initial_nodes != 0:
            raise ValidationError(
                "initial_nodes must be 0: nodes require explicit content/provenance"
            )
        boolean(self.directed, "graph.directed")
        boolean(self.auto_save, "graph.auto_save")


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    validation_enabled: bool = True
    require_provenance: bool = True

    def __post_init__(self) -> None:
        boolean(self.validation_enabled, "memory.validation_enabled")
        boolean(self.require_provenance, "memory.require_provenance")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.device != "cpu" or self.dtype != "float32":
            raise ValidationError("This foundation supports only device=cpu and dtype=float32")


@dataclass(frozen=True, slots=True)
class RUTAMConfig:
    project_name: str = "RUTAM"
    backbone: str = "LiquidAI/LFM3-1.7B-Base"
    graph: GraphConfig = field(default_factory=GraphConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        nonempty_string(self.project_name, "project_name")
        nonempty_string(self.backbone, "backbone")
        for name, kind in (
            ("graph", GraphConfig),
            ("memory", MemoryConfig),
            ("runtime", RuntimeConfig),
        ):
            if not isinstance(getattr(self, name), kind):
                raise ValidationError(f"{name} must be a {kind.__name__}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=True)

    def with_overrides(self, overrides: Mapping[str, Any]) -> RUTAMConfig:
        return self.from_dict(_merge(self.to_dict(), overrides))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RUTAMConfig:
        if not isinstance(data, Mapping):
            raise ValidationError("Configuration must be a mapping")
        values = dict(data)
        try:
            for name, kind in (
                ("graph", GraphConfig),
                ("memory", MemoryConfig),
                ("runtime", RuntimeConfig),
            ):
                if name in values:
                    if not isinstance(values[name], Mapping):
                        raise ValidationError(f"{name} must be a mapping")
                    values[name] = kind(**values[name])
            return cls(**values)
        except TypeError as exc:
            raise ValidationError(f"Unknown or invalid configuration field: {exc}") from exc

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, overrides: Mapping[str, Any] | None = None
    ) -> RUTAMConfig:
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValidationError(f"Cannot load configuration {path}: {exc}") from exc
        config = cls.from_dict({} if data is None else data)
        return config if overrides is None else config.with_overrides(overrides)


def _merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, Mapping):
        raise ValidationError("Overrides must be a mapping")
    for key, value in overrides.items():
        if key not in base:
            raise ValidationError(f"Unknown configuration field: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, Mapping):
                raise ValidationError(f"{key} override must be a mapping")
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base
