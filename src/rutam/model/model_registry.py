"""Model registry managing RUTAM configurations, operational modes, and versioned architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.model_factory import RUTAMModelFactory
from rutam.model.rutam_model import ModelMode, RUTAMForCausalLM


@dataclass
class ModelRegistryEntry:
    """Represents a registered RUTAM model configuration with version and metadata."""

    name: str
    version: str
    config: RUTAMConfig
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Central registry for versioned RUTAM model configurations and architectures."""

    def __init__(self, populate_defaults: bool = True) -> None:
        self._entries: dict[tuple[str, str], ModelRegistryEntry] = {}
        if populate_defaults:
            self._register_default_architectures()

    def _register_default_architectures(self) -> None:
        """Register the four standard RUTAM v0.1.0 architecture configurations."""
        base = RUTAMConfig()

        defaults = [
            (
                "baseline",
                "0.1.0",
                base.with_mode("baseline"),
                "baseline",
                {"description": "Neural LM backbone without graph memory or recurrent loops"},
            ),
            (
                "rutam-v0.1-baseline",
                "0.1.0",
                base.with_mode("baseline"),
                "baseline",
                {"description": "Neural LM backbone without graph memory or recurrent loops"},
            ),
            (
                "graph_native",
                "0.1.0",
                base.with_mode("graph_native"),
                "graph_native",
                {"description": "Graph-native model with neural querying, traversal, and fusion"},
            ),
            (
                "rutam-v0.1-graph-native",
                "0.1.0",
                base.with_mode("graph_native"),
                "graph_native",
                {"description": "Graph-native model with neural querying, traversal, and fusion"},
            ),
            (
                "recurrent",
                "0.1.0",
                base.with_mode("recurrent"),
                "recurrent",
                {"description": "Recurrent architecture with parameter-shared reasoning and ACT"},
            ),
            (
                "rutam-v0.1-recurrent",
                "0.1.0",
                base.with_mode("recurrent"),
                "recurrent",
                {"description": "Recurrent architecture with parameter-shared reasoning and ACT"},
            ),
            (
                "full_rutam",
                "0.1.0",
                base.with_mode("full_rutam"),
                "full_rutam",
                {"description": "Unified RUTAM with graph memory fusion and recurrent reasoning"},
            ),
            (
                "rutam-v0.1-full",
                "0.1.0",
                base.with_mode("full_rutam"),
                "full_rutam",
                {"description": "Unified RUTAM with graph memory fusion and recurrent reasoning"},
            ),
        ]

        for name, version, cfg, _mode, meta in defaults:
            self.register(name, cfg, version=version, metadata=meta, overwrite=True)

    def register(
        self,
        name: str,
        config: RUTAMConfig,
        version: str = "0.1.0",
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a new configuration under a name and version.

        Args:
            name: Model configuration name identifier.
            config: Full RUTAMConfig instance.
            version: Semantic version string.
            metadata: Optional arbitrary dictionary of metadata.
            overwrite: Whether to allow overwriting an existing (name, version) entry.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Model name must be a non-empty string")
        if not isinstance(version, str) or not version.strip():
            raise ValidationError("Version must be a non-empty string")
        if not isinstance(config, RUTAMConfig):
            raise ValidationError(f"config must be a RUTAMConfig, got {type(config)}")

        key = (name.strip(), version.strip())
        if key in self._entries and not overwrite:
            raise ValidationError(
                f"Model '{name}' at version '{version}' is already registered. "
                "Set overwrite=True to replace."
            )

        # Infer mode from config
        mode = getattr(config.rutam_model, "mode", "full_rutam")
        valid_modes = ("baseline", "graph_native", "recurrent", "full_rutam")
        if not isinstance(mode, str) or mode not in valid_modes:
            graph_on = config.rutam_model.graph_enabled and config.rutam_model.graph_fusion_enabled
            recurrent_on = config.recurrent.enabled or config.rutam_model.recurrent_loops_enabled
            if graph_on and recurrent_on:
                mode = ModelMode.FULL_RUTAM.value
            elif graph_on:
                mode = ModelMode.GRAPH_NATIVE.value
            elif recurrent_on:
                mode = ModelMode.RECURRENT.value
            else:
                mode = ModelMode.BASELINE.value

        entry = ModelRegistryEntry(
            name=name.strip(),
            version=version.strip(),
            config=config,
            mode=mode,
            metadata=dict(metadata or {}),
        )
        self._entries[key] = entry

    def get(self, name: str, version: str | None = None) -> RUTAMConfig:
        """Retrieve a registered configuration.

        If version is None, selects the latest registered version.
        """
        entry = self._resolve_entry(name, version)
        return entry.config

    def get_metadata(self, name: str, version: str | None = None) -> dict[str, Any]:
        """Retrieve metadata for a registered model."""
        entry = self._resolve_entry(name, version)
        return {
            "name": entry.name,
            "version": entry.version,
            "mode": entry.mode,
            **entry.metadata,
        }

    def has_model(self, name: str, version: str | None = None) -> bool:
        """Check whether a model configuration exists."""
        clean_name = name.strip()
        if version is not None:
            return (clean_name, version.strip()) in self._entries
        return any(k[0] == clean_name for k in self._entries)

    def list_models(self) -> list[str]:
        """Return a sorted list of unique registered model names."""
        return sorted({k[0] for k in self._entries})

    def list_versions(self, name: str) -> list[str]:
        """Return all available versions for a given model name."""
        clean_name = name.strip()
        versions = [k[1] for k in self._entries if k[0] == clean_name]
        if not versions:
            raise ValidationError(f"Model '{name}' not found in registry")
        return sorted(versions)

    def list_entries(self) -> list[dict[str, Any]]:
        """Return a summary of all registered entries."""
        return [
            {
                "name": entry.name,
                "version": entry.version,
                "mode": entry.mode,
                "metadata": entry.metadata,
            }
            for entry in self._entries.values()
        ]

    def unregister(self, name: str, version: str | None = None) -> None:
        """Remove a registered entry."""
        clean_name = name.strip()
        if version is not None:
            key = (clean_name, version.strip())
            if key not in self._entries:
                raise ValidationError(f"Model '{name}' version '{version}' not found")
            del self._entries[key]
        else:
            matching = [k for k in self._entries if k[0] == clean_name]
            if not matching:
                raise ValidationError(f"Model '{name}' not found")
            for k in matching:
                del self._entries[k]

    def create_model(
        self,
        name: str,
        version: str | None = None,
        **kwargs: Any,
    ) -> RUTAMForCausalLM:
        """Instantiate a RUTAMForCausalLM model directly from a registered configuration."""
        config = self.get(name, version)
        return RUTAMModelFactory.create_model(config=config, **kwargs)

    def _resolve_entry(self, name: str, version: str | None = None) -> ModelRegistryEntry:
        clean_name = name.strip()
        if version is not None:
            key = (clean_name, version.strip())
            if key not in self._entries:
                raise ValidationError(
                    f"Model '{name}' at version '{version}' not found in registry"
                )
            return self._entries[key]

        matching = [entry for (n, _), entry in self._entries.items() if n == clean_name]
        if not matching:
            raise ValidationError(f"Model '{name}' not found in registry")
        # Return latest registered (last in list)
        return matching[-1]


# Module-level default singleton registry
default_registry = ModelRegistry()


def register_model(
    name: str,
    config: RUTAMConfig,
    version: str = "0.1.0",
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a configuration with the default global registry."""
    default_registry.register(name, config, version=version, metadata=metadata, overwrite=overwrite)


def get_model_config(name: str, version: str | None = None) -> RUTAMConfig:
    """Retrieve a configuration from the default global registry."""
    return default_registry.get(name, version=version)


def create_registered_model(
    name: str,
    version: str | None = None,
    **kwargs: Any,
) -> RUTAMForCausalLM:
    """Create a model instance from the default global registry."""
    return default_registry.create_model(name, version=version, **kwargs)


def list_registered_models() -> list[str]:
    """List all models in the default global registry."""
    return default_registry.list_models()
