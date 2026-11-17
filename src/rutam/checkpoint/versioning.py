"""Versioning definitions and compatibility checks for checkpoints."""

from __future__ import annotations

CHECKPOINT_SCHEMA_VERSION = "1.0.0"
MODEL_ARCHITECTURE_VERSION = "1.0.0"
SUPPORTED_MAJOR_VERSIONS = {1}


def parse_semver(version_str: str) -> tuple[int, int, int]:
    """Parse semver string into (major, minor, patch) integer tuple."""
    parts = version_str.strip().split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver string: '{version_str}'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def is_compatible_version(version_str: str) -> bool:
    """Check if checkpoint version shares supported major version."""
    try:
        major, _, _ = parse_semver(version_str)
        return major in SUPPORTED_MAJOR_VERSIONS
    except (ValueError, TypeError):
        return False
