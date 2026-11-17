"""Schema version migration boundaries and compatibility enforcement."""

from __future__ import annotations

from rutam.checkpoint.manifest import CheckpointManifest
from rutam.checkpoint.versioning import CHECKPOINT_SCHEMA_VERSION, parse_semver
from rutam.exceptions import ValidationError


def check_migration_needed(manifest: CheckpointManifest) -> bool:
    """Determine whether a loaded checkpoint requires schema migration."""
    cur_major, cur_minor, _ = parse_semver(CHECKPOINT_SCHEMA_VERSION)
    src_major, src_minor, _ = parse_semver(manifest.schema_version)

    if src_major != cur_major:
        raise ValidationError(
            f"Breaking schema version change: current {CHECKPOINT_SCHEMA_VERSION} vs "
            f"checkpoint {manifest.schema_version}. Manual migration required."
        )

    # Minor version differences within same major can be loaded directly
    return src_minor < cur_minor


def apply_checkpoint_migration(manifest: CheckpointManifest) -> CheckpointManifest:
    """Migrate a valid checkpoint manifest to the latest schema version."""
    if not check_migration_needed(manifest):
        return manifest

    # For v1.0.0 no transformations are currently needed
    return manifest
