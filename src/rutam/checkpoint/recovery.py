"""Recovery procedures for corrupted, missing, or interrupted checkpoints."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from rutam.checkpoint.integrity import verify_checkpoint_integrity

_STEP_PATTERN = re.compile(r"^checkpoint_(\d+)$")


def parse_checkpoint_step(dir_name: str) -> int | None:
    """Extract step number from checkpoint directory name."""
    m = _STEP_PATTERN.match(dir_name)
    if m:
        return int(m.group(1))
    return None


def find_latest_valid_checkpoint(run_dir: str | Path) -> Path | None:
    """Scan checkpoints in run_dir descending by step and return latest intact checkpoint."""
    base = Path(run_dir)
    if not base.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for item in base.iterdir():
        if item.is_dir():
            step = parse_checkpoint_step(item.name)
            if step is not None:
                candidates.append((step, item))

    # Sort descending by step
    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, c_path in candidates:
        is_valid, _ = verify_checkpoint_integrity(c_path)
        if is_valid:
            return c_path

    return None


def clean_incomplete_checkpoints(run_dir: str | Path) -> list[str]:
    """Identify and remove interrupted temporary checkpoint directories."""
    base = Path(run_dir)
    cleaned: list[str] = []
    if not base.is_dir():
        return cleaned

    for item in base.iterdir():
        if item.is_dir() and (item.name.startswith(".tmp_") or item.name.startswith("tmp_")):
            shutil.rmtree(item)
            cleaned.append(str(item))

    return cleaned
