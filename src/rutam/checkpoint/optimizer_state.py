"""Safe, pickle-free optimizer state serialization using safetensors and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch.optim import Optimizer

from rutam.exceptions import ValidationError


def save_optimizer_state(
    optimizer: Optimizer,
    tensor_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Serialize optimizer state dict without pickle.

    Tensors are stored in safetensors; parameter groups and scalar metadata in JSON.
    """
    t_path = Path(tensor_path)
    m_path = Path(metadata_path)
    t_path.parent.mkdir(parents=True, exist_ok=True)
    m_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = optimizer.state_dict()
    param_groups = state_dict.get("param_groups", [])
    raw_state = state_dict.get("state", {})

    state_tensors: dict[str, torch.Tensor] = {}
    state_scalars: dict[str, Any] = {}

    for param_id, p_state in raw_state.items():
        param_key = str(param_id)
        state_scalars[param_key] = {}
        for k, v in p_state.items():
            if isinstance(v, torch.Tensor):
                tensor_name = f"{param_key}__{k}"
                state_tensors[tensor_name] = v.detach().cpu().contiguous()
            else:
                state_scalars[param_key][k] = v

    # Write tensors via safetensors
    if state_tensors:
        save_safetensors(state_tensors, str(t_path))
    else:
        # Save a dummy scalar tensor so valid safetensors file exists
        save_safetensors({"__empty_state__": torch.zeros(1)}, str(t_path))

    # Write metadata JSON
    meta_content = {
        "param_groups": param_groups,
        "state_scalars": state_scalars,
    }
    m_path.write_text(json.dumps(meta_content, indent=2, sort_keys=True), encoding="utf-8")


def load_optimizer_state(
    optimizer: Optimizer,
    tensor_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Restore optimizer state from safetensors and JSON metadata."""
    t_path = Path(tensor_path)
    m_path = Path(metadata_path)

    if not m_path.is_file():
        raise FileNotFoundError(f"Optimizer metadata file not found: {m_path}")

    meta = json.loads(m_path.read_text(encoding="utf-8"))
    param_groups = meta.get("param_groups", [])
    state_scalars = meta.get("state_scalars", {})

    state_tensors: dict[str, torch.Tensor] = {}
    if t_path.is_file():
        try:
            state_tensors = load_safetensors(str(t_path), device="cpu")
        except Exception as exc:
            raise ValidationError(f"Failed to read optimizer safetensors: {exc}") from exc

    # Reconstruct nested state dict
    reconstructed_state: dict[int, dict[str, Any]] = {}

    # 1. Fill scalars
    for param_key, p_scalars in state_scalars.items():
        param_id = int(param_key)
        reconstructed_state[param_id] = dict(p_scalars)

    # 2. Fill tensors
    for name, tensor in state_tensors.items():
        if name == "__empty_state__":
            continue
        parts = name.split("__", 1)
        if len(parts) == 2:
            param_id = int(parts[0])
            field_name = parts[1]
            if param_id not in reconstructed_state:
                reconstructed_state[param_id] = {}
            reconstructed_state[param_id][field_name] = tensor

    restored_dict = {
        "param_groups": param_groups,
        "state": reconstructed_state,
    }

    try:
        optimizer.load_state_dict(restored_dict)
    except Exception as exc:
        raise ValidationError(f"Failed to load reconstructed optimizer state: {exc}") from exc
