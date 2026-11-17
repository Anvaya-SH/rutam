"""Safetensors neural model weight serialization and deserialization."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn

from rutam.exceptions import ValidationError


def save_model_weights(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Serialize model state dict to safetensors format on CPU without pickle."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    cpu_tensors: dict[str, torch.Tensor] = {}
    seen_ptrs: set[int] = set()

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.detach().cpu().contiguous()
        # Clone if tensor shares storage with another parameter
        if t.data_ptr() in seen_ptrs:
            t = t.clone()
        seen_ptrs.add(t.data_ptr())
        cpu_tensors[name] = t

    meta = metadata if metadata is not None else {}
    str_metadata = {str(k): str(v) for k, v in meta.items()}

    save_safetensors(cpu_tensors, str(target_path), metadata=str_metadata)
    return target_path


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load neural model weights from a safetensors file into an instantiated model."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Model weight file not found: {source_path}")

    try:
        tensors = load_safetensors(str(source_path), device="cpu")
    except Exception as exc:
        raise ValidationError(f"Failed to read safetensors model weights: {exc}") from exc

    incompatible = model.load_state_dict(tensors, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
