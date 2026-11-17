"""Primary checkpoint manager coordinating atomic saves, verification, and restoration."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer

from rutam.checkpoint.graph_snapshot import load_graph_snapshot, save_graph_snapshot
from rutam.checkpoint.integrity import create_file_record, verify_checkpoint_integrity
from rutam.checkpoint.manifest import CheckpointManifest
from rutam.checkpoint.model_weights import load_model_weights, save_model_weights
from rutam.checkpoint.optimizer_state import load_optimizer_state, save_optimizer_state
from rutam.checkpoint.recovery import find_latest_valid_checkpoint, parse_checkpoint_step
from rutam.config import RUTAMConfig
from rutam.exceptions import CorruptSnapshotError, ValidationError
from rutam.graph.operations import Graph


class RUTAMCheckpointManager:
    """Manages versioned, cryptographic checkpoints of neural weights and graph memory."""

    def __init__(
        self,
        base_dir: str | Path = "checkpoints",
        run_id: str = "run_default",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        graph: Graph | None = None,
        optimizer: Optimizer | None = None,
        step: int = 0,
        rutam_config: RUTAMConfig | None = None,
        metrics: dict[str, Any] | None = None,
        trainer_state: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically persist complete checkpoint with cryptographic manifest."""
        target_name = f"checkpoint_{step:06d}"
        final_dir = self.run_dir / target_name
        if final_dir.exists():
            raise ValidationError(f"Checkpoint already exists at {final_dir}")

        tmp_id = uuid.uuid4().hex[:8]
        tmp_dir = self.run_dir / f"tmp_{target_name}_{tmp_id}"
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            # 1. Save neural weights
            save_model_weights(
                model=model,
                path=tmp_dir / "model.safetensors",
                metadata={"step": str(step), "run_id": self.run_id},
            )

            # 2. Save graph snapshot
            if graph is not None:
                save_graph_snapshot(graph, tmp_dir / "graph_snapshot")

            # 3. Save optimizer state
            if optimizer is not None:
                save_optimizer_state(
                    optimizer=optimizer,
                    tensor_path=tmp_dir / "optimizer.safetensors",
                    metadata_path=tmp_dir / "optimizer_metadata.json",
                )

            # 4. Save metadata JSONs
            cfg_dict = rutam_config.to_dict() if rutam_config is not None else {}
            (tmp_dir / "rutam_config.json").write_text(
                json.dumps(cfg_dict, indent=2, sort_keys=True), encoding="utf-8"
            )

            m_dict = metrics if metrics is not None else {}
            (tmp_dir / "metrics.json").write_text(
                json.dumps(m_dict, indent=2, sort_keys=True), encoding="utf-8"
            )

            t_state = trainer_state if trainer_state is not None else {}
            t_state["step"] = step
            (tmp_dir / "trainer_state.json").write_text(
                json.dumps(t_state, indent=2, sort_keys=True), encoding="utf-8"
            )

            # 5. Build cryptographic manifest
            file_records = {}
            for file_path in tmp_dir.rglob("*"):
                if file_path.is_file():
                    rel_p = str(file_path.relative_to(tmp_dir))
                    file_records[rel_p] = create_file_record(file_path)

            manifest = CheckpointManifest(
                checkpoint_id=target_name,
                run_id=self.run_id,
                step=step,
                backbone_model_id=(
                    rutam_config.backbone
                    if rutam_config is not None
                    else "LiquidAI/LFM3-1.7B-Base"
                ),
                architecture_metadata={
                    "total_params": sum(p.numel() for p in model.parameters()),
                    "trainable_params": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                },
                files=file_records,
            )
            manifest.save(tmp_dir / "manifest.json")

            # 6. Atomic directory commit
            tmp_dir.rename(final_dir)

            # 7. Update latest.json pointer
            latest_info = {
                "latest_checkpoint": target_name,
                "step": step,
                "run_id": self.run_id,
            }
            (self.run_dir / "latest.json").write_text(
                json.dumps(latest_info, indent=2, sort_keys=True), encoding="utf-8"
            )

            return final_dir

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path,
        model: nn.Module | None = None,
        graph: Graph | None = None,
        optimizer: Optimizer | None = None,
        rutam_config: RUTAMConfig | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Validate integrity and load checkpoint components into target objects."""
        dir_path = Path(checkpoint_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {dir_path}")

        is_valid, errors = verify_checkpoint_integrity(dir_path)
        if not is_valid:
            raise CorruptSnapshotError(
                f"Checkpoint integrity verification failed for {dir_path}: {errors}"
            )

        manifest = CheckpointManifest.load(dir_path / "manifest.json")

        result: dict[str, Any] = {
            "manifest": manifest,
            "step": manifest.step,
        }

        # 1. Load model weights
        if model is not None and (dir_path / "model.safetensors").is_file():
            missing, unexpected = load_model_weights(
                model=model,
                path=dir_path / "model.safetensors",
                strict=strict,
            )
            result["missing_keys"] = missing
            result["unexpected_keys"] = unexpected

        # 2. Load graph snapshot
        if (dir_path / "graph_snapshot").is_dir():
            effective_cfg = rutam_config if rutam_config is not None else RUTAMConfig()
            loaded_graph = load_graph_snapshot(
                config=effective_cfg,
                source_dir=dir_path / "graph_snapshot",
            )
            result["graph"] = loaded_graph

        # 3. Load optimizer state
        if optimizer is not None and (dir_path / "optimizer_metadata.json").is_file():
            load_optimizer_state(
                optimizer=optimizer,
                tensor_path=dir_path / "optimizer.safetensors",
                metadata_path=dir_path / "optimizer_metadata.json",
            )

        # 4. Load metadata
        if (dir_path / "metrics.json").is_file():
            result["metrics"] = json.loads((dir_path / "metrics.json").read_text(encoding="utf-8"))
        if (dir_path / "trainer_state.json").is_file():
            result["trainer_state"] = json.loads(
                (dir_path / "trainer_state.json").read_text(encoding="utf-8")
            )
        if (dir_path / "rutam_config.json").is_file():
            result["rutam_config"] = json.loads(
                (dir_path / "rutam_config.json").read_text(encoding="utf-8")
            )

        return result

    def verify_checkpoint(self, checkpoint_dir: str | Path) -> bool:
        """Verify checksums and completeness of a checkpoint."""
        is_valid, _ = verify_checkpoint_integrity(checkpoint_dir)
        return is_valid

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all versioned checkpoints in the run directory ordered by step."""
        checkpoints: list[dict[str, Any]] = []
        for item in self.run_dir.iterdir():
            if item.is_dir():
                step = parse_checkpoint_step(item.name)
                if step is not None:
                    manifest_path = item / "manifest.json"
                    created_at = None
                    if manifest_path.is_file():
                        try:
                            m = CheckpointManifest.load(manifest_path)
                            created_at = m.created_at
                        except Exception:
                            pass
                    is_valid, _ = verify_checkpoint_integrity(item)
                    checkpoints.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "step": step,
                            "created_at": created_at,
                            "is_valid": is_valid,
                        }
                    )
        checkpoints.sort(key=lambda x: x["step"])
        return checkpoints

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to the latest intact checkpoint, rolling back if latest is corrupted."""
        latest_file = self.run_dir / "latest.json"
        if latest_file.is_file():
            try:
                info = json.loads(latest_file.read_text(encoding="utf-8"))
                candidate = self.run_dir / info["latest_checkpoint"]
                if candidate.is_dir() and self.verify_checkpoint(candidate):
                    return candidate
            except Exception:
                pass

        # Fallback to scanning run directory
        return find_latest_valid_checkpoint(self.run_dir)

    def restore_checkpoint(
        self,
        target_step: int | None = None,
        model: nn.Module | None = None,
        graph: Graph | None = None,
        optimizer: Optimizer | None = None,
        rutam_config: RUTAMConfig | None = None,
    ) -> dict[str, Any]:
        """Restore model, graph, and optimizer from latest or specific step checkpoint."""
        if target_step is None:
            latest = self.get_latest_checkpoint()
            if latest is None:
                raise FileNotFoundError(f"No valid checkpoint found in {self.run_dir}")
            target_dir = latest
        else:
            target_dir = self.run_dir / f"checkpoint_{target_step:06d}"

        return self.load_checkpoint(
            checkpoint_dir=target_dir,
            model=model,
            graph=graph,
            optimizer=optimizer,
            rutam_config=rutam_config,
        )

    def delete_checkpoint(self, step: int, confirmed: bool = False) -> bool:
        """Delete specific checkpoint folder. Requires explicit confirmed=True."""
        if not confirmed:
            raise ValidationError(
                "Deletion requires confirmed=True to prevent accidental data loss."
            )

        target = self.run_dir / f"checkpoint_{step:06d}"
        if not target.is_dir():
            return False

        shutil.rmtree(target)
        return True

    def export_architecture(
        self, export_path: str | Path, rutam_config: RUTAMConfig | None = None
    ) -> Path:
        """Export architecture schema, dimensions, and versioning info to JSON."""
        target = Path(export_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        cfg = rutam_config if rutam_config is not None else RUTAMConfig()

        arch_spec = {
            "foundation_backbone": cfg.backbone,
            "project_name": cfg.project_name,
            "architecture_version": "1.0.0",
            "components": {
                "encoder": cfg.encoder.to_dict() if hasattr(cfg.encoder, "to_dict") else {},
                "graph_attention": cfg.graph_attention.to_dict()
                if hasattr(cfg.graph_attention, "to_dict")
                else {},
                "selector": cfg.selector.to_dict() if hasattr(cfg.selector, "to_dict") else {},
                "query_head": cfg.query_head.to_dict()
                if hasattr(cfg.query_head, "to_dict")
                else {},
                "graph_fusion": cfg.graph_fusion.to_dict()
                if hasattr(cfg.graph_fusion, "to_dict")
                else {},
                "memory_write": cfg.memory_write.to_dict()
                if hasattr(cfg.memory_write, "to_dict")
                else {},
            },
        }
        target.write_text(json.dumps(arch_spec, indent=2, sort_keys=True), encoding="utf-8")
        return target
