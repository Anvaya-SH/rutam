"""RUTAMBundle generator and manager for self-contained package architecture bundles."""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rutam.checkpoint.graph_snapshot import load_graph_snapshot, save_graph_snapshot
from rutam.checkpoint.model_weights import load_model_weights, save_model_weights
from rutam.compatibility import check_platform_compatibility
from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage
from rutam.model.architecture_manifest import ArchitectureManifest, generate_architecture_manifest
from rutam.model.model_factory import RUTAMModelFactory
from rutam.model.rutam_model import RUTAMForCausalLM


def _compute_sha256(filepath: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RUTAMBundle:
    """Self-contained RUTAM architecture bundle."""

    bundle_dir: Path
    config: RUTAMConfig
    manifest_data: dict[str, Any]
    weights_path: Path
    graph_dir: Path
    verification_report: dict[str, Any]

    @classmethod
    def create(
        cls,
        target_dir: str | Path = "rutam_bundle",
        config: RUTAMConfig | None = None,
        overwrite: bool = False,
        model: RUTAMForCausalLM | None = None,
    ) -> RUTAMBundle:
        """Create or update a self-contained RUTAM architecture bundle idempotently.

        Args:
            target_dir: Target directory path for bundle.
            config: Optional RUTAMConfig instance. Defaults to default RUTAMConfig().
            overwrite: If True, forces re-initialization of weights and graph snapshots.
                       If False (default), preserves existing trained weights and graph snapshots.
            model: Optional pre-instantiated RUTAMForCausalLM.

        Returns:
            Instantiated and verified RUTAMBundle.
        """
        bundle_path = Path(target_dir).resolve()
        weights_dir = bundle_path / "weights"
        graph_dir = bundle_path / "graph_snapshot"
        weights_path = weights_dir / "model.safetensors"

        bundle_path.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)

        idempotent_actions: list[str] = []

        # 1. Resolve configuration
        config_file = bundle_path / "config.yaml"
        if config_file.exists() and config is None and not overwrite:
            active_config = RUTAMConfig.from_yaml(config_file)
            idempotent_actions.append(f"Loaded existing configuration from {config_file.name}")
        else:
            active_config = RUTAMConfig() if config is None else config
            # Cleanly dump YAML without Python tags
            cfg_dict = json.loads(json.dumps(active_config.to_dict()))
            config_file.write_text(
                "# RUTAM Architecture Bundle Configuration\n"
                + yaml.safe_dump(cfg_dict, sort_keys=False),
                encoding="utf-8",
            )
            idempotent_actions.append(f"Wrote configuration to {config_file.name}")

        # 2. Manage Model Weights (Idempotent)
        active_model = model
        if weights_path.exists() and not overwrite:
            idempotent_actions.append(
                f"Retained existing model weights at {weights_path.relative_to(bundle_path)}"
            )
            if active_model is None:
                active_model = RUTAMModelFactory.create_model(config=active_config)
                load_model_weights(active_model, weights_path, strict=False)
        else:
            if active_model is None:
                active_model = RUTAMModelFactory.create_model(config=active_config)
            save_model_weights(
                active_model,
                weights_path,
                metadata={"bundle_version": "0.1.0", "author": "Anvaya.sh"},
            )
            idempotent_actions.append(
                f"Generated and persisted model weights to {weights_path.relative_to(bundle_path)}"
            )

        # 3. Manage Persistent Graph Snapshot (Idempotent)
        manifest_path = graph_dir / "graph.manifest.json"
        if manifest_path.exists() and not overwrite:
            idempotent_actions.append(
                f"Retained existing graph snapshot in {graph_dir.relative_to(bundle_path)}"
            )
        else:
            # Create fresh initial graph snapshot
            init_graph = Graph(active_config)
            save_graph_snapshot(init_graph, graph_dir)
            idempotent_actions.append(
                f"Initialized new graph memory snapshot in {graph_dir.relative_to(bundle_path)}"
            )

        # 4. Generate Architecture Manifest
        manifest_file = bundle_path / "architecture_manifest.json"
        arch_manifest: ArchitectureManifest = generate_architecture_manifest(
            active_model, active_config
        )
        manifest_dict = arch_manifest.to_dict()
        manifest_file.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
        idempotent_actions.append(f"Generated architecture manifest at {manifest_file.name}")

        # 5. Build Verification Report
        compat_report = check_platform_compatibility()
        files_catalog: dict[str, dict[str, Any]] = {}
        for f in bundle_path.glob("**/*"):
            if f.is_file() and f.name != "verification_report.json":
                rel_path = str(f.relative_to(bundle_path))
                files_catalog[rel_path] = {
                    "sha256": _compute_sha256(f),
                    "size_bytes": f.stat().st_size,
                }

        verification_report = {
            "bundle_title": "RUTAM Architecture Bundle",
            "version": "0.1.0",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "VERIFIED" if compat_report.is_compatible else "COMPATIBILITY_WARNING",
            "idempotent_actions": idempotent_actions,
            "compatibility": compat_report.to_dict(),
            "files": files_catalog,
            "parameter_summary": arch_manifest.to_dict().get("parameter_counts", {}),
        }

        report_file = bundle_path / "verification_report.json"
        report_file.write_text(json.dumps(verification_report, indent=2), encoding="utf-8")

        return cls(
            bundle_dir=bundle_path,
            config=active_config,
            manifest_data=manifest_dict,
            weights_path=weights_path,
            graph_dir=graph_dir,
            verification_report=verification_report,
        )

    @classmethod
    def load(cls, bundle_dir: str | Path = "rutam_bundle") -> RUTAMBundle:
        """Load and parse an existing RUTAM architecture bundle.

        Args:
            bundle_dir: Path to directory containing rutam_bundle.

        Returns:
            RUTAMBundle instance.
        """
        bundle_path = Path(bundle_dir).resolve()
        if not bundle_path.exists():
            raise ValidationError(f"Bundle directory does not exist: {bundle_path}")

        config_file = bundle_path / "config.yaml"
        if not config_file.exists():
            raise ValidationError(f"Missing config.yaml in bundle: {bundle_path}")
        config = RUTAMConfig.from_yaml(config_file)

        manifest_file = bundle_path / "architecture_manifest.json"
        if not manifest_file.exists():
            raise ValidationError(f"Missing architecture_manifest.json in bundle: {bundle_path}")
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

        weights_path = bundle_path / "weights" / "model.safetensors"
        if not weights_path.exists():
            raise ValidationError(f"Missing model.safetensors in bundle: {weights_path}")

        graph_dir = bundle_path / "graph_snapshot"
        if not (graph_dir / "graph.manifest.json").exists():
            raise ValidationError(f"Missing graph snapshot manifest in bundle: {graph_dir}")

        report_file = bundle_path / "verification_report.json"
        report_data = (
            json.loads(report_file.read_text(encoding="utf-8"))
            if report_file.exists()
            else {}
        )

        return cls(
            bundle_dir=bundle_path,
            config=config,
            manifest_data=manifest_data,
            weights_path=weights_path,
            graph_dir=graph_dir,
            verification_report=report_data,
        )

    def verify(self) -> dict[str, Any]:
        """Verify checksums and structural integrity of this bundle.

        Returns:
            Dictionary with validation status, matched files, and any errors.
        """
        errors: list[str] = []
        verified_files: list[str] = []
        if not isinstance(self.verification_report, dict):
            return {
                "is_valid": False,
                "verified_files": [],
                "errors": ["Invalid verification report"],
            }
        recorded_files = self.verification_report.get("files", {})

        required_files = {
            "config.yaml", "architecture_manifest.json", "weights/model.safetensors",
            "graph_snapshot/graph.manifest.json",
        }
        if not isinstance(recorded_files, dict):
            return {"is_valid": False, "verified_files": [], "errors": ["Invalid file catalog"]}
        for missing in sorted(required_files - recorded_files.keys()):
            errors.append(f"Missing checksum record: {missing}")

        for rel_path, meta in recorded_files.items():
            if not isinstance(rel_path, str) or not isinstance(meta, dict):
                errors.append(f"Invalid checksum entry: {rel_path!r}")
                continue
            expected_sha = meta.get("sha256")
            if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                    or any(char not in "0123456789abcdef" for char in expected_sha)):
                errors.append(f"Invalid checksum entry: {rel_path}")
                continue
            target = self.bundle_dir / rel_path
            if not target.resolve().is_relative_to(self.bundle_dir.resolve()):
                errors.append(f"Checksum path escapes bundle: {rel_path}")
                continue
            if not target.is_file():
                errors.append(f"Missing file: {rel_path}")
                continue
            try:
                curr_sha = _compute_sha256(target)
            except OSError as exc:
                errors.append(f"Cannot read file {rel_path}: {exc}")
                continue
            if curr_sha != expected_sha:
                errors.append(
                    f"Checksum mismatch for {rel_path}: expected {expected_sha}, got {curr_sha}"
                )
            else:
                verified_files.append(rel_path)

        # Verify graph storage is readable
        try:
            storage = GraphStorage(path=self.graph_dir)
            storage.load()
        except Exception as exc:
            errors.append(f"Graph snapshot verification failed: {exc}")

        return {
            "is_valid": len(errors) == 0,
            "verified_files": verified_files,
            "errors": errors,
        }

    def load_model(self, strict: bool = True) -> RUTAMForCausalLM:
        """Instantiate RUTAMForCausalLM model with bundle configuration and weights."""
        model = RUTAMModelFactory.create_model(config=self.config)
        load_model_weights(model, self.weights_path, strict=strict)
        return model

    def load_graph(self) -> Graph:
        """Load persistent Graph memory snapshot from bundle."""
        return load_graph_snapshot(self.config, self.graph_dir)
