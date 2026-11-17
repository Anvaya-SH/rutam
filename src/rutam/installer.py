"""Installer, environment validation, diagnostics, and CLI runners for RUTAM."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from rutam.compatibility import (
    CompatibilityReport,
    check_memory_headroom,
    check_platform_compatibility,
)
from rutam.config import RUTAMConfig
from rutam.exceptions import StorageError, ValidationError


def validate_environment(min_memory_bytes: int = 1 * 1024**3) -> CompatibilityReport:
    """Validate runtime environment, platform, CPU capabilities, and memory headroom.

    Args:
        min_memory_bytes: Recommended minimum available RAM in bytes (default: 1 GB).

    Returns:
        Structured CompatibilityReport with status, details, warnings, and errors.
    """
    return check_platform_compatibility(min_memory_bytes=min_memory_bytes)


def check_dependencies() -> dict[str, dict[str, Any]]:
    """Audit installed Python dependencies against RUTAM requirements.

    Returns:
        Dictionary mapping package name to installed status and detected version.
    """
    required_packages = {
        "torch": ">=2.5.0",
        "safetensors": ">=0.4.5",
        "numpy": ">=1.26.0",
        "yaml": ">=6.0.2",
    }
    optional_packages = {
        "pytest": ">=8.3.0",
        "ruff": ">=0.9.0",
    }

    results: dict[str, dict[str, Any]] = {}

    for pkg, req in {**required_packages, **optional_packages}.items():
        is_required = pkg in required_packages
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            results[pkg] = {
                "installed": True,
                "version": version,
                "required": is_required,
                "spec": req,
                "status": "OK",
            }
        except ImportError:
            results[pkg] = {
                "installed": False,
                "version": None,
                "required": is_required,
                "spec": req,
                "status": "MISSING" if is_required else "OPTIONAL_MISSING",
            }

    return results


def check_graph_storage(graph_path: str | Path) -> dict[str, Any]:
    """Audit health and accessibility of persistent graph storage directory.

    Args:
        graph_path: Path to graph storage directory.

    Returns:
        Dictionary with existence, accessibility, snapshot status, and any errors.
    """
    p = Path(graph_path).resolve()
    status: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "writable": os.access(p, os.W_OK) if p.exists() else os.access(p.parent, os.W_OK),
        "has_snapshot": (p / "graph.manifest.json").exists(),
        "is_healthy": False,
        "errors": [],
    }

    if status["has_snapshot"]:
        try:
            from rutam.graph.storage import GraphStorage

            storage = GraphStorage(path=p)
            storage.load()
            status["is_healthy"] = True
        except (StorageError, ValidationError, Exception) as exc:
            status["errors"].append(f"Storage load failed: {exc}")
    elif status["writable"]:
        status["is_healthy"] = True
    else:
        status["errors"].append("Directory not writable.")

    return status


def run_synthetic_forward_smoke_test(
    config: RUTAMConfig | None = None,
) -> dict[str, Any]:
    """Execute a single synthetic forward pass on CPU to verify end-to-end tensor pipeline.

    Args:
        config: Optional RUTAMConfig instance. Defaults to default configuration.

    Returns:
        Dictionary containing smoke test results, output shape, and latency.
    """
    import time

    from rutam.model.model_factory import RUTAMModelFactory

    cfg = RUTAMConfig() if config is None else config
    start_time = time.perf_counter()

    try:
        model = RUTAMModelFactory.create_model(config=cfg)
        model.eval()

        # Synthetic input sequence
        batch_size = 1
        seq_len = 8
        vocab_size = cfg.query_head.backbone_hidden_dim  # SyntheticBackbone vocab
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logits = output.logits
        has_nans = bool(torch.isnan(logits).any().item())
        has_infs = bool(torch.isinf(logits).any().item())

        success = not has_nans and not has_infs and logits.shape[0] == batch_size

        return {
            "success": success,
            "logits_shape": list(logits.shape),
            "has_nans": has_nans,
            "has_infs": has_infs,
            "latency_ms": round(elapsed_ms, 2),
            "error": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "logits_shape": None,
            "has_nans": None,
            "has_infs": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": str(exc),
        }


def run_prepare(
    bundle_dir: str | Path = "rutam_bundle",
    config_path: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare RUTAM architecture bundle idempotently.

    Args:
        bundle_dir: Output bundle directory path.
        config_path: Optional path to YAML configuration file.
        overwrite: If True, overwrite existing weights and graph snapshots.
        dry_run: If True, simulate actions without writing to disk.

    Returns:
        Dictionary summary of preparation actions.
    """
    from rutam.bundle import RUTAMBundle

    p = Path(bundle_dir).resolve()
    cfg = (
        RUTAMConfig.from_yaml(config_path)
        if config_path is not None
        else (RUTAMConfig.from_yaml(p / "config.yaml") if (p / "config.yaml").exists() and not overwrite else RUTAMConfig())
    )

    if dry_run:
        weights_exist = (p / "weights" / "model.safetensors").exists()
        graph_exist = (p / "graph_snapshot" / "graph.manifest.json").exists()
        return {
            "dry_run": True,
            "bundle_dir": str(p),
            "will_overwrite": overwrite,
            "weights_exist": weights_exist,
            "graph_exist": graph_exist,
            "status": "SIMULATED",
        }

    bundle = RUTAMBundle.create(
        target_dir=p,
        config=cfg,
        overwrite=overwrite,
    )

    return {
        "bundle_dir": str(bundle.bundle_dir),
        "status": "SUCCESS",
        "actions": bundle.verification_report.get("idempotent_actions", []),
        "weights_path": str(bundle.weights_path),
        "graph_dir": str(bundle.graph_dir),
        "files_count": len(bundle.verification_report.get("files", {})),
    }


def run_verification(
    bundle_dir: str | Path = "rutam_bundle",
    synthetic_forward: bool = True,
) -> dict[str, Any]:
    """Run diagnostics verifying environment, dependencies, bundle integrity, and CPU forward execution.

    Args:
        bundle_dir: Path to rutam_bundle directory.
        synthetic_forward: Whether to execute synthetic forward pass smoke test.

    Returns:
        Structured diagnostics report dictionary.
    """
    from rutam.bundle import RUTAMBundle

    p = Path(bundle_dir).resolve()
    report: dict[str, Any] = {
        "environment": validate_environment().to_dict(),
        "dependencies": check_dependencies(),
        "bundle": {"exists": p.exists(), "is_valid": False, "errors": []},
        "smoke_test": None,
        "all_passed": False,
    }

    # Bundle validation if present
    if p.exists():
        try:
            bundle = RUTAMBundle.load(bundle_dir=p)
            bundle_status = bundle.verify()
            report["bundle"]["is_valid"] = bundle_status["is_valid"]
            report["bundle"]["errors"] = bundle_status["errors"]
            report["bundle"]["verified_files"] = bundle_status.get("verified_files", [])
        except Exception as exc:
            report["bundle"]["errors"].append(str(exc))
    else:
        report["bundle"]["errors"].append(f"Bundle directory does not exist: {p}")

    # Forward smoke test
    if synthetic_forward:
        report["smoke_test"] = run_synthetic_forward_smoke_test()

    env_ok = report["environment"]["is_compatible"]
    deps_ok = all(
        d["installed"] for d in report["dependencies"].values() if d.get("required")
    )
    smoke_ok = report["smoke_test"]["success"] if report["smoke_test"] else True
    bundle_ok = report["bundle"]["is_valid"] or not p.exists()

    report["all_passed"] = env_ok and deps_ok and smoke_ok and (not p.exists() or bundle_ok)
    return report


def run_attach_lfm3(
    checkpoint_path: str | Path,
    target_dir: str | Path = "weights/lfm3",
    symlink: bool = True,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Validate external LFM3 checkpoint safely and attach it without full tensor instantiation.

    Safely inspects safetensors headers using zero-copy metadata reading without allocating
    1.7B parameters in system memory.

    Args:
        checkpoint_path: Source directory containing LFM3 checkpoint.
        target_dir: Target attachment destination directory.
        symlink: If True (default), create symlink; if False, copy files.
        dry_run: If True, validate headers and report plan without modifying files.
        force: If True, overwrite target directory if already populated.

    Returns:
        Structured attachment report with metadata, header validation, and attachment status.
    """
    from safetensors import safe_open

    src = Path(checkpoint_path).resolve()
    # Keep the final path component unresolved: an existing attachment may be a
    # symlink to the checkpoint, and --force must unlink it, never delete its target.
    dst = Path(os.path.abspath(target_dir))
    resolved_dst = dst.resolve()
    resolved_parent = dst.parent.resolve()
    if resolved_parent == src or src in resolved_parent.parents:
        raise ValidationError("Attachment target must not be inside the checkpoint source")
    if resolved_dst == src and not dst.is_symlink():
        raise ValidationError("Attachment target must not be the checkpoint source")
    if not dst.is_symlink() and (src in resolved_dst.parents or resolved_dst in src.parents):
        raise ValidationError("Checkpoint source and attachment target must not contain each other")

    if not src.exists() or not src.is_dir():
        raise ValidationError(f"Checkpoint directory does not exist or is not a directory: {src}")

    # 1. Inspect config.json
    config_file = src / "config.json"
    if not config_file.exists():
        raise ValidationError(f"Missing config.json in checkpoint directory: {src}")

    try:
        cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Invalid config.json in checkpoint: {exc}") from exc

    # 2. Inspect safetensors files safely via header inspection
    safetensors_files = list(src.glob("*.safetensors"))
    if not safetensors_files:
        # Check for PyTorch bin fallback
        bin_files = list(src.glob("*.bin"))
        if not bin_files:
            raise ValidationError(
                f"No weight files (.safetensors or .bin) found in checkpoint directory: {src}"
            )
        weight_format = "pytorch_bin"
        header_info: dict[str, Any] = {"file_count": len(bin_files)}
    else:
        weight_format = "safetensors"
        header_info = {"file_count": len(safetensors_files), "tensor_shapes": {}}
        # Inspect headers in zero-copy mode without allocating tensor memory
        for st_file in safetensors_files:
            try:
                with safe_open(str(st_file), framework="pt", device="cpu") as f:
                    keys = f.keys()
                    # Inspect a sample of tensor shapes for validation
                    sample_keys = list(keys)[:10]
                    for k in sample_keys:
                        header_info["tensor_shapes"][k] = list(f.get_slice(k).get_shape())
            except Exception as exc:
                raise ValidationError(
                    f"Corrupt or unreadable safetensors header in {st_file.name}: {exc}"
                ) from exc

    # 3. Check system memory headroom
    # LFM3-1.7B float32 requires ~7GB, bfloat16 requires ~3.5GB
    has_mem, avail_bytes, mem_msg = check_memory_headroom(required_bytes=7 * 1024**3)

    attachment_manifest = {
        "source_path": str(src),
        "target_path": str(dst),
        "weight_format": weight_format,
        "config": {
            "model_type": cfg_data.get("model_type", "unknown"),
            "hidden_size": cfg_data.get("hidden_size", cfg_data.get("d_model", "unknown")),
            "num_hidden_layers": cfg_data.get(
                "num_hidden_layers", cfg_data.get("n_layer", "unknown")
            ),
            "vocab_size": cfg_data.get("vocab_size", "unknown"),
        },
        "header_validation": header_info,
        "memory_check": {
            "sufficient_for_float32": has_mem,
            "available_gb": round(avail_bytes / (1024**3), 2),
            "message": mem_msg,
        },
    }

    if dry_run:
        attachment_manifest["status"] = "SIMULATED"
        return attachment_manifest

    # 4. Attach to target directory
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            raise ValidationError(
                f"Attachment target {dst} already exists. Pass force=True to overwrite."
            )
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    if symlink:
        try:
            dst.symlink_to(src, target_is_directory=True)
            attachment_manifest["link_type"] = "symlink"
        except OSError:
            # Fallback to copy if symlink creation is not permitted
            shutil.copytree(src, dst)
            attachment_manifest["link_type"] = "copied_fallback"
    else:
        shutil.copytree(src, dst)
        attachment_manifest["link_type"] = "copied"

    # Use an exclusively created sibling, never a predictable path that could
    # already be a symlink or an unrelated file.  This also avoids writing into
    # either the external source or a copied checkpoint directory.
    fd, manifest_name = tempfile.mkstemp(
        prefix=f"{dst.name}.attachment_manifest.", suffix=".json", dir=dst.parent
    )
    attachment_manifest["manifest_path"] = manifest_name
    attachment_manifest["status"] = "ATTACHED"
    with os.fdopen(fd, "w", encoding="utf-8") as manifest_file:
        json.dump(attachment_manifest, manifest_file, indent=2)

    return attachment_manifest


# ---------------------------------------------------------------------------
# CLI Entrypoints
# ---------------------------------------------------------------------------


def main_prepare() -> None:
    """CLI entrypoint for rutam-prepare."""
    parser = argparse.ArgumentParser(
        description="Prepare RUTAM architecture bundle idempotently."
    )
    parser.add_argument(
        "--output-dir",
        default="rutam_bundle",
        help="Target directory for rutam_bundle (default: rutam_bundle)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to custom YAML configuration file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing trained weights and graph snapshot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate preparation without writing files",
    )
    args = parser.parse_args()

    print(f"[RUTAM] Preparing bundle at '{args.output_dir}' (overwrite={args.force})...")
    res = run_prepare(
        bundle_dir=args.output_dir,
        config_path=args.config,
        overwrite=args.force,
        dry_run=args.dry_run,
    )
    print(f"[RUTAM] Bundle preparation complete: {res['status']}")
    for action in res.get("actions", []):
        print(f"  - {action}")


def main_verify() -> None:
    """CLI entrypoint for rutam-verify."""
    parser = argparse.ArgumentParser(
        description="Verify RUTAM installation, environment, and bundle integrity."
    )
    parser.add_argument(
        "--bundle-dir",
        default="rutam_bundle",
        help="Path to rutam_bundle directory (default: rutam_bundle)",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip synthetic forward pass smoke test",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics in raw JSON format",
    )
    args = parser.parse_args()

    res = run_verification(
        bundle_dir=args.bundle_dir,
        synthetic_forward=not args.skip_forward,
    )

    if args.json:
        print(json.dumps(res, indent=2))
        sys.exit(0 if res["all_passed"] else 1)

    print("=" * 60)
    print("RUTAM INSTALLATION & ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"Environment Compatible: {'[PASS]' if res['environment']['is_compatible'] else '[FAIL]'}")
    for k, v in res["dependencies"].items():
        status_tag = "[PASS]" if v["installed"] else ("[FAIL]" if v["required"] else "[OPTIONAL]")
        ver_str = f"v{v['version']}" if v["version"] else "not installed"
        print(f"  - {k} ({v['spec']}): {ver_str} {status_tag}")

    bundle_info = res["bundle"]
    if bundle_info["exists"]:
        bundle_tag = "[PASS]" if bundle_info["is_valid"] else "[FAIL]"
        print(f"Bundle Integrity: {bundle_tag}")
        for err in bundle_info.get("errors", []):
            print(f"    [ERROR] {err}")
    else:
        print(f"Bundle Directory: not present at '{args.bundle_dir}' (Run `python scripts/prepare_rutam.py` to create)")

    if res.get("smoke_test"):
        st = res["smoke_test"]
        st_tag = "[PASS]" if st["success"] else "[FAIL]"
        print(f"Synthetic CPU Forward Pass: {st_tag} (latency: {st['latency_ms']} ms)")
        if st.get("error"):
            print(f"    [ERROR] {st['error']}")

    print("=" * 60)
    overall = "[ALL CHECKS PASSED]" if res["all_passed"] else "[DIAGNOSTICS FAILED]"
    print(f"OVERALL STATUS: {overall}")
    print("=" * 60)

    sys.exit(0 if res["all_passed"] else 1)


def main_attach_lfm3() -> None:
    """CLI entrypoint for rutam-attach-lfm3."""
    parser = argparse.ArgumentParser(
        description="Safely validate and attach external LFM3-1.7B-Base checkpoint without loading weights."
    )
    parser.add_argument(
        "checkpoint_path",
        help="Path to external directory containing LFM3 checkpoint",
    )
    parser.add_argument(
        "--target-dir",
        default="weights/lfm3",
        help="Target directory for attached checkpoint (default: weights/lfm3)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy checkpoint files instead of creating a symlink",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint headers without attaching",
    )
    args = parser.parse_args()

    print(f"[RUTAM] Inspecting external checkpoint at '{args.checkpoint_path}'...")
    try:
        res = run_attach_lfm3(
            checkpoint_path=args.checkpoint_path,
            target_dir=args.target_dir,
            symlink=not args.copy,
            dry_run=args.dry_run,
            force=args.force,
        )
        print(f"[RUTAM] Checkpoint validation: SUCCESS (format: {res['weight_format']})")
        print(f"  Model Type: {res['config'].get('model_type')}")
        print(f"  Hidden Size: {res['config'].get('hidden_size')}")
        print(f"  Layers: {res['config'].get('num_hidden_layers')}")
        print(f"  Memory Advisory: {res['memory_check']['message']}")
        print(f"[RUTAM] Attachment Status: {res['status']} -> {res['target_path']}")
    except Exception as exc:
        print(f"[RUTAM] Attachment Error: {exc}", file=sys.stderr)
        sys.exit(1)
