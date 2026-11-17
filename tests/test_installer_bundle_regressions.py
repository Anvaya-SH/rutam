"""Regressions for attachment safety and bundle verification."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from rutam.bundle import RUTAMBundle
from rutam.exceptions import ValidationError
from rutam.installer import run_attach_lfm3


def test_force_reattach_never_deletes_or_modifies_source(tmp_path):
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "lfm3"}))
    save_file({"weight": torch.ones(1)}, str(source / "model.safetensors"))
    original_files = {p.name: p.read_bytes() for p in source.iterdir()}
    target = tmp_path / "attached"
    sidecar = tmp_path / "attached.attachment_manifest.json"
    sidecar.symlink_to(source / "config.json")
    first = run_attach_lfm3(source, target)
    assert target.is_symlink()
    assert sidecar.is_symlink()
    assert Path(first["manifest_path"]).is_file()
    second = run_attach_lfm3(source, target, force=True)
    assert second["manifest_path"] != first["manifest_path"]
    assert sidecar.is_symlink()
    assert target.is_symlink()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files
    other_target = tmp_path / "other"
    unrelated_sidecar = tmp_path / "other.attachment_manifest.json"
    unrelated_sidecar.write_text("leave me alone")
    run_attach_lfm3(source, other_target)
    assert unrelated_sidecar.read_text() == "leave me alone"
    with pytest.raises(ValidationError):
        run_attach_lfm3(source, source, force=True)
    with pytest.raises(ValidationError):
        run_attach_lfm3(source, source / "nested", force=True)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_files


def test_missing_bundle_catalog_fails_verification(tmp_path, monkeypatch):
    class Storage:
        def __init__(self, path):
            pass

        def load(self):
            return None

    monkeypatch.setattr("rutam.bundle.GraphStorage", Storage)
    bundle = RUTAMBundle(tmp_path, None, {}, tmp_path / "weights", tmp_path / "graph", {})
    report = bundle.verify()
    assert not report["is_valid"]
    assert any("Missing checksum record" in error for error in report["errors"])


def test_real_bundle_cli_and_corruption_detection(tmp_path):
    bundle_path = tmp_path / "bundle"
    binary_dir = Path(sys.executable).parent
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(
        [str(binary_dir / "rutam-prepare"), "--output-dir", str(bundle_path)],
        check=True, capture_output=True, text=True, env=env,
    )
    result = subprocess.run(
        [str(binary_dir / "rutam-verify"), "--bundle-dir", str(bundle_path), "--json"],
        check=True, capture_output=True, text=True, env=env,
    )
    assert json.loads(result.stdout)["all_passed"]
    assert RUTAMBundle.load(bundle_path).verify()["is_valid"]

    weights = bundle_path / "weights" / "model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original + b"tampered")
    result = RUTAMBundle.load(bundle_path).verify()
    assert not result["is_valid"]
    assert any(
        "Checksum mismatch for weights/model.safetensors" in error for error in result["errors"]
    )
    weights.write_bytes(original)

    report_path = bundle_path / "verification_report.json"
    report_data = json.loads(report_path.read_text())
    report_data["files"]["weights/model.safetensors"] = None
    report_path.write_text(json.dumps(report_data))
    result = RUTAMBundle.load(bundle_path).verify()
    assert not result["is_valid"]
    assert any("Invalid checksum entry" in error for error in result["errors"])

    report_path.unlink()
    result = RUTAMBundle.load(bundle_path).verify()
    assert not result["is_valid"]
    assert any("Missing checksum record" in error for error in result["errors"])
