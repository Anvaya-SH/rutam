"""Complete architecture audit coordinating contract, gradient, lifecycle, and recovery."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import safetensors
import torch

from rutam.checkpoint.integrity import verify_checkpoint_integrity
from rutam.checkpoint.manager import RUTAMCheckpointManager
from rutam.config import RUTAMConfig
from rutam.diagnostics.gradient_checks import FullGradientFlowReport, run_all_gradient_checks
from rutam.diagnostics.integration import (
    TensorContractReport,
    create_verification_model,
    verify_forward_contracts,
)
from rutam.diagnostics.memory_checks import (
    MemoryLifecycleReport,
    run_memory_lifecycle_audit,
)


@dataclass
class EnvironmentInfo:
    """System, runtime, and hardware execution environment metadata."""

    platform: str
    os_name: str
    python_version: str
    torch_version: str
    torch_threads: int
    safetensors_version: str
    cpu_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert environment info to dictionary."""
        return asdict(self)


@dataclass
class ParameterAuditItem:
    """Parameter count and memory footprint for a submodule."""

    module_name: str
    total_params: int
    trainable_params: int
    frozen_params: int
    footprint_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Convert item to dictionary."""
        return asdict(self)


@dataclass
class ParameterAuditReport:
    """Aggregated parameter audit across model submodules."""

    total_params: int
    total_trainable: int
    total_frozen: int
    total_footprint_mb: float
    modules: list[ParameterAuditItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert parameter report to dictionary."""
        return {
            "total_params": self.total_params,
            "total_trainable": self.total_trainable,
            "total_frozen": self.total_frozen,
            "total_footprint_mb": self.total_footprint_mb,
            "modules": [m.to_dict() for m in self.modules],
        }


@dataclass
class CheckpointAuditReport:
    """Audit report for checkpoint serialization, integrity verification, and tamper detection."""

    passed: bool
    save_verified: bool
    manifest_verified: bool
    tamper_detection_verified: bool
    rollback_verified: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint report to dictionary."""
        return asdict(self)


@dataclass
class ArchitectureAuditReport:
    """Complete consolidated architecture audit report for RUTAM."""

    audit_timestamp: str
    passed: bool
    duration_seconds: float
    environment: EnvironmentInfo
    parameter_audit: ParameterAuditReport
    contract_audit: TensorContractReport
    gradient_audit: FullGradientFlowReport
    lifecycle_audit: MemoryLifecycleReport
    checkpoint_audit: CheckpointAuditReport
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert complete audit report to dictionary."""
        return {
            "audit_timestamp": self.audit_timestamp,
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
            "environment": self.environment.to_dict(),
            "parameter_audit": self.parameter_audit.to_dict(),
            "contract_audit": self.contract_audit.to_dict(),
            "gradient_audit": self.gradient_audit.to_dict(),
            "lifecycle_audit": self.lifecycle_audit.to_dict(),
            "checkpoint_audit": self.checkpoint_audit.to_dict(),
            "summary": self.summary,
        }


def collect_environment_info() -> EnvironmentInfo:
    """Query runtime, Python, PyTorch, and CPU details."""
    return EnvironmentInfo(
        platform=platform.platform(),
        os_name=platform.system(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        safetensors_version=safetensors.__version__,
        cpu_count=os.cpu_count(),
    )


def audit_parameters(model: Any) -> ParameterAuditReport:
    """Inspect and calculate parameter counts and memory usage across submodules."""
    module_dict = {
        "backbone": model.backbone,
        "encoder": model.encoder,
        "selector": model.selector,
        "query_head": model.query_head,
        "traversal": model.traversal,
        "fusion": model.fusion,
        "write_head": model.write_head,
    }

    items: list[ParameterAuditItem] = []
    tot_params = 0
    tot_trainable = 0
    tot_frozen = 0
    tot_footprint = 0.0

    for name, submodule in module_dict.items():
        if submodule is None:
            continue
        params = list(submodule.parameters())
        count = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        frozen = count - trainable
        footprint = sum(p.numel() * p.element_size() for p in params) / (1024 * 1024)

        tot_params += count
        tot_trainable += trainable
        tot_frozen += frozen
        tot_footprint += footprint

        items.append(
            ParameterAuditItem(
                module_name=name,
                total_params=count,
                trainable_params=trainable,
                frozen_params=frozen,
                footprint_mb=round(footprint, 3),
            )
        )

    return ParameterAuditReport(
        total_params=tot_params,
        total_trainable=tot_trainable,
        total_frozen=tot_frozen,
        total_footprint_mb=round(tot_footprint, 3),
        modules=items,
    )


def audit_checkpoints_and_recovery() -> CheckpointAuditReport:
    """Verify end-to-end checkpoint save, manifest SHA-256 verification, and corruption rollback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = RUTAMCheckpointManager(base_dir=tmp_dir, run_id="audit_run")
        model, graph = create_verification_model(num_nodes=4, hidden_dim=16)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)

        # 1. Save checkpoint 10
        c10_path = manager.save_checkpoint(
            model=model,
            graph=graph,
            optimizer=opt,
            step=10,
            rutam_config=model.config,
        )
        save_ok = c10_path.is_dir() and (c10_path / "model.safetensors").is_file()

        # 2. Manifest verification
        is_intact, errors = verify_checkpoint_integrity(c10_path)
        manifest_ok = is_intact and len(errors) == 0

        # 3. Save checkpoint 20
        c20_path = manager.save_checkpoint(
            model=model,
            graph=graph,
            optimizer=opt,
            step=20,
            rutam_config=model.config,
        )

        # 4. Inject corruption into checkpoint 20
        (c20_path / "model.safetensors").write_bytes(b"tampered_bytes_signature")
        c20_valid, c20_errors = verify_checkpoint_integrity(c20_path)
        tamper_ok = not c20_valid and len(c20_errors) > 0

        # 5. Rollback to latest intact checkpoint
        latest_valid = manager.get_latest_checkpoint()
        rollback_ok = latest_valid is not None and latest_valid.name == "checkpoint_000010"

        passed = save_ok and manifest_ok and tamper_ok and rollback_ok
        return CheckpointAuditReport(
            passed=passed,
            save_verified=save_ok,
            manifest_verified=manifest_ok,
            tamper_detection_verified=tamper_ok,
            rollback_verified=rollback_ok,
        )


def run_architecture_audit(config: RUTAMConfig | None = None) -> ArchitectureAuditReport:
    """Execute complete architecture audit across contracts, gradients, lifecycle, and recovery."""
    start_time = time.perf_counter()
    timestamp = datetime.now(UTC).isoformat()

    env_info = collect_environment_info()

    # 1. Parameter audit
    model, _ = create_verification_model(num_nodes=4, hidden_dim=16)
    param_audit = audit_parameters(model)

    # 2. Tensor contract audit across scenarios
    b, t = 2, 8
    input_ids = torch.randint(0, 100, (b, t), dtype=torch.long)
    labels = torch.randint(0, 100, (b, t), dtype=torch.long)
    contract_report = verify_forward_contracts(
        model=model,
        input_ids=input_ids,
        labels=labels,
        generate_proposals=True,
    )

    # 3. Gradient flow audit across all 8 submodules & e2e
    gradient_report = run_all_gradient_checks()

    # 4. 14-step memory lifecycle audit
    lifecycle_report = run_memory_lifecycle_audit()

    # 5. Checkpoint & recovery audit
    checkpoint_report = audit_checkpoints_and_recovery()

    duration = time.perf_counter() - start_time
    overall_passed = (
        contract_report.passed
        and gradient_report.passed
        and lifecycle_report.passed
        and checkpoint_report.passed
    )

    summary = {
        "overall_passed": overall_passed,
        "contract_checks_passed": contract_report.passed,
        "total_contract_checks": contract_report.total_checks,
        "gradient_checks_passed": gradient_report.passed,
        "gradient_submodules_passed": gradient_report.summary.get("passed_components", 0),
        "lifecycle_checks_passed": lifecycle_report.passed,
        "lifecycle_steps_passed": lifecycle_report.passed_steps,
        "checkpoint_recovery_passed": checkpoint_report.passed,
        "duration_seconds": round(duration, 3),
    }

    return ArchitectureAuditReport(
        audit_timestamp=timestamp,
        passed=overall_passed,
        duration_seconds=round(duration, 3),
        environment=env_info,
        parameter_audit=param_audit,
        contract_audit=contract_report,
        gradient_audit=gradient_report,
        lifecycle_audit=lifecycle_report,
        checkpoint_audit=checkpoint_report,
        summary=summary,
    )
