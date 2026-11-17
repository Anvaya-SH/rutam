"""Platform and PyTorch CPU compatibility validation for RUTAM."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class CompatibilityReport:
    """Detailed report on platform, runtime, and CPU compatibility."""

    is_compatible: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "is_compatible": self.is_compatible,
            "checks": self.checks,
            "details": self.details,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Produce human-readable terminal summary."""
        lines = [
            f"RUTAM Platform Compatibility: {'PASS' if self.is_compatible else 'FAIL'}",
            f"  Python: {self.details.get('python_version', 'unknown')} (min: 3.11)",
            f"  OS / Arch: {self.details.get('os', 'unknown')} / {self.details.get('machine', 'unknown')}",
            f"  CPU Cores: {self.details.get('cpu_count', 'unknown')} logical",
            f"  PyTorch: {self.details.get('torch_version', 'unknown')} (CPU math: {'OK' if self.checks.get('torch_cpu', False) else 'FAIL'})",
            f"  Total RAM: {self.details.get('total_memory_gb', 0.0):.2f} GB",
            f"  Available RAM: {self.details.get('available_memory_gb', 0.0):.2f} GB",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - [WARN] {w}")
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    - [ERROR] {e}")
        return "\n".join(lines)


def get_system_info() -> dict[str, Any]:
    """Retrieve platform and operating system information."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_version_tuple": sys.version_info[:3],
        "cpu_count": os.cpu_count() or 1,
    }


def get_memory_info() -> dict[str, Any]:
    """Retrieve system physical and available memory in bytes.

    Uses /proc/meminfo on Linux, os.sysconf where available, or fallback.
    """
    total_bytes = 0
    available_bytes = 0

    # 1. Try reading /proc/meminfo (Linux)
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            meminfo: dict[str, int] = {}
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_parts = parts[1].strip().split()
                    if val_parts and val_parts[0].isdigit():
                        # values in /proc/meminfo are typically in kB
                        meminfo[key] = int(val_parts[0]) * 1024
            total_bytes = meminfo.get("MemTotal", 0)
            available_bytes = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        except (OSError, ValueError):
            pass

    # 2. Try os.sysconf if total_bytes not detected
    if total_bytes == 0 and hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_bytes = page_size * phys_pages
            if hasattr(os, "sysconf_names") and "SC_AVPHYS_PAGES" in os.sysconf_names:
                available_bytes = page_size * os.sysconf("SC_AVPHYS_PAGES")
            else:
                available_bytes = total_bytes
        except (ValueError, OSError):
            pass

    # 3. Default fallback if platform does not expose memory
    if total_bytes == 0:
        # Fallback approximation: 8 GB default assume
        total_bytes = 8 * (1024**3)
        available_bytes = 4 * (1024**3)

    return {
        "total_memory_bytes": total_bytes,
        "available_memory_bytes": available_bytes,
        "total_memory_gb": round(total_bytes / (1024**3), 2),
        "available_memory_gb": round(available_bytes / (1024**3), 2),
    }


def check_python_version(min_version: tuple[int, int] = (3, 11)) -> tuple[bool, str]:
    """Verify current Python interpreter version satisfies minimum required."""
    curr = sys.version_info[:2]
    if curr >= min_version:
        return True, f"Python {platform.python_version()} >= {min_version[0]}.{min_version[1]}"
    return False, f"Python {platform.python_version()} is below minimum required {min_version[0]}.{min_version[1]}"


def check_pytorch_cpu() -> tuple[bool, str, dict[str, Any]]:
    """Verify PyTorch installation, CPU backend, threads, and tensor arithmetic."""
    details: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }

    try:
        # Verify basic CPU tensor operations
        x = torch.ones((2, 2), dtype=torch.float32, device="cpu")
        y = torch.ones((2, 2), dtype=torch.float32, device="cpu")
        z = torch.matmul(x, y)
        if float(z.sum().item()) != 8.0:
            return False, "PyTorch CPU tensor multiplication test failed.", details

        # Verify bfloat16 support (vital for lightweight LFM3 inspection & forward)
        bf = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, device="cpu")
        bf_sum = float(bf.sum().item())
        details["bfloat16_supported"] = bool(abs(bf_sum - 3.0) < 1e-3)
        return True, f"PyTorch {torch.__version__} CPU verified successfully.", details
    except Exception as exc:
        details["error"] = str(exc)
        return False, f"PyTorch CPU verification failed: {exc}", details


def check_memory_headroom(
    required_bytes: int = 4 * 1024**3,
) -> tuple[bool, int, str]:
    """Check whether system has enough available memory for memory-intensive operations.

    Args:
        required_bytes: Number of bytes requested (e.g. ~7GB for float32 LFM3, ~3.5GB for bfloat16).

    Returns:
        tuple of (has_sufficient_headroom, available_bytes, message)
    """
    mem = get_memory_info()
    avail = mem["available_memory_bytes"]
    req_gb = required_bytes / (1024**3)
    avail_gb = avail / (1024**3)

    if avail >= required_bytes:
        return True, avail, f"Sufficient memory available ({avail_gb:.2f} GB >= {req_gb:.2f} GB required)."
    return (
        False,
        avail,
        f"Available memory ({avail_gb:.2f} GB) is below the recommended threshold ({req_gb:.2f} GB).",
    )


def check_platform_compatibility(
    min_memory_bytes: int = 1 * 1024**3,
) -> CompatibilityReport:
    """Run complete platform and runtime compatibility audit.

    Returns:
        Structured CompatibilityReport containing status of all checks.
    """
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []

    # 1. System info
    sys_info = get_system_info()
    details.update(sys_info)

    # 2. Python version check
    py_ok, py_msg = check_python_version((3, 11))
    checks["python_version"] = py_ok
    if not py_ok:
        errors.append(py_msg)

    # 3. Memory info and headroom check
    mem_info = get_memory_info()
    details.update(mem_info)
    mem_ok, avail_bytes, mem_msg = check_memory_headroom(min_memory_bytes)
    checks["minimum_memory"] = mem_ok
    if not mem_ok:
        warnings.append(mem_msg)

    # 4. PyTorch CPU check
    torch_ok, torch_msg, torch_details = check_pytorch_cpu()
    checks["torch_cpu"] = torch_ok
    details.update(torch_details)
    if not torch_ok:
        errors.append(torch_msg)

    # 5. Core dependencies availability check
    dep_checks: list[str] = ["safetensors", "numpy", "yaml"]
    for dep in dep_checks:
        try:
            __import__(dep)
            checks[f"dep_{dep}"] = True
        except ImportError:
            checks[f"dep_{dep}"] = False
            errors.append(f"Missing required dependency: {dep}")

    is_compatible = len(errors) == 0 and all(
        checks.get(k, False) for k in ["python_version", "torch_cpu", "dep_safetensors", "dep_numpy", "dep_yaml"]
    )

    return CompatibilityReport(
        is_compatible=is_compatible,
        checks=checks,
        details=details,
        warnings=warnings,
        errors=errors,
    )
