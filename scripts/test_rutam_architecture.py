#!/usr/bin/env python3
"""Run complete architecture integration audit, generating JSON and Markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rutam.diagnostics.architecture_audit import run_architecture_audit  # noqa: E402
from rutam.diagnostics.report import generate_audit_json, generate_audit_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RUTAM Architecture Integration Audit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save audit reports",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="architecture_audit.json",
        help="JSON report file name",
    )
    parser.add_argument(
        "--md-name",
        type=str,
        default="architecture_audit.md",
        help="Markdown report file name",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / args.json_name
    md_path = args.output_dir / args.md_name

    print("=" * 70)
    print("Starting RUTAM Architecture Integration Audit")
    print("=" * 70)

    report = run_architecture_audit()

    # Generate reports
    generate_audit_json(report, json_path)
    generate_audit_markdown(report, filepath=md_path)

    print(f"\nAudit completed in {report.duration_seconds:.2f}s")
    print(f"Overall Status: {'PASSED' if report.passed else 'FAILED'}")
    print("-" * 70)
    contract_status = "PASS" if report.contract_audit.passed else "FAIL"
    print(
        f"1. Tensor Contracts:     {contract_status} "
        f"({report.contract_audit.passed_checks}/{report.contract_audit.total_checks} checks)"
    )
    grad_status = "PASS" if report.gradient_audit.passed else "FAIL"
    passed_comp = report.gradient_audit.summary.get("passed_components", 0)
    print(f"2. Gradient Flow:        {grad_status} ({passed_comp}/8 submodules)")
    bb_status = "PASS" if report.gradient_audit.frozen_backbone_verified else "FAIL"
    print(f"   - Backbone Isolation: {bb_status}")
    life_status = "PASS" if report.lifecycle_audit.passed else "FAIL"
    passed_steps = report.lifecycle_audit.passed_steps
    print(f"3. Memory Lifecycle:     {life_status} ({passed_steps}/14 steps)")
    ckpt_status = "PASS" if report.checkpoint_audit.passed else "FAIL"
    print(f"4. Checkpoint Recovery:  {ckpt_status}")
    print("-" * 70)
    print(f"Generated JSON Report:     {json_path}")
    print(f"Generated Markdown Report: {md_path}")
    print("=" * 70)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
