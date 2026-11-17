#!/usr/bin/env python3
"""Diagnostics checking environment, imports, graph storage, manifests, and synthetic forward smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rutam.installer import run_verification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify RUTAM environment, dependencies, bundle integrity, and CPU forward execution."
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

    print("=" * 65)
    print("  RUTAM INSTALLATION & ENVIRONMENT DIAGNOSTICS")
    print("=" * 65)

    env = res["environment"]
    env_str = "[PASS]" if env["is_compatible"] else "[FAIL]"
    print(f"Platform Compatibility: {env_str}")
    details = env.get("details", {})
    print(f"  * Python: {details.get('python_version')} ({details.get('os')} {details.get('machine')})")
    print(f"  * CPU Cores: {details.get('cpu_count')} logical cores")
    print(f"  * Memory: {details.get('available_memory_gb', 0)} GB available / {details.get('total_memory_gb', 0)} GB total")

    print("\nCore Dependencies:")
    for pkg, info in res["dependencies"].items():
        tag = "[PASS]" if info["installed"] else ("[FAIL]" if info["required"] else "[OPTIONAL]")
        ver = f"v{info['version']}" if info["version"] else "not installed"
        print(f"  * {pkg} ({info['spec']}): {ver:<14} {tag}")

    print("\nArchitecture Bundle Integrity:")
    b_info = res["bundle"]
    if b_info["exists"]:
        b_tag = "[PASS]" if b_info["is_valid"] else "[FAIL]"
        print(f"  * Bundle directory: {args.bundle_dir} {b_tag}")
        for vf in b_info.get("verified_files", []):
            print(f"    - Verified: {vf}")
        for err in b_info.get("errors", []):
            print(f"    - [ERROR] {err}")
    else:
        print(f"  * Bundle directory '{args.bundle_dir}' not found (run `python scripts/prepare_rutam.py` to create)")

    if res.get("smoke_test"):
        st = res["smoke_test"]
        st_tag = "[PASS]" if st["success"] else "[FAIL]"
        print(f"\nSynthetic CPU Forward Pass: {st_tag}")
        print(f"  * Output shape: {st.get('logits_shape')}")
        print(f"  * Execution latency: {st.get('latency_ms')} ms")
        if st.get("error"):
            print(f"  * [ERROR] {st['error']}")

    print("=" * 65)
    status_msg = "ALL CHECKS PASSED - RUTAM IS READY FOR USE" if res["all_passed"] else "DIAGNOSTICS ENCOUNTERED ISSUES"
    print(f"RESULT: [{status_msg}]")
    print("=" * 65)

    sys.exit(0 if res["all_passed"] else 1)


if __name__ == "__main__":
    main()
