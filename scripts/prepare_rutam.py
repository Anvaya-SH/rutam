#!/usr/bin/env python3
"""One-command local architecture preparation script generating rutam_bundle/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rutam.installer import run_prepare


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command local architecture preparation script for RUTAM."
    )
    parser.add_argument(
        "--output-dir",
        default="rutam_bundle",
        help="Target directory for rutam_bundle (default: rutam_bundle)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to custom YAML configuration file (default: configs/rutam_default.yaml if present)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing graph snapshot and trained model weights",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate bundle preparation without modifying the file system",
    )
    args = parser.parse_args()

    # Default to configs/rutam_default.yaml if available and not explicitly provided
    config_path = args.config
    if config_path is None:
        default_cfg = _REPO_ROOT / "configs" / "rutam_default.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)

    print("=" * 65)
    print("  RUTAM ARCHITECTURE PREPARATION")
    print(f"  Target: {args.output_dir} | Mode: {'FORCE OVERWRITE' if args.force else 'IDEMPOTENT'}")
    print("=" * 65)

    try:
        res = run_prepare(
            bundle_dir=args.output_dir,
            config_path=config_path,
            overwrite=args.force,
            dry_run=args.dry_run,
        )

        if res.get("dry_run"):
            print("[DRY-RUN] Preparation plan:")
            print(f"  Target: {res['bundle_dir']}")
            print(f"  Will overwrite: {res['will_overwrite']}")
            print(f"  Weights exist: {res['weights_exist']}")
            print(f"  Graph exists: {res['graph_exist']}")
            return

        print(f"[SUCCESS] RUTAM architecture bundle ready at: {res['bundle_dir']}")
        print("Bundle components:")
        for action in res.get("actions", []):
            print(f"  * {action}")
        print(f"Total bundle files: {res.get('files_count', 0)}")
        print("\nNext step: Run `python scripts/verify_installation.py` to verify diagnostics.")
    except Exception as exc:
        print(f"\n[ERROR] Preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
