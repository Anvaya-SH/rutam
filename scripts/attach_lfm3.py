#!/usr/bin/env python3
"""CLI for validating and attaching external LFM3-1.7B-Base pretrained checkpoints without downloading during local dev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rutam.installer import run_attach_lfm3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely validate and attach external LFM3-1.7B-Base checkpoint without loading weights."
    )
    parser.add_argument(
        "checkpoint_path",
        help="Path to external directory containing LFM3 checkpoint (config.json + model.safetensors)",
    )
    parser.add_argument(
        "--target-dir",
        default="weights/lfm3",
        help="Target directory for attached checkpoint (default: weights/lfm3)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy checkpoint files instead of creating a symlink (default: create symlink)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint headers and report plan without modifying files",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  RUTAM LFM3-1.7B CHECKPOINT ATTACHMENT")
    print("=" * 65)
    print(f"Source Checkpoint: {args.checkpoint_path}")
    print(f"Target Directory:  {args.target_dir}")
    print(f"Mode:              {'COPY' if args.copy else 'SYMLINK'} (force={args.force})")
    print("-" * 65)

    try:
        res = run_attach_lfm3(
            checkpoint_path=args.checkpoint_path,
            target_dir=args.target_dir,
            symlink=not args.copy,
            dry_run=args.dry_run,
            force=args.force,
        )

        print("[SUCCESS] Checkpoint Header Validation Passed:")
        print(f"  * Weight format: {res['weight_format']}")
        cfg = res.get("config", {})
        print(f"  * Model Architecture: {cfg.get('model_type')}")
        print(f"  * Hidden Size: {cfg.get('hidden_size')}")
        print(f"  * Number of Layers: {cfg.get('num_hidden_layers')}")
        print(f"  * Vocab Size: {cfg.get('vocab_size')}")

        mem = res.get("memory_check", {})
        print(f"\nMemory Advisory:\n  * {mem.get('message')}")

        if res.get("dry_run"):
            print("\n[DRY-RUN] No files were attached.")
        else:
            print(f"\n[ATTACHED] Status: {res['status']} -> {res['target_path']}")
            print("To use attached backbone in RUTAM, point config.yaml `backbone` to the target path.")
    except Exception as exc:
        print(f"\n[ERROR] Attachment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
