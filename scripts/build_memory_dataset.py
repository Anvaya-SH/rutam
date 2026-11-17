#!/usr/bin/env python3
"""CLI script to build the RUTAM memory training instruction dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rutam.data.memory_dataset import RUTAMMemoryDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RUTAM native memory training dataset.")
    parser.add_argument(
        "--count", type=int, default=1000, help="Total number of examples to generate."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/memory_training",
        help="Output directory for dataset splits.",
    )
    args = parser.parse_args()

    print(
        f"Building RUTAM memory dataset: {args.count} examples (seed={args.seed}) -> {args.output}"
    )
    builder = RUTAMMemoryDatasetBuilder(seed=args.seed)
    manifest = builder.build_dataset(args.output, total_count=args.count)

    print("Dataset generation complete!")
    print(f"Total examples: {manifest['total_count']}")
    print(f"Splits: {manifest['split_counts']}")
    print(f"Manifest written to: {Path(args.output) / 'manifest.json'}")


if __name__ == "__main__":
    main()
