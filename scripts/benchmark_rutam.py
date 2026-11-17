#!/usr/bin/env python3
"""Execute CPU latency, throughput, and scaling benchmarks for RUTAM architecture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rutam.diagnostics.architecture_audit import run_architecture_audit  # noqa: E402
from rutam.diagnostics.benchmark import run_all_benchmarks  # noqa: E402
from rutam.diagnostics.report import generate_audit_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RUTAM Architecture Benchmarking")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations per benchmark target",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Measurement iterations per benchmark target",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save benchmark reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Starting RUTAM Architecture CPU Benchmarks")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 70)

    bench_report = run_all_benchmarks(warmup=args.warmup, iterations=args.iterations)

    bench_json_path = args.output_dir / "benchmark_report.json"
    bench_json_path.write_text(
        json.dumps(bench_report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nBenchmark Summary:")
    print("-" * 70)
    print(f"Total benchmarks executed: {bench_report.summary.get('total_benchmarks')}")
    print(f"Total runtime:             {bench_report.summary.get('total_duration_seconds'):.2f}s")
    print(
        f"Mean forward latency:      {bench_report.summary.get('mean_forward_latency_ms'):.2f} ms"
    )
    print(
        f"Mean attention latency:    {bench_report.summary.get('mean_attention_latency_ms'):.2f} ms"
    )
    print(f"Mean training step time:   {bench_report.summary.get('mean_training_step_ms'):.2f} ms")
    print("-" * 70)

    # Update or generate comprehensive architecture_audit.md with benchmarks
    md_path = args.output_dir / "architecture_audit.md"
    audit_report = run_architecture_audit()
    generate_audit_markdown(audit_report, benchmark_report=bench_report, filepath=md_path)
    print(f"Updated Architecture Audit Markdown: {md_path}")
    print(f"Generated Benchmark JSON Report:     {bench_json_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
