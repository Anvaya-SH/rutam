#!/usr/bin/env python3
"""CLI utility to evaluate trained RUTAM models on held-out memory test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rutam.config import RUTAMConfig
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.training.dataloader import MemoryDataLoader
from rutam.training.evaluator import RUTAMEvaluator


def create_cpu_model(hidden_dim: int = 16) -> RUTAMForCausalLM:
    """Create lightweight CPU-compatible RUTAM model with synthetic backbone."""
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": hidden_dim, "num_relation_types": 4},
            "selector": {
                "query_dim": 8,
                "hidden_dim": hidden_dim,
                "score_dim": 8,
                "top_k": 2,
            },
            "query_head": {
                "backbone_hidden_dim": hidden_dim,
                "graph_hidden_dim": hidden_dim,
                "query_dim": 8,
                "intermediate_dim": hidden_dim,
            },
            "traversal": {
                "hidden_dim": hidden_dim,
                "path_hidden_dim": hidden_dim,
                "query_dim": 8,
                "num_relation_types": 4,
                "max_visited_nodes": 4,
                "max_candidate_paths": 8,
                "max_selected_paths": 2,
            },
            "graph_fusion": {
                "hidden_dim": hidden_dim,
                "graph_dim": hidden_dim,
                "num_attention_heads": 2,
            },
            "memory_write": {
                "backbone_hidden_dim": hidden_dim,
                "hidden_dim": hidden_dim,
                "projection_dim": hidden_dim,
            },
            "rutam_model": {
                "return_memory_diagnostics": True,
                "backbone_frozen": True,
            },
        }
    )
    return RUTAMForCausalLM(config=config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RUTAM model on memory dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/memory_training",
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--simulate", action="store_true", help="Enable memory simulation")
    args = parser.parse_args()

    split_file = Path(args.data_dir) / f"{args.split}.jsonl"
    if not split_file.exists():
        print(f"Error: Split file not found at {split_file}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"Evaluating RUTAM Model on {split_file}")
    print("=" * 60)

    model = create_cpu_model(hidden_dim=16)
    loader = MemoryDataLoader(dataset=split_file, batch_size=args.batch_size, shuffle=False)
    evaluator = RUTAMEvaluator(simulate_memory=args.simulate)

    res = evaluator.evaluate(model, loader)

    print(f"\nEvaluated Examples: {res.num_examples}")
    print(f"Mean Loss: {res.mean_loss:.4f}")
    print("\nLoss Components:")
    for k, v in res.loss_components.items():
        print(f"  {k}: {v:.4f}")

    print("\nMetrics:")
    for k, v in res.metrics.items():
        print(f"  {k}: {v:.4f}")

    if res.simulation_metrics:
        print("\nSimulation Metrics:")
        for k, v in res.simulation_metrics.items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
