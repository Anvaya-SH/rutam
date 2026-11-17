#!/usr/bin/env python3
"""CLI utility to execute local CPU neural training on memory datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rutam.config import RUTAMConfig
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.training.train_config import TrainingConfig
from rutam.training.trainer import RUTAMTrainer


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
    parser = argparse.ArgumentParser(description="Train RUTAM memory modules on local CPU.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/memory_training",
        help="Directory containing train.jsonl and validation.jsonl",
    )
    parser.add_argument("--steps", type=int, default=30, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--mode", type=str, default="joint_memory", help="Training mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"

    if not train_path.exists():
        print(f"Error: train dataset not found at {train_path}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("RUTAM CPU Neural Training Engine")
    print(f"Data directory: {data_dir}")
    print(f"Training mode: {args.mode}")
    print(f"Max steps: {args.steps}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    model = create_cpu_model(hidden_dim=16)

    train_config = TrainingConfig(
        mode=args.mode,
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        eval_interval_steps=10,
        seed=args.seed,
    )

    trainer = RUTAMTrainer(
        model=model,
        config=train_config,
        train_dataset=train_path,
        eval_dataset=val_path if val_path.exists() else None,
    )

    summary = trainer.train()

    print("\n--- Training Results ---")
    print(f"Steps executed: {summary.total_steps}")
    print(f"Initial loss: {summary.initial_loss:.4f}")
    print(f"Final loss: {summary.final_loss:.4f}")
    print(f"Runtime: {summary.runtime_seconds:.2f}s")
    print(f"Trainable parameters: {summary.trainable_parameters_count:,}")
    print(f"Frozen parameters: {summary.frozen_parameters_count:,}")

    if summary.final_eval_result is not None:
        print(f"Validation loss: {summary.final_eval_result.mean_loss:.4f}")
        for k, v in summary.final_eval_result.metrics.items():
            print(f"  {k}: {v:.4f}")

    if summary.warnings:
        print(f"Warnings: {summary.warnings}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
