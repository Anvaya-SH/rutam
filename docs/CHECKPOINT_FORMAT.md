# RUTAM Checkpoint Format Specification (v1.0.0)

This document specifies the exact layout, file schemas, and tensor key conventions for RUTAM checkpoints.

## Directory Layout

```
checkpoints/
  <run_id>/
    checkpoint_000100/
      model.safetensors             # Differentiable neural weights
      graph_snapshot/
        graph.safetensors           # Persistent node & edge embeddings
        graph.json                  # Node contents, types, provenance metadata
        graph.manifest.json         # Graph-level SHA-256 manifest
      optimizer.safetensors         # First & second moment tensors
      optimizer_metadata.json       # Optimizer param groups and hyperparameters
      trainer_state.json            # Step, epoch, and training history
      rutam_config.json             # Full model & subsystem configuration
      metrics.json                  # Training and evaluation metrics
      manifest.json                 # Checkpoint-level SHA-256 manifest
    latest.json                     # Symlink/pointer to latest valid step
```

## Manifest Schema (`manifest.json`)

```json
{
  "schema_version": "1.0.0",
  "checkpoint_id": "checkpoint_000100",
  "run_id": "run_001",
  "step": 100,
  "created_at": "2026-11-14T04:45:00.000000Z",
  "backbone_model_id": "LiquidAI/LFM3-1.7B-Base",
  "architecture_metadata": {
    "total_params": 51192,
    "trainable_params": 13528
  },
  "files": {
    "model.safetensors": {
      "sha256": "3a7b...",
      "size_bytes": 204768,
      "content_type": "application/octet-stream"
    },
    "graph_snapshot/graph.safetensors": {
      "sha256": "f9c1...",
      "size_bytes": 1024,
      "content_type": "application/octet-stream"
    }
  }
}
```
