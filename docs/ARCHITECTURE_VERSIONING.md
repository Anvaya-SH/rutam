# Architecture Versioning & Migration Guidelines

RUTAM maintains strict separation between its neural model architecture, graph storage schemas, and checkpoint envelope versions.

## Version Hierarchy

- **Checkpoint Envelope Version (`CHECKPOINT_SCHEMA_VERSION = "1.0.0"`)**:
  - Covers the structure of `manifest.json`, file naming conventions, and cryptographic validation.
  - Minor version increments (e.g. `1.1.0`) guarantee backwards compatibility.
  - Major version increments indicate breaking directory structure or encryption changes.

- **Model Architecture Version (`MODEL_ARCHITECTURE_VERSION = "1.0.0"`)**:
  - Tracks neural layer names, projection dimensions, and parameter shapes.
  - Stored in checkpoint metadata.

- **Persistent Graph Schema Version (`SCHEMA_VERSION = 1`)**:
  - Defined in `src/rutam/graph/storage.py`.
  - Covers `graph.safetensors`, `graph.json`, and graph-level integrity.

## Future Foundation Model Integration: `LiquidAI/LFM3-1.7B-Base`

When transitioning from the lightweight synthetic CPU backbone to the genuine `LiquidAI/LFM3-1.7B-Base` pretrained foundation:

1. `backbone_model_id` in `manifest.json` will identify `"LiquidAI/LFM3-1.7B-Base"`.
2. Backbone weights remain external or can be loaded from HuggingFace cache/Safetensors.
3. Checkpoint manager saves and restores all adapter layers, query heads, graph encoders, selectors, traversal modules, and memory write heads without needing to duplicate 1.7B parameters in each training checkpoint step.
