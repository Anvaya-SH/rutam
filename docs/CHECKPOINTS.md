# RUTAM Native Model Checkpoint & Persistence System

The RUTAM checkpoint system provides safe, robust, and reproducible persistence of both neural parameters and persistent graph memory without relying on unsafe Python pickle.

## Architecture Diagram

```mermaid
flowchart TD
    subgraph RUTAM Checkpoint System
        Manager[RUTAMCheckpointManager] --> RunDir[run_xxxx/]
        RunDir --> CkptDir[checkpoint_000100/]
        
        CkptDir --> ModelWeights[model.safetensors\nTrainable Neural Weights]
        CkptDir --> GraphSnap[graph_snapshot/\nPersistent Graph Memory]
        GraphSnap --> GraphTensors[graph.safetensors]
        GraphSnap --> GraphMeta[graph.json]
        GraphSnap --> GraphManifest[graph.manifest.json]
        
        CkptDir --> OptTensors[optimizer.safetensors\nOptimizer Momentum Buffers]
        CkptDir --> OptMeta[optimizer_metadata.json\nParam Groups & Hyperparameters]
        CkptDir --> TrainState[trainer_state.json\nStep, Epoch, History]
        CkptDir --> ConfigJSON[rutam_config.json\nModel & Hardware Config]
        CkptDir --> MetricsJSON[metrics.json\nTrain & Validation Metrics]
        
        CkptDir --> Manifest[manifest.json\nCryptographic SHA-256 Hashes of All Files]
        RunDir --> LatestJSON[latest.json\nPointer to Latest Valid Checkpoint]
    end

    subgraph Verification & Recovery
        Verifier[Integrity Verifier] --> Manifest
        Verifier --> Hashes[Validate Real File Hashes vs Manifest]
        Recovery[Recovery Engine] --> Rollback[Automatic Rollback to Previous Valid Checkpoint]
        Recovery --> Cleanup[Clean Incomplete tmp_* Directories]
    end
```

## Checkpoint Philosophy

1. **Logical Separation of Assets**:
   - Neural parameters are model weights stored in `model.safetensors`.
   - Persistent graph memory is factual knowledge stored in `graph_snapshot/`.
   - Optimizer buffers are training runtime states stored in `optimizer.safetensors`.
2. **Zero-Pickle Security**:
   - All tensors are serialized using the safe, memory-mapped `safetensors` format.
   - All metadata, hyperparameters, and schemas are serialized using standard JSON.
3. **Atomic Operations**:
   - Saves write to an isolated temporary directory (`tmp_*`) before atomic rename.
   - Interrupted processes cannot produce partially valid checkpoints.
4. **Cryptographic Provenance**:
   - Every file within a checkpoint is hashed with SHA-256 and recorded in `manifest.json`.
