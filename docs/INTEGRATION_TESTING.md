# RUTAM Integration Testing Suite

## Overview

The RUTAM integration testing suite validates end-to-end interactions across the neural language backbone, graph memory, transactional storage, neural attention mechanisms, and optimization loops. It guarantees that individual modules (query head, graph encoder, graph attention, selector, path traversal, graph fusion, memory write head, and link predictor) collaborate seamlessly without regressions, numeric instability, or cache desynchronization.

All tests in `tests/integration/` execute against **real production components** with synthetic CPU configurations—no mocks or placeholders are permitted.

---

## Test Suite Architecture

```
tests/integration/
├── test_full_forward.py          # Unified causal LM forward passes & tensor contracts
├── test_memory_lifecycle.py      # Complete 14-step transactional memory lifecycle
├── test_gradient_flow.py         # Backward passes & gradient flow across all 8 submodules
├── test_graph_consistency.py     # Graph immutability, snapshot parity, & cache sync
├── test_model_serialization.py   # Checkpoint persistence, SHA-256 manifests, & reload
├── test_training_step.py         # Multi-step optimization, loss decrease, & parameter updates
└── test_failure_recovery.py      # Corruption detection, tamper defense, & automatic rollback
```

---

## Detailed Test Module Descriptions

### 1. `test_full_forward.py`
Validates end-to-end forward computation via `RUTAMForCausalLM.forward()`:
- **Empty Graph Fallback**: Confirms that when the graph has no nodes ($N=0$), the model gracefully bypasses retrieval, preserves identity fusion ($\mathbf{h}_{\text{fused}} = \mathbf{h}$), produces valid logits $[B, T, V]$, and outputs transparent diagnostics (`has_nodes=False`).
- **Populated Memory Retrieval & Fusion**: Validates that when graph memory is populated, query vectors trigger neural selection, path traversal synthesizes relational context, and cross-attention fusion modulates decoder hidden representations.
- **Shifted Causal LM Loss**: Verifies next-token prediction loss calculation via shifted cross-entropy:
  $$\mathcal{L}_{\text{LM}} = \text{CrossEntropy}(\text{logits}_{:, :-1, :}, \text{labels}_{:, 1:})$$
- **Attention Masking**: Ensures sequences with dynamic padding masks ignore pad positions.
- **Memory Write Proposals**: Tests candidate memory proposal generation (`CREATE_NODE`, `UPDATE_NODE`, etc.) when `generate_proposals=True`.
- **Tensor Contract Compliance**: Runs `verify_forward_contracts` verifying rank, dimensions, dtypes, CPU placement, contiguity, and numerical finiteness across all intermediate and output tensors.

### 2. `test_memory_lifecycle.py`
Validates the complete **14-step transactional memory lifecycle**:
1. *Empty graph initialization*: Verified at version 0 with zero nodes and edges.
2. *Candidate proposal generation*: Write head constructs staged proposals with discrete operations and embeddings.
3. *Discrete multi-policy validation*: Validator enforces schema, security provenance, and non-stale graph versions.
4. *Atomic transactional commitment*: `GraphTransaction` commits accepted mutations; version increments monotonically to $V=1$.
5. *Cache invalidation & bridge sync*: Neural caches clear; bridge automatically exposes version 1 state.
6. *Retrieval & read verification*: Next forward pass retrieves active memory and modulates token representations.
7. *Snapshot persistence*: Complete graph state serialized to `graph.safetensors`, `graph.json`, and `graph.manifest.json`.
8. *Snapshot integrity verification*: SHA-256 checksums and file lengths verified against manifest.
9. *Snapshot round-trip reload*: Graph reloaded into a clean instance matches original topology and embeddings.
10. *Multi-node & edge update*: Staged creation of second node and relational link committed to reach version 3.
11. *Monotonic version advance*: Version increases monotonically ($0 \to 1 \to 2 \to 3$) with audit history.
12. *Stale update rejection*: Proposal originating from version 0 rejected against version 3 with `ValidationDecision.STALE`.
13. *Conflict & constraint validation*: Dangling target node references rejected cleanly with `ValidationDecision.REJECT`.
14. *Rollback & recovery execution*: Transactional abort rolls back working copy without mutating target graph; corrupted checkpoints roll back to earlier valid snapshots.

### 3. `test_gradient_flow.py`
Validates differentiability and backward gradient propagation:
- **Submodule Gradient Propagation**: Verifies non-zero, finite gradients flow through all 8 production submodules:
  1. `RUTAMNeuralQueryHead` (query projections, decision gate)
  2. `RUTAMGraphEncoder` (MPNN node and edge feature transformations)
  3. `RUTAMGraphAttention` (edge attention, relation attention, attention bias)
  4. `RUTAMGraphSelector` (relevance scoring projections)
  5. `RUTAMPathScorer` (path evaluation and attention)
  6. `RUTAMGraphFusion` (cross-attention $Q, K, V$ projections and fusion gate)
  7. `RUTAMMemoryWriteHead` (write gate, operation predictor, candidate projections)
  8. `RUTAMGraphLinker` (structural feature projections, link scoring, relation classifier)
- **Frozen Backbone Isolation**: When `backbone_frozen=True`, verifies backbone weights receive zero gradients (`p.grad is None` or `requires_grad=False`) while graph memory parameters receive valid gradients.
- **Unfrozen Full Fine-Tuning**: Confirms full model backpropagation updates backbone parameters during end-to-end training.
- **Combined Objective Backprop**: Backpropagates joint causal language modeling loss and memory write loss simultaneously.

### 4. `test_graph_consistency.py`
Ensures state consistency, immutability, and cache coherence:
- **Inference Immutability**: Multiple forward passes never mutate nodes, edges, or version numbers.
- **Snapshot Parity**: Binary safetensors serialization and reload preserves exact tensor values.
- **Bridge Cache Invalidation**: Committing memory updates automatically invalidates bridge tensor caches.
- **Neighborhood Cache Coherence**: Graph edits correctly update adjacency caches without stale edges.

### 5. `test_model_serialization.py`
Tests versioned cryptographic persistence via `RUTAMCheckpointManager`:
- **Atomic Compound Saves**: Persists neural model weights (`model.safetensors`), graph snapshot (`graph_snapshot/`), optimizer states (`optimizer.safetensors`), metadata, and cryptographic manifest (`manifest.json`).
- **Cryptographic Manifest Verification**: Verifies SHA-256 hashes and file byte sizes across all component files.
- **Model Output Invariance**: Verifies restored model produces identical logits (`torch.allclose`) to the source model.

### 6. `test_training_step.py`
Tests optimization dynamics and convergence:
- **Weight Update Verification**: Trainable parameters change after an optimizer step; frozen backbone remains identical.
- **Loss Reduction**: Multi-step training via `RUTAMTrainer` achieves monotonic loss reduction on synthetic batches.
- **Parameter Group Partitioning**: Parameters are properly grouped by module and weight decay policy.

### 7. `test_failure_recovery.py`
Validates robustness against hardware crashes, bit flips, and aborted transactions:
- **Bit-Level Tamper Detection**: Injected corruptions in safetensors trigger cryptographic mismatch and raise `CorruptSnapshotError`.
- **Automatic Rollback**: Checkpoint manager automatically bypasses corrupted steps to recover from the latest intact checkpoint.
- **Exception Rollback**: Unhandled runtime exceptions inside a transaction block safely abort and roll back all working graph mutations.
- **Orphan Cleanup**: Abandoned temporary directories (`.tmp_*`, `tmp_*`) are identified and purged.

---

## Execution Guide

### Running All Integration Tests
```bash
uv run pytest tests/integration/ -v
```

### Running Specific Test Modules
```bash
# Full forward pass integration
uv run pytest tests/integration/test_full_forward.py -v

# 14-step memory lifecycle
uv run pytest tests/integration/test_memory_lifecycle.py -v

# Gradient flow & differentiability
uv run pytest tests/integration/test_gradient_flow.py -v

# Failure recovery & rollback
uv run pytest tests/integration/test_failure_recovery.py -v
```

### Running the Diagnostic Architecture Audit Script
```bash
python scripts/test_rutam_architecture.py
```
This executes all diagnostic checks and writes:
- `reports/architecture_audit.json`
- `reports/architecture_audit.md`

### Running the Benchmark Suite
```bash
python scripts/benchmark_rutam.py --warmup 2 --iterations 5
```
This executes CPU benchmarks and generates `reports/benchmark_report.json`.
