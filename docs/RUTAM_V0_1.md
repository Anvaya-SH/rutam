# RUTAM v0.1: Architecture & Foundation Overview

RUTAM (Relational Unified Thought and Memory) is a neural architecture combining causal language model backbones with native differentiable graph memory, multi-hop path traversal, parameter-shared recurrent reasoning, and transactional memory mutations.

---

## Key Pillars of RUTAM v0.1

### 1. Differentiable Graph Memory
Instead of treating memory as flat vectors or external document chunks, RUTAM maintains an active relational knowledge graph $G = (V, E)$. Input tokens query the graph dynamically, retrieving pertinent entities and multi-hop relation paths, which modulate the language model's representations through cross-attention fusion gates.

### 2. Parameter-Shared Recurrent Reasoning
Inspired by dynamic depth computation, RUTAM integrates parameter-shared recurrent blocks after graph fusion. The model iterates recurrently over complex relational contexts, conditioning on loop position embeddings and dynamic Adaptive Computation Time (ACT) halting probabilities.

### 3. Discrete Transactional Safety
Neural networks propose mutations; deterministic transaction managers commit them. The neural write head emits candidate proposals, which are validated against schema invariants, access control permissions, and semantic conflict detectors before being committed to disk with rollback protection.

### 4. The 4 Operational Modes
RUTAM natively supports 4 distinct operational modes:
- **`baseline`**: Standard causal language model without graph or recurrence.
- **`graph_native`**: Causal language model augmented with graph memory querying, traversal, and cross-attention fusion.
- **`recurrent`**: Causal language model augmented with parameter-shared recurrent reasoning.
- **`full_rutam`**: Unified model combining both graph memory and recurrent reasoning.

---

## Quick Start Guide

### 1. Model Assembly via Factory
```python
from rutam import RUTAMConfig, RUTAMModelFactory

# Load default config and select mode
config = RUTAMConfig().with_mode("full_rutam")

# Construct model with zero duplicate instances
model = RUTAMModelFactory.create_model(config)
print(f"Model Mode: {model.mode}")
```

### 2. Multi-Turn Session Execution
```python
import torch
from rutam import RUTAMSession

# Spin up interactive session
session = RUTAMSession.create(mode="full_rutam")

# Run an inference step
prompt_ids = torch.tensor([[101, 205, 309, 412]], dtype=torch.long)
result = session.step(prompt_ids, generate_proposals=True, auto_commit=True)

print(f"Turn: {result.turn_index}")
print(f"Logits shape: {result.output.logits.shape}")
print(f"Proposals generated: {len(result.proposals)}")
```

### 3. Autoregressive Text Generation
```python
# Generate subsequent tokens autoregressively
generated_tokens = session.generate(prompt_ids, max_new_tokens=10, temperature=0.7)
print(f"Generated tokens: {generated_tokens[0].tolist()}")
```

### 4. Persisting and Restoring Memory Snapshots
```python
# Persist current graph memory state
session.save_snapshot("checkpoints/my_snapshot")

# Restore in a new session
new_session = RUTAMSession.create(mode="full_rutam")
new_session.restore_snapshot("checkpoints/my_snapshot")
```

---

## Codebase Organization

- `src/rutam/config.py`: Central typed configurations for all subsystems.
- `src/rutam/graph/`: Relational schema, operations, graph encoder, traversal, and storage.
- `src/rutam/memory/`: Write head, discrete validator, controller, and transaction management.
- `src/rutam/model/`: Neural query head, fusion gate, recurrent block, causal LM assembly, factory, registry.
- `src/rutam/runtime/`: Interactive session, inference engine, memory runtime coordinator, capabilities.
- `scripts/demo_rutam.py`: Complete 14-step runnable demonstration.
- `tests/`: 719-test automated test suite.
