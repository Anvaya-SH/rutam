# RUTAM Runtime Capabilities & Telemetry Reference

`RuntimeCapabilities` provides explicit architectural introspection for RUTAM models and runtime sessions, enabling client applications to query, assert, and enforce supported operational features before execution.

---

## 1. Capability Matrix by Architecture Mode

| Capability Name | Accessor Property | `baseline` | `graph_native` | `recurrent` | `full_rutam` | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Graph Memory** | `supports_graph_memory` | ✗ | **✓** | ✗ | **✓** | Dynamic retrieval of relational entities from active graph. |
| **Multi-Hop Traversal** | `supports_multi_hop_traversal` | ✗ | **✓** | ✗ | **✓** | Traversal engine discovering relation paths across graph edges. |
| **Recurrent Reasoning** | `supports_recurrent_reasoning` | ✗ | ✗ | **✓** | **✓** | Parameter-shared depth iteration with loop position embeddings. |
| **Dynamic Halting (ACT)** | `supports_dynamic_halting` | ✗ | ✗ | **✓** | **✓** | Adaptive computation time terminating loops on confidence bounds. |
| **Memory Proposals** | `supports_memory_proposals` | ✗ | **✓** | ✗ | **✓** | Neural write head generating candidate memory mutations. |
| **Discrete Validation** | `supports_memory_validation` | **✓** | **✓** | **✓** | **✓** | Deterministic checking against schema and provenance rules. |
| **Memory Commits** | `supports_memory_commits` | ✗ | **✓** | ✗ | **✓** | Atomic commit of validated mutations to graph memory. |
| **State Snapshots** | `supports_state_snapshots` | ✗ | **✓** | ✗ | **✓** | Serialization of graph topology and neural caches to disk. |
| **Causal Generation** | `supports_generation` | **✓** | **✓** | **✓** | **✓** | Autoregressive next-token sampling loop. |

---

## 2. Inspecting and Verifying Capabilities

### Extracting Capabilities
```python
from rutam.runtime import RuntimeCapabilities

# From a model instance
caps = RuntimeCapabilities.from_model(model)

# From a configuration instance
caps_cfg = RuntimeCapabilities.from_config(config)

print(f"Current mode: {caps.mode}")
print(f"Supports Graph Memory: {caps.supports_graph_memory}")
print(f"Supports Recurrence:   {caps.supports_recurrent_reasoning}")
```

### Asserting Runtime Requirements
Applications requiring specific architectural features can verify compliance programmatically:

```python
# Pass single or multiple requirements
caps.verify(["graph_memory", "recurrent_reasoning"])

# Raises ValidationError if a capability is missing in the active mode
try:
    caps.verify("supports_multi_hop_traversal")
except ValidationError as e:
    print(f"Feature unsupported: {e}")
```

---

## 3. Runtime Telemetry and Diagnostics Schema

`RUTAMInferenceEngine` and `RUTAMSession` automatically emit performance and behavioral diagnostics with every step:

```json
{
  "session_id": "session_rutam_v01_demo",
  "turn_index": 1,
  "model_mode": "full_rutam",
  "is_graph_enabled": true,
  "is_recurrent_enabled": true,
  "batch_size": 1,
  "seq_len": 6,
  "total_tokens": 6,
  "engine_latency_ms": 2.45,
  "has_nodes": true,
  "num_nodes": 4,
  "num_edges": 2,
  "graph_version": 6,
  "selection_active": true,
  "traversal_active": true,
  "fusion_active": true,
  "recurrent_active": true,
  "write_proposals_active": false
}
```
