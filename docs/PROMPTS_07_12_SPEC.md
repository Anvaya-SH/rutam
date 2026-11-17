# RUTAM — MEGA ENGINEERING SPECIFICATION (PROMPTS 07–12)
**Project:** RUTAM  
**Organization:** Anvaya.sh  
**Foundation:** LiquidAI/LFM3-1.7B-Base  
**Development:** Local CPU-only environment  
**Scope:** Six consecutive architecture milestones (07 -> 08 -> 09 -> 10 -> 11 -> 12)  
**Version Target:** RUTAM Phase 1 — Native Graph-Memory Integration

---

## Hardware Constraints
- Intel Core i3 CPU, 12 GB RAM, no GPU, Python 3.11+, PyTorch CPU, safetensors.
- Synthetic tensors and lightweight test doubles only. No downloading full LFM3-1.7B weights.

---

## Execution Protocol
For each milestone:
1. Announce `STARTING PROMPT XX`.
2. Inspect relevant existing modules.
3. Define compatible interfaces.
4. Implement requested functionality.
5. Write comprehensive CPU tests.
6. Run full test suite (`uv run pytest`).
7. Run `uv run ruff check .`.
8. Fix introduced regressions.
9. Update architecture documentation (`docs/ARCHITECTURE.md`, etc.).
10. Report implemented functionality and limitations.
11. Announce `PROMPT XX COMPLETE`.
12. Announce `STARTING PROMPT YY`.

---

## PROMPT 07 — NATIVE MEMORY WRITE HEAD
**Milestone:** 07/20  
**Deliverable:** `RUTAMMemoryWriteHead`

### Required Files
- `src/rutam/memory/write_head.py`
- `src/rutam/memory/write_gate.py`
- `src/rutam/memory/memory_projection.py`
- `src/rutam/memory/operation_predictor.py`
- `src/rutam/memory/write_proposal.py`
- `src/rutam/memory/write_losses.py`
- `src/rutam/memory/__init__.py`
Tests:
- `tests/test_memory_write_head.py`
- `tests/test_memory_write_gate.py`
- `tests/test_memory_proposals.py`
- `tests/test_memory_write_integration.py`

### Specifications
`class RUTAMMemoryWriteHead(torch.nn.Module)`
Inputs:
- `hidden_states`: `[B,T,D]`
- `attention_mask`: optional `[B,T]`
- `graph_summary`: optional `[B,H]`
- `selected_memory_states`: optional `[B,K,H]`
- `selected_memory_ids`: optional `[B,K]`
- `source_references`: structured metadata supplied outside autograd
- `memory_available`: optional `[B]`
Outputs:
- `write_logits`: `[B,1]`
- `write_probabilities`: `[B,1]`
- `operation_logits`: `[B,O]`
- `operation_probabilities`: `[B,O]`
- `proposed_embeddings`: `[B,M,H]`
- `target_memory_scores`: optional `[B,K]`
- `proposal_mask`: `[B,M]`
- `write_decisions`: `[B]` in inference mode
Where B=batch, T=sequence length, D=LM hidden dim, H=graph embedding dim, M=max proposals, O=operations count.
Returns typed `MemoryWriteOutput`.
Supported operations (stable versioned mapping):
- NO_OP (0)
- CREATE_NODE (1)
- UPDATE_NODE (2)
- PROPOSE_LINK (3)
- PROPOSE_MERGE (4)
- SUPERSEDE_NODE (5)
- REQUEST_VALIDATION (6)

Pooling:
- Last valid token, masked mean, learnable attention pooling (reuse `QueryPooling` where appropriate).

Write Gate:
- `Pooled Hidden State -> LayerNorm -> Linear -> GELU -> Linear -> Write Logit`
- Sigmoid probabilities, configurable threshold.

Operation Predictor:
- `Pooled Hidden State -> LayerNorm -> Linear -> GELU -> Linear -> Operation Logits`

Neural Memory Projection:
- `Pooled Hidden State -> Linear(D, Intermediate) -> GELU -> Linear(Intermediate, M*H)` reshaped to `[B, M, H]` (or candidate slot embeddings).
- Normalize embeddings when configured.

Target Memory Selection:
- Trainable compatibility scoring between proposed memory and existing selected nodes `[B,K,H]`.

Structured `MemoryProposal`:
- `proposal_id`: str (uuid4)
- `operation`: str or enum
- `candidate_embedding`: Tensor [H] or None
- `target_node_ids`: tuple[int, ...]
- `proposed_link_ids`: tuple[int, ...]
- `provenance`: ProvenanceRecord or dict
- `confidence`: float
- `source_graph_version`: int | None
- `timestamp`: datetime
- `origin_category`: str
- `validation_status`: str ("pending", etc.)

Losses:
- `L_write`: BCEWithLogits
- `L_operation`: CrossEntropy
- `L_embedding`: Cosine similarity / contrastive
- `L_target`: CrossEntropy / BCE over candidate nodes
- `L = αL_write + βL_operation + γL_embedding + δL_target`

Extend `RUTAMConfig`:
```yaml
memory_write:
  enabled: true
  hidden_dim: 128
  projection_dim: 256
  max_proposals: 4
  write_threshold: 0.70
  operation_count: 7
  use_graph_context: true
  target_selection: true
  normalize_embeddings: true
  dropout: 0.1
  commit_automatically: false
```
`commit_automatically` must remain false. Proposal generation ONLY. No persistent mutations.

---

## PROMPT 08 — MEMORY VALIDATION & INTEGRITY
**Milestone:** 08/20  
**Deliverable:** `RUTAMMemoryValidator`

### Required Files
- `src/rutam/memory/validator.py`
- `src/rutam/memory/validation_rules.py`
- `src/rutam/memory/conflict_detection.py`
- `src/rutam/memory/duplicate_detection.py`
- `src/rutam/memory/provenance.py`
- `src/rutam/memory/validation_output.py`
Tests:
- `tests/test_memory_validator.py`
- `tests/test_memory_conflicts.py`
- `tests/test_memory_provenance.py`
- `tests/test_memory_integrity.py`
- `tests/test_validation_integration.py`

### Specifications
`class RUTAMMemoryValidator` (composable Python class, not necessarily nn.Module).
Inputs:
- Memory proposals / Link proposals
- Current graph snapshot (GraphState or Graph)
- Graph schema / config
- Provenance info
- Authorization context
- Validation policy
Outputs:
`MemoryValidationResult`:
- `proposal_id`: str
- `decision`: "ACCEPT", "REJECT", "DEFER", "DUPLICATE", "CONFLICT", "STALE"
- `reason_codes`: tuple[str, ...]
- `confidence_metadata`: dict
- `detected_conflicts`: tuple[str, ...]
- `existing_match_ids`: tuple[int, ...]
- `source_graph_version`: int | None
- `sanitized_proposal`: Any | None
- `requires_further_verification`: bool

Validation checks:
1. Structural validation against graph schema (valid op IDs, existing target node IDs, no dangling edges, embedding dims match and no NaN/Inf, dtype float32, capacity limits, relation direction).
2. Provenance enforcement: structured provenance schema (source_id, source_category, observation_timestamp, creation_timestamp, content_hash, evidence, authorization_category, originating_proposal_id). Models cannot fabricate sources.
3. Duplicate detection: exact stable-ID matches, exact normalized content, duplicate relationships, bounded similarity candidates.
4. Contradiction/conflict detection: mutually exclusive relations, version changed on node update, unique property violations.
5. Authorization and privacy boundaries: user-authorized, imported, unverified model hypothesis, sensitive data rejection (e.g. auth secrets).
6. Version and concurrency checks: compare proposal version with snapshot version; reject stale updates (optimistic concurrency).
7. Confidence handling: thresholds filter candidates, but neural confidence never overrides missing authorization or invalid structure.
8. No-commit requirement: validator never mutates persistent storage.

Extend `RUTAMConfig`:
```yaml
memory_validation:
  enabled: true
  require_provenance: true
  allow_unverified_hypotheses: false
  detect_duplicates: true
  detect_structural_conflicts: true
  reject_stale_updates: true
  require_authorization: true
  max_proposals_per_batch: 64
  max_proposal_bytes: 65536
  semantic_verification_enabled: false
  default_decision: defer
```

---

## PROMPT 09 — NATIVE MEMORY CONTROLLER
**Milestone:** 09/20  
**Deliverable:** `RUTAMMemoryController`

### Required Files
- `src/rutam/memory/controller.py`
- `src/rutam/memory/read_pipeline.py`
- `src/rutam/memory/write_pipeline.py`
- `src/rutam/memory/transaction.py`
- `src/rutam/memory/audit.py`
- `src/rutam/memory/memory_state.py`
- `src/rutam/memory/controller_output.py`
Tests:
- `tests/test_memory_controller.py`
- `tests/test_memory_transactions.py`
- `tests/test_memory_reads.py`
- `tests/test_memory_commits.py`
- `tests/test_memory_controller_integration.py`

### Specifications
`class RUTAMMemoryController(torch.nn.Module)`
Coordinates neural query, graph selection, multi-hop traversal, write proposals, validation, and transactional graph commits.
Methods:
- `read_memory(...)`: Neural Query -> Graph Selection -> Graph Traversal -> Memory Context. Differentiable.
- `propose_memory_update(...)`: LM Hidden States -> Write Head -> Proposals. Differentiable.
- `validate_updates(...)`: Proposals -> Validator -> ValidationResults. Discrete.
- `commit_validated_updates(...)`: Transactional graph commit outside autograd. Non-differentiable.
- `refresh_graph_state(...)`: Invalidate cache, reload active snapshot.
- `get_memory_status(...)`: Status of memory system.

Transactional Commits:
- Isolated working state -> validate -> write safetensors + metadata + manifest -> atomic switch.
- All-or-nothing batch semantics.
- Preserves previous snapshots (keep `preserve_snapshots` recent).
- Auditing: persistent audit log of committed operations.
- Single-writer synchronization.
- Memory states: EMPTY, LOADED, READ_READY, WRITE_PENDING, VALIDATING, COMMITTING, READY, ERROR.

Extend `RUTAMConfig`:
```yaml
memory_controller:
  enabled: true
  read_enabled: true
  write_enabled: true
  auto_commit: false
  require_validation: true
  max_reads_per_request: 4
  max_pending_proposals: 128
  transaction_mode: atomic
  preserve_snapshots: 5
  audit_enabled: true
  cache_graph_states: true
  allow_destructive_updates: false
```

---

## PROMPT 10 — NATIVE GRAPH FUSION
**Milestone:** 10/20  
**Deliverable:** `RUTAMGraphFusion`

### Required Files
- `src/rutam/model/fusion.py`
- `src/rutam/model/graph_attention.py`
- `src/rutam/model/fusion_gate.py`
- `src/rutam/model/fusion_projection.py`
- `src/rutam/model/fusion_output.py`
Tests:
- `tests/test_graph_fusion.py`
- `tests/test_graph_attention.py`
- `tests/test_fusion_gate.py`
- `tests/test_fusion_gradients.py`
- `tests/test_fusion_integration.py`

### Specifications
`class RUTAMGraphFusion(torch.nn.Module)`
Inputs:
- `hidden_states`: `[B,T,D]`
- `graph_states`: `[B,K,H]`
- `graph_mask`: `[B,K]`
- `reasoning_context`: optional `[B,H]`
- `query_gate`: optional `[B,1]`
- `token_attention_mask`: optional `[B,T]`
Outputs:
- `fused_hidden_states`: `[B,T,D]`
- `fusion_gate`: `[B,T,1]` or `[B,1]`
- `attention_weights`: optional `[B,Heads,T,K]`
- `utilization_metrics`: dict

Cross-Attention:
`Q = Wq(hidden_states)`, `K = Wk(graph_states)`, `V = Wv(graph_states)`
`Attention = softmax(QK^T / sqrt(d_k))` with masking for padded graph slots.
`Context = Attention @ V`.
Includes optional projection of `reasoning_context`.

Fusion Gate:
`Gate = sigmoid(MLP([hidden_states, projected_context]))`
`Fused = hidden_states + alpha * Gate * projected_context`
`alpha` is residual scale parameter, initialized to near-zero (e.g. 0.001 or configurable) to preserve identity while allowing nonzero initial gradients.

Empty Graph:
When graph is empty / all masked: returns `hidden_states` unchanged, no NaNs, preserves dtype/device.

Extend `RUTAMConfig`:
```yaml
graph_fusion:
  enabled: true
  hidden_dim: 256
  graph_dim: 128
  num_attention_heads: 4
  attention_dropout: 0.0
  projection_dropout: 0.1
  fusion_mode: cross_attention
  gate_mode: token
  residual_scale_init: 0.001
  use_reasoning_context: true
  preserve_empty_identity: true
  return_attention: false
```

---

## PROMPT 11 — LFM3 BACKBONE ADAPTER
**Milestone:** 11/20  
**Deliverable:** `RUTAMLFM3Adapter`

### Required Files
- `src/rutam/model/lfm3_adapter.py`
- `src/rutam/model/backbone_protocol.py`
- `src/rutam/model/layer_integration.py`
- `src/rutam/model/architecture_inspection.py`
- `src/rutam/model/adapter_config.py`
- `src/rutam/model/adapter_output.py`
Tests:
- `tests/test_backbone_protocol.py`
- `tests/test_lfm3_adapter.py`
- `tests/test_layer_integration.py`
- `tests/test_backbone_compatibility.py`
- `tests/test_adapter_gradients.py`

### Specifications
`class BackboneProtocol(Protocol)`:
- Hidden dim discovery, forward call, layer interface discovery, token embeddings access, logits extraction, parameter enumeration.

`class RUTAMLFM3Adapter(torch.nn.Module)`:
- Manages integration between backbone and graph fusion.
- Inspects backbone, validates integration points.
- Freezing/unfreezing utilities: freeze backbone, keep graph modules trainable, selective layer unfreezing.
- Explicit layer integration vs module forward hooks.
- Synthetic testing backbone (tiny test double) for testing without full 1.7B checkpoint.

Config:
```yaml
lfm3_adapter:
  backbone_id: LiquidAI/LFM3-1.7B-Base
  enabled: true
  fusion_locations: []
  backbone_frozen: true
  graph_modules_trainable: true
  integration_mode: explicit
  require_verified_layers: true
  preserve_output_interface: true
  enable_cache: false
  allow_remote_code: false
```

---

## PROMPT 12 — COMPLETE NATIVE GRAPH-AWARE FORWARD PASS
**Milestone:** 12/20  
**Deliverable:** `RUTAMForCausalLM`

### Required Files
- `src/rutam/model/rutam_model.py`
- `src/rutam/model/rutam_forward.py`
- `src/rutam/model/rutam_outputs.py`
- `src/rutam/model/rutam_config.py`
- `src/rutam/model/graph_integration.py`
- `src/rutam/model/forward_context.py`
- `src/rutam/model/parameter_groups.py`
Tests:
- `tests/test_rutam_model.py`
- `tests/test_rutam_forward.py`
- `tests/test_rutam_graph_integration.py`
- `tests/test_rutam_memory_integration.py`
- `tests/test_rutam_gradients.py`
- `tests/test_rutam_compatibility.py`
- `tests/test_rutam_end_to_end.py`

### Specifications
`class RUTAMForCausalLM(torch.nn.Module)`:
Unified forward computation:
1. Token inputs -> initial backbone layers.
2. Neural Query Head -> query vectors & gate.
3. Graph Encoder & Selector -> selected node states.
4. Multi-hop Traversal -> reasoning context & paths.
5. Graph Fusion -> fused hidden states.
6. Remaining backbone layers -> output logits.
7. Next-token causal LM loss (shifted).
8. Neural Memory Write Head -> memory proposals (optional).
9. Memory Controller -> explicit validation & commit API (separate from forward).

Parameter groups:
- Frozen backbone
- Trainable graph encoder, selector, query head, traversal, linker, write head, fusion.

Model Config:
```yaml
rutam_model:
  backbone: LiquidAI/LFM3-1.7B-Base
  graph_enabled: true
  graph_encoder_enabled: true
  graph_selector_enabled: true
  graph_traversal_enabled: true
  graph_fusion_enabled: true
  memory_write_enabled: true
  memory_auto_commit: false
  backbone_frozen: true
  recurrent_loops_enabled: false
  use_cached_graph_states: true
  return_memory_diagnostics: false
```
