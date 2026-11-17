# RUTAM

Persistent graph-memory infrastructure developed by **Anvaya.sh**. Core CPU-only
components provide typed graph records, deterministic graph
operations, configuration, versioned safetensors storage, and a trainable
relation-aware graph neural encoder, trainable neural graph selector and native
neural query head, bounded path reasoning and a proposal-only neural linker.

The intended backbone identifier is `LiquidAI/LFM3-1.7B-Base`. It is a configuration
value only: this repository does not download, load, or claim integration with
that model. Its internal APIs have not been verified. The graph encoder and
selector, generic query head, graph traversal and linker are implemented; the LFM3 adapter,
graph-aware backbone attention, fusion, model training and recurrent reasoning
are future work.
See [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install

Python 3.11+; Linux/POSIX local filesystem for the storage implementation. No GPU,
model download, account token, or external database is required. Use an isolated
environment on the target CPU machine.

Recommended with `uv` (the package configuration selects CPU PyTorch):

```bash
uv sync --python 3.11 --extra dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Or with pip (install CPU PyTorch first):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
```

Runtime dependencies: PyTorch, safetensors, NumPy, PyYAML. NumPy is required by
the installed safetensors PyTorch serializer. Development dependencies:
pytest and ruff. `uv.lock` pins the resolved environment, including the CPU wheel.
No Transformers, PyG, graph database, or model artifacts are needed.

## Example

Run from the repository root after installation:

```python
from rutam import Graph, GraphStorage, RUTAMConfig

config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 4}})
graph = Graph(config, metadata={"project": "RUTAM", "owner": "Anvaya.sh"})
project = graph.create_node(
    "project",
    "RUTAM",
    provenance="manual:project-description",
    embedding=[0.1, 0.2, 0.3, 0.4],
)
developer = graph.create_node(
    "organization",
    "Anvaya.sh",
    provenance="manual:project-description",
)
graph.add_relationship(
    project.id,
    developer.id,
    "developed_by",
    weight=0.9,
    provenance="manual:project-description",
)
graph.update_node(project.id, content="RUTAM: persistent native graph memory")
print(graph.neighbors(project.id))
print(graph.traverse(project.id, max_hops=2))

storage = GraphStorage(config.graph.path)
storage.save(graph)
restored = storage.load()
assert restored.nodes == graph.nodes
assert restored.edges == graph.edges
assert restored.version == graph.version
```

## APIs and behavior

- `Node`, `Edge`: immutable, validated records with timezone-aware timestamps and
  provenance. Confidence is in `[0, 1]`; weights are finite, nonnegative float32.
- `Graph(config=None, metadata=None, storage=None)`: starts completely empty.
  `create_node`, `add_node`, `get_node`, `update_node`, `delete_node` implement node
  CRUD. Deletion removes all incident edges, including incoming edges/self-loops.
- `add_relationship`, `add_edge`, `get_relationship`, `update_relationship`,
  `remove_relationship` implement typed edge CRUD. An edge is uniquely keyed by
  `(source_id, target_id, relationship)`; different types may connect the same pair.
  To change an edge's type or endpoints, remove it and add a new edge.
- `neighbors` and `traverse` support outgoing/incoming/both directions and optional
  relationship filtering. Traversal is bounded breadth-first search, visits each
  node once, sorts each node's neighbors, and excludes the starting node.
- `Graph.to_state()` / `Graph.from_state(state)` convert to/from detached tensor
  snapshots; `GraphState.validate()` checks tensor contracts and structural data.
- `GraphStorage(path).save(graph)` / `.load()` persist snapshots with manifest-generation
  conflict detection across independent writers. A stale writer raises
  `SnapshotConflictError`; `unconditional=True` explicitly opts into last-writer-wins.
  `Graph.save()` uses an attached storage object or the configured path.
- `nodes`, `edges`, `version`, `metadata`, `set_metadata` expose canonical records,
  mutation version and detached JSON graph metadata.
- `RUTAMGraphEncoder(config.encoder)` reads a `GraphState` and returns typed
  `GraphEncoderOutput` with contextualized `[N,H]` node states, an optional `[H]`
  mean summary (enabled by default), and `has_nodes`. It has no disk/model access.
- `RUTAMGraphSelector(config.selector)` accepts queries, node states or the actual
  encoder output, and matching persistent node IDs. It returns fixed-size masked
  top-k selections and a differentiable memory context in soft mode.
- `NeuralRelevanceScorer`, `masked_softmax`, `stable_topk`, `node_retrieval_loss`
  and `attention_supervision_loss` provide reusable scoring/selection objectives.
- `RUTAMNeuralQueryHead`, `QueryPooling`, `QueryDecisionGate`, `query_decision_loss`
  and `query_norm_loss` implement token pooling, neural queries, graph conditioning
  and utility supervision. The select_graph method composes existing neural APIs.

IDs are nonnegative signed int64 values. Automatic allocation starts at zero and
always advances beyond the highest allocated ID. Deleted IDs remain retired and
cannot be reused, including after restart. IDs never shift to fill deleted rows.
Relationship type IDs are also retained when unused. Undirected graphs canonicalize
endpoint order. All graph mutations increment the graph version once.

Missing provenance is rejected by default; references are strings, not fetched or
fact-checked. Optional embeddings are explicitly marked absent in metadata;
their tensor rows are zero-filled. Actual zero vectors remain distinguishable.
Embeddings and weights are rounded to float32 on insertion/update so round trips
preserve exact graph values. Graph objects are single-threaded; callers should
coordinate concurrent in-memory mutation.

## Five-node neural encoder example

```bash
uv run python examples/encode_graph.py
```

This executable example constructs five nodes and typed weighted edges, encodes
the graph entirely on CPU, and prints:

```text
node_ids: [10, 20, 30, 40, 50]
node_states: (5, 128) torch.float32
graph_summary: (128,)
has_nodes: True
trainable_parameters: 119936
```

For an existing persistent graph:

```python
from rutam import EncoderConfig, GraphStorage, RUTAMGraphEncoder

state = GraphStorage("data/graph").load().to_state()  # Disk access before forward.
encoder = RUTAMGraphEncoder(EncoderConfig(input_dim=state.config.graph.embedding_dim))
output = encoder(state)  # Differentiable; use .eval() and torch.no_grad() for inference.
assert output.node_states.shape == (state.node_ids.numel(), 128)
```

The default encoder has 119,936 trainable parameters: node projection, a shared
relation vocabulary and two sparse message-passing layers. It uses no dense
adjacency or additional dependencies. Encoder weights belong in a separate
neural checkpoint; forward never modifies graph data. Set `pool_graph=False`
to omit the summary, and ensure all persisted relationship IDs fit the encoder's
configured vocabulary. These are randomly initialized neural representations,
with no claim of trained knowledge quality or LFM3 integration.

## Neural graph selection

For the ten-node/two-query encoder-to-selector demonstration, run:

```bash
uv run python examples/select_graph.py
```

It prints selected persistent IDs and row indices for two continuous query
prototypes. The encoder is seeded/random; selector projections are initialized
to identity for this interpretable smoke test. No training is performed.

For supervised neural reads, use `selection_mode: soft`. The returned
`memory_context: [B,H]` is differentiable through scoring and node projections.
`relevance_logits` and `eligible_mask` support the reusable retrieval/KL losses;
pass the eligibility mask into the loss. Hard ranking is discrete; gathered
representations and selected scores retain their gradients. No straight-through
ranking estimator is used.

## Native neural query head

```bash
uv run python examples/query_graph.py
```

This runs a synthetic `[2,5,256]` hidden-state batch (two inputs) through the query
head, graph encoder and selector. The output queries are `[2,128]`. Modules are seeded
but their weights are unedited and untrained: the first gate declines querying,
while the second selects IDs `[179,166,192]`. Different vectors do not guarantee
useful or distinct retrieval before training. No LFM3 checkpoint is used.

`head(hidden_states, attention_mask, graph_summary=..., memory_available=...)`
returns typed QueryHeadOutput. Query/gate tensors remain differentiable; eval
mode additionally returns threshold decisions. `query_confidence` aliases the
uncalibrated gate score. Token masks are boolean. An explicit
empty_sequence_policy=zeros enables safe zero-context fallbacks; otherwise
all-masked sequences are rejected.

`head.select_graph(hidden_states, selector, encoded, node_ids, ...)` derives memory
availability, verifies query dimensions, and applies inference gate eligibility.
During training soft reads, gated_memory_context multiplies the context by the
continuous gate. Raw gate logits support query_decision_loss; selector logits
support the existing retrieval loss. See ARCHITECTURE.md for gradient semantics
and the adapter's O(B*N) eligibility-mask cost.

## Native path reasoning and link proposals

```bash
.venv/bin/python examples/traverse_graph.py
.venv/bin/python examples/propose_links.py
```

The traversal demo uses **8 nodes, 10 directed typed edges and two neural queries**.
It prints bounded paths as node/relation IDs, query-dependent path logits, a
reasoning-context shape, and an actual warmed CPU runtime. The linker demo uses
**6 nodes and 8 typed edges**, explicitly sets exploratory thresholds to zero,
prints unverified proposals with provenance, and checks saved-file hashes before
and after inference. Random weights do not establish semantic reasoning or factual
relationship discovery.

Public APIs include RUTAMGraphTraversal, RUTAMPathEncoder, RUTAMPathScorer,
NeighborhoodCache, path_retrieval_loss, endpoint_supervision_loss,
pairwise_path_ranking_loss, RUTAMGraphLinker, RUTAMLinkScorer,
RUTAMRelationPredictor, RelationVocabulary, generate_link_candidates,
sample_negative_pairs, binary_link_prediction_loss, relation_classification_loss,
combined_link_loss and validate_proposal_snapshot. Typed TraversalOutput and
LinkerOutput preserve row indices separately from stable IDs. LinkProposal is
always unverified or marked for conflict review; the linker has no commit API.

The default traversal has **170,112** parameters; the default linker has **275,882**.
Reasoning context is differentiable over bounded candidate paths. Neighborhood
enumeration and proposal decisions remain discrete. See ARCHITECTURE.md for exact
contracts, schema migration, gradient boundaries and memory limits.

## Configuration

`RUTAMConfig.from_yaml("config.yaml", overrides={...})` accepts partial YAML and
validated nested overrides. `from_dict`, `to_dict`, `to_yaml` and `with_overrides`
also work without a configuration file. Serialization sorts keys deterministically.
Unknown fields, wrong scalar types and invalid capacities are rejected.

```yaml
project_name: RUTAM
backbone: LiquidAI/LFM3-1.7B-Base
graph:
  path: data/graph
  embedding_dim: 128
  directed: true
  max_nodes: 100000
  max_edges: 500000
  initial_nodes: 0
  auto_save: false
memory:
  validation_enabled: true
  require_provenance: true
runtime:
  device: cpu
  dtype: float32
encoder:
  input_dim: 128
  hidden_dim: 128
  num_layers: 2
  num_relation_types: 32
  dropout: 0.1
  activation: gelu
  aggregation: mean
  residual: true
  layer_norm: true
selector:
  query_dim: 128
  hidden_dim: 128
  score_dim: 128
  scoring_method: scaled_dot
  top_k: 16
  selection_mode: hard
  temperature: 1.0
  normalize_vectors: false
  chunk_size: 4096
  return_attention: false
  max_dense_elements: 1000000
query_head:
  backbone_hidden_dim: 256  # Synthetic; not a verified LFM3 hidden size.
  graph_hidden_dim: 128
  query_dim: 128
  intermediate_dim: 256
  pooling: attention
  dropout: 0.1
  normalize_queries: true
  graph_conditioning: true
  conditioning_mode: gated_add
  query_gate_enabled: true
  gate_threshold: 0.5
  return_attention: false
  empty_sequence_policy: error
  gate_use_availability: true
  residual: false
  residual_projection: false
  norm_regularization_weight: 0.0
traversal:
  max_hops: 3
  max_neighbors_per_node: 16
  max_visited_nodes: 128
  max_candidate_paths: 64
  max_selected_paths: 8
  direction: outgoing
  deterministic: true
  path_encoder: gru
  path_hidden_dim: 128
  path_scoring: additive
  temperature: 1.0
  include_seed_nodes: true
  hidden_dim: 128
  query_dim: 128
  num_relation_types: 32
  max_path_elements: 1000000
linker:
  hidden_dim: 128
  scoring_hidden_dim: 256
  dropout: 0.1
  directed: true
  allow_self_links: false
  max_candidate_pairs: 512
  candidate_strategy: neighborhood
  max_candidate_hops: 2
  link_threshold: 0.75
  relation_threshold: 0.70
  max_proposals_per_call: 32
  use_structural_features: true
  query_conditioning: false
  negative_samples_per_positive: 3
  max_candidate_expansions: 4096
  max_link_elements: 1000000
  relation_labels: [related_to, depends_on, part_of, causes, supports, contradicts, derived_from, precedes]
  loss_weights:
    link: 1.0
    relation: 1.0
    ranking: 0.2
```

`initial_nodes` must be zero: no fabricated placeholder memories are created.
`auto_save=true` saves each successful mutation; an error restores the previous
in-memory state. An interrupted disk publication still requires retry/recovery as
described below. A loaded graph saves back to its actual storage directory even
if the directory was moved. Paths in configuration otherwise resolve against the
working directory. `validation_enabled` reserves the policy switch for future
semantic memory validation; core shape, reference, schema and provenance checks
always run, even when it is false.

Older YAML and schema-v1 snapshots without `encoder` still load with defaults.
Graph storage dimensions remain independent of neural dimensions; encoding
requires `encoder.input_dim == graph.embedding_dim`. Activations `gelu`, `relu`
and `silu`, and aggregations `mean` and `sum` are supported. Mean aggregation
normalizes by incoming edge count, preserving the effect of edge weights.

Old configurations without `selector` also load with defaults. Selector hidden_dim
must match encoder hidden_dim when used together. Query dimensions remain
independent of the future LFM3 interface. The default scaled-dot selector adds
32,768 trainable parameters; additive scoring adds 33,024. Persistent IDs break
equal-score ties deterministically, independent of chunks or row ordering.

Hard mode scores chunks and recomputes winning scores with autograd to avoid
retaining losing-chunk graphs. Soft mode and requested attention require dense
global normalization; max_dense_elements provides an explicit workspace limit
and a clear error for oversized requests. Additive hidden activations count
toward this limit. See ARCHITECTURE.md for formulas and CPU limitations.

Older configurations without query_head load with defaults. The default head
has 182,402 trainable parameters. Pooling modes are last_valid, mean and attention;
conditioning modes are add and gated_add. Selector/query dimensional compatibility
is checked when constructing with selector_config and always in select_graph,
so unrelated graph-only configurations remain loadable. Summary dimensions are
validated directly; no backbone dimensions are guessed.

## Storage and failure behavior

Traversal and linker dimensions are checked when components are combined; older
graph snapshots load their new configuration sections with defaults. For linking
an existing graph, configure relation_labels in its persistent relation-ID order.
RelationVocabulary.from_state(state) can derive an existing contiguous schema;
unknown names are rejected rather than silently relabeled. Classifier class 0
is NO_RELATION, while real class r+1 maps to graph relation ID r. LinkProposal
always carries the graph relation ID, not the classifier class ID. Explicit
migrate_relation_types appends validated schema labels and preserves old IDs.

The traversal context uses soft attention over all bounded candidate paths;
max_selected_paths bounds the separate inspection list. Traversal/path retrieval,
endpoint and margin losses are available without a training engine. The linker
offers BCE, relation CE, optional ranking and a weighted combined objective.
Deterministic sampled negatives exclude known positives, but an unobserved pair
is not automatically a false fact. Proposal thresholds, argmax and enumeration
remain discrete; use returned tensors for neural supervision.

Each checkpoint consists of:

```text
data/graph/
├── graph.safetensors
├── graph.json
├── graph.manifest.json
└── .graph.lock
```

The safetensors payload contains exactly:

| Tensor | dtype | Shape |
|---|---|---|
| `node_ids` | int64 | `[N]` |
| `node_embeddings` | float32 | `[N, D]` |
| `edge_index` | int64 | `[2, E]` |
| `edge_types` | int64 | `[E]` |
| `edge_weights` | float32 | `[E]` |

`edge_index` uses **row offsets into `node_ids`**, not persistent node IDs. Nodes
and edges are sorted in canonical order. The JSON sidecar stores text, timestamps,
confidence, provenance, missing-embedding flags, ID-to-row mappings, retired IDs,
relationship labels/IDs, configuration and graph metadata. Schema version `1`
and a shared snapshot UUID bind the manifest, JSON and tensor header together.

Writes use temporary files in the destination directory, `fsync`, atomic
`os.replace`, and a manifest published last. SHA-256 checksums and sizes bind both
payloads to that manifest. Advisory locks serialize writers and coordinate readers
using this API. Checksums detect corruption/mixed files; they are not signatures.
No pickle or executable deserialization is used.

This is a **single-checkpoint store**, not a transactional history/database. An
interruption between replacements may invalidate the current checkpoint; load
then fails explicitly instead of accepting partial/mismatched data. Retry saving
the still-live graph or restore all three files from a backup. No previous
generation is automatically retained. Temporary crash leftovers are ignored.
Back up the three files together while writers are stopped. The filesystem must
support POSIX locks, rename and directory fsync; network/object storage is not
supported or tested.

Public errors in `rutam.exceptions` distinguish invalid data, duplicate IDs,
missing graph references, absent snapshots, corrupt snapshots and storage I/O
failures. Graph files and local environments are excluded from Git.

## Development and scope

Tests exercise empty/complete snapshots, CRUD, deletion cleanup, directed and
undirected traversal, dtype/shape validation, repeated updates, missing/corrupted
files, recomputed-checksum semantic corruption, interrupted writes, auto-save
rollback and concurrent writers. All run locally on CPU without model downloads.

Encoder tests additionally check analytical weighted aggregation and gradients,
directed hop limits, undirected edges, zero/isolated nodes, type/weight sensitivity,
eval determinism, input/parameter gradients, a real synthetic optimizer step,
separate neural checkpoints, old snapshot compatibility and runtime detection
of dense adjacency allocations. The synthetic objective tests trainability only.

Selector tests cover both scoring formulas, masks, batched independent queries,
deterministic ID ties, exact chunk merging, hard-gradient limitations, soft-context
and projection gradients, retrieval/KL losses, encoder integration and empty or
fully masked memory. There is no full-graph performance benchmark or trained
semantic memory-quality claim.

Query tests cover padding positions, pooling gradients, all-masked policy,
normalization, graph conditioning, gate boundaries and thresholds, decision BCE,
retrieval gradients into the query head and encoder, storage immutability and a
tiny joint CPU optimizer step. These tests establish functionality and gradient
flow; they do not establish calibrated decisions or reliable semantic retrieval.

No neural integration or model quality is claimed. The graph is deliberately
stored separately so future trained native memory modules can evolve without
rewriting backbone weights for every knowledge update.

Traversal/linking tests additionally cover cycles, alternative paths, direction
and relation order, typed duplicates, conflict annotations, budget enforcement,
content-checked index invalidation, schema/checkpoint migration, negative sampling,
encoder/query gradients, sparse allocation guards and saved-graph immutability.
