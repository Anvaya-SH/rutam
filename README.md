# RUTAM

Persistent graph-memory infrastructure developed by **Anvaya.sh**. This first
phase is working CPU-only Python code: typed graph records, deterministic graph
operations, configuration, and versioned safetensors storage independent of model
weights.

The intended backbone identifier is `LiquidAI/LFM3-1.7B-Base`. It is a configuration
value only: this repository does not download, load, or claim integration with
that model. Its internal APIs have not been verified. Neural graph encoders,
graph-aware attention, learned memory selection, fusion, training and recurrent
reasoning are future work. See [ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
- `GraphStorage(path).save(graph)` / `.load()` persist snapshots. `Graph.save()`
  uses an attached storage object or the configured path.
- `nodes`, `edges`, `version`, `metadata`, `set_metadata` expose canonical records,
  mutation version and detached JSON graph metadata.

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
```

`initial_nodes` must be zero: no fabricated placeholder memories are created.
`auto_save=true` saves each successful mutation; an error restores the previous
in-memory state. An interrupted disk publication still requires retry/recovery as
described below. A loaded graph saves back to its actual storage directory even
if the directory was moved. Paths in configuration otherwise resolve against the
working directory. `validation_enabled` reserves the policy switch for future
semantic memory validation; core shape, reference, schema and provenance checks
always run, even when it is false.

## Storage and failure behavior

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

No neural integration or model quality is claimed. The graph is deliberately
stored separately so future trained native memory modules can evolve without
rewriting backbone weights for every knowledge update.
