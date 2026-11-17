# RUTAM v1 repair log

The checkout already contained extensive uncommitted source, configuration, docs and tests. Those were preserved; git diff against HEAD is not a task-only change list. The following source paths were modified by the root and Luna workers during this audit, including consolidated duplicate-worker edits. No commit/push or foundation checkpoint overwrite occurred.

- `src/rutam/graph/storage.py`: Immutable generation publication, fsync/checksums, legacy preservation, rollback, alias recovery, retired edge IDs.
- `src/rutam/graph/schema.py`: Node validity/provenance/scope/permissions metadata and stable edge identity metadata.
- `src/rutam/graph/state.py`: Retired edge identity state and validation.
- `src/rutam/graph/operations.py`: Edge identity lifecycle, metadata conversion, permissions JSON conversion and per-graph synchronization.
- `src/rutam/graph/encoder.py`: Visibility-aware feature/message masking so hidden nodes cannot influence visible summaries.
- `src/rutam/graph/traversal.py`: Visibility/eligibility propagation for traversal.
- `src/rutam/model/rutam_forward.py`: Prefix-only graph querying/fusion, snapshot/visibility handling, shape checks and safe layer invocation.
- `src/rutam/model/graph_integration.py`: Inference-only parameter/version/mask keyed cache, coherent snapshots and identity-index reload.
- `src/rutam/model/rutam_model.py`: Public snapshot/mask forwarding and commit authorization propagation.
- `src/rutam/memory/transaction.py`: Mandatory revalidation/authorization, detached durable-first publication, locking, replay/version checks, structured content/delete operations and truthful post-commit failures.
- `src/rutam/memory/controller.py`: Validator and authorization propagation into explicit and automatic commits.
- `src/rutam/memory/provenance.py`: Explicit authorized scopes and permission value validation.
- `src/rutam/memory/validation_rules.py`: Sensitive-data scan includes structured proposal content.
- `src/rutam/memory/write_proposal.py`: Structured content/relation/scope fields, delete aliases and detached candidate cloning.
- `src/rutam/memory/write_head.py`: Ignore negative selected-memory sentinel IDs when choosing targets.
- `src/rutam/runtime/inference.py`: Hold a graph snapshot across autoregressive generation.
- `src/rutam/runtime/memory_runtime.py`: Carry authorization to model commit and fix empty validation result construction.
- `src/rutam/diagnostics/integration.py`: Keep verification model retrieval config independent of initial graph size.
- `src/rutam/diagnostics/memory_checks.py`: Explicit authorized commits in diagnostic lifecycle.

New regression files: `tests/test_v1_storage_atomicity.py`, `tests/test_v1_causal_integration.py`, `tests/test_v1_memory_commit_safety.py`, `tests/test_v1_end_to_end.py`.

Existing tests updated: `tests/test_storage.py`, `tests/test_memory_transactions.py`, `tests/test_memory_commits.py`, `tests/integration/test_failure_recovery.py`, `tests/integration/test_graph_consistency.py`, `tests/integration/test_memory_lifecycle.py`, `tests/test_memory_controller_integration.py`, `tests/test_model_assembly.py`, `tests/test_rutam_end_to_end.py`, `tests/test_rutam_graph_integration.py`, `tests/test_rutam_model.py`, `tests/test_memory_proposals.py`. Changes explicitly authorize intended writes, keep similarity policy outside transaction-only fixtures, check inference-only caching, recognize new operation IDs and expect interrupted saves to retain the prior committed snapshot. Tests were not disabled to hide defects.

Added `scripts/benchmark_graph_v1.py` and evidence under `reports/v1_validation/`. Added the four required reports and preserved the Luna storage findings and independent review. Installation was exercised with `uv pip install --python .venv/bin/python -e . --no-deps`.

Second review: Luna inspected consolidated persistence/transaction/neural code read-only and recorded findings in `RUTAM_V1_INDEPENDENT_REVIEW.md`. Parent fixes: ordinary alias failure restores compatibility exports; direct commits and graph snapshots share the graph lock; process-crash consumers must follow the manifest API. A regression checks two direct concurrent commits cannot both win. Residual separate-process/distributed mutation and raw fixed-file consumer guarantees are not claimed.

## Subsequent continuation and independent review

The following files were modified in this continuation, on top of an already dirty checkout; the pre-existing files and unrelated user work were not reset. No checkpoint or foundation weights were modified.

- `src/rutam/graph/storage.py`, `src/rutam/graph/operations.py`: detach compatibility exports from immutable generations; serialize saves of the same Graph; report post-publication durability uncertainty and retain matching auto-save state.
- `src/rutam/memory/transaction.py`, `src/rutam/memory/controller.py`: prevent staging auto-save, require baseline validation, abort failed apply, preserve durability and maintenance warnings, independently persist archives, and restore lifecycle after failures.
- `src/rutam/model/rutam_forward.py`, `src/rutam/model/graph_integration.py`, `src/rutam/model/layer_integration.py`, `src/rutam/model/lfm3_adapter.py`: prefix-safe optional recurrence, finite empty supervision loss, graph-cache and reload synchronization, and adapter fusion-context cleanup.
- `src/rutam/installer.py`, `src/rutam/bundle.py`: protect checkpoint attachment against symlink/source overlap and sidecar overwrites; require and validate bundle checksum records.
- `src/rutam/model/rutam_forward.py`, `src/rutam/model/query_head.py`, `src/rutam/graph/selector.py`, `src/rutam/graph/embeddings.py`, and `src/rutam/config.py`: make hard selection sparse-first by default, rank persisted features in chunks before relational encoding, and encode only the selected bounded neighborhood. Soft/full-attention and explicit dense compatibility modes retain dense execution.
- `tests/test_storage.py`, `tests/test_graph_encoder.py`: corruption fixtures now target the manifest-selected generation rather than independently copied compatibility exports.
- `tests/test_v1_causal_integration.py`, `tests/test_storage_snapshot_safety.py`, `tests/test_memory_transaction_regressions.py`, `tests/test_memory_controller_regressions.py`, `tests/test_model_bridge_concurrency.py`, `tests/test_installer_bundle_regressions.py`: regression coverage for the subsequent findings.
- `tests/test_v1_sparse_execution.py`: added a 2,000-node default-path ordering/resource-bound regression and query-head gradient assertion.
- `docs/RUTAM_V1_ARCHITECTURE.md`, `docs/RUTAM_V1_ARCHITECTURE_AUDIT.md`, `docs/RUTAM_V1_REPAIR_LOG.md`, `docs/RUTAM_V1_VALIDATION.md`: updated implementation, findings, and evidence.

V1-010 is repaired for the default hard-selector path; candidate scoring and edge-list traversal still scale with graph size. The real LFM3 bridge remains open, and no foundation checkpoint was modified. Scope and results of checks are recorded in the validation report.

## V1-020 storage compare-and-swap

- `src/rutam/exceptions.py`, `src/rutam/graph/storage.py`: define and export `SnapshotConflictError`; remove implicit adoption of the latest manifest for unbound graphs; keep the generation comparison inside the exclusive publication lock. `unconditional=True` remains the explicit last-writer-wins escape hatch.
- `src/rutam/memory/transaction.py` already captured and passed its base generation through commit; `GraphStorage.rollback` already used the loaded active generation. Verified both paths and retained `SnapshotPublicationError` handling for post-publication fsync uncertainty.
- `src/rutam/checkpoint/graph_snapshot.py`: keep independent checkpoint exports from rebinding the source graph.
- Regression coverage: independent stale writers, continued saves by the accepted writer, explicit LWW, unbound-writer safety, stale rollback and transaction conflict. Targeted storage suite passed; full-suite counts and unrelated/environmental failures are in `RUTAM_V1_VALIDATION.md`.
- `.venv/lib/python3.11/site-packages/rutam-local-src.pth`: restore imports from the relocated checkout so the requested plain full-suite command and installed entry points run.
