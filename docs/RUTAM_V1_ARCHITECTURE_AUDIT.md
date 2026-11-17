# RUTAM v1 architecture gap register

Severity describes the discovered issue, including repaired findings. Known open HIGH gaps prevent an architecture-ready verdict. Actual defects are separated from unsupported future capabilities.

## V1-001 — CRITICAL

- File/symbol: `src/rutam/graph/storage.py:GraphStorage.save`
- Intended: A failed/interrupted save preserves a readable committed state.
- Actual/root cause: Fixed payload replacement destroyed the previous generation before manifest publication.
- Reproduction: Fault-inject alias/manifest replacement.
- Repair: Immutable generations, checksummed manifest, fsync and legacy preservation.
- Outcome/compatibility: FIXED for manifest API; raw exports are explicitly nontransactional.
- Verification: `test_v1_storage_atomicity.py`

## V1-002 — HIGH

- File/symbol: `src/rutam/model/rutam_forward.py:execute_rutam_forward`
- Intended: Earlier logits cannot depend on future tokens.
- Actual/root cause: Full-sequence pooled query influenced every token via shared selection.
- Reproduction: Compare full output against each separately evaluated prefix.
- Repair: Per-token prefix query/selection/traversal/fusion.
- Outcome/compatibility: FIXED on tested synthetic causal path; query cost increases.
- Verification: `test_v1_causal_integration.py`

## V1-003 — CRITICAL

- File/symbol: `src/rutam/memory/transaction.py:apply_proposal/commit`
- Intended: Only authorized validated operations alter live/durable state.
- Actual/root cause: Direct transaction bypassed validator; live state transfer preceded save.
- Reproduction: Direct unapproved commit and injected save failure.
- Repair: Revalidate at public boundary; explicit commit permission; persist isolated copy first.
- Outcome/compatibility: FIXED; callers must now pass trusted authorization.
- Verification: `test_v1_memory_commit_safety.py`

## V1-004 — HIGH

- File/symbol: `src/rutam/model/graph_integration.py:get_active_encoding`
- Intended: Cache invalidation includes autograd and encoder changes.
- Actual/root cause: Training encoding reuse could retain a freed autograd graph.
- Reproduction: Run two independent backward passes on unchanged graph.
- Repair: Cache only eval/no-grad with version/parameter/mask keys.
- Outcome/compatibility: FIXED; training recomputes encoder.
- Verification: `test_v1_causal_integration.py`

## V1-005 — HIGH

- File/symbol: `src/rutam/memory/transaction.py:commit`
- Intended: Stale/repeated/concurrent requests cannot lose accepted writes.
- Actual/root cause: Version comparison was not serialized for non-context-managed commit.
- Reproduction: Concurrent direct transactions; stale/replayed proposal IDs after reload.
- Repair: Graph RLock, exact source version and bounded durable replay ledger.
- Outcome/compatibility: FIXED for one shared Graph; distinct processes are unsupported.
- Verification: `test_v1_memory_commit_safety.py`

## V1-006 — HIGH

- File/symbol: `src/rutam/graph/operations.py:to_state; model/rutam_forward.py`
- Intended: Metadata round-trips and forbidden scopes do not influence retrieval.
- Actual/root cause: Permissions tuple was not JSON-safe; private features could reach message passing.
- Reproduction: Serialize new metadata; put all fixture nodes into private scope.
- Repair: JSON list conversion, mask before graph encoding and traversal, target-scope write checks.
- Outcome/compatibility: FIXED for implemented scope/permission contract.
- Verification: `test_storage.py; test_v1_causal_integration.py`

## V1-007 — MEDIUM

- File/symbol: `src/rutam/memory/write_head.py:_build_proposals`
- Intended: Padding IDs never become stable node IDs.
- Actual/root cause: Negative selection sentinel could win target argmax.
- Reproduction: Exercise memory runtime proposal lifecycle.
- Repair: Mask negative IDs before target ranking.
- Outcome/compatibility: FIXED.
- Verification: `test_model_assembly.py`

## V1-008 — MEDIUM

- File/symbol: `src/rutam/diagnostics/integration.py:create_verification_model`
- Intended: Checkpoint comparison uses identical architecture/config.
- Actual/root cause: Helper changed selector top_k with initial node count, so restored behavior differed despite identical weights.
- Reproduction: Restore populated checkpoint into helper initialized empty.
- Repair: Fixed retrieval configuration independent of graph data.
- Outcome/compatibility: FIXED; graph data no longer changes model config.
- Verification: `tests/integration/test_model_serialization.py`

## V1-009 — HIGH

- File/symbol: `src/rutam/model/model_factory.py; model/lfm3_adapter.py`
- Intended: Verified graph fusion in real intended LFM3 and cached decoding.
- Actual/root cause: Default execution uses SyntheticBackbone; generic protocol/adapter does not prove installed LFM3 compatibility.
- Reproduction: Set local RUTAM_LFM3_CHECKPOINT and run guarded integration test.
- Repair: Implement and validate a model-specific bridge against actual implementation.
- Outcome/compatibility: OPEN/BLOCKED: no local checkpoint configured; no real-model claim.
- Verification: `test_v1_end_to_end.py::test_real_lfm3_local_integration`

## V1-010 — HIGH

- File/symbol: `src/rutam/model/rutam_forward.py; src/rutam/model/query_head.py; src/rutam/graph/selector.py; src/rutam/graph/embeddings.py; src/rutam/graph/encoder.py`
- Intended: Bounded selection should precede relational encoding.
- Actual/root cause: The default path encoded every visible node before query selection; the earlier sparse option was opt-in and ranked already projected node states.
- Reproduction: `test_default_forward_selects_before_encoding_2000_node_graph` instruments selector and encoder calls over a 2,000-node snapshot.
- Repair: Make hard selection sparse-first by default. Score persisted features in bounded chunks, expand top-k seeds by the configured hop limit, then run the relational encoder only on the bounded induced subgraph. Preserve dense execution for soft/full-attention selectors and the explicit compatibility opt-out.
- Outcome/compatibility: FIXED for the default hard-selector path. Candidate scoring and edge-list traversal still scan graph data, so this bounds relational encoding, not total retrieval complexity.
- Verification: `tests/test_v1_sparse_execution.py::test_default_forward_selects_before_encoding_2000_node_graph`; causality, scope, empty parity and gradient tests. No real LFM3 checkpoint was used.

## V1-011 — MEDIUM

- File/symbol: `src/rutam/memory/transaction.py; graph/storage.py`
- Intended: Retention/evidence/operations have explicit semantics.
- Actual/root cause: History remains accessible; external evidence truth is not verified; merge/supersede were misleading partial operations.
- Reproduction: Inspect archive/load and unsupported operation commits.
- Repair: Explicit rejection; active deletes/content/link operations supported; document historical retention.
- Outcome/compatibility: LIMITATION: secure erase and external evidence verification are not implemented.
- Verification: `test_v1_memory_commit_safety.py`

## V1-012 — LOW

- File/symbol: `src/rutam/graph/storage.py:compatibility exports`
- Intended: Compatibility readers have a defined consistency boundary.
- Actual/root cause: Independently read fixed root payloads can mix around process crash.
- Reproduction: Alias replacement fault injection; inspect raw-file API assumptions.
- Repair: Restore exports on ordinary errors; require manifest/locking for transactional reads.
- Outcome/compatibility: DOCUMENTED; independent raw-file pairs are not a supported transactional API.
- Verification: `test_failed_alias_replacement_restores_compatibility_exports`

## Second-pass findings (2026-11-16)

Each reproduction uses disposable graph/checkpoint fixtures; no foundation checkpoint was modified.

- **V1-013 — CRITICAL, fixed:** `graph/storage.py:GraphStorage.save` made fixed-name exports hard links to supposedly immutable generation files. Truncating an export corrupted the committed snapshot. Root cause: alias creation used hard links. Repair: independently written, fsynced copies; compatibility exports remain nontransactional and legacy snapshots may retain old links until a successful save. Test: `test_storage_snapshot_safety.py::test_compatibility_export_write_cannot_corrupt_committed_generation` (reproduced before repair).
- **V1-014 — CRITICAL, fixed:** `memory/transaction.py:GraphTransaction` staged an auto-save-enabled working graph that could write to the configured durable path during `apply_proposal`, before commit. Repair: disable auto-save for staging and restore the original config during persistence; resolve implicit storage for an auto-save target. Compatibility: explicitly supplied storage still takes precedence. Test: `test_autosave_staging_never_publishes_before_commit` (reproduced before repair) and implicit-storage regression in `test_memory_transaction_regressions.py`.
- **V1-015 — CRITICAL, fixed:** `installer.py:run_attach_lfm3` resolved an existing target symlink before `--force`, potentially deleting the source checkpoint, and wrote a manifest through the symlink. Reproduction was established by source inspection, not destructive execution. Repair: preserve the final target component, reject overlapping source/target directories, unlink symlinks themselves, and create uniquely named sibling manifests without following preexisting sidecars. Compatibility: manifest is now a returned sibling path. Test: `test_force_reattach_never_deletes_or_modifies_source` in `test_installer_bundle_regressions.py`.
- **V1-016 — HIGH, fixed:** `graph/storage.py:save`, `graph/operations.py:_mutation`, and `memory/transaction.py:commit` treated an fsync failure after manifest publication as if nothing had committed. Fault injection showed the new state was already visible. Repair: typed publication-uncertain error, consistent live state, and commit warning preserved through maintenance errors. Compatibility: callers must handle committed-but-durability-uncertain results. Tests: `test_storage_snapshot_safety.py` and `test_memory_transaction_regressions.py`.
- **V1-017 — HIGH, fixed:** `memory/transaction.py:apply_proposal` could publish a partially mutated working graph after an apply exception, and an injected permissive validator could bypass baseline checks. Repair: abort on any apply exception and always run baseline validation before additional validators. Tests: ledger exhaustion and injected-validator regressions in `test_memory_transaction_regressions.py`.
- **V1-018 — HIGH, fixed:** `model/graph_integration.py:refresh` could associate an old snapshot with a newer live version, and `reload_from_storage` replaced fields without graph locking. Repair: synchronize snapshot/reload with the graph lock and key cache to the snapshot version. Reload still explicitly replaces earlier state, not merges it. Test: `test_model_bridge_concurrency.py`.
- **V1-019 — HIGH, fixed:** `bundle.py:RUTAMBundle.verify` could report success for an absent checksum catalog and unverified model weights. Repair: require checksums for core files and reject malformed or escaping paths. Compatibility: old bundles without a complete catalog fail verification instead of reporting success. Test: `test_installer_bundle_regressions.py`, including actual temporary CLI bundle and tampered weights.
- **V1-020 — HIGH, fixed (2026-11-17):** `GraphStorage.load` binds the loaded graph to the manifest snapshot ID; `GraphStorage.save` compares that expected generation under the exclusive publication lock and raises catchable `SnapshotConflictError` for stale writers. `GraphTransaction` carries its base generation into commit, and `GraphStorage.rollback` captures and checks the active generation before republishing history. `unconditional=True` explicitly retains last-writer-wins behavior; an unbound graph cannot silently adopt the current generation. Post-publication fsync failures still raise `SnapshotPublicationError`. Tests: `test_independent_writers_reject_stale_manifest_generation`, `test_explicit_lww_and_newer_generation_writer_are_supported`, `test_unbound_graph_does_not_adopt_latest_generation_implicitly`, `test_stale_rollback_cannot_replace_a_newer_writer`, and `test_concurrent_transactions_on_distinct_graphs_detect_storage_conflict`. This claim is limited to the supported local POSIX storage API.
- **V1-021 — MEDIUM, fixed:** `memory/transaction.py:_preserve_snapshot_archive` copied a manifest without its referenced generation; archive load failed. Repair creates an independent snapshot with GraphStorage. Test: `test_snapshot_archive_is_independently_loadable`.
- **V1-022 — MEDIUM, fixed for synthetic path:** optional recurrence reused final-prefix graph selection for earlier tokens, violating causality; empty shifted supervision yielded NaN; explicit adapter could reuse stale fusion context. Repair uses prefix recurrence, differentiable zero for no supervised targets, and clears adapter context. Compatibility: recurrence is more expensive. Tests: `test_v1_causal_integration.py`; real LFM3 remains unverified.

## Mandatory invariants

| Invariant | Evidence and remaining limit |
|---|---|
| 1 No graph-derived facts without a graph | Synthetic empty-graph bypass tested; learned behavior of real LFM3 is unverified. |
| 2 Empty baseline parity | Tested against synthetic backbone on CPU; real LFM3 not run. |
| 3 Stable persistent IDs | Storage round-trip tests; manifest-based reader. |
| 4 No silently remapped edges | Lifecycle, identity and corrupt-state tests; manifest-based reader. |
| 5 Fusion on real intended model path | Synthetic/protocol path tested; **real LFM3 not run and bridge unresolved**. |
| 6 Graph features influence hidden state | Controlled synthetic integration tests; no learned QA claim. |
| 7 Gradients never commit memory | Forward/transaction separation tested; staging auto-save regression added. |
| 8 Invalid/unauthorized writes rejected | Scoped regression tests; external evidence truth is not authenticated. |
| 9 Crash cannot publish silently inconsistent snapshot | Generation/manifest and injected I/O tests on supported local POSIX API; post-publication fsync reports uncertainty. Hardware power-loss proof is not claimed. |
| 10 Coherent request snapshot | Snapshot pinning tested on synthetic generation; real cached LFM3 decoding not run. |
| 11 Cache invalidation | Training recomputation, version/tensor keys and bridge concurrency regressions; broader concurrent multi-request behavior remains unverified. |
| 12 Separate model, module, graph data | Separate checkpoint and graph formats tested; no foundation checkpoint was overwritten. |
| 13 Untrusted content remains data | Continuous graph encoding rather than prompt privilege; adversarial learned-model behavior unverified. |
| 14 Scope isolation | Visibility and write-scope tests; comprehensive multi-user isolation not certified. |
| 15 Explicit fallback or diagnostic | Unsupported paths documented or rejected in tested cases; real LFM3 bridge and KV-cache integration remain unavailable. |

See validation report for acceptance gates and current test results. V1-020 is repaired for the supported local POSIX storage API, and V1-010 is repaired for the default hard-selector path with graph-wide candidate/edge scans still present. Real LFM3 integration remains open.
