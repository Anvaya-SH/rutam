# Rutam v1 independent review

Scope reviewed: graph snapshot persistence and serialization, transaction commit path, controller integration, causal forward path/cache integration, and `tests/test_v1_*.py`. Checkout: `/run/media/anvaya/New Volume/.linux-moved/Desktop/Rutam`. Review was read-only; no training or test suite was run. The requested full-suite run was already in progress and was not polled. LFM3 checkpoint is absent, so HF backbone protocol behavior remains unverified; defaults use `SyntheticBackbone` and must not be described as validated against a real LFM3 model.

## Findings

### High: manifest publication can leave compatibility aliases inconsistent after a reported save failure

`src/rutam/graph/storage.py`, `GraphStorage.save`: the tensor and metadata fixed-name aliases are individually replaced before the authoritative manifest is replaced. If replacement of the second alias or manifest fails, save raises `StorageError` while fixed-name files can represent the new generation and the manifest still names the old generation. Manifest-based `load()` remains protected, but legacy consumers reading fixed names can observe a mixed/uncommitted snapshot. The failure injection test covers manifest replacement and verifies active `load()`, but does not assert alias rollback/consistency. Reproduce by injecting `os.replace` failure specifically on metadata alias after tensor alias replacement, then read fixed-name tensor+metadata.

### Medium: process-local transaction lock does not serialize concurrent commits

`src/rutam/memory/transaction.py`, `_TRANSACTION_LOCK` and `GraphTransaction.commit`: the lock is acquired only by `__enter__`, so callers may construct/use/commit transactions without a context manager; commit itself never acquires it. Two such transactions can both pass the version check before either publishes, and both overwrite the target graph, losing one update. The global lock also serializes unrelated graphs when context-managed. Reproduce with two transactions based on the same graph and synchronize both immediately after version check; both can publish. The current stale-version guard is not an atomic compare-and-swap.

### Medium: graph save is not coordinated with concurrent graph mutation

`src/rutam/graph/storage.py`, `GraphStorage.save` captures `state = graph.to_state()` before taking the storage lock. The lock protects filesystem publication only, not the graph state. A concurrent mutation during state capture can produce a stale snapshot, and a transaction may save its isolated working graph then overwrite a newer live graph if version changes after its check. Snapshot isolation therefore applies to a captured state, but not atomically to the live graph's version transition.

## Positive evidence and limitations

The generation manifest is the authoritative load path and hashes/sizes are checked before decoding; generation payloads are immutable and fsynced before manifest replacement. Transactions stage changes in a working graph, revalidate proposals, require explicit `commit` permission for non-no-op proposals, check source version and replay IDs, and persist before transferring working state into the target. Causal tests compare full-prefix logits with separately evaluated prefixes, and forward code invokes graph conditioning per token prefix. These are meaningful protections, but the review did not execute tests and cannot certify runtime behavior beyond the inspected code. No confirmed permission bypass or suffix-leak defect was found in the reviewed paths.

## Parent disposition after review

The original findings above are preserved as the independent review record. Direct commits and public graph snapshots/mutations now acquire the same per-graph RLock, covering the version check, persistence and live publication. New concurrent-direct-commit regression requires exactly one winner. Ordinary alias I/O failures restore exports to the manifest's committed generation and have a fault-injection regression. Process-crash consistency is guaranteed only through the manifest reader API; raw root-file consumers and separate-process live Graph coordination remain unsupported. Parent reran affected tests and the full suite; see RUTAM_V1_VALIDATION.md for final counts. These changes do not establish real LFM3 compatibility.
