# RUTAM v1 validation and acceptance verdict

Date: 2026-11-17. **ARCHITECTURE NOT READY.** V1-010's default hard-selector path and V1-020's independent-writer lost updates are repaired. The open HIGH finding V1-009 (real LFM3 bridge) still prevents acceptance. Passing synthetic tests is not real foundation-model validation.

## Latest continuation evidence

- Baseline before this continuation's fixes: `.venv/bin/python -m pytest -q --tb=short` — **738 passed, 1 skipped** in 58.95s.
- Post-repair full suite: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q --tb=short` — **761 passed, 1 skipped, 0 failed** in 73.94s. This includes experimental tests, not a training run. Skip: `test_real_lfm3_local_integration`, missing configured checkpoint. The virtual environment also has no Transformers installation and the local Hugging Face hub cache is absent.
- Following only formatting changes to three previously tested paths, `OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_v1_causal_integration.py tests/test_model_bridge_concurrency.py tests/test_installer_bundle_regressions.py --tb=short` — **15 passed** in 12.80s. Counts overlap the full suite.
- The full run includes real temporary `rutam-prepare`/`rutam-verify` CLI execution and a tampered bundle, but uses a synthetic model; it does not validate LFM3.
- `python -m compileall -q src/rutam scripts/benchmark_graph_v1.py` using `.venv/bin/python` passed. Scoped Ruff over modified source and new regressions failed with **six remaining style diagnostics**: five E501 in `installer.py` and one UP017 in `bundle.py`. Newly introduced findings in the first lint run were corrected; the same focused Ruff check excluding `installer.py` and `bundle.py` passed, as did package imports. `git diff --check` still reports the pre-existing EOF blank line in `docs/ARCHITECTURE.md`.
- The deterministic four-node benchmark was rerun in a temporary working directory so the previous artifact was not overwritten. Prior single cold run vs this continuation: initialization 31.83/69.00 ms; selection 5.18/7.40 ms; encoding 6.27/11.20 ms; tensor serialization 0.54/0.78 ms; save 12.98/16.43 ms; reload 3.71/7.94 ms; fusion 0.096/0.231 ms; full graph forward 70.38/127.80 ms; peak process RSS 259632/259508 KiB. These are single cold samples, not a controlled performance study or evidence of optimization/regression; hardware, contention and other run conditions were not controlled. They do not measure LFM3.

## V1-020 storage compare-and-swap continuation

- `PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short tests/test_storage_snapshot_safety.py tests/test_v1_storage_atomicity.py tests/test_storage.py tests/test_v1_end_to_end.py`: **49 passed, 1 skipped, 0 failed** in 2.77s. The skip is the unconfigured real-LFM3 integration.
- After adding `.venv/lib/python3.11/site-packages/rutam-local-src.pth` to point to this checkout's `src`, plain `.venv/bin/python -c 'import rutam'` succeeds and the installed `rutam-prepare` / `rutam-verify` commands run. Full suite command `/home/anvaya/Desktop/Rutam/.venv/bin/python -m pytest -q --tb=short`: **782 passed, 1 skipped, 0 failed** in 165.68s. The skip is `test_real_lfm3_local_integration`, because no local LFM3 checkpoint is configured.
- To verify the regression's red case, `GraphStorage.save` was temporarily forced to unconditional last-writer-wins behavior in-process; `test_independent_writers_reject_stale_manifest_generation` failed with `DID NOT RAISE SnapshotConflictError`, as expected. Under the repaired implementation it passes along with the four LWW, newer-writer, rollback and transaction checks (`5 passed in 0.48s`).
- Regression coverage verifies two separately loaded writers cannot overwrite the accepted generation, the accepted writer can save again using its updated generation, explicit `unconditional=True` still overwrites, an unbound graph cannot silently adopt an existing manifest, rollback cannot replace a concurrent newer writer, and transaction commit detects a cross-instance conflict. `SnapshotConflictError` is importable from `rutam` and remains a `StorageError` subclass. No foundation checkpoint was read or modified.

## V1-010 sparse-first continuation

- `/home/anvaya/Desktop/Rutam/.venv/bin/python -m pytest -q tests/test_v1_sparse_execution.py tests/test_v1_causal_integration.py tests/test_rutam_forward.py tests/test_v1_security_scope_isolation.py --tb=short`: **31 passed, 0 failed** in 6.87s. This covers the 2,000-node ordering/resource regression, prefix causality, private-scope exclusion, empty-graph parity, encoder/query-head gradients, and scoped reads.
- `test_default_forward_selects_before_encoding_2000_node_graph`: **1 passed** in 1.70s. It instruments `select_projected` and `RUTAMGraphEncoder.forward` on a 2,000-node snapshot; all three causal prefixes select before encoding, each encoder input is at most 24 nodes, and total encoded nodes are at most 72.
- In the sparse LM-loss gradient check, query-head pooling/projections/conditioning/gate parameters and selected-subgraph encoder projection/relation/message-passing parameters receive gradients. Top-k indices and candidate ranking are nondifferentiable. Selector scorer parameters receive no gradient from the LM objective because selected scores are not consumed by fusion; an explicit retrieval-score loss can train the recomputed selected scores.
- Candidate scoring and neighborhood extraction still scan visible node and edge data, respectively. This closes the relational-encoding bound only; it is not a sublinear graph retrieval claim. The real LFM3 bridge remains unverified.

## Earlier executed evidence

Working directory: `/run/media/anvaya/New Volume/.linux-moved/Desktop/Rutam`.

- `uv pip install --python .venv/bin/python -e . --no-deps`: PASS, editable rutam 0.1.0 installed against existing environment. A new clean dependency installation was not tested.
- `.venv/bin/python -m pytest -q --tb=short --junitxml=reports/v1_validation/pytest.xml`: **737 passed, 1 skipped, 0 failed, 0 errors, 59.85 seconds**. This full run includes existing experimental unit tests; it was not a training pipeline.
- After that full run, corrected empty `MemoryRuntime.validate([])` result construction and added its regression. `.venv/bin/python -m pytest tests/test_v1_memory_commit_safety.py tests/test_model_assembly.py -q --junitxml=reports/v1_validation/final_affected.xml`: **19 passed, 0 skipped/failed/errors**. This overlaps the full run; counts must not be added together. The final empty-result correction received this affected-file run, not another complete run.
- `.venv/bin/python -m compileall -q src/rutam scripts/benchmark_graph_v1.py`: PASS.
- Focused Ruff check of storage, graph operations, transaction, unified forward and new benchmark: PASS (`reports/v1_validation/lint.log`). No claim of a repository-wide lint cleanup. `git diff --check` still reports a pre-existing extra EOF blank line in unrelated `docs/ARCHITECTURE.md`.
- `rutam-prepare --output-dir <temporary-directory>/bundle`: exit 0. `rutam-verify --bundle-dir <temporary-directory>/bundle --json`: exit 0, `all_passed=true`, valid bundle hashes and finite synthetic smoke logits `[1,8,1000]`. Raw outputs: `reports/v1_validation/cli.json`. Temporary verification bundle removed afterward.
- `.venv/bin/python scripts/benchmark_graph_v1.py`: PASS, measurements in `reports/v1_validation/benchmark.json`. Single cold measurements on a deterministic four-node/three-edge graph, batch 1, four tokens, hidden size 16, one CPU thread. Initialization, encoding, selection, fusion, graph forward, save, reload and tensor serialization timings are recorded. Peak process RSS was 259632 KiB (includes imports/runtime). There is **no before/after baseline** and no optimization claim. No full-model speed or memory estimate is inferred from this tiny workload.

Complete logs and JUnit artifacts are in `reports/v1_validation/`; command inventory is `commands.txt`.

## Hardware and runtime

Intel Core i3-8130U CPU @ 2.20 GHz; `/proc/meminfo` MemTotal 12090320 kB; Linux 7.2.15-1-omarchy x86_64; Python 3.11.17; PyTorch 2.15.0+cpu; CUDA unavailable. See `hardware.json`. This is CPU validation, not GPU/dtype backend certification.

## Real versus synthetic coverage

Actual Graph, GraphStorage, safetensors serialization, neural encoder/query/selection/traversal/fusion, transaction/controller/model public paths execute in tests. Most neural integration tests use the real RUTAM modules with SyntheticBackbone; file/commit failure tests deliberately inject I/O exceptions. They do not replace the intended model boundary with fake successful LFM3 outputs.

`tests/test_v1_end_to_end.py::test_real_lfm3_local_integration`: **NOT RUN / skipped** because `RUTAM_LFM3_CHECKPOINT` does not identify a local `LiquidAI/LFM3-1.7B-Base` checkpoint. No foundation weights were downloaded or modified. The guarded test uses local-only Transformers loading and the public factory if configured; it does not silently substitute a synthetic model. A compatible model-specific bridge is still needed, so supplying weights alone is not claimed sufficient to pass it.

Storage proof: empty initialization; stable identities and CRUD; save/reload; legacy schema-v1 loading; fault-injected interruption before commit; corruption rejection; verified rollback; retired IDs. This validates supported POSIX manifest-based access, not independent raw fixed-file readers, a distributed graph service or power-loss behavior on every filesystem.

Neural proof: finite shapes, graph fusion on the protocol model path, empty-graph baseline parity, graph relation tests, repeated backward passes, prefix invariance, hidden-scope exclusion and versioned inference caching. Learned natural-language memory QA, tokenizer extensions, real LFM3 attention interception, real KV-cache decoding, half precision and GPU integration are unverified.

Memory proof: proposals separate from writes; explicit commit permission; target scope/write checks; invalid provenance rejection; failed-save isolation of live and durable states; exact expected version; repeated proposal rejection across reload; direct concurrent commit serialization; structured content and delete persistence; NO_OP. External evidence truth and secure historical erasure are not certified. Unsupported merge/supersede/request-validation operations fail explicitly.

`test_storage_neural_commit_reload_and_empty_bypass` starts empty, creates/saves/reloads a controlled graph, invokes actual query/selection/encoder/fusion, validates/commits a structured proposal, verifies reload, rejects an unauthorized update without durable change, then restores empty state and verifies bypass. This proves storage correctness and synthetic neural integration, not learned behavior.

## Acceptance gates

| Gate | Status | Reason and evidence |
|---|---|---|
| 1 Install/import | PASS | Current venv imports through a local `.pth`, CLI entry points run, and full tests pass. Clean dependency rebuild not tested. |
| 2 Storage/persistence tests | PASS | Current full suite and `test_v1_storage_atomicity.py`; manifest-based local POSIX API. |
| 3 Node/edge lifecycle | PASS | `test_graph.py`, `test_storage.py`, transaction/content/delete regressions; stable IDs and dangling-reference rejection. |
| 4 Query/selection/encoder tests | PASS | Existing encoder/selection/traversal tests plus CPU causal/gradient regressions. V1-010 bounds relational encoding; candidate scoring and edge-list traversal still scan graph data. |
| 5 Fusion in intended real model path | NOT RUN | Synthetic/protocol path proven; actual LFM3 checkpoint/bridge unverified (V1-009). |
| 6 Empty graph | PASS | Empty tensor round-trip, baseline parity and end-to-end empty restoration. |
| 7 Shapes/finite graph tensors | PASS | CPU synthetic model contracts and integration tests; real LFM3 not covered. |
| 8 Registered parameters/gradients | PASS | Existing gradient tests and two independent backward-step regression on actual RUTAM modules. |
| 9 Proposals separate from commits | PASS | Forward does not write; explicit runtime/controller/transaction boundary. |
| 10 Invalid writes cannot corrupt graph | PASS | Authorization/provenance, staging isolation, save-failure and corruption regressions on the supported single-Graph API. |
| 11 Version/state consistency | PASS | Manifest-generation CAS rejects stale independent writers under the local POSIX publication lock; current-writer continuation, explicit LWW, and transaction conflicts are covered. |
| 12 Save/reload/package interfaces | PASS | Model and graph serialization suite; CLI bundle preparation/verification exit 0. |
| 13 No unresolved critical/high gaps | FAIL | V1-009 real LFM3 bridge remains open. |
| 14 Tests/docs match implementation | PASS | Reports disclose synthetic scope, supported commit operations, manifest API, measured results and open gaps. |
| 15 No false mock-only verification | PASS | Real-model test explicitly skipped and architecture-ready claim withheld. |

## Second review and readiness

GPT-6 Luna's independent read-only review recorded three findings in `RUTAM_V1_INDEPENDENT_REVIEW.md`; it did not run tests. Parent fixed supported-API concurrency and ordinary alias-failure recovery, added regressions, reran affected tests and the complete suite. Original review findings and parent disposition are both retained. The final empty-result fix subsequently passed the affected-file suite.

The package supports local POSIX graph storage with manifest-generation compare-and-swap and synthetic architecture experiments. It is **not ready for the requested next model-development/training stage** as an accepted LFM3-native architecture. The next architecture work is a verified model-specific LFM3 bridge with checkpoint/tokenizer/cache tests and bounded candidate/subgraph selection before relational encoding. Also retain explicit limits for historical deletion, evidence truth verification and generation-file retention. No claim of learned skills, self-improvement or improved natural-language answers is made.
