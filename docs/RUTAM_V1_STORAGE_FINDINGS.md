# RUTAM v1 graph storage findings

`GraphStorage` previously replaced the two fixed payload files before committing the manifest. An interruption during either replacement could invalidate the prior committed graph. The writer now writes each snapshot's payloads under an immutable `snapshot-<uuid>` directory, fsyncs payloads and the directory, then publishes the manifest last. The manifest remains the atomic commit record; loading follows the committed snapshot directory. Fixed-name payload aliases are retained for existing integrations. Legacy schema v1 directories without a generation directory continue to load from `graph.safetensors` and `graph.json`.

Validation run: `.venv/bin/python -m pytest tests/test_storage.py tests/test_v1_storage_atomicity.py`.

Parent integration: restored the worker's generation design after overlapping edits removed storage.py; added legacy preservation, explicit generation validation, retired edge ID serialization, verified rollback, parent-directory fsync and ordinary-error export recovery. Interrupted saves now retain the previous committed generation. Root fixed-file pairs require the manifest/locking protocol; they are not an independent transactional reader API. Final executable evidence is recorded in RUTAM_V1_VALIDATION.md and reports/v1_validation/pytest.xml.
