# Checkpoint Verification & Recovery Procedures

RUTAM incorporates fault-tolerant recovery procedures designed to withstand power loss, SIGKILL, or disk corruption.

## Verification Protocol

Before any checkpoint is loaded into memory, `RUTAMCheckpointManager` runs `verify_checkpoint_integrity`:

1. Verifies `manifest.json` exists and parses cleanly.
2. Checks that every file registered in `manifest.files` exists on disk.
3. Checks that the exact file size in bytes matches the manifest record.
4. Reads each file in 64KB blocks, computes the SHA-256 digest, and matches it against the manifest hash.
5. If any byte was altered or truncated, loading is aborted with `CorruptSnapshotError`.

## Automatic Rollback Mechanism

When calling `get_latest_checkpoint()` or `restore_checkpoint()`:

1. The manager inspects `latest.json` for the recorded step.
2. If `latest.json` points to a corrupt or missing checkpoint, the manager falls back to `find_latest_valid_checkpoint()`.
3. The directory is scanned for all `checkpoint_<step>` directories sorted descending by step.
4. Each checkpoint is verified against its manifest until the newest fully intact checkpoint is found.
5. Incomplete saves left in temporary directories (`tmp_*`) are cleaned using `clean_incomplete_checkpoints()`.
