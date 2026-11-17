"""Versioned safetensors + JSON persistence, with a manifest as commit record.

No model weights are accessed. No pickle is used. POSIX advisory locks coordinate
readers/writers using this API. A process interrupted between file replacements
may leave an unavailable snapshot, but its files cannot silently pass validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as serialize_tensors

from rutam.config import RUTAMConfig
from rutam.exceptions import (
    CorruptSnapshotError,
    SnapshotConflictError,
    SnapshotNotFoundError,
    StorageError,
    ValidationError,
)
from rutam.graph.operations import Graph
from rutam.graph.state import GraphState

SCHEMA_VERSION = 1
TENSOR_FILE = "graph.safetensors"
METADATA_FILE = "graph.json"
MANIFEST_FILE = "graph.manifest.json"
TENSOR_NAMES = {"node_ids", "node_embeddings", "edge_index", "edge_types", "edge_weights"}


_UNSET = object()


class SnapshotPublicationError(StorageError):
    """Manifest was published but its directory fsync failed; durability is uncertain."""


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(data: bytes) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON numeric constant: {value}")

    value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Snapshot JSON must be an object")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class GraphStorage:
    """Local POSIX filesystem snapshots. Hashes detect corruption, not hostile edits.

    Immutable generations are committed by one atomic manifest replacement.
    Fixed-name payloads are compatibility aliases. Legacy schema v1 is readable.
    POSIX local filesystems only; hashes detect damage, not hostile modification.
    """

    def __init__(self, path: str | Path = "data/graph") -> None:
        self.path = Path(path)

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        import fcntl

        with (self.path / ".graph.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def save(
        self, graph: Graph, *, expected_generation: str | None | object = _UNSET,
        unconditional: bool = False, bind: bool = True,
    ) -> None:
        """Publish with a manifest CAS; a new destination must have no manifest.

        Loaded/saved graphs retain their generation for this path. Explicit
        ``unconditional=True`` opts into unsafe last-writer-wins compatibility.
        ``bind=False`` is for independent exports and does not rebind the graph.
        """
        if unconditional and expected_generation is not _UNSET:
            raise ValueError("unconditional and expected_generation are mutually exclusive")
        with graph._mutation_lock:
            path = str(self.path.resolve())
            if expected_generation is _UNSET:
                expected_generation = (
                    graph._storage_generation if graph._storage_path == path else None
                )
            self._save_locked(graph, expected_generation, unconditional, bind, path)

    def _save_locked(
        self, graph: Graph, expected_generation: str | None | object,
        unconditional: bool, bind: bool, path: str,
    ) -> None:
        state = graph.to_state()
        # Validate textual records as well as tensors, including caller-created states.
        Graph.from_state(state)
        snapshot_id = uuid.uuid4().hex
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "graph_version": state.graph_version,
            "config": state.config.to_dict(),
            "metadata": state.metadata,
            "next_node_id": state.next_node_id,
            "retired_node_ids": list(state.retired_node_ids),
            "retired_edge_ids": list(state.retired_edge_ids),
            "relationship_types": state.relationship_types,
            "id_to_row": {str(value): row for row, value in enumerate(state.node_ids.tolist())},
            "nodes": state.node_metadata,
            "edges": state.edge_metadata,
        }
        temporary: list[Path] = []
        published = False
        try:
            tensor_bytes = serialize_tensors(
                state.tensors(),
                metadata={
                    "snapshot_id": snapshot_id,
                    "schema_version": str(SCHEMA_VERSION),
                    "graph_version": str(state.graph_version),
                },
            )
            metadata_bytes = _json_bytes(metadata)
            manifest_bytes = _json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "snapshot_id": snapshot_id,
                    "graph_version": state.graph_version,
                    "payload_directory": f"snapshot-{snapshot_id}",
                    "files": {
                        TENSOR_FILE: {"sha256": _digest(tensor_bytes), "size": len(tensor_bytes)},
                        METADATA_FILE: {
                            "sha256": _digest(metadata_bytes),
                            "size": len(metadata_bytes),
                        },
                    },
                }
            )
            self.path.mkdir(parents=True, exist_ok=True)
            with self._lock(exclusive=True):
                active_manifest = self.path / MANIFEST_FILE
                old = _read_json(active_manifest.read_bytes()) if active_manifest.exists() else None
                current = old["snapshot_id"] if old is not None else None
                if current is not None and (
                    not isinstance(current, str) or uuid.UUID(hex=current).hex != current
                ):
                    raise ValueError("Invalid active snapshot ID")
                if not unconditional and current != expected_generation:
                    raise SnapshotConflictError(
                        f"STALE_SNAPSHOT_GENERATION: expected {expected_generation!r}, "
                        f"found {current!r} at {self.path}"
                    )
                if old is not None:
                    old_dir = self.path / f"snapshot-{old['snapshot_id']}"
                    if not old_dir.exists():
                        old_dir.mkdir()
                        for name in (TENSOR_FILE, METADATA_FILE, MANIFEST_FILE):
                            with (old_dir / name).open("xb") as out:
                                out.write((self.path / name).read_bytes())
                                out.flush()
                                os.fsync(out.fileno())
                        _sync_directory(old_dir)
                snapshot_dir = self.path / f"snapshot-{snapshot_id}"
                snapshot_dir.mkdir()
                staged = []
                for name, payload in ((TENSOR_FILE, tensor_bytes), (METADATA_FILE, metadata_bytes)):
                    target = snapshot_dir / name
                    with target.open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    staged.append(target)
                with (snapshot_dir / MANIFEST_FILE).open("xb") as handle:
                    handle.write(manifest_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                _sync_directory(snapshot_dir)
                _sync_directory(self.path)
                # Keep fixed-name compatibility aliases for existing integrations.
                # Readers use the immutable generation named by the manifest.
                for name, source in zip((TENSOR_FILE, METADATA_FILE), staged, strict=True):
                    alias = self.path / f".{name}.{snapshot_id}.tmp"
                    temporary.append(alias)
                    with alias.open("xb") as handle:
                        handle.write(source.read_bytes())
                        handle.flush()
                        os.fsync(handle.fileno())
                os.replace(temporary[-2], self.path / TENSOR_FILE)
                os.replace(temporary[-1], self.path / METADATA_FILE)
                with tempfile.NamedTemporaryFile(
                    dir=self.path, prefix=".graph-", suffix=".tmp", delete=False
                ) as handle:
                    temporary.append(Path(handle.name))
                    handle.write(manifest_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary[-1], self.path / MANIFEST_FILE)
                published = True
                if bind:
                    graph._storage_generation = snapshot_id
                    graph._storage_path = path
                    graph._storage = self
                _sync_directory(self.path)
        except (OSError, SafetensorError, ValueError) as exc:
            if published:
                raise SnapshotPublicationError(
                    f"Graph snapshot published at {self.path}, but durability is uncertain: {exc}"
                ) from exc
            # Restore compatibility exports to the committed generation on ordinary
            # I/O failures. Crash consistency is defined by manifest-based load().
            if self.path.exists() and (self.path / MANIFEST_FILE).exists():
                try:
                    with self._lock(exclusive=True):
                        committed = _read_json((self.path / MANIFEST_FILE).read_bytes())
                        source = self.path / f"snapshot-{committed['snapshot_id']}"
                        if source.is_dir():
                            for name in (TENSOR_FILE, METADATA_FILE):
                                alias = self.path / name
                                with tempfile.NamedTemporaryFile(
                                    dir=self.path,
                                    prefix=".graph-recover-",
                                    suffix=".tmp",
                                    delete=False,
                                ) as handle:
                                    temp = Path(handle.name)
                                    temporary.append(temp)
                                    handle.write((source / name).read_bytes())
                                    handle.flush()
                                    os.fsync(handle.fileno())
                                os.rename(temp, alias)
                            _sync_directory(self.path)
                except OSError:
                    # Keep the original error. load() still follows immutable files.
                    pass
            raise StorageError(f"Cannot save graph snapshot at {self.path}: {exc}") from exc
        finally:
            for path in temporary:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # Cleanup must not turn a published snapshot into an apparent failure.
                    pass

    def load(self) -> Graph:
        """Load exactly the persisted configuration, including auto_save and provenance policy.

        If this directory is relocated, the returned graph remains attached to this
        storage object, so auto-save targets the actual loaded directory.
        """
        if not self.path.exists():
            raise SnapshotNotFoundError(f"Graph snapshot directory does not exist: {self.path}")
        try:
            with self._lock(exclusive=False):
                if not (self.path / MANIFEST_FILE).exists():
                    if any((self.path / name).exists() for name in (TENSOR_FILE, METADATA_FILE)):
                        raise CorruptSnapshotError(
                            "Snapshot payloads exist without a commit manifest"
                        )
                    raise SnapshotNotFoundError(f"No graph snapshot manifest at {self.path}")
                manifest = _read_json((self.path / MANIFEST_FILE).read_bytes())
                if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
                    raise CorruptSnapshotError("Unsupported manifest schema_version")
                snapshot_id = manifest["snapshot_id"]
                if (
                    not isinstance(snapshot_id, str)
                    or uuid.UUID(hex=snapshot_id).hex != snapshot_id
                ):
                    raise CorruptSnapshotError("Invalid snapshot ID")
                if set(manifest["files"]) != {TENSOR_FILE, METADATA_FILE}:
                    raise CorruptSnapshotError(
                        "Manifest must reference exactly the two snapshot files"
                    )
                payloads = {}
                generation = self.path / f"snapshot-{snapshot_id}"
                directory = manifest.get("payload_directory")
                if directory is not None and directory != generation.name:
                    raise CorruptSnapshotError("Invalid snapshot payload directory")
                if directory is not None and not generation.is_dir():
                    raise CorruptSnapshotError("Committed snapshot generation is missing")
                source_dir = generation if generation.is_dir() else self.path
                for name in (TENSOR_FILE, METADATA_FILE):
                    data = (source_dir / name).read_bytes()
                    entry = manifest["files"][name]
                    if (
                        type(entry["size"]) is not int
                        or entry["size"] != len(data)
                        or entry["sha256"] != _digest(data)
                    ):
                        raise CorruptSnapshotError(f"Checksum or size mismatch for {name}")
                    payloads[name] = data
                metadata = _read_json(payloads[METADATA_FILE])
                if (
                    type(metadata["schema_version"]) is not int
                    or metadata["schema_version"] != SCHEMA_VERSION
                    or metadata["snapshot_id"] != snapshot_id
                    or type(manifest["graph_version"]) is not int
                    or type(metadata["graph_version"]) is not int
                    or metadata["graph_version"] != manifest["graph_version"]
                ):
                    raise CorruptSnapshotError(
                        "Snapshot identity, schema or graph versions disagree"
                    )
                tensors = load_tensors(payloads[TENSOR_FILE])
                if set(tensors) != TENSOR_NAMES:
                    raise CorruptSnapshotError("Snapshot tensor names do not match schema v1")
                # Verify the safetensors header metadata against the same bytes we hashed.
                header_length = int.from_bytes(payloads[TENSOR_FILE][:8], "little")
                header = _read_json(payloads[TENSOR_FILE][8 : 8 + header_length])
                if header.get("__metadata__") != {
                    "snapshot_id": snapshot_id,
                    "schema_version": str(SCHEMA_VERSION),
                    "graph_version": str(metadata["graph_version"]),
                }:
                    raise CorruptSnapshotError("Tensor header does not match snapshot identity")
                if not isinstance(metadata["retired_node_ids"], list):
                    raise CorruptSnapshotError("retired_node_ids must be a list")
                state = GraphState(
                    **tensors,
                    node_metadata=metadata["nodes"],
                    edge_metadata=metadata["edges"],
                    relationship_types=metadata["relationship_types"],
                    graph_version=metadata["graph_version"],
                    config=RUTAMConfig.from_dict(metadata["config"]),
                    next_node_id=metadata["next_node_id"],
                    retired_node_ids=tuple(metadata["retired_node_ids"]),
                    retired_edge_ids=tuple(metadata.get("retired_edge_ids", [])),
                    metadata=metadata["metadata"],
                )
                state.validate()
                mapping = {str(value): row for row, value in enumerate(state.node_ids.tolist())}
                if metadata["id_to_row"] != mapping or any(
                    type(row) is not int for row in metadata["id_to_row"].values()
                ):
                    raise CorruptSnapshotError("Node ID-to-row mapping is inconsistent")
                graph = Graph.from_state(state, storage=self)
                graph._storage_generation = snapshot_id
                graph._storage_path = str(self.path.resolve())
                return graph
        except (SnapshotNotFoundError, CorruptSnapshotError):
            raise
        except (
            SafetensorError,
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            OverflowError,
            RuntimeError,
            UnicodeError,
        ) as exc:
            raise CorruptSnapshotError(f"Invalid graph snapshot at {self.path}: {exc}") from exc
        except FileNotFoundError as exc:
            raise CorruptSnapshotError(
                f"Snapshot is incomplete; missing file: {exc.filename}"
            ) from exc
        except OSError as exc:
            raise StorageError(f"Cannot read graph snapshot at {self.path}: {exc}") from exc

    def rollback(self, snapshot_id: str) -> Graph:
        """Validate an older completed generation and publish it as a new snapshot."""
        if not isinstance(snapshot_id, str) or uuid.UUID(hex=snapshot_id).hex != snapshot_id:
            raise ValidationError("Invalid rollback snapshot ID")
        # Capture the active generation before validating the historical payload;
        # publishing later must fail if any other writer has advanced it.
        active = self.load()
        expected = active._storage_generation
        generation = self.path / f"snapshot-{snapshot_id}"
        manifest = _read_json((generation / MANIFEST_FILE).read_bytes())
        manifest.pop("payload_directory", None)
        with tempfile.TemporaryDirectory(dir=self.path) as tmp:
            check = Path(tmp)
            for name in (TENSOR_FILE, METADATA_FILE):
                os.link(generation / name, check / name)
            (check / MANIFEST_FILE).write_bytes(_json_bytes(manifest))
            graph = GraphStorage(check).load()
        self.save(graph, expected_generation=expected)
        return graph
