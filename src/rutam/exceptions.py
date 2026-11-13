"""Public errors raised by the graph-memory subsystem."""


class RUTAMError(Exception):
    """Base class for errors callers can handle without hiding programming errors."""


class ValidationError(RUTAMError, ValueError):
    """Invalid configuration, schema data, or tensor state."""


class DuplicateIDError(ValidationError):
    """A live or retired node identifier was reused."""


class GraphReferenceError(ValidationError):
    """An operation refers to a missing node or relationship."""


class StorageError(RUTAMError):
    """A snapshot could not be read or written."""


class SnapshotNotFoundError(StorageError, FileNotFoundError):
    """No complete snapshot exists at the requested location."""


class CorruptSnapshotError(StorageError):
    """Snapshot files, hashes, metadata, or tensor invariants disagree."""
