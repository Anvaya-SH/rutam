"""Integration of persistent graph memory snapshots with model checkpoints."""

from __future__ import annotations

from pathlib import Path

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage


def save_graph_snapshot(graph: Graph, target_dir: str | Path) -> None:
    """Save persistent graph memory snapshot into designated checkpoint directory."""
    dir_path = Path(target_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    storage = GraphStorage(path=dir_path)
    storage.save(graph, bind=False)


def load_graph_snapshot(config: RUTAMConfig | None, source_dir: str | Path) -> Graph:
    """Load and validate persistent graph snapshot from checkpoint directory."""
    dir_path = Path(source_dir)
    storage = GraphStorage(path=dir_path)
    return storage.load()
