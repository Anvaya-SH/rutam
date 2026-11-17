"""Small deterministic CPU benchmark, no model download or training.

Run: .venv/bin/python scripts/benchmark_graph_v1.py
Times are current implementation measurements, not optimization claims.
"""

import json
import platform
import resource
import tempfile
import time
from pathlib import Path

import torch

from rutam.diagnostics.integration import create_verification_model
from rutam.graph.storage import GraphStorage


def main():
    torch.manual_seed(73)
    torch.set_num_threads(1)
    measurements = {}

    def timed(name, fn):
        start = time.perf_counter()
        value = fn()
        measurements[name] = (time.perf_counter() - start) * 1000
        return value

    model, graph = timed(
        "initialization_ms",
        lambda: create_verification_model(num_nodes=4, hidden_dim=16, vocab_size=60),
    )
    model.eval()
    state = graph.to_state()
    with torch.no_grad():
        encoded = timed("encoding_ms", lambda: model.encoder(state))
        selected = timed(
            "selection_ms",
            lambda: model.query_head.select_graph(
                torch.ones(1, 4, 16), model.selector, encoded, state.node_ids
            ),
        )
        timed(
            "fusion_ms",
            lambda: model.fusion(
                torch.ones(1, 4, 16),
                selected.selection.selected_states,
                graph_mask=selected.selection.selection_mask,
            ),
        )
        timed("forward_with_graph_ms", lambda: model(torch.tensor([[1, 2, 3, 4]])))
    with tempfile.TemporaryDirectory() as path:
        storage = GraphStorage(path)
        timed("save_ms", lambda: storage.save(graph))
        timed("reload_ms", storage.load)
    from safetensors.torch import save

    timed("tensor_serialization_ms", lambda: save(state.tensors()))
    report = {
        "workload": {
            "nodes": 4,
            "edges": 3,
            "hidden_dim": 16,
            "tokens": 4,
            "batch": 1,
            "threads": 1,
            "repeats": 1,
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "measurements": measurements,
        "limits": ("Single cold measurement; no before/after baseline, no LFM3. "
                   "Peak RSS covers process and imports."),
    }
    out = Path("reports/v1_validation/benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
