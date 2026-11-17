"""6 nodes and 8 typed edges; exploratory unverified proposals without graph mutation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rutam import Graph, GraphStorage, RUTAMConfig, RUTAMGraphEncoder, RUTAMGraphLinker


def main() -> None:
    torch.manual_seed(7)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 8},
            "encoder": {"input_dim": 8, "hidden_dim": 8, "dropout": 0},
            "linker": {
                "hidden_dim": 8,
                "scoring_hidden_dim": 16,
                "structural_feature_dim": 4,
                "dropout": 0,
                "relation_labels": ["related_to", "depends_on", "supports"],
                "link_threshold": 0,
                "relation_threshold": 0,
                "max_proposals_per_call": 4,
            },
        }
    )
    graph = Graph(config)
    for row in range(6):
        graph.create_node(
            "synthetic",
            str(row),
            node_id=(row + 1) * 10,
            provenance=f"synthetic:node:{row}",
            embedding=torch.randn(8).tolist(),
        )
    labels = ("related_to", "depends_on", "supports")
    for source, target, relation in (
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (0, 2, 0),
        (3, 4, 1),
        (4, 5, 2),
        (5, 0, 0),
        (1, 4, 1),
    ):
        graph.add_relationship(
            (source + 1) * 10, (target + 1) * 10, labels[relation], provenance="synthetic:edge"
        )
    encoder = RUTAMGraphEncoder(config.encoder).eval()
    linker = RUTAMGraphLinker(config.linker).eval()
    with TemporaryDirectory(prefix="rutam-link-proposals-") as directory:
        storage = GraphStorage(directory)
        storage.save(graph)  # Fixture initialization, before neural proposal generation.
        files = list(Path(directory).glob("graph.*"))
        before = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
        state = storage.load().to_state()
        with torch.no_grad():
            output = linker.from_state(encoder(state), state)
        after = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
        print(
            "Untrained proposals: exploratory thresholds=0, unedited random weights; "
            "predictions need validation."
        )
        print("nodes=6, typed directed edges=8, candidate pairs=", output.candidate_pairs.shape[0])
        for proposal in output.proposal_batches[0]:
            print(
                f"  {proposal.source_node_id} -> {proposal.destination_node_id}: "
                f"{proposal.relation_name} (graph relation ID={proposal.relation_id}), "
                f"confidence={proposal.combined_confidence:.6f}, "
                f"status={proposal.validation_status}, evidence={proposal.evidence_references}"
            )
        print("saved graph unchanged:", before == after)
        print("proposal count:", sum(map(len, output.proposal_batches)))
        assert before == after


if __name__ == "__main__":
    main()
