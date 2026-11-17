"""Demonstration of native graph-aware causal language modeling with RUTAM.

Demonstrates:
  Case A: Causal LM forward computation with empty graph (identity preservation).
  Case B: Causal LM forward pass with active graph querying, traversal, and cross-attention fusion.
  Case C: Dynamic memory write loop (proposal -> discrete validation -> atomic commit -> reload).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.graph.storage import GraphStorage
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.write_proposal import MemoryOperation
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


def main() -> None:
    torch.manual_seed(42)
    print("=" * 70)
    print("RUTAM: Complete Native Graph-Aware Causal Language Model Demonstration")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir) / "graph_storage"
        config = RUTAMConfig().with_overrides(
            {
                "graph": {"embedding_dim": 16, "path": str(storage_path)},
                "encoder": {"input_dim": 16, "hidden_dim": 16, "num_relation_types": 4},
                "selector": {"query_dim": 8, "hidden_dim": 16, "score_dim": 8, "top_k": 2},
                "query_head": {
                    "backbone_hidden_dim": 16,
                    "graph_hidden_dim": 16,
                    "query_dim": 8,
                    "intermediate_dim": 16,
                },
                "traversal": {
                    "hidden_dim": 16,
                    "path_hidden_dim": 16,
                    "query_dim": 8,
                    "num_relation_types": 4,
                },
                "graph_fusion": {
                    "hidden_dim": 16,
                    "graph_dim": 16,
                    "num_attention_heads": 2,
                    "residual_scale_init": 0.2,
                },
                "memory_write": {
                    "backbone_hidden_dim": 16,
                    "hidden_dim": 16,
                    "projection_dim": 16,
                    "write_threshold": 0.01,
                },
                "rutam_model": {
                    "return_memory_diagnostics": True,
                },
            }
        )

        storage = GraphStorage(storage_path)
        graph = Graph(config)
        backbone = SyntheticBackbone(hidden_dim=16, vocab_size=100, num_layers=3)
        model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph, storage=storage)

        # -------------------------------------------------------------
        # CASE A: Causal LM Forward with Empty Graph
        # -------------------------------------------------------------
        print("\n--- CASE A: Empty Graph Baseline ---")
        input_ids = torch.randint(0, 100, (1, 8), dtype=torch.long)
        out_a = model(input_ids)

        print(f"Input shape: {input_ids.shape}")
        print(f"Logits shape: {out_a.logits.shape}")
        print(f"Graph nodes present: {out_a.diagnostics.get('has_nodes')}")
        print(f"Graph fusion applied: {out_a.fusion_output is not None}")
        assert out_a.fusion_output is None
        assert not torch.isnan(out_a.logits).any()

        # -------------------------------------------------------------
        # CASE B: Graph-Aware Forward with Active Knowledge Graph
        # -------------------------------------------------------------
        print("\n--- CASE B: Knowledge Graph Memory Integration ---")
        gen = torch.Generator().manual_seed(99)
        graph.create_node(
            node_type="concept",
            content="quantum_computing",
            node_id=1,
            provenance="doc:physics_101",
            embedding=torch.randn(16, generator=gen).tolist(),
        )
        graph.create_node(
            node_type="concept",
            content="qubits",
            node_id=2,
            provenance="doc:physics_101",
            embedding=torch.randn(16, generator=gen).tolist(),
        )
        graph.add_relationship(1, 2, "utilizes", provenance="doc:physics_101")
        storage.save(graph)

        out_b = model(input_ids)
        print(f"Graph nodes present: {out_b.diagnostics.get('has_nodes')}")
        print(f"Graph version: {out_b.graph_version}")
        print(f"Selection active: {out_b.diagnostics.get('selection_active')}")
        print(f"Traversal active: {out_b.diagnostics.get('traversal_active')}")
        print(f"Fusion active: {out_b.diagnostics.get('fusion_active')}")
        diff = torch.norm(out_b.logits - out_a.logits).item()
        print(f"L2 Logit modulation delta from graph fusion: {diff:.6f}")
        assert diff > 0.0

        # -------------------------------------------------------------
        # CASE C: Memory Write Lifecycle (Propose -> Validate -> Commit -> Reload)
        # -------------------------------------------------------------
        print("\n--- CASE C: Neural Memory Write Lifecycle ---")
        with torch.no_grad():
            model.write_head.operation_predictor.linear2.bias[MemoryOperation.CREATE_NODE.value] = (
                10.0
            )
        model.write_head.eval()

        auth_ctx = AuthorizationContext(user_id="research_agent", role="admin")

        # 1. Propose
        proposals = model.propose_memory_updates(input_ids, authorization_context=auth_ctx)
        print(f"Generated {len(proposals)} memory update proposal(s)")
        print(f"First proposal op: {proposals[0].operation.name}, ID: {proposals[0].proposal_id}")

        # 2. Validate
        val_result = model.validate_memory_updates(proposals, authorization_context=auth_ctx)
        accepted = [p for p, r in zip(proposals, val_result.results, strict=True) if r.is_accepted]
        print(
            f"Validation decision: {val_result.accepted_count} accepted, "
            f"{val_result.rejected_count} rejected"
        )
        assert len(accepted) > 0

        # 3. Commit
        commit_res = model.commit_memory_updates(accepted)
        print(
            f"Commit result: committed={commit_res.committed}, "
            f"new graph version={commit_res.new_graph_version}"
        )

        # 4. Storage sync & Reload
        storage.save(graph)
        model.reload_graph()
        print(f"Reloaded graph from storage. Active version: {model.graph.version}")

        # 5. Subsequent Forward Pass
        out_c = model(input_ids)
        print(f"New forward pass graph version: {out_c.graph_version}")
        diff_bc = torch.norm(out_c.logits - out_b.logits).item()
        print(f"L2 Logit delta after committing new memory: {diff_bc:.6f}")
        assert diff_bc > 0.0

    print("\nAll demonstration cases succeeded cleanly!")


if __name__ == "__main__":
    main()
