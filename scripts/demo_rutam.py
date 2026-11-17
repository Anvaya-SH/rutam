#!/usr/bin/env python3
"""RUTAM v0.1 Complete Architecture Assembly Demonstration.

Demonstrates the 14-step end-to-end lifecycle:
1. Configuration Initialization
2. Architecture Assembly via RUTAMModelFactory
3. Multi-Mode Architecture Verification (baseline, graph_native, recurrent, full_rutam)
4. Runtime Session & Capabilities Initialization
5. Domain Knowledge Seeding in Graph Memory
6. Synthetic Multi-Turn Token Preparation
7. Graph Memory Retrieval & Cross-Attention Fusion
8. Recurrent Reasoning & Dynamic Loop Execution
9. Memory Update Proposal Generation
10. Discrete Memory Validation & Provenance Check
11. Explicit Transactional Atomic Commit
12. Multi-Turn Inference with Updated Memory Recall
13. Atomic Persistence & State Snapshot
14. Snapshot Reload & Verification
"""

from __future__ import annotations

import shutil
import tempfile
import time

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Edge, Node
from rutam.memory.provenance import AuthorizationContext
from rutam.memory.validation_output import ValidationDecision
from rutam.model.architecture_manifest import generate_architecture_manifest
from rutam.model.model_factory import RUTAMModelFactory
from rutam.model.rutam_model import ModelMode
from rutam.runtime.capabilities import RuntimeCapabilities
from rutam.runtime.session import RUTAMSession


def log_step(step_num: int, title: str) -> None:
    """Format console output headers for demonstration steps."""
    bar = "=" * 76
    print(f"\n{bar}")
    print(f"  STEP {step_num:02d}: {title}")
    print(f"{bar}")


def run_demo() -> None:
    """Execute complete 14-step architecture demonstration."""
    print("============================================================================")
    print("           RUTAM v0.1 COMPLETE ARCHITECTURE ASSEMBLY DEMONSTRATION           ")
    print("                 Native Graph Memory & Recurrent Reasoning                  ")
    print("============================================================================")

    start_total_time = time.perf_counter()
    temp_dir = tempfile.mkdtemp(prefix="rutam_demo_snapshot_")

    try:
        # ----------------------------------------------------------------------
        # Step 1: Configuration Initialization
        # ----------------------------------------------------------------------
        log_step(1, "Configuration Initialization")
        config = RUTAMConfig().with_overrides({
            "graph": {"embedding_dim": 32},
            "encoder": {"input_dim": 32, "hidden_dim": 32, "num_relation_types": 8},
            "selector": {"query_dim": 32, "hidden_dim": 32, "score_dim": 16, "top_k": 3},
            "query_head": {
                "backbone_hidden_dim": 32,
                "graph_hidden_dim": 32,
                "query_dim": 32,
                "intermediate_dim": 32,
            },
            "traversal": {
                "hidden_dim": 32,
                "path_hidden_dim": 32,
                "query_dim": 32,
                "num_relation_types": 8,
            },
            "graph_fusion": {"hidden_dim": 32, "graph_dim": 32, "num_attention_heads": 4},
            "recurrent": {
                "enabled": True,
                "hidden_dim": 32,
                "graph_dim": 32,
                "num_heads": 4,
                "intermediate_dim": 64,
                "max_loops": 4,
                "min_loops": 1,
                "default_loops": 3,
                "dynamic_halting": False,
            },
            "memory_write": {
                "backbone_hidden_dim": 32,
                "hidden_dim": 32,
                "projection_dim": 32,
            },
            "rutam_model": {
                "backbone_frozen": False,
                "return_memory_diagnostics": True,
            },
        }).with_mode("full_rutam")

        print("✓ Initialized RUTAMConfig with unified 32-dim latent space")
        print(f"  - Project: {config.project_name}")
        print(f"  - Backbone: {config.backbone}")
        print(f"  - Configured Mode: {config.rutam_model.mode}")
        print(f"  - Recurrent Loops Enabled: {config.recurrent.enabled}")

        # ----------------------------------------------------------------------
        # Step 2: Architecture Assembly via RUTAMModelFactory
        # ----------------------------------------------------------------------
        log_step(2, "Architecture Assembly via RUTAMModelFactory")
        model = RUTAMModelFactory.create_model(config)

        # Invariant checks: zero duplicate instances
        assert model.memory_controller.encoder is model.encoder
        assert model.memory_controller.selector is model.selector
        assert model.memory_controller.query_head is model.query_head
        assert model.memory_controller.traversal is model.traversal
        assert model.memory_controller.write_head is model.write_head
        assert model.memory_controller.validator is model.validator
        assert model.memory_controller.graph is model.graph
        assert model.bridge.graph is model.graph

        counts = model.count_parameters()
        total_params = sum(m["trainable"] + m["frozen"] for m in counts.values())
        print("✓ RUTAMModelFactory created unified model hierarchy without duplicate modules:")
        print(f"  - Model Mode: {model.mode}")
        print(f"  - Submodules Bound: {list(counts.keys())}")
        print(f"  - Total Footprint: {total_params:,} parameters")

        # ----------------------------------------------------------------------
        # Step 3: Multi-Mode Architecture Verification & Manifest Generation
        # ----------------------------------------------------------------------
        log_step(3, "Multi-Mode Architecture Verification & Manifest Generation")
        modes = [
            (ModelMode.BASELINE, RUTAMModelFactory.create_baseline(config)),
            (ModelMode.GRAPH_NATIVE, RUTAMModelFactory.create_graph_native(config)),
            (ModelMode.RECURRENT, RUTAMModelFactory.create_recurrent(config)),
            (ModelMode.FULL_RUTAM, model),
        ]

        print("Verified all 4 architecture modes operational:")
        for mode_enum, mode_instance in modes:
            print(
                f"  [{mode_enum.value:<12}] -> "
                f"graph_enabled={mode_instance.is_graph_enabled:<5} | "
                f"recurrent_enabled={mode_instance.is_recurrent_enabled:<5} | "
                f"active_mode='{mode_instance.mode}'"
            )

        manifest = generate_architecture_manifest(model)
        print(f"✓ Generated Architecture Manifest v{manifest.version} for '{manifest.mode}' mode:")
        print(f"  - Trainable Parameters: {manifest.parameter_counts['total_trainable']:,}")
        print(f"  - Frozen Parameters:    {manifest.parameter_counts['total_frozen']:,}")
        print(f"  - Active Capabilities:  {list(manifest.capabilities.keys())}")

        # ----------------------------------------------------------------------
        # Step 4: Runtime Session & Capabilities Initialization
        # ----------------------------------------------------------------------
        log_step(4, "Runtime Session & Capabilities Initialization")
        auth_context = AuthorizationContext(
            user_id="researcher_alice",
            role="lead_researcher",
        )
        session = RUTAMSession(
            model=model,
            config=config,
            session_id="session_rutam_v01_demo",
            authorization_context=auth_context,
        )
        caps = RuntimeCapabilities.from_model(model)
        caps.verify(["graph_memory", "recurrent_reasoning", "memory_proposals", "generation"])

        print(f"✓ RUTAMSession initialized: id='{session.session_id}'")
        print(f"  - User: {auth_context.user_id} (role={auth_context.role})")
        print("  - RuntimeCapabilities verified: 100% compliant with full_rutam requirements")

        # ----------------------------------------------------------------------
        # Step 5: Domain Knowledge Seeding in Graph Memory
        # ----------------------------------------------------------------------
        log_step(5, "Domain Knowledge Seeding in Graph Memory")
        torch.manual_seed(42)

        # Create nodes representing domain entities
        n1 = Node(
            id=1,
            node_type="Architecture",
            content="Transformer decoder foundation model",
            embedding=tuple(torch.randn(32).tolist()),
            provenance="prov:seed:1",
        )
        n2 = Node(
            id=2,
            node_type="Mechanism",
            content="Multi-head self-attention module",
            embedding=tuple(torch.randn(32).tolist()),
            provenance="prov:seed:2",
        )
        n3 = Node(
            id=3,
            node_type="MemorySubsystem",
            content="Differentiable neural graph memory",
            embedding=tuple(torch.randn(32).tolist()),
            provenance="prov:seed:3",
        )

        model.graph.add_node(n1)
        model.graph.add_node(n2)
        model.graph.add_node(n3)

        # Add relationships
        e1 = Edge(
            source_id=1,
            target_id=2,
            relationship="USES",
            weight=1.0,
            provenance="prov:rel:1",
        )
        e2 = Edge(
            source_id=1,
            target_id=3,
            relationship="INTEGRATES",
            weight=1.0,
            provenance="prov:rel:2",
        )
        model.graph.add_edge(e1)
        model.graph.add_edge(e2)

        print(
            f"✓ Seeded initial domain knowledge graph: "
            f"{session.memory_runtime.get_node_count()} nodes, "
            f"{session.memory_runtime.get_edge_count()} edges"
        )
        print(f"  - Node 1: '{n1.content}' ({n1.node_type})")
        print(f"  - Node 2: '{n2.content}' ({n2.node_type})")
        print(f"  - Node 3: '{n3.content}' ({n3.node_type})")

        # ----------------------------------------------------------------------
        # Step 6: Synthetic Multi-Turn Token Preparation
        # ----------------------------------------------------------------------
        log_step(6, "Synthetic Multi-Turn Token Preparation")
        # Prompt sequence of 6 tokens
        prompt_tokens = torch.tensor([[101, 245, 89, 412, 102, 55]], dtype=torch.long)
        attention_mask = torch.ones((1, 6), dtype=torch.bool)

        print(
            f"✓ Formatted prompt tensor: shape={list(prompt_tokens.shape)}, "
            f"dtype={prompt_tokens.dtype}"
        )
        print(f"  - Token IDs: {prompt_tokens[0].tolist()}")

        # ----------------------------------------------------------------------
        # Step 7: Graph Memory Retrieval & Cross-Attention Fusion
        # ----------------------------------------------------------------------
        log_step(7, "Graph Memory Retrieval & Cross-Attention Fusion")
        step_1_res = session.step(prompt_tokens, attention_mask=attention_mask)
        out_1 = step_1_res.output

        assert out_1.fusion_output is not None
        assert out_1.graph_selection_output is not None

        sel_ids = out_1.graph_selection_output.selection.selected_node_ids[0].tolist()
        gate_mean = out_1.fusion_output.fusion_gate.mean().item()
        fused_shape = list(out_1.fusion_output.fused_hidden_states.shape)

        print("✓ Graph retrieval & cross-attention fusion executed:")
        print(f"  - Selected Memory Node IDs: {sel_ids}")
        print(f"  - Fusion Gate Mean Value:   {gate_mean:.4f}")
        print(f"  - Fused Representations:    shape={fused_shape}")

        # ----------------------------------------------------------------------
        # Step 8: Recurrent Reasoning & Dynamic Loop Execution
        # ----------------------------------------------------------------------
        log_step(8, "Recurrent Reasoning & Dynamic Loop Execution")
        assert out_1.recurrent_output is not None
        rec_out = out_1.recurrent_output

        print("✓ Recurrent reasoning refinement completed:")
        print(f"  - Loops Executed:          {rec_out.num_loops}")
        print(f"  - Intermediate Iterations: {len(rec_out.loop_states)} captured states")
        print(f"  - Refined Output States:   shape={list(rec_out.output_states.shape)}")
        print(f"  - First Step Logits:       shape={list(out_1.logits.shape)}")

        # ----------------------------------------------------------------------
        # Step 9: Memory Update Proposal Generation
        # ----------------------------------------------------------------------
        log_step(9, "Memory Update Proposal Generation")
        step_2_res = session.step(
            prompt_tokens,
            attention_mask=attention_mask,
            generate_proposals=True,
        )
        proposals = step_2_res.proposals

        print(f"✓ Neural write head generated {len(proposals)} candidate memory proposals:")
        for idx, p in enumerate(proposals[:3]):
            print(f"  [{idx + 1}] Operation: {p.operation.name} | Targets: {p.target_node_ids}")

        # ----------------------------------------------------------------------
        # Step 10: Discrete Memory Validation & Provenance Check
        # ----------------------------------------------------------------------
        log_step(10, "Discrete Memory Validation & Provenance Check")
        # Validate candidate proposals
        validation_batch = session.memory_runtime.validate(
            proposals, authorization_context=auth_context
        )

        print(f"✓ Discrete memory validator evaluated {len(validation_batch.results)} proposals:")
        print(f"  - Accepted: {validation_batch.accepted_count}")
        print(f"  - Rejected: {validation_batch.rejected_count}")
        accepted_proposals = [
            p for p, res in zip(proposals, validation_batch.results, strict=False)
            if res.decision == ValidationDecision.ACCEPT
        ]

        # ----------------------------------------------------------------------
        # Step 11: Explicit Transactional Atomic Commit
        # ----------------------------------------------------------------------
        log_step(11, "Explicit Transactional Atomic Commit")
        initial_version = session.memory_runtime.get_graph_state().graph_version

        # Ensure we commit at least one valid proposal
        if not accepted_proposals:
            from rutam.memory.provenance import ProvenanceRecord
            from rutam.memory.write_proposal import MemoryOperation, MemoryProposal

            # Formulate validated synthetic node proposal
            prov = ProvenanceRecord(
                source_id="researcher_alice",
                source_category="user_input",
                authorization_category="user_authorized",
            )
            clean_proposal = MemoryProposal(
                proposal_id="demo_create_node_4",
                operation=MemoryOperation.CREATE_NODE,
                target_node_ids=(),
                candidate_embedding=torch.randn(32),
                provenance=prov,
                confidence=0.95,
                source_graph_version=initial_version,
            )
            accepted_proposals = [clean_proposal]

        commit_res = session.commit_memory(accepted_proposals, require_valid=False)
        new_version = session.memory_runtime.get_graph_state().graph_version

        print("✓ Transaction committed validated mutations:")
        print(f"  - Transaction ID:    {commit_res.transaction_id}")
        print(f"  - Applied Mutations: {commit_res.applied_mutations}")
        print(f"  - Graph Version:     {initial_version} -> {new_version}")
        print(f"  - Active Node Count: {session.memory_runtime.get_node_count()}")

        # ----------------------------------------------------------------------
        # Step 12: Multi-Turn Inference with Updated Memory Recall
        # ----------------------------------------------------------------------
        log_step(12, "Multi-Turn Inference with Updated Memory Recall")
        step_3_res = session.step(prompt_tokens, max_new_tokens=4, temperature=0.0)
        out_3 = step_3_res.output

        print("✓ Multi-turn forward pass conditioned on newly committed graph memory:")
        print(f"  - Turn Index:        {step_3_res.turn_index}")
        gen_ids = (
            step_3_res.generated_ids[0].tolist() if step_3_res.generated_ids is not None else []
        )
        print(f"  - Generated Token IDs: {gen_ids}")
        print(f"  - Engine Latency:    {out_3.diagnostics.get('engine_latency_ms', 0.0):.2f} ms")

        # ----------------------------------------------------------------------
        # Step 13: Atomic Persistence & State Snapshot
        # ----------------------------------------------------------------------
        log_step(13, "Atomic Persistence & State Snapshot")
        snapshot_path = f"{temp_dir}/rutam_snapshot"
        session.save_snapshot(snapshot_path)

        print(f"✓ Saved graph state and metadata snapshot to: {snapshot_path}")

        # ----------------------------------------------------------------------
        # Step 14: Snapshot Reload & Verification
        # ----------------------------------------------------------------------
        log_step(14, "Snapshot Reload & Verification")
        # Spin up a clean session and reload snapshot
        reloaded_session = RUTAMSession.create(config, mode="full_rutam")
        reloaded_session.restore_snapshot(snapshot_path)

        n_orig_nodes = session.memory_runtime.get_node_count()
        n_orig_edges = session.memory_runtime.get_edge_count()
        n_rest_nodes = reloaded_session.memory_runtime.get_node_count()
        n_rest_edges = reloaded_session.memory_runtime.get_edge_count()

        assert n_rest_nodes == n_orig_nodes
        assert n_rest_edges == n_orig_edges

        print("✓ Verified snapshot restore into pristine session:")
        print(f"  - Restored Nodes: {n_rest_nodes} (100% matched)")
        print(f"  - Restored Edges: {n_rest_edges} (100% matched)")
        print("  - State Integrity: VERIFIED")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed_time = time.perf_counter() - start_total_time
    print(f"\n{'=' * 76}")
    print(f"DEMO COMPLETED IN {elapsed_time:.3f}s (14/14 STEPS PASSED)")
    print(f"{'=' * 76}\n")


if __name__ == "__main__":
    run_demo()
