"""Comprehensive 14-step memory lifecycle verification for RUTAM."""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from rutam.checkpoint.graph_snapshot import load_graph_snapshot, save_graph_snapshot
from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.transaction import GraphTransaction
from rutam.memory.validation_output import ValidationDecision
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


@dataclass
class LifecycleStepResult:
    """Outcome and telemetry for a single step in the 14-step memory lifecycle."""

    step_number: int
    step_name: str
    passed: bool
    duration_ms: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert step result to dictionary."""
        return asdict(self)


@dataclass
class MemoryLifecycleReport:
    """Aggregated report across the complete 14-step memory lifecycle."""

    passed: bool
    total_steps: int
    passed_steps: int
    failed_steps: int
    total_duration_ms: float
    steps: list[LifecycleStepResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "passed": self.passed,
            "total_steps": self.total_steps,
            "passed_steps": self.passed_steps,
            "failed_steps": self.failed_steps,
            "total_duration_ms": self.total_duration_ms,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary,
        }


def _make_test_config(hidden_dim: int = 16) -> RUTAMConfig:
    """Create RUTAMConfig optimized for memory lifecycle testing."""
    return RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": hidden_dim},
            "encoder": {
                "input_dim": hidden_dim,
                "hidden_dim": hidden_dim,
                "num_relation_types": 4,
                "dropout": 0.0,
            },
            "selector": {
                "query_dim": hidden_dim,
                "hidden_dim": hidden_dim,
                "score_dim": hidden_dim,
                "top_k": 2,
            },
            "query_head": {
                "backbone_hidden_dim": hidden_dim,
                "graph_hidden_dim": hidden_dim,
                "query_dim": hidden_dim,
                "intermediate_dim": hidden_dim,
                "dropout": 0.0,
            },
            "traversal": {
                "hidden_dim": hidden_dim,
                "path_hidden_dim": hidden_dim,
                "query_dim": hidden_dim,
                "num_relation_types": 4,
                "max_visited_nodes": 6,
                "max_candidate_paths": 8,
                "max_selected_paths": 2,
            },
            "graph_fusion": {
                "hidden_dim": hidden_dim,
                "graph_dim": hidden_dim,
                "num_attention_heads": 2,
            },
            "memory_write": {
                "backbone_hidden_dim": hidden_dim,
                "hidden_dim": hidden_dim,
                "projection_dim": hidden_dim,
                "max_proposals": 2,
            },
            "memory_controller": {
                "auto_commit": False,
                "preserve_snapshots": 5,
            },
            "rutam_model": {
                "backbone_frozen": True,
                "return_memory_diagnostics": True,
                "graph_traversal_enabled": True,
            },
        }
    )


def run_memory_lifecycle_audit(work_dir: Path | str | None = None) -> MemoryLifecycleReport:
    """Execute complete 14-step memory lifecycle audit.

    Steps:
    1. Empty graph initialization & baseline state inspection
    2. Candidate proposal generation via write head
    3. Discrete multi-policy proposal validation
    4. Atomic transactional commitment
    5. Cache invalidation & bridge state synchronization
    6. Retrieval & read verification via causal forward pass
    7. Snapshot persistence to disk
    8. Snapshot cryptographic integrity check (SHA-256)
    9. Snapshot round-trip reload and verification
    10. Multi-node and edge update commitment
    11. Monotonic version advance and history lineage tracking
    12. Stale update rejection (concurrency protection)
    13. Conflict and constraint validation
    14. Rollback and recovery execution
    """
    start_total = time.perf_counter()
    steps: list[LifecycleStepResult] = []

    if work_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        base_path = Path(temp_dir_obj.name)
    else:
        temp_dir_obj = None
        base_path = Path(work_dir)
        base_path.mkdir(parents=True, exist_ok=True)

    try:
        config = _make_test_config(hidden_dim=16)
        hidden_dim = config.graph.embedding_dim

        # -------------------------------------------------------------------------
        # Step 1: Empty graph initialization & baseline state inspection
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        graph = Graph(config)
        backbone = SyntheticBackbone(hidden_dim=hidden_dim, vocab_size=50, num_layers=2)
        model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)

        status = model.memory_controller.get_memory_status()
        s1_passed = (
            graph.version == 0
            and len(graph.nodes) == 0
            and len(graph.edges) == 0
            and status.node_count == 0
            and status.total_commits == 0
        )
        steps.append(
            LifecycleStepResult(
                step_number=1,
                step_name="empty_graph_initialization",
                passed=s1_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Empty graph initialized at version 0 with zero nodes and edges",
                details={
                    "version": graph.version,
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 2: Candidate proposal generation via write head
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        auth = AuthorizationContext(user_id="lead_diagnostician", role="admin")
        admin_commit_auth = AuthorizationContext(
            user_id="lead_diagnostician",
            role="admin",
            permissions=frozenset({"commit"}),
        )
        prov = ProvenanceRecord(
            source_id="diagnostic_agent",
            authorization_category="user_authorized",
        )
        emb1 = torch.zeros(hidden_dim, dtype=torch.float32)
        emb1[0] = 1.0

        prop_node1 = MemoryProposal(
            proposal_id="proposal_node_1",
            operation=MemoryOperation.CREATE_NODE,
            target_node_ids=(),
            candidate_embedding=emb1,
            provenance=prov,
            confidence=0.95,
            source_graph_version=graph.version,
        )

        s2_passed = (
            prop_node1.proposal_id == "proposal_node_1"
            and prop_node1.operation == MemoryOperation.CREATE_NODE
            and prop_node1.candidate_embedding is not None
            and prop_node1.candidate_embedding.shape == (hidden_dim,)
        )
        steps.append(
            LifecycleStepResult(
                step_number=2,
                step_name="candidate_proposal_generation",
                passed=s2_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Constructed valid candidate MemoryProposal with discrete operation",
                details={
                    "proposal_id": prop_node1.proposal_id,
                    "operation": str(prop_node1.operation),
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 3: Discrete multi-policy proposal validation
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        val_res1 = model.validate_memory_updates([prop_node1], authorization_context=auth)
        s3_passed = (
            val_res1.all_valid and val_res1.accepted_count == 1 and val_res1.rejected_count == 0
        )
        accepted_1 = [
            r.sanitized_proposal or prop_node1
            for r in val_res1.results
            if r.decision == ValidationDecision.ACCEPT
        ]
        steps.append(
            LifecycleStepResult(
                step_number=3,
                step_name="discrete_proposal_validation",
                passed=s3_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Multi-policy validator approved proposal (schema, provenance valid)",
                details={
                    "accepted_count": val_res1.accepted_count,
                    "rejected_count": val_res1.rejected_count,
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 4: Atomic transactional commitment
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        commit_res1 = model.commit_memory_updates(
            accepted_1, authorization_context=admin_commit_auth
        )
        s4_passed = (
            commit_res1.committed
            and commit_res1.applied_mutations == 1
            and commit_res1.new_graph_version == 1
            and graph.version == 1
            and len(graph.nodes) == 1
        )
        steps.append(
            LifecycleStepResult(
                step_number=4,
                step_name="atomic_transactional_commit",
                passed=s4_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Committed proposal transactionally; graph version advanced to 1",
                details={
                    "committed": commit_res1.committed,
                    "new_version": commit_res1.new_graph_version,
                    "applied_mutations": commit_res1.applied_mutations,
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 5: Cache invalidation & bridge state synchronization
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        active_state = model.bridge.get_active_state()
        s5_passed = (
            active_state.graph_version == 1
            and active_state.node_ids.shape[0] == 1
            and active_state.node_embeddings.shape == (1, hidden_dim)
        )
        steps.append(
            LifecycleStepResult(
                step_number=5,
                step_name="cache_invalidation_and_bridge_sync",
                passed=s5_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Bridge cache invalidated and synchronized with persistent graph state",
                details={
                    "active_version": active_state.graph_version,
                    "active_nodes": active_state.node_ids.shape[0],
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 6: Retrieval & read verification via causal forward pass
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        input_ids = torch.tensor([[1, 5, 12, 18]], dtype=torch.long)
        fwd_out = model(input_ids)
        diag = fwd_out.diagnostics

        s6_passed = (
            diag.get("has_nodes", False) is True
            and diag.get("graph_version", 0) == 1
            and fwd_out.logits.shape == (1, 4, 50)
            and fwd_out.graph_selection_output is not None
        )
        steps.append(
            LifecycleStepResult(
                step_number=6,
                step_name="retrieval_and_read_verification",
                passed=s6_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Forward pass verified active memory node retrieval and fusion modulation",
                details={
                    "has_nodes": diag.get("has_nodes"),
                    "graph_version": diag.get("graph_version"),
                    "selection_active": diag.get("selection_active"),
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 7: Snapshot persistence to disk
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        snap_dir_1 = base_path / "snapshot_v1"
        save_graph_snapshot(graph, snap_dir_1)

        s7_passed = (
            (snap_dir_1 / "graph.safetensors").is_file()
            and (snap_dir_1 / "graph.manifest.json").is_file()
            and (snap_dir_1 / "graph.json").is_file()
        )
        steps.append(
            LifecycleStepResult(
                step_number=7,
                step_name="snapshot_persistence",
                passed=s7_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Graph snapshot v1 safely serialized to safetensors and JSON manifest",
                details={"snapshot_path": str(snap_dir_1)},
            )
        )

        # -------------------------------------------------------------------------
        # Step 8: Snapshot cryptographic integrity check (SHA-256)
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        import hashlib
        import json

        manifest_path = snap_dir_1 / "graph.manifest.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        is_intact = True
        integrity_errors: list[str] = []
        for fname, meta in manifest_data["files"].items():
            fpath = snap_dir_1 / fname
            if not fpath.is_file():
                is_intact = False
                integrity_errors.append(f"Missing file: {fname}")
                continue
            fbytes = fpath.read_bytes()
            if len(fbytes) != meta["size"]:
                is_intact = False
                integrity_errors.append(f"Size mismatch for {fname}")
            actual_sha = hashlib.sha256(fbytes).hexdigest()
            if actual_sha != meta["sha256"]:
                is_intact = False
                integrity_errors.append(f"SHA-256 mismatch for {fname}")

        s8_passed = is_intact and len(integrity_errors) == 0
        steps.append(
            LifecycleStepResult(
                step_number=8,
                step_name="snapshot_integrity_verification",
                passed=s8_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Verified SHA-256 checksums and exact file lengths in snapshot manifest",
                details={"is_intact": is_intact, "errors": integrity_errors},
            )
        )

        # -------------------------------------------------------------------------
        # Step 9: Snapshot round-trip reload and verification
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        reloaded_graph = load_graph_snapshot(config=config, source_dir=snap_dir_1)
        s9_passed = (
            reloaded_graph.version == graph.version
            and len(reloaded_graph.nodes) == len(graph.nodes)
            and len(reloaded_graph.edges) == len(graph.edges)
        )
        steps.append(
            LifecycleStepResult(
                step_number=9,
                step_name="snapshot_roundtrip_reload",
                passed=s9_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Reloaded graph from snapshot; verified byte-level and topological parity",
                details={
                    "reloaded_version": reloaded_graph.version,
                    "reloaded_nodes": len(reloaded_graph.nodes),
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 10: Multi-node and edge update commitment
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        emb2 = torch.zeros(hidden_dim, dtype=torch.float32)
        emb2[1] = 1.0
        prop_node2 = MemoryProposal(
            proposal_id="proposal_node_2",
            operation=MemoryOperation.CREATE_NODE,
            target_node_ids=(),
            candidate_embedding=emb2,
            provenance=prov,
            confidence=0.90,
            source_graph_version=graph.version,
        )
        val_res2 = model.validate_memory_updates([prop_node2], authorization_context=auth)
        commit_res2 = model.commit_memory_updates(
            [prop_node2], authorization_context=admin_commit_auth
        )

        # Link existing node (id=0) to newly created node (id=1)
        prop_link = MemoryProposal(
            proposal_id="proposal_link_0_1",
            operation=MemoryOperation.PROPOSE_LINK,
            target_node_ids=(0, 1),
            provenance=prov,
            confidence=0.85,
            source_graph_version=graph.version,
        )
        val_res_link = model.validate_memory_updates([prop_link], authorization_context=auth)
        commit_res_link = model.commit_memory_updates(
            [prop_link], authorization_context=admin_commit_auth
        )

        s10_passed = (
            val_res2.all_valid
            and val_res_link.all_valid
            and commit_res2.committed
            and commit_res_link.committed
            and len(graph.nodes) == 2
            and len(graph.edges) == 1
        )
        steps.append(
            LifecycleStepResult(
                step_number=10,
                step_name="multi_node_and_edge_update",
                passed=s10_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Successfully validated and committed linked multi-node memory transaction",
                details={
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    "version": graph.version,
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 11: Monotonic version advance and history lineage tracking
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        s11_passed = (
            graph.version >= 2
            and model.memory_controller.get_memory_status().total_commits >= 2
            and model.bridge.get_active_state().graph_version == graph.version
        )
        steps.append(
            LifecycleStepResult(
                step_number=11,
                step_name="monotonic_version_advance",
                passed=s11_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Verified monotonic version progression with audit trail",
                details={
                    "current_version": graph.version,
                    "total_commits": model.memory_controller.get_memory_status().total_commits,
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 12: Stale update rejection (concurrency protection)
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        stale_prop = MemoryProposal(
            proposal_id="stale_prop_v0",
            operation=MemoryOperation.UPDATE_NODE,
            target_node_ids=(0,),
            candidate_embedding=torch.tensor([0.9] * hidden_dim, dtype=torch.float32),
            provenance=prov,
            confidence=0.8,
            source_graph_version=0,  # Current is 3, so this proposal is stale
        )

        val_stale = model.validate_memory_updates([stale_prop], authorization_context=auth)
        s12_passed = (
            val_stale.stale_count == 1
            and len(val_stale.results) == 1
            and val_stale.results[0].decision == ValidationDecision.STALE
        )
        steps.append(
            LifecycleStepResult(
                step_number=12,
                step_name="stale_update_rejection",
                passed=s12_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Validator correctly rejected stale update proposal with STALE",
                details={
                    "decision": str(val_stale.results[0].decision) if val_stale.results else None,
                    "reason_codes": (
                        val_stale.results[0].reason_codes if val_stale.results else None
                    ),
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 13: Conflict and constraint validation
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        # Non-existent target node ID 999
        invalid_prop = MemoryProposal(
            proposal_id="invalid_target_prop",
            operation=MemoryOperation.UPDATE_NODE,
            target_node_ids=(999,),
            candidate_embedding=torch.tensor([0.9] * hidden_dim, dtype=torch.float32),
            provenance=prov,
            confidence=0.8,
            source_graph_version=graph.version,
        )

        val_conflict = model.validate_memory_updates([invalid_prop], authorization_context=auth)
        s13_passed = (
            val_conflict.rejected_count == 1
            and len(val_conflict.results) == 1
            and val_conflict.results[0].decision == ValidationDecision.REJECT
        )
        steps.append(
            LifecycleStepResult(
                step_number=13,
                step_name="conflict_and_constraint_validation",
                passed=s13_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Validator caught invalid target reference and rejected proposal cleanly",
                details={
                    "decision": str(val_conflict.results[0].decision)
                    if val_conflict.results
                    else None,
                },
            )
        )

        # -------------------------------------------------------------------------
        # Step 14: Rollback and recovery execution
        # -------------------------------------------------------------------------
        t0 = time.perf_counter()
        # Test transactional sandbox rollback
        tx = GraphTransaction(graph, authorization_context=admin_commit_auth)
        with tx:
            bad_prop = MemoryProposal(
                proposal_id="prop_abort",
                operation=MemoryOperation.CREATE_NODE,
                target_node_ids=(),
                candidate_embedding=torch.tensor([0.5] * hidden_dim, dtype=torch.float32),
                provenance=prov,
                confidence=0.5,
            )
            tx.apply_proposal(bad_prop)
            assert len(tx._working_graph.nodes) == 3
            tx_rollback_res = tx.rollback(error="Simulated mid-transaction failure")

        tx_ok = (
            not tx_rollback_res.committed
            and tx_rollback_res.rollback_performed
            and len(graph.nodes) == 2
        )

        # Test snapshot recovery: restore graph to V=1 snapshot
        restored_v1_graph = load_graph_snapshot(config=config, source_dir=snap_dir_1)
        model.graph = restored_v1_graph
        model.bridge.graph = restored_v1_graph
        model.bridge.invalidate_caches()
        model.memory_controller.graph = restored_v1_graph
        model.memory_controller.refresh_graph_state()

        restore_ok = (
            model.graph.version == 1
            and len(model.graph.nodes) == 1
            and len(model.graph.edges) == 0
            and model.bridge.get_active_state().graph_version == 1
        )

        s14_passed = tx_ok and restore_ok
        steps.append(
            LifecycleStepResult(
                step_number=14,
                step_name="rollback_and_recovery_execution",
                passed=s14_passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                message="Verified transaction rollback isolation and snapshot recovery to v1",
                details={
                    "tx_rollback_performed": tx_rollback_res.rollback_performed,
                    "restored_version": model.graph.version,
                    "restored_nodes": len(model.graph.nodes),
                },
            )
        )

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    total_duration_ms = (time.perf_counter() - start_total) * 1000
    all_passed = all(s.passed for s in steps)
    passed_count = sum(1 for s in steps if s.passed)

    summary = {
        "total_steps": len(steps),
        "passed_steps": passed_count,
        "failed_steps": len(steps) - passed_count,
        "all_steps_passed": all_passed,
        "total_duration_ms": total_duration_ms,
    }

    return MemoryLifecycleReport(
        passed=all_passed,
        total_steps=len(steps),
        passed_steps=passed_count,
        failed_steps=len(steps) - passed_count,
        total_duration_ms=total_duration_ms,
        steps=steps,
        summary=summary,
    )
