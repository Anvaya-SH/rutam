"""Tensor contract verification across model, graph, and memory interfaces in RUTAM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from rutam.config import RUTAMConfig
from rutam.graph.operations import Graph
from rutam.graph.state import GraphState
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM

if TYPE_CHECKING:
    from rutam.memory.provenance import AuthorizationContext


@dataclass
class ContractCheckResult:
    """Individual tensor contract check outcome."""

    name: str
    passed: bool
    message: str
    actual_shape: tuple[int, ...] | None = None
    expected_shape: tuple[int | None, ...] | str | None = None
    actual_dtype: str | None = None
    expected_dtype: str | None = None
    actual_device: str | None = None
    expected_device: str | None = None
    is_contiguous: bool | None = None
    is_finite: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)


@dataclass
class TensorContractReport:
    """Aggregated contract verification report covering interfaces and state."""

    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    checks: list[ContractCheckResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "passed": self.passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "checks": [c.to_dict() for c in self.checks],
            "errors": list(self.errors),
        }


def verify_tensor_contract(
    tensor: torch.Tensor | None,
    name: str,
    *,
    expected_shape: tuple[int | None, ...] | None = None,
    expected_dtype: torch.dtype | tuple[torch.dtype, ...] | None = None,
    expected_device: str = "cpu",
    allow_none: bool = False,
    check_contiguous: bool = True,
    check_finite: bool = True,
    value_range: tuple[float | None, float | None] | None = None,
) -> ContractCheckResult:
    """Verify that a tensor adheres strictly to shape, dtype, device, and layout contracts."""
    if tensor is None:
        if allow_none:
            return ContractCheckResult(
                name=name,
                passed=True,
                message=f"{name} is None (allowed)",
            )
        return ContractCheckResult(
            name=name,
            passed=False,
            message=f"{name} is None but contract requires a valid tensor",
        )

    if not isinstance(tensor, torch.Tensor):
        return ContractCheckResult(
            name=name,
            passed=False,
            message=f"{name} expected torch.Tensor, got {type(tensor).__name__}",
        )

    actual_shape = tuple(tensor.shape)
    actual_dtype = str(tensor.dtype)
    actual_device = tensor.device.type
    is_contiguous = tensor.is_contiguous()

    # Shape validation
    if expected_shape is not None:
        if len(actual_shape) != len(expected_shape):
            return ContractCheckResult(
                name=name,
                passed=False,
                message=(
                    f"{name} shape dimension mismatch: expected rank {len(expected_shape)} "
                    f"({expected_shape}), got rank {len(actual_shape)} ({actual_shape})"
                ),
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                actual_dtype=actual_dtype,
                actual_device=actual_device,
                is_contiguous=is_contiguous,
            )

        for dim_idx, (act_dim, exp_dim) in enumerate(
            zip(actual_shape, expected_shape, strict=False)
        ):
            if exp_dim is not None and exp_dim != -1 and act_dim != exp_dim:
                return ContractCheckResult(
                    name=name,
                    passed=False,
                    message=(
                        f"{name} dim {dim_idx} mismatch: expected {exp_dim}, got {act_dim} "
                        f"(full actual: {actual_shape}, expected: {expected_shape})"
                    ),
                    actual_shape=actual_shape,
                    expected_shape=expected_shape,
                    actual_dtype=actual_dtype,
                    actual_device=actual_device,
                    is_contiguous=is_contiguous,
                )

    # Dtype validation
    if expected_dtype is not None:
        allowed_dtypes = expected_dtype if isinstance(expected_dtype, tuple) else (expected_dtype,)
        if tensor.dtype not in allowed_dtypes:
            exp_str = ", ".join(str(d) for d in allowed_dtypes)
            return ContractCheckResult(
                name=name,
                passed=False,
                message=f"{name} dtype mismatch: expected {exp_str}, got {tensor.dtype}",
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                actual_dtype=actual_dtype,
                expected_dtype=exp_str,
                actual_device=actual_device,
                is_contiguous=is_contiguous,
            )

    # Device validation
    if expected_device is not None and tensor.device.type != expected_device:
        return ContractCheckResult(
            name=name,
            passed=False,
            message=(
                f"{name} device mismatch: expected {expected_device}, got {tensor.device.type}"
            ),
            actual_shape=actual_shape,
            expected_shape=expected_shape,
            actual_dtype=actual_dtype,
            actual_device=actual_device,
            expected_device=expected_device,
            is_contiguous=is_contiguous,
        )

    # Contiguity validation
    if check_contiguous and not is_contiguous:
        return ContractCheckResult(
            name=name,
            passed=False,
            message=f"{name} layout violation: tensor is not memory-contiguous",
            actual_shape=actual_shape,
            expected_shape=expected_shape,
            actual_dtype=actual_dtype,
            actual_device=actual_device,
            is_contiguous=False,
        )

    # Finiteness validation (for float dtypes)
    is_finite: bool | None = None
    if check_finite and tensor.is_floating_point():
        is_finite = bool(torch.isfinite(tensor).all().item())
        if not is_finite:
            return ContractCheckResult(
                name=name,
                passed=False,
                message=f"{name} numerical violation: contains NaN or Inf values",
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                actual_dtype=actual_dtype,
                actual_device=actual_device,
                is_contiguous=is_contiguous,
                is_finite=False,
            )

    # Value range validation
    if value_range is not None and tensor.numel() > 0:
        min_val, max_val = value_range
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        if min_val is not None and t_min < min_val:
            return ContractCheckResult(
                name=name,
                passed=False,
                message=(
                    f"{name} value range violation: min value {t_min:.4f} < lower bound {min_val}"
                ),
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                actual_dtype=actual_dtype,
                actual_device=actual_device,
                is_contiguous=is_contiguous,
                is_finite=is_finite,
            )
        if max_val is not None and t_max > max_val:
            return ContractCheckResult(
                name=name,
                passed=False,
                message=(
                    f"{name} value range violation: max value {t_max:.4f} > upper bound {max_val}"
                ),
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                actual_dtype=actual_dtype,
                actual_device=actual_device,
                is_contiguous=is_contiguous,
                is_finite=is_finite,
            )

    return ContractCheckResult(
        name=name,
        passed=True,
        message=(
            f"{name} satisfied contract "
            f"(shape={actual_shape}, dtype={actual_dtype}, device={actual_device})"
        ),
        actual_shape=actual_shape,
        expected_shape=expected_shape,
        actual_dtype=actual_dtype,
        actual_device=actual_device,
        is_contiguous=is_contiguous,
        is_finite=is_finite,
    )


def verify_graph_contracts(state: GraphState) -> TensorContractReport:
    """Verify that a GraphState object adheres to tensor contracts."""
    checks: list[ContractCheckResult] = []
    errors: list[str] = []

    # 1. node_ids
    checks.append(
        verify_tensor_contract(
            state.node_ids,
            "GraphState.node_ids",
            expected_shape=(None,),
            expected_dtype=torch.long,
            expected_device="cpu",
            check_contiguous=True,
        )
    )

    # 2. node_embeddings
    n_nodes = state.node_ids.shape[0]
    checks.append(
        verify_tensor_contract(
            state.node_embeddings,
            "GraphState.node_embeddings",
            expected_shape=(n_nodes, None),
            expected_dtype=torch.float32,
            expected_device="cpu",
            check_contiguous=True,
            check_finite=True,
        )
    )

    # 3. edge_index
    checks.append(
        verify_tensor_contract(
            state.edge_index,
            "GraphState.edge_index",
            expected_shape=(2, None),
            expected_dtype=torch.long,
            expected_device="cpu",
            check_contiguous=True,
        )
    )

    # 4. edge_types
    n_edges = state.edge_index.shape[1]
    checks.append(
        verify_tensor_contract(
            state.edge_types,
            "GraphState.edge_types",
            expected_shape=(n_edges,),
            expected_dtype=torch.long,
            expected_device="cpu",
            check_contiguous=True,
        )
    )

    # 5. edge_weights
    checks.append(
        verify_tensor_contract(
            state.edge_weights,
            "GraphState.edge_weights",
            expected_shape=(n_edges,),
            expected_dtype=torch.float32,
            expected_device="cpu",
            check_contiguous=True,
            check_finite=True,
            value_range=(0.0, 1.0),
        )
    )

    # Graph index invariant check: edge_index entries must be valid indices < n_nodes
    if n_edges > 0 and n_nodes > 0:
        max_idx = state.edge_index.max().item()
        min_idx = state.edge_index.min().item()
        if min_idx < 0 or max_idx >= n_nodes:
            chk = ContractCheckResult(
                name="GraphState.edge_index_bounds",
                passed=False,
                message=f"edge_index bounds violation: [{min_idx}, {max_idx}] with {n_nodes} nodes",
            )
        else:
            chk = ContractCheckResult(
                name="GraphState.edge_index_bounds",
                passed=True,
                message=f"edge_index in valid bounds [0, {n_nodes - 1}]",
            )
        checks.append(chk)

    for c in checks:
        if not c.passed:
            errors.append(c.message)

    passed_count = sum(1 for c in checks if c.passed)
    return TensorContractReport(
        passed=len(errors) == 0,
        total_checks=len(checks),
        passed_checks=passed_count,
        failed_checks=len(errors),
        checks=checks,
        errors=errors,
    )


def verify_forward_contracts(
    model: RUTAMForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    memory_available: torch.Tensor | None = None,
    generate_proposals: bool = False,
    authorization_context: AuthorizationContext | None = None,
) -> TensorContractReport:
    """Verify input, intermediate, and output tensor contracts across the forward pass."""
    checks: list[ContractCheckResult] = []
    errors: list[str] = []

    b, t = input_ids.shape
    cfg = model.config
    hidden_dim = cfg.graph_fusion.hidden_dim
    vocab_size = getattr(model.backbone, "vocab_size", 1000)

    # 1. Input IDs contract
    checks.append(
        verify_tensor_contract(
            input_ids,
            "input_ids",
            expected_shape=(b, t),
            expected_dtype=torch.long,
            expected_device="cpu",
            check_contiguous=True,
            value_range=(0, vocab_size - 1),
        )
    )

    # 2. Labels contract
    if labels is not None:
        checks.append(
            verify_tensor_contract(
                labels,
                "labels",
                expected_shape=(b, t),
                expected_dtype=torch.long,
                expected_device="cpu",
                check_contiguous=True,
            )
        )

    # 3. Attention mask contract
    if attention_mask is not None:
        checks.append(
            verify_tensor_contract(
                attention_mask,
                "attention_mask",
                expected_shape=(b, t),
                expected_dtype=(torch.bool, torch.float32, torch.long),
                expected_device="cpu",
                check_contiguous=True,
            )
        )

    # 4. Memory available contract
    if memory_available is not None:
        checks.append(
            verify_tensor_contract(
                memory_available,
                "memory_available",
                expected_shape=(b,),
                expected_dtype=torch.bool,
                expected_device="cpu",
                check_contiguous=True,
            )
        )

    # 5. Graph contracts (if populated)
    state = model.bridge.get_active_state()
    graph_report = verify_graph_contracts(state)
    checks.extend(graph_report.checks)

    # 6. Execute forward pass with diagnostics
    diagnostics: dict[str, Any] = {}
    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        memory_available=memory_available,
        generate_proposals=generate_proposals,
        authorization_context=authorization_context,
        diagnostics=diagnostics,
    )

    # 7. Logits contract
    checks.append(
        verify_tensor_contract(
            output.logits,
            "output.logits",
            expected_shape=(b, t, vocab_size),
            expected_dtype=torch.float32,
            expected_device="cpu",
            check_contiguous=True,
            check_finite=True,
        )
    )

    # 8. Loss contract (when labels provided)
    if labels is not None:
        checks.append(
            verify_tensor_contract(
                output.loss,
                "output.loss",
                expected_shape=(),
                expected_dtype=torch.float32,
                expected_device="cpu",
                check_contiguous=True,
                check_finite=True,
                value_range=(0.0, None),
            )
        )

    # 9. Query selection contract (when memory is selected)
    if output.graph_selection_output is not None:
        sel = output.graph_selection_output.selection
        top_k = cfg.selector.top_k
        checks.append(
            verify_tensor_contract(
                sel.selected_states,
                "output.graph_selection_output.selection.selected_states",
                expected_shape=(b, top_k, hidden_dim),
                expected_dtype=torch.float32,
                expected_device="cpu",
                check_contiguous=True,
                check_finite=True,
            )
        )
        checks.append(
            verify_tensor_contract(
                sel.selected_scores,
                "output.graph_selection_output.selection.selected_scores",
                expected_shape=(b, top_k),
                expected_dtype=torch.float32,
                expected_device="cpu",
                check_contiguous=True,
                check_finite=True,
            )
        )
        checks.append(
            verify_tensor_contract(
                sel.selection_mask,
                "output.graph_selection_output.selection.selection_mask",
                expected_shape=(b, top_k),
                expected_dtype=torch.bool,
                expected_device="cpu",
                check_contiguous=True,
            )
        )
        if output.graph_selection_output.gated_memory_context is not None:
            checks.append(
                verify_tensor_contract(
                    output.graph_selection_output.gated_memory_context,
                    "output.graph_selection_output.gated_memory_context",
                    expected_shape=(b, hidden_dim),
                    expected_dtype=torch.float32,
                    expected_device="cpu",
                    check_contiguous=True,
                    check_finite=True,
                )
            )

    # 10. Traversal contract (when active)
    if output.traversal_output is not None:
        p_count = output.traversal_output.path_representations.shape[1]
        p_dim = output.traversal_output.path_representations.shape[2]
        checks.append(
            verify_tensor_contract(
                output.traversal_output.path_representations,
                "output.traversal_output.path_representations",
                expected_shape=(b, p_count, p_dim),
                expected_dtype=torch.float32,
                expected_device="cpu",
                check_contiguous=True,
                check_finite=True,
            )
        )
        checks.append(
            verify_tensor_contract(
                output.traversal_output.reasoning_context,
                "output.traversal_output.reasoning_context",
                expected_shape=(b, hidden_dim),
                expected_dtype=torch.float32,
                expected_device="cpu",
                check_contiguous=True,
                check_finite=True,
            )
        )

    # 11. Fusion output contract (when fusion modulated)
    if output.fusion_output is not None:
        checks.append(
            verify_tensor_contract(
                output.fusion_output.fused_hidden_states,
                "output.fusion_output.fused_hidden_states",
                expected_shape=(b, t, hidden_dim),
                expected_dtype=torch.float32,
                expected_device="cpu",
                check_contiguous=True,
                check_finite=True,
            )
        )
        checks.append(
            verify_tensor_contract(
                output.fusion_output.fusion_gate,
                "output.fusion_output.fusion_gate",
                expected_shape=(b, t, 1),
                expected_dtype=torch.float32,
                expected_device="cpu",
                check_contiguous=True,
                check_finite=True,
                value_range=(0.0, 1.0),
            )
        )

    # 12. Write proposals contract (when generated)
    if generate_proposals and output.write_proposals:
        for idx, prop in enumerate(output.write_proposals):
            if prop.candidate_embedding is not None:
                checks.append(
                    verify_tensor_contract(
                        prop.candidate_embedding,
                        f"proposal[{idx}].candidate_embedding",
                        expected_shape=(hidden_dim,),
                        expected_dtype=torch.float32,
                        expected_device="cpu",
                        check_contiguous=True,
                        check_finite=True,
                    )
                )

    for c in checks:
        if not c.passed:
            errors.append(c.message)

    passed_count = sum(1 for c in checks if c.passed)
    return TensorContractReport(
        passed=len(errors) == 0,
        total_checks=len(checks),
        passed_checks=passed_count,
        failed_checks=len(errors),
        checks=checks,
        errors=errors,
    )


def create_verification_model(
    num_nodes: int = 4,
    hidden_dim: int = 16,
    vocab_size: int = 100,
    backbone_frozen: bool = True,
) -> tuple[RUTAMForCausalLM, Graph]:
    """Helper creating a fully configured RUTAM model and graph for diagnostic verification."""
    config = RUTAMConfig().with_overrides(
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
            },
            "rutam_model": {
                "backbone_frozen": backbone_frozen,
                "return_memory_diagnostics": True,
                "graph_traversal_enabled": True,
            },
        }
    )

    graph = Graph(config)
    gen = torch.Generator().manual_seed(42)
    for i in range(num_nodes):
        graph.create_node(
            node_type="concept",
            content=f"diagnostic_node_{i}",
            node_id=(i + 1) * 10,
            provenance="diag:provenance",
            embedding=torch.randn(hidden_dim, generator=gen).tolist(),
        )

    if num_nodes >= 2:
        graph.add_relationship(10, 20, "rel_0", provenance="diag:rel", weight=0.8)
    if num_nodes >= 3:
        graph.add_relationship(20, 30, "rel_1", provenance="diag:rel", weight=0.6)
    if num_nodes >= 4:
        graph.add_relationship(30, 40, "rel_2", provenance="diag:rel", weight=0.9)

    backbone = SyntheticBackbone(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_layers=3,
    )
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)
    return model, graph
