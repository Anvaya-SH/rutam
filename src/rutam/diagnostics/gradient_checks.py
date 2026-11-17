"""Gradient flow and backward pass verification across RUTAM submodules and end-to-end model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from rutam.config import (
    GraphAttentionConfig,
    GraphFusionConfig,
    MemoryWriteConfig,
    QueryHeadConfig,
    RUTAMConfig,
    SelectorConfig,
    TraversalConfig,
)
from rutam.graph.attention import RUTAMGraphAttention
from rutam.graph.encoder import RUTAMGraphEncoder
from rutam.graph.linker import RUTAMGraphLinker
from rutam.graph.operations import Graph
from rutam.graph.path_scoring import RUTAMPathScorer
from rutam.graph.selector import RUTAMGraphSelector
from rutam.graph.state import GraphState
from rutam.memory.write_head import RUTAMMemoryWriteHead
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.query_head import RUTAMNeuralQueryHead
from rutam.model.rutam_model import RUTAMForCausalLM


@dataclass
class ParameterGradientStats:
    """Gradient summary statistics for an individual parameter tensor."""

    name: str
    shape: tuple[int, ...]
    requires_grad: bool
    has_grad: bool
    is_finite: bool
    grad_norm: float
    grad_mean: float
    grad_min: float
    grad_max: float

    def to_dict(self) -> dict[str, Any]:
        """Convert parameter stats to dictionary."""
        return asdict(self)


@dataclass
class ComponentGradientReport:
    """Gradient report for an individual neural component."""

    component_name: str
    passed: bool
    trainable_params_count: int
    params_with_gradients_count: int
    all_finite: bool
    all_non_zero: bool
    parameters: list[ParameterGradientStats] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert component report to dictionary."""
        return {
            "component_name": self.component_name,
            "passed": self.passed,
            "trainable_params_count": self.trainable_params_count,
            "params_with_gradients_count": self.params_with_gradients_count,
            "all_finite": self.all_finite,
            "all_non_zero": self.all_non_zero,
            "parameters": [p.to_dict() for p in self.parameters],
            "error_message": self.error_message,
        }


@dataclass
class FullGradientFlowReport:
    """Aggregated gradient report across all 8 submodules and end-to-end model."""

    passed: bool
    components: dict[str, ComponentGradientReport] = field(default_factory=dict)
    frozen_backbone_verified: bool = False
    unfrozen_backbone_verified: bool = False
    end_to_end_passed: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert full report to dictionary."""
        return {
            "passed": self.passed,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "frozen_backbone_verified": self.frozen_backbone_verified,
            "unfrozen_backbone_verified": self.unfrozen_backbone_verified,
            "end_to_end_passed": self.end_to_end_passed,
            "summary": self.summary,
        }


def extract_gradient_stats(module: nn.Module) -> list[ParameterGradientStats]:
    """Collect gradient statistics for all parameters in a module."""
    stats: list[ParameterGradientStats] = []
    for name, param in module.named_parameters():
        has_grad = param.grad is not None
        is_finite = True
        grad_norm = 0.0
        grad_mean = 0.0
        grad_min = 0.0
        grad_max = 0.0

        if has_grad and param.grad is not None:
            g = param.grad.detach()
            is_finite = bool(torch.isfinite(g).all().item())
            grad_norm = float(g.norm().item())
            grad_mean = float(g.mean().item())
            grad_min = float(g.min().item())
            grad_max = float(g.max().item())

        stats.append(
            ParameterGradientStats(
                name=name,
                shape=tuple(param.shape),
                requires_grad=param.requires_grad,
                has_grad=has_grad,
                is_finite=is_finite,
                grad_norm=grad_norm,
                grad_mean=grad_mean,
                grad_min=grad_min,
                grad_max=grad_max,
            )
        )
    return stats


def check_query_head_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMNeuralQueryHead."""
    torch.manual_seed(42)
    cfg = QueryHeadConfig(
        backbone_hidden_dim=16,
        graph_hidden_dim=16,
        query_dim=16,
        intermediate_dim=16,
        dropout=0.0,
    )
    module = RUTAMNeuralQueryHead(cfg)
    hidden_states = torch.randn(2, 4, 16, requires_grad=True)
    graph_summary = torch.randn(2, 16, requires_grad=True)

    out = module(hidden_states, graph_summary=graph_summary)
    loss = out.query_vectors.sum() + out.query_gate.sum()
    loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="query_head",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_graph_encoder_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMGraphEncoder."""
    torch.manual_seed(42)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16},
            "encoder": {
                "input_dim": 16,
                "hidden_dim": 16,
                "num_layers": 2,
                "num_relation_types": 4,
                "dropout": 0.0,
            },
        }
    )
    module = RUTAMGraphEncoder(config.encoder)

    graph = Graph(config)
    for i in range(4):
        graph.create_node(
            node_type="concept",
            content=f"enc_node_{i}",
            node_id=(i + 1) * 10,
            provenance="diag:enc",
            embedding=torch.randn(16).tolist(),
        )
    graph.add_relationship(10, 20, "rel_0", provenance="diag:enc", weight=1.0)
    graph.add_relationship(20, 30, "rel_1", provenance="diag:enc", weight=1.0)
    graph.add_relationship(30, 40, "rel_2", provenance="diag:enc", weight=1.0)

    state = graph.to_state()
    # Make node embeddings differentiable
    node_embeddings = state.node_embeddings.clone().detach().requires_grad_(True)
    differentiable_state = GraphState(
        node_ids=state.node_ids,
        node_embeddings=node_embeddings,
        edge_index=state.edge_index,
        edge_types=state.edge_types,
        edge_weights=state.edge_weights,
        node_metadata=state.node_metadata,
        edge_metadata=state.edge_metadata,
        relationship_types=state.relationship_types,
        graph_version=state.graph_version,
        config=state.config,
        next_node_id=state.next_node_id,
        retired_node_ids=state.retired_node_ids,
        metadata=state.metadata,
    )

    out = module(differentiable_state)
    loss = out.node_states.sum()
    loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="encoder",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_graph_attention_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMGraphAttention."""
    torch.manual_seed(42)
    cfg = GraphAttentionConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_relation_types=8,
        query_dim=16,
        relation_bias=True,
        query_conditioning=True,
        return_attention=True,
        dropout=0.0,
    )
    module = RUTAMGraphAttention(cfg)

    nodes = torch.randn(3, 32, requires_grad=True)
    edges = torch.tensor([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    types = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    weights = torch.tensor([0.2, 0.5, 0.8, 0.3, 0.7, 0.9], dtype=torch.float32)
    query = torch.randn(16, requires_grad=True)

    out = module(nodes, edges, types, weights, query=query)
    attn_sum = out.edge_attention.sum() if out.edge_attention is not None else 0.0
    loss = attn_sum + out.node_states.sum() + out.graph_summary.sum()
    loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="attention",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_graph_selector_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMGraphSelector."""
    torch.manual_seed(42)
    cfg = SelectorConfig(
        query_dim=16,
        hidden_dim=16,
        score_dim=16,
        top_k=2,
        selection_mode="soft",
    )
    module = RUTAMGraphSelector(cfg)

    query_states = torch.randn(2, 16, requires_grad=True)
    node_states = torch.randn(4, 16, requires_grad=True)
    node_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)

    out = module(query_states, node_states, node_ids)
    loss = out.selected_states.sum() + out.selected_scores.sum()
    loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="selector",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_path_scorer_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMPathScorer."""
    torch.manual_seed(42)
    cfg = TraversalConfig(
        query_dim=16,
        hidden_dim=16,
        score_hidden_dim=16,
        path_scoring="additive",
        max_candidate_paths=4,
        max_selected_paths=2,
    )
    module = RUTAMPathScorer(cfg)

    queries = torch.randn(2, 16, requires_grad=True)
    paths = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    out = module(queries, paths, mask)
    loss = (out.attention[..., None] * paths).sum()
    loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="path_scorer",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_fusion_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMGraphFusion."""
    torch.manual_seed(42)
    cfg = GraphFusionConfig(
        hidden_dim=16,
        graph_dim=16,
        num_attention_heads=2,
        use_reasoning_context=True,
    )
    module = RUTAMGraphFusion(cfg)

    b, t, k = 2, 4, 2
    hidden_states = torch.randn(b, t, 16, requires_grad=True)
    graph_states = torch.randn(b, k, 16, requires_grad=True)
    reasoning_context = torch.randn(b, 16, requires_grad=True)
    query_gate = torch.rand(b, 1, requires_grad=True)
    graph_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

    out = module(
        hidden_states,
        graph_states,
        graph_mask=graph_mask,
        reasoning_context=reasoning_context,
        query_gate=query_gate,
    )
    loss = out.fused_hidden_states.sum()
    loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="fusion",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_write_head_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMMemoryWriteHead."""
    torch.manual_seed(42)
    cfg = MemoryWriteConfig(
        backbone_hidden_dim=16,
        hidden_dim=16,
        projection_dim=16,
        max_proposals=2,
        target_selection=True,
    )
    module = RUTAMMemoryWriteHead(cfg)

    b, t, d, h, k = 2, 4, 16, 16, 2
    hidden = torch.randn(b, t, d, requires_grad=True)
    states = torch.randn(b, k, h, requires_grad=True)
    summary = torch.randn(b, h, requires_grad=True)

    out = module(hidden, graph_summary=summary, selected_memory_states=states)
    losses = module.loss(
        out,
        write_targets=torch.tensor([1.0, 0.0]),
        operation_targets=torch.tensor([1, 0], dtype=torch.long),
        target_embeddings=torch.randn(b, h),
        target_node_labels=torch.tensor([0, 1], dtype=torch.long),
    )
    # Also verify backward pass through memory projection slot gate
    context = module.pooling(hidden).context
    proj_out = module.memory_projection(context)
    total_loss = losses.total + proj_out.slot_logits.sum()
    total_loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="write_head",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_link_predictor_gradients() -> ComponentGradientReport:
    """Verify backward pass and gradient flow through RUTAMGraphLinker."""
    torch.manual_seed(42)
    config = RUTAMConfig().with_overrides(
        {
            "graph": {"embedding_dim": 16},
            "linker": {
                "hidden_dim": 16,
                "scoring_hidden_dim": 16,
                "structural_feature_dim": 4,
                "query_dim": 16,
                "dropout": 0.0,
                "relation_labels": ["rel_0", "rel_1", "rel_2"],
                "max_candidate_pairs": 8,
            },
        }
    )
    module = RUTAMGraphLinker(config.linker)

    graph = Graph(config)
    for i in range(3):
        graph.create_node(
            node_type="concept",
            content=f"link_node_{i}",
            node_id=(i + 1) * 10,
            provenance="diag:link",
            embedding=torch.randn(16).tolist(),
        )
    graph.add_relationship(10, 20, "rel_0", provenance="diag:link", weight=1.0)
    graph.add_relationship(20, 30, "rel_1", provenance="diag:link", weight=1.0)

    state = graph.to_state()
    node_states = torch.randn(3, 16, requires_grad=True)
    candidate_pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)

    out = module.from_state(node_states, state, candidate_pairs=candidate_pairs)
    loss = out.link_logits.sum() + out.relation_logits.sum()
    loss.backward()

    stats = extract_gradient_stats(module)
    trainable = [p for p in stats if p.requires_grad]
    with_grad = [p for p in trainable if p.has_grad]
    all_finite = all(p.is_finite for p in with_grad)
    all_non_zero = all(p.grad_norm > 0.0 for p in with_grad)
    passed = len(trainable) > 0 and len(with_grad) == len(trainable) and all_finite and all_non_zero

    return ComponentGradientReport(
        component_name="link_predictor",
        passed=passed,
        trainable_params_count=len(trainable),
        params_with_gradients_count=len(with_grad),
        all_finite=all_finite,
        all_non_zero=all_non_zero,
        parameters=stats,
    )


def check_end_to_end_gradients(
    model: RUTAMForCausalLM | None = None,
) -> tuple[bool, bool, bool]:
    """Verify end-to-end backward pass, frozen backbone isolation, and full fine-tuning."""
    torch.manual_seed(42)
    from rutam.diagnostics.integration import create_verification_model

    if model is None:
        m, _ = create_verification_model(num_nodes=4, hidden_dim=16, backbone_frozen=True)
    else:
        m = model

    input_ids = torch.randint(0, 50, (2, 6), dtype=torch.long)
    labels = torch.randint(0, 50, (2, 6), dtype=torch.long)

    # 1. Test with frozen backbone
    m.freeze_backbone()
    out = m(input_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()

    # Backbone parameters must have no gradients
    backbone_frozen_verified = all(
        p.grad is None or not p.requires_grad for p in m.backbone.parameters()
    )

    # Graph/fusion parameters must have received valid gradients
    fusion_grads = [p.grad for p in m.fusion.parameters() if p.requires_grad]
    graph_has_grads = len(fusion_grads) > 0 and all(
        g is not None and torch.isfinite(g).all() and g.norm().item() > 0.0 for g in fusion_grads
    )

    # 2. Test with unfrozen backbone
    m.zero_grad()
    m.bridge.invalidate_caches()
    m.unfreeze_backbone()
    out2 = m(input_ids, labels=labels)
    assert out2.loss is not None
    out2.loss.backward()

    backbone_grads = [p.grad for p in m.backbone.parameters() if p.requires_grad]
    unfrozen_verified = len(backbone_grads) > 0 and all(
        g is not None and torch.isfinite(g).all() for g in backbone_grads
    )

    end_to_end_passed = backbone_frozen_verified and graph_has_grads and unfrozen_verified
    return end_to_end_passed, backbone_frozen_verified, unfrozen_verified


def run_all_gradient_checks() -> FullGradientFlowReport:
    """Execute isolated and end-to-end gradient checks across all 8 submodules."""
    components: dict[str, ComponentGradientReport] = {}

    components["query_head"] = check_query_head_gradients()
    components["encoder"] = check_graph_encoder_gradients()
    components["attention"] = check_graph_attention_gradients()
    components["selector"] = check_graph_selector_gradients()
    components["path_scorer"] = check_path_scorer_gradients()
    components["fusion"] = check_fusion_gradients()
    components["write_head"] = check_write_head_gradients()
    components["link_predictor"] = check_link_predictor_gradients()

    e2e_passed, frozen_ok, unfrozen_ok = check_end_to_end_gradients()

    all_components_passed = all(c.passed for c in components.values())
    overall_passed = all_components_passed and e2e_passed

    summary = {
        "total_components": len(components),
        "passed_components": sum(1 for c in components.values() if c.passed),
        "all_components_passed": all_components_passed,
        "frozen_backbone_verified": frozen_ok,
        "unfrozen_backbone_verified": unfrozen_ok,
        "end_to_end_passed": e2e_passed,
        "overall_passed": overall_passed,
    }

    return FullGradientFlowReport(
        passed=overall_passed,
        components=components,
        frozen_backbone_verified=frozen_ok,
        unfrozen_backbone_verified=unfrozen_ok,
        end_to_end_passed=e2e_passed,
        summary=summary,
    )
