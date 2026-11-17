"""Comprehensive CPU benchmarking suite for RUTAM architecture and components."""

from __future__ import annotations

import gc
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from rutam.checkpoint.manager import RUTAMCheckpointManager
from rutam.config import GraphAttentionConfig, RUTAMConfig
from rutam.graph.attention import RUTAMGraphAttention
from rutam.graph.operations import Graph
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.write_proposal import MemoryOperation, MemoryProposal
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.rutam_model import RUTAMForCausalLM


@dataclass
class LatencyMetric:
    """Latency distribution and throughput metrics for a benchmark target."""

    name: str
    iterations: int
    min_ms: float
    mean_ms: float
    median_ms: float
    max_ms: float
    p90_ms: float
    p95_ms: float
    throughput_per_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert latency metric to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSuiteReport:
    """Aggregated benchmark suite report covering forward, backward, attention, and memory."""

    forward_benchmarks: list[LatencyMetric] = field(default_factory=list)
    component_benchmarks: list[LatencyMetric] = field(default_factory=list)
    attention_scaling_benchmarks: list[LatencyMetric] = field(default_factory=list)
    training_step_benchmarks: list[LatencyMetric] = field(default_factory=list)
    memory_operation_benchmarks: list[LatencyMetric] = field(default_factory=list)
    checkpoint_benchmarks: list[LatencyMetric] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert suite report to dictionary."""
        return {
            "forward_benchmarks": [m.to_dict() for m in self.forward_benchmarks],
            "component_benchmarks": [m.to_dict() for m in self.component_benchmarks],
            "attention_scaling_benchmarks": [
                m.to_dict() for m in self.attention_scaling_benchmarks
            ],
            "training_step_benchmarks": [m.to_dict() for m in self.training_step_benchmarks],
            "memory_operation_benchmarks": [m.to_dict() for m in self.memory_operation_benchmarks],
            "checkpoint_benchmarks": [m.to_dict() for m in self.checkpoint_benchmarks],
            "summary": self.summary,
        }


def _measure_latency(
    fn: Any,
    warmup: int = 2,
    iterations: int = 5,
    name: str = "operation",
    throughput_items: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> LatencyMetric:
    """Run warmup and timed iterations, calculating robust percentile metrics."""
    for _ in range(warmup):
        fn()

    durations: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - t0) * 1000.0)

    durations.sort()
    min_ms = durations[0]
    max_ms = durations[-1]
    mean_ms = statistics.mean(durations)
    median_ms = statistics.median(durations)

    p90_idx = min(int(len(durations) * 0.9), len(durations) - 1)
    p95_idx = min(int(len(durations) * 0.95), len(durations) - 1)
    p90_ms = durations[p90_idx]
    p95_ms = durations[p95_idx]

    throughput: float | None = None
    if throughput_items is not None and mean_ms > 0:
        throughput = float(throughput_items / (mean_ms / 1000.0))

    return LatencyMetric(
        name=name,
        iterations=iterations,
        min_ms=round(min_ms, 3),
        mean_ms=round(mean_ms, 3),
        median_ms=round(median_ms, 3),
        max_ms=round(max_ms, 3),
        p90_ms=round(p90_ms, 3),
        p95_ms=round(p95_ms, 3),
        throughput_per_sec=round(throughput, 2) if throughput is not None else None,
        metadata=dict(metadata or {}),
    )


def benchmark_forward_passes(
    warmup: int = 2,
    iterations: int = 5,
) -> list[LatencyMetric]:
    """Benchmark full model forward pass latency across graph sizes and sequence lengths."""
    metrics: list[LatencyMetric] = []
    from rutam.diagnostics.integration import create_verification_model

    scenarios = [
        # (num_nodes, batch_size, seq_len)
        (0, 1, 8),
        (0, 1, 32),
        (10, 1, 8),
        (10, 1, 32),
        (10, 4, 32),
        (50, 1, 32),
        (100, 1, 32),
    ]

    for n_nodes, b, t in scenarios:
        model, _ = create_verification_model(num_nodes=n_nodes, hidden_dim=16, vocab_size=100)
        input_ids = torch.randint(0, 100, (b, t), dtype=torch.long)

        metric = _measure_latency(
            fn=lambda m=model, inp=input_ids: m(inp),
            warmup=warmup,
            iterations=iterations,
            name=f"forward_N{n_nodes}_B{b}_T{t}",
            throughput_items=b * t,
            metadata={"num_nodes": n_nodes, "batch_size": b, "seq_len": t},
        )
        metrics.append(metric)

    return metrics


def benchmark_components(
    warmup: int = 2,
    iterations: int = 5,
) -> list[LatencyMetric]:
    """Benchmark isolated latency of individual neural graph submodules."""
    metrics: list[LatencyMetric] = []
    from rutam.diagnostics.integration import create_verification_model

    model, _ = create_verification_model(num_nodes=10, hidden_dim=16, vocab_size=100)
    hidden_states = torch.randn(2, 16, 16)
    active_state = model.bridge.get_active_state()

    # 1. Query Head
    metrics.append(
        _measure_latency(
            fn=lambda: model.query_head(hidden_states),
            warmup=warmup,
            iterations=iterations,
            name="submodule_query_head",
            metadata={"component": "query_head"},
        )
    )

    # 2. Graph Encoder
    metrics.append(
        _measure_latency(
            fn=lambda: model.encoder(active_state),
            warmup=warmup,
            iterations=iterations,
            name="submodule_graph_encoder",
            metadata={"component": "encoder"},
        )
    )

    # 3. Graph Selector
    q_out = model.query_head(hidden_states)
    enc_out = model.encoder(active_state)
    query_vectors = q_out.query_vectors.view(-1, 16)
    metrics.append(
        _measure_latency(
            fn=lambda: model.selector(query_vectors, enc_out.node_states, active_state.node_ids),
            warmup=warmup,
            iterations=iterations,
            name="submodule_graph_selector",
            metadata={"component": "selector"},
        )
    )

    # 4. Graph Fusion
    sel_out = model.selector(query_vectors, enc_out.node_states, active_state.node_ids)
    metrics.append(
        _measure_latency(
            fn=lambda: model.fusion(hidden_states, sel_out.selected_states),
            warmup=warmup,
            iterations=iterations,
            name="submodule_graph_fusion",
            metadata={"component": "fusion"},
        )
    )

    # 5. Memory Write Head
    metrics.append(
        _measure_latency(
            fn=lambda: model.write_head(hidden_states),
            warmup=warmup,
            iterations=iterations,
            name="submodule_memory_write_head",
            metadata={"component": "write_head"},
        )
    )

    return metrics


def benchmark_graph_attention_scaling(
    warmup: int = 2,
    iterations: int = 5,
) -> list[LatencyMetric]:
    """Benchmark RUTAMGraphAttention CPU scaling across node counts N and edge densities."""
    metrics: list[LatencyMetric] = []
    hidden_dim = 32
    query_dim = 16

    cases = [
        (10, 2),
        (50, 4),
        (100, 4),
        (500, 2),
    ]

    for num_nodes, degree in cases:
        config = GraphAttentionConfig(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            num_relation_types=8,
            query_dim=query_dim,
            relation_bias=True,
            query_conditioning=True,
            return_attention=False,
            dropout=0.0,
        )
        module = RUTAMGraphAttention(config)

        # Build random graph
        nodes = torch.randn(num_nodes, hidden_dim)
        src_list: list[int] = []
        dst_list: list[int] = []
        for i in range(num_nodes):
            for d in range(1, degree + 1):
                dst_list.append(i)
                src_list.append((i + d) % num_nodes)

        num_edges = len(src_list)
        edges = torch.tensor([src_list, dst_list], dtype=torch.long)
        types = torch.randint(0, 8, (num_edges,), dtype=torch.long)
        weights = torch.rand(num_edges, dtype=torch.float32)
        query = torch.randn(query_dim)

        # Unconditioned attention
        metrics.append(
            _measure_latency(
                fn=lambda mod=module, nd=nodes, ed=edges, ty=types, wt=weights: mod(
                    nd, ed, ty, wt, query=None
                ),
                warmup=warmup,
                iterations=iterations,
                name=f"attention_unconditioned_N{num_nodes}_E{num_edges}",
                metadata={"nodes": num_nodes, "edges": num_edges, "query_conditioned": False},
            )
        )

        # Query-conditioned attention
        metrics.append(
            _measure_latency(
                fn=lambda mod=module, nd=nodes, ed=edges, ty=types, wt=weights, q=query: mod(
                    nd, ed, ty, wt, query=q
                ),
                warmup=warmup,
                iterations=iterations,
                name=f"attention_query_conditioned_N{num_nodes}_E{num_edges}",
                metadata={"nodes": num_nodes, "edges": num_edges, "query_conditioned": True},
            )
        )

    return metrics


def benchmark_training_step(
    warmup: int = 2,
    iterations: int = 5,
) -> list[LatencyMetric]:
    """Benchmark end-to-end training step latency (forward + loss + backward + optimizer step)."""
    metrics: list[LatencyMetric] = []
    from rutam.diagnostics.integration import create_verification_model

    model, _ = create_verification_model(num_nodes=8, hidden_dim=16, vocab_size=100)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)

    input_ids = torch.randint(0, 100, (2, 16), dtype=torch.long)
    labels = torch.randint(0, 100, (2, 16), dtype=torch.long)

    def train_step() -> None:
        optimizer.zero_grad()
        model.bridge.invalidate_caches()
        out = model(input_ids, labels=labels)
        if out.loss is not None:
            out.loss.backward()
        optimizer.step()

    metrics.append(
        _measure_latency(
            fn=train_step,
            warmup=warmup,
            iterations=iterations,
            name="training_step_B2_T16",
            throughput_items=32,
            metadata={"batch_size": 2, "seq_len": 16},
        )
    )

    return metrics


def benchmark_memory_operations(
    warmup: int = 2,
    iterations: int = 5,
) -> list[LatencyMetric]:
    """Benchmark discrete memory proposals, multi-policy validation, and transactional commits."""
    metrics: list[LatencyMetric] = []
    config = RUTAMConfig().with_overrides({"graph": {"embedding_dim": 16}})
    graph = Graph(config)
    backbone = SyntheticBackbone(16, 50, 2)
    model = RUTAMForCausalLM(config=config, backbone=backbone, graph=graph)

    auth = AuthorizationContext(user_id="bench_user", role="admin")
    prov = ProvenanceRecord(source_id="bench_source", authorization_category="user_authorized")

    # Proposal validation benchmark
    def validate_batch() -> None:
        props = [
            MemoryProposal(
                proposal_id=f"bench_prop_{i}",
                operation=MemoryOperation.CREATE_NODE,
                target_node_ids=(),
                candidate_embedding=torch.tensor([0.1] * 16),
                provenance=prov,
                confidence=0.9,
                source_graph_version=model.graph.version,
            )
            for i in range(10)
        ]
        model.validate_memory_updates(props, authorization_context=auth)

    metrics.append(
        _measure_latency(
            fn=validate_batch,
            warmup=warmup,
            iterations=iterations,
            name="memory_validate_batch_10_proposals",
            metadata={"proposal_count": 10},
        )
    )

    return metrics


def benchmark_checkpoints(
    warmup: int = 1,
    iterations: int = 3,
) -> list[LatencyMetric]:
    """Benchmark checkpoint save, manifest SHA-256 integrity verification, and reload."""
    metrics: list[LatencyMetric] = []
    from rutam.diagnostics.integration import create_verification_model

    model, graph = create_verification_model(num_nodes=8, hidden_dim=16, vocab_size=100)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)

    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = RUTAMCheckpointManager(base_dir=tmp_dir, run_id="bench_run")

        # 1. Save checkpoint
        step_counter = [0]

        def save_step() -> Path:
            step_counter[0] += 1
            return manager.save_checkpoint(
                model=model,
                graph=graph,
                optimizer=opt,
                step=step_counter[0],
                rutam_config=model.config,
            )

        saved_path = save_step()
        metrics.append(
            _measure_latency(
                fn=save_step,
                warmup=warmup,
                iterations=iterations,
                name="checkpoint_atomic_save",
                metadata={"components_persisted": ["model", "graph", "optimizer", "manifest"]},
            )
        )

        # 2. Verify integrity
        metrics.append(
            _measure_latency(
                fn=lambda: manager.load_checkpoint(saved_path, strict=False),
                warmup=warmup,
                iterations=iterations,
                name="checkpoint_verify_and_load",
                metadata={"checkpoint_dir": str(saved_path)},
            )
        )

    return metrics


def run_all_benchmarks(
    warmup: int = 2,
    iterations: int = 5,
) -> BenchmarkSuiteReport:
    """Execute complete benchmarking suite across model, submodules, attention, and storage."""
    gc.collect()
    start_time = time.perf_counter()

    fwd = benchmark_forward_passes(warmup=warmup, iterations=iterations)
    comps = benchmark_components(warmup=warmup, iterations=iterations)
    attn = benchmark_graph_attention_scaling(warmup=warmup, iterations=iterations)
    train = benchmark_training_step(warmup=warmup, iterations=iterations)
    mem = benchmark_memory_operations(warmup=warmup, iterations=iterations)
    ckpt = benchmark_checkpoints(warmup=1, iterations=iterations)

    total_duration_sec = time.perf_counter() - start_time
    total_metrics = len(fwd) + len(comps) + len(attn) + len(train) + len(mem) + len(ckpt)

    summary = {
        "total_benchmarks": total_metrics,
        "total_duration_seconds": round(total_duration_sec, 2),
        "mean_forward_latency_ms": round(statistics.mean(m.mean_ms for m in fwd), 2)
        if fwd
        else 0.0,
        "mean_attention_latency_ms": round(statistics.mean(m.mean_ms for m in attn), 2)
        if attn
        else 0.0,
        "mean_training_step_ms": round(train[0].mean_ms, 2) if train else 0.0,
    }

    return BenchmarkSuiteReport(
        forward_benchmarks=fwd,
        component_benchmarks=comps,
        attention_scaling_benchmarks=attn,
        training_step_benchmarks=train,
        memory_operation_benchmarks=mem,
        checkpoint_benchmarks=ckpt,
        summary=summary,
    )
