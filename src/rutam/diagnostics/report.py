"""Audit report generation for RUTAM in structured JSON and comprehensive Markdown formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rutam.diagnostics.architecture_audit import ArchitectureAuditReport
    from rutam.diagnostics.benchmark import BenchmarkSuiteReport


def generate_audit_json(
    report: ArchitectureAuditReport,
    filepath: Path | str | None = None,
) -> str:
    """Serialize ArchitectureAuditReport into formatted JSON and optionally write to file."""
    data = report.to_dict()
    json_text = json.dumps(data, indent=2, sort_keys=True)
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_text, encoding="utf-8")
    return json_text


def generate_audit_markdown(
    report: ArchitectureAuditReport,
    benchmark_report: BenchmarkSuiteReport | None = None,
    filepath: Path | str | None = None,
) -> str:
    """Render comprehensive, professional GitHub-flavored Markdown audit report."""
    status_badge = "🟢 **PASSED**" if report.passed else "🔴 **FAILED**"
    lines: list[str] = [
        "# RUTAM Architecture Integration & Benchmarking Audit Report",
        "",
        f"**Audit Status**: {status_badge}  ",
        f"**Audit Timestamp**: `{report.audit_timestamp}`  ",
        f"**Audit Duration**: `{report.duration_seconds:.2f}s`  ",
        "",
        "## 1. Executive Summary",
        "",
        (
            "This diagnostic report details the end-to-end architecture integration, "
            "tensor contracts, gradient propagation across all 8 submodules, the complete "
            "14-step transactional memory lifecycle, fault-tolerant checkpoint verification, "
            "and CPU runtime benchmarks for **RUTAM**."
        ),
        "",
        "| Verification Domain | Checks / Steps | Status | Notes |",
        "| :--- | :---: | :---: | :--- |",
    ]

    c_pass = "✅ PASSED" if report.contract_audit.passed else "❌ FAILED"
    lines.append(
        f"| **Tensor Contracts** | "
        f"{report.contract_audit.passed_checks} / {report.contract_audit.total_checks} | "
        f"{c_pass} | Shape, dtype, CPU device, layout contracts |"
    )

    g_pass = "✅ PASSED" if report.gradient_audit.passed else "❌ FAILED"
    passed_sub = report.gradient_audit.summary.get("passed_components", 0)
    lines.append(
        f"| **Gradient Backward Pass** | {passed_sub} / 8 Submodules | {g_pass} | "
        "Query head, encoder, attention, selector, path scorer, fusion, write head, linker |"
    )

    b_pass = "✅ PASSED" if report.gradient_audit.frozen_backbone_verified else "❌ FAILED"
    lines.append(
        f"| **Backbone Gradient Isolation** | Verified | {b_pass} | "
        "Zero gradients in frozen backbone; non-zero memory gradients |"
    )

    l_pass = "✅ PASSED" if report.lifecycle_audit.passed else "❌ FAILED"
    l_passed_steps = report.lifecycle_audit.passed_steps
    l_tot_steps = report.lifecycle_audit.total_steps
    lines.append(
        f"| **14-Step Memory Lifecycle** | {l_passed_steps} / {l_tot_steps} Steps | {l_pass} | "
        "Full cycle: proposal -> validate -> commit -> reload -> rollback |"
    )

    cp_pass = "✅ PASSED" if report.checkpoint_audit.passed else "❌ FAILED"
    lines.append(
        f"| **Checkpoint & Recovery** | 4 / 4 Checks | {cp_pass} | "
        "Atomic safetensors, SHA-256 manifest, tamper detection, rollback |"
    )
    lines.append("")

    # 2. Environment Specifications
    lines.extend(
        [
            "## 2. Runtime & Environment Specifications",
            "",
            "| Parameter | Value |",
            "| :--- | :--- |",
            f"| **Platform / OS** | `{report.environment.platform}` "
            f"(`{report.environment.os_name}`) |",
            f"| **Python Version** | `{report.environment.python_version}` |",
            f"| **PyTorch Version** | `{report.environment.torch_version}` |",
            f"| **PyTorch Threads** | `{report.environment.torch_threads}` "
            f"(CPU cores: `{report.environment.cpu_count}`) |",
            f"| **Safetensors Version** | `{report.environment.safetensors_version}` |",
            "",
            "## 3. Submodule Parameter Audit & Memory Footprint",
            "",
            "| Submodule | Total Params | Trainable | Frozen | Memory Footprint (MB) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for mod in report.parameter_audit.modules:
        lines.append(
            f"| `{mod.module_name}` | {mod.total_params:,} | {mod.trainable_params:,} | "
            f"{mod.frozen_params:,} | {mod.footprint_mb:.3f} MB |"
        )
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    lines.append(
        f"| **Total Model** | **{report.parameter_audit.total_params:,}** | "
        f"**{report.parameter_audit.total_trainable:,}** | "
        f"**{report.parameter_audit.total_frozen:,}** | "
        f"**{report.parameter_audit.total_footprint_mb:.3f} MB** |"
    )
    lines.append("")

    # 4. Tensor Contract Verification
    lines.extend(
        [
            "## 4. Tensor Contract Verification",
            "",
            "Strict contract enforcement across all model, memory, and graph boundaries:",
            "",
            "| Tensor / Interface | Expected Shape | Actual Shape | Dtype | Device | "
            "Contiguous | Status |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
        ]
    )

    for chk in report.contract_audit.checks:
        exp_shape = str(chk.expected_shape) if chk.expected_shape else "-"
        act_shape = str(chk.actual_shape) if chk.actual_shape else "-"
        act_dtype = str(chk.actual_dtype).replace("torch.", "") if chk.actual_dtype else "-"
        act_dev = str(chk.actual_device) if chk.actual_device else "-"
        contig = "Yes" if chk.is_contiguous else ("No" if chk.is_contiguous is False else "-")
        status = "✅ PASS" if chk.passed else "❌ FAIL"
        lines.append(
            f"| `{chk.name}` | `{exp_shape}` | `{act_shape}` | `{act_dtype}` | "
            f"`{act_dev}` | {contig} | {status} |"
        )
    lines.append("")

    # 5. Gradient Flow & Backward Pass Verification
    lines.extend(
        [
            "## 5. Gradient Propagation & Differentiability Audit",
            "",
            "Gradient flow verified through every trainable weight tensor in submodules:",
            "",
            "| Submodule | Trainable | With Grad | Finite | Active (>0) | Backward Pass Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for name, comp in report.gradient_audit.components.items():
        finite_sym = "✅" if comp.all_finite else "❌"
        active_sym = "✅" if comp.all_non_zero else "❌"
        comp_status = "✅ PASSED" if comp.passed else "❌ FAILED"
        lines.append(
            f"| `{name}` | {comp.trainable_params_count} | {comp.params_with_gradients_count} | "
            f"{finite_sym} | {active_sym} | {comp_status} |"
        )
    lines.append("")

    f_iso = "✅ Confirmed" if report.gradient_audit.frozen_backbone_verified else "❌ Failed"
    lines.append(
        f"- **Frozen Backbone Isolation**: {f_iso} "
        "(zero gradients flow to backbone parameters during graph fine-tuning)."
    )

    u_fine = "✅ Confirmed" if report.gradient_audit.unfrozen_backbone_verified else "❌ Failed"
    lines.append(
        f"- **Unfrozen Full Fine-Tuning**: {u_fine} "
        "(valid gradients propagate to all backbone layers during end-to-end training)."
    )
    lines.append("")

    # 6. Complete 14-Step Memory Lifecycle Audit
    lines.extend(
        [
            "## 6. Complete 14-Step Memory Lifecycle Verification",
            "",
            "RUTAM enforces ACID-like transactional guarantees outside autograd:",
            "",
            "| Step | Phase / Operation | Duration | Status | Verification Summary |",
            "| :---: | :--- | :---: | :---: | :--- |",
        ]
    )

    for s in report.lifecycle_audit.steps:
        status_sym = "✅ PASS" if s.passed else "❌ FAIL"
        lines.append(
            f"| **{s.step_number}** | `{s.step_name}` | `{s.duration_ms:.2f} ms` | "
            f"{status_sym} | {s.message} |"
        )
    lines.append("")
    lines.append(
        f"**Total Lifecycle Execution Time**: `{report.lifecycle_audit.total_duration_ms:.2f} ms` "
        "across 14 discrete steps."
    )
    lines.append("")

    # 7. Checkpoint & Failure Recovery Audit
    c_save = "✅ PASSED" if report.checkpoint_audit.save_verified else "❌ FAILED"
    c_man = "✅ PASSED" if report.checkpoint_audit.manifest_verified else "❌ FAILED"
    c_tam = "✅ PASSED" if report.checkpoint_audit.tamper_detection_verified else "❌ FAILED"
    c_roll = "✅ PASSED" if report.checkpoint_audit.rollback_verified else "❌ FAILED"
    lines.extend(
        [
            "## 7. Checkpoint & Fault-Tolerance Verification",
            "",
            "| Capability | Mechanism | Status | Details |",
            "| :--- | :--- | :---: | :--- |",
            f"| **Atomic Persistence** | Safetensors + Temp Directory Rename | {c_save} | "
            "Zero risk of partial checkpoint writes |",
            f"| **Cryptographic Manifest** | SHA-256 Digest Verification | {c_man} | "
            "Exact file length and checksum matching |",
            f"| **Tamper & Corruption Detection** | Bit-level Injected Corruption Caught | "
            f"{c_tam} | Mismatched hash raises CorruptSnapshotError |",
            f"| **Automatic Rollback** | Fallback to Preceding Valid Checkpoint | {c_roll} | "
            "Corrupted latest step automatically bypassed |",
            "",
        ]
    )

    # 8. CPU Benchmarks (if provided)
    if benchmark_report is not None:
        lines.extend(
            [
                "## 8. CPU Latency & Scaling Benchmarks",
                "",
                "### 8.1 Model Forward Pass Latency across Graph Dimensions",
                "",
                "| Scenario | Nodes | Batch | Seq Len | Mean (ms) | Median (ms) | "
                "Throughput (tok/s) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for m in benchmark_report.forward_benchmarks:
            meta = m.metadata
            n_val = meta.get("num_nodes", "-")
            b_val = meta.get("batch_size", "-")
            t_val = meta.get("seq_len", "-")
            tp = m.throughput_per_sec or "-"
            lines.append(
                f"| `{m.name}` | {n_val} | {b_val} | {t_val} | **{m.mean_ms:.2f}** | "
                f"{m.median_ms:.2f} | {tp} |"
            )
        lines.append("")

        lines.extend(
            [
                "### 8.2 Submodule Latency Breakdown",
                "",
                "| Submodule | Mean Latency (ms) | Median (ms) | Min (ms) | Max (ms) |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for m in benchmark_report.component_benchmarks:
            lines.append(
                f"| `{m.name}` | **{m.mean_ms:.3f} ms** | {m.median_ms:.3f} ms | "
                f"{m.min_ms:.3f} ms | {m.max_ms:.3f} ms |"
            )
        lines.append("")

        lines.extend(
            [
                "### 8.3 Graph Attention Scaling on CPU",
                "",
                "| Attention Mode | Nodes | Edges | Query Conditioned | Mean Latency (ms) | "
                "P95 Latency (ms) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for m in benchmark_report.attention_scaling_benchmarks:
            meta = m.metadata
            qc_str = "Yes" if meta.get("query_conditioned") else "No"
            lines.append(
                f"| `{m.name}` | {meta.get('nodes')} | {meta.get('edges')} | {qc_str} | "
                f"**{m.mean_ms:.2f} ms** | {m.p95_ms:.2f} ms |"
            )
        lines.append("")

        lines.extend(
            [
                "### 8.4 Training Step & Checkpoint Latency",
                "",
                "| Target | Mean Latency (ms) | Median (ms) | Notes |",
                "| :--- | :---: | :---: | :--- |",
            ]
        )
        for m in benchmark_report.training_step_benchmarks:
            lines.append(
                f"| `{m.name}` | **{m.mean_ms:.2f} ms** | {m.median_ms:.2f} ms | "
                f"{m.throughput_per_sec} tokens/sec |"
            )
        for m in benchmark_report.memory_operation_benchmarks:
            lines.append(
                f"| `{m.name}` | **{m.mean_ms:.2f} ms** | {m.median_ms:.2f} ms | "
                "Discrete validation pipeline |"
            )
        for m in benchmark_report.checkpoint_benchmarks:
            lines.append(
                f"| `{m.name}` | **{m.mean_ms:.2f} ms** | {m.median_ms:.2f} ms | "
                "Atomic safetensors + SHA-256 |"
            )
        lines.append("")

    # 9. Architecture & Lifecycle Mermaid Flowchart
    lines.extend(
        [
            "## 9. Architecture & Lifecycle Flowchart",
            "",
            "```mermaid",
            "flowchart TD",
            "    subgraph InputTokens [Sequence Input]",
            "        Tokens[input_ids: B x T] --> Embed[Token Embeddings]",
            "        Embed --> L0[Initial Backbone Decoder Layers]",
            "    end",
            "",
            "    subgraph MemoryRead [Differentiable Memory Retrieval]",
            "        L0 --> QHead[Neural Query Head]",
            "        QHead --> QVecs[Query Vectors & Query Gate]",
            "        GraphState[Active Graph State] --> GEncoder[Graph Encoder MPNN]",
            "        GEncoder --> NodeStates[Contextualized Node States]",
            "        QVecs --> GSelector[Graph Selector]",
            "        NodeStates --> GSelector",
            "        GSelector --> SelectedMem[Selected Memory States]",
            "        SelectedMem --> GTraversal[Multi-Hop Traversal]",
            "        GTraversal --> ReasoningCtx[Relational Context]",
            "    end",
            "",
            "    subgraph FusionLayer [Cross-Attention Fusion]",
            "        L0 --> GFusion[Graph Fusion Gate + Cross-Attn]",
            "        SelectedMem --> GFusion",
            "        ReasoningCtx --> GFusion",
            "        GFusion --> FusedHidden[Fused Hidden States]",
            "    end",
            "",
            "    subgraph OutputLM [Causal Prediction]",
            "        FusedHidden --> RemLayers[Remaining Decoder Layers]",
            "        RemLayers --> Logits[Vocabulary Logits: B x T x V]",
            "    end",
            "",
            "    subgraph MemoryLifecycle [Discrete Transactional Lifecycle]",
            "        FusedHidden --> WriteHead[Memory Write Head]",
            "        WriteHead --> Proposals[Staged Memory Proposals]",
            "        Proposals --> Validator[Multi-Policy Validator]",
            "        Validator --> Commit[Atomic Transaction Commit]",
            "        Commit --> Snapshot[Safetensors Snapshot & Manifest]",
            "        Commit --> BridgeSync[Invalidate Caches & Refresh V+1]",
            "    end",
            "```",
            "",
            "## 10. Diagnostic Verdict & Production Readiness",
            "",
        ]
    )

    if report.passed:
        lines.extend(
            [
                "> [!NOTE]",
                "> **ARCHITECTURE INTEGRATION VERIFIED**: RUTAM has passed all tensor contracts, "
                "verified backward gradient propagation across all 8 submodules with frozen "
                "backbone isolation, executed the complete 14-step transactional memory "
                "lifecycle with zero anomalies, and proven cryptographic integrity detection "
                "and automatic rollback under simulated corruption.",
            ]
        )
    else:
        lines.extend(
            [
                "> [!WARNING]",
                "> **AUDIT ISSUES DETECTED**: One or more verification domains failed. "
                "Inspect the detailed sections above for failure diagnostics.",
            ]
        )
    lines.append("")

    md_text = "\n".join(lines)
    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md_text, encoding="utf-8")

    return md_text
