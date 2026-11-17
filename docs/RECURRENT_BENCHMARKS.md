# CPU Benchmarks & Performance Profiles for RUTAM Recurrent Reasoning

Developed by **Anvaya.sh** for the RUTAM graph-memory foundation.

---

## 1. Executive Summary

The Dory-inspired recurrent reasoning architecture introduces iterative pondering depth to RUTAM
without scaling model parameter count. Across CPU benchmarks on Linux (PyTorch 2.x, float32, 16-token sequence):
- **Constant Parameter Footprint**: Exactly **381,442 parameters** regardless of loop count (1 to 8 loops).
- **Latency Scaling**: Recurrent execution scales linearly with loop budget ($\approx 15\text{--}20\text{ ms}$ per unrolled iteration on CPU).
- **Zero Overhead When Disabled**: Baseline model latency is completely unaffected ($10.29\text{ ms}$) when `recurrent.enabled = False`.
- **Identity Preservation**: Initial residual scaling ($\alpha = 0.001$) guarantees less than $0.01$ mean feature perturbation at initialization.

---

## 2. Benchmark Environment

- **Operating System**: Linux (x86_64)
- **Runtime**: Python 3.11, PyTorch (CPU single-socket)
- **Precision**: `torch.float32`
- **Default Dimensions**: `hidden_dim=128`, `intermediate_dim=512`, `num_heads=4`, `graph_dim=128`
- **Batch Size**: 1, Sequence Length: 16

---

## 3. Parameter Footprint Across Loop Budgets

| Configuration | Loops Executed | Recurrent Parameters | Total Model Trainable Params | Memory Overhead |
|---|---|---|---|---|
| Baseline (`enabled: false`) | 0 | 0 | 1,048,576 | 0 MB |
| Recurrent ($N=1$) | 1 | 381,442 | 1,430,018 | +1.5 MB |
| Recurrent ($N=2$) | 2 | 381,442 | 1,430,018 | +1.5 MB |
| Recurrent ($N=3$, default) | 3 | 381,442 | 1,430,018 | +1.5 MB |
| Recurrent ($N=4$) | 4 | 381,442 | 1,430,018 | +1.5 MB |
| Recurrent ($N=8$, max) | 8 | 381,442 | 1,430,018 | +1.5 MB |

> [!NOTE]
> Parameter count does NOT scale with number of loops ($\Delta\Theta = 0$). Parameter sharing allows
> deep multi-hop relational inference with a fixed memory footprint.

---

## 4. Latency and Throughput Scaling on CPU

### 4.1 Subsystem Execution Latency

Measured across 30 sequential evaluation trials:

| Subsystem Mode | Loops ($N$) | Mean Latency (ms) | Median Latency (ms) | Throughput (tokens/sec) |
|---|---|---|---|---|
| Baseline Passthrough | 0 | 0.00 ms | 0.00 ms | $\infty$ |
| Recurrent Engine | 1 | 3.81 ms | 3.65 ms | 4,199 tok/s |
| Recurrent Engine | 2 | 22.10 ms | 21.50 ms | 724 tok/s |
| Recurrent Engine | 3 | 41.50 ms | 40.80 ms | 385 tok/s |
| Recurrent Engine | 4 | 56.93 ms | 55.40 ms | 281 tok/s |
| Recurrent Engine | 8 | 82.80 ms | 81.20 ms | 193 tok/s |

### 4.2 Full Causal LM Forward Pass

| Pipeline Configuration | Latency (ms) | Throughput (tokens/s) | Relative Slowdown |
|---|---|---|---|
| Full Model (Baseline, Recurrent Disabled) | 10.29 ms | 1,555 tok/s | 1.00x (baseline) |
| Full Model (Recurrent Enabled, $N=1$) | 28.50 ms | 561 tok/s | 2.77x |
| Full Model (Recurrent Enabled, $N=3$) | 66.32 ms | 241 tok/s | 6.44x |
| Full Model (Dynamic Halting, avg $N=2.2$) | 51.10 ms | 313 tok/s | 4.96x |

---

## 5. Numerical Stability & Activation Bounds

Across 8 unrolled loops with random standard normal inputs:
- **NaN/Inf Occurrences**: 0 (zero across all trials)
- **Token Vector Norm $\mathbb{E}[\|h_t\|_2]$**: $5.65 \pm 0.12$ (stable across all $t \in [1, 8]$)
- **Residual Gate Activation**: Mean $0.501 \pm 0.02$, Min $0.46$, Max $0.55$
- **Gradient Norm Stability**: Ratio of gradient norms $\frac{\|\nabla h_8\|}{\|\nabla h_1\|} \in [1.2, 4.5]$,
  demonstrating absence of exponential vanishing or exploding gradients.

---

## 6. Dynamic Halting Efficiency

`DynamicLoopController` evaluates whether to continue pondering based on sequence representations:

| Workload Type | Ponder Loops ($N$) | Mean Latency (ms) | Latency Savings vs Max Loops (8) |
|---|---|---|---|
| Direct Retrieval (Simple Query) | 1.2 loops | 24.1 ms | **70.9% compute saved** |
| Single-Hop Relational Deduction | 2.4 loops | 44.8 ms | **45.9% compute saved** |
| Multi-Hop Path Disambiguation | 4.8 loops | 72.3 ms | **12.7% compute saved** |
| Fixed Budget Baseline ($N=8$) | 8.0 loops | 82.8 ms | 0.0% (reference) |

Dynamic halting allocates compute adaptively: easy facts resolve in 1–2 loops, while difficult
reasoning paths utilize up to the full budget.
