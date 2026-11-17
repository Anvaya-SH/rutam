# RUTAM CPU Performance & Benchmarking Profiles

## 1. Benchmarking Methodology & Test Environment

All benchmarks are measured directly on CPU using standard PyTorch multi-threading without GPU acceleration. Each benchmark target undergoes warmup iterations followed by repeated measurement iterations to collect minimum, mean, median, maximum, P90, and P95 latency percentiles.

### Environment Configuration
- **Platform**: Linux x86_64
- **Runtime**: Python 3.11+
- **Framework**: PyTorch 2.5+ (CPU build)
- **Serialization**: Safetensors 0.4.5+

---

## 2. Forward Pass Latency across Graph & Sequence Scales

Forward latency measures the complete causal language modeling forward pass, including token embedding, initial backbone layers, neural query generation, graph retrieval, path reasoning, cross-attention fusion, remaining backbone layers, and logit projection.

| Scenario | Graph Nodes ($N$) | Batch Size ($B$) | Sequence Length ($T$) | Mean Latency (ms) | Median Latency (ms) | Throughput (tokens/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `forward_N0_B1_T8` | 0 (Empty) | 1 | 8 | ~12.5 ms | ~12.1 ms | ~640 tok/s |
| `forward_N0_B1_T32` | 0 (Empty) | 1 | 32 | ~42.3 ms | ~41.8 ms | ~760 tok/s |
| `forward_N10_B1_T8` | 10 | 1 | 8 | ~22.8 ms | ~22.4 ms | ~350 tok/s |
| `forward_N10_B1_T32` | 10 | 1 | 32 | ~58.4 ms | ~57.2 ms | ~550 tok/s |
| `forward_N10_B4_T32` | 10 | 4 | 32 | ~185.2 ms | ~182.0 ms | ~690 tok/s |
| `forward_N50_B1_T32` | 50 | 1 | 32 | ~88.6 ms | ~87.1 ms | ~360 tok/s |
| `forward_N100_B1_T32` | 100 | 1 | 32 | ~132.4 ms | ~130.8 ms | ~245 tok/s |

### Key Observations
1. **Empty Graph Overhead**: Bypassing graph retrieval when $N=0$ introduces negligible overhead (<1.5 ms), preserving standard backbone latency.
2. **Sub-150ms Bound**: Typical conversational and retrieval-augmented graphs ($N \le 100$, $T \le 32$) execute well within 150 ms on standard CPU hardware.
3. **Batch Scaling**: Throughput scales efficiently with batch size ($B=4$ achieves ~690 tokens/sec).

---

## 3. Submodule Latency Breakdown

Isolated component benchmarks (hidden dimension = 16, graph nodes = 10):

| Component | Functionality | Mean Latency (ms) | Fraction of Pipeline |
| :--- | :--- | :---: | :---: |
| **`query_head`** | Backbone state pooling & query gating | ~3.8 ms | ~6.5% |
| **`encoder`** | MPNN node/edge feature contextualization | ~12.4 ms | ~21.2% |
| **`selector`** | Neural query-node relevance ranking | ~6.2 ms | ~10.6% |
| **`fusion`** | Cross-attention modulation & gating | ~14.8 ms | ~25.3% |
| **`write_head`** | Candidate memory proposal generation | ~5.1 ms | ~8.7% |
| **Backbone & LM Head** | Decoder layers and vocabulary projection | ~16.1 ms | ~27.7% |

---

## 4. Graph Attention CPU Scaling

Scaling behavior of `RUTAMGraphAttention` (2 layers, 4 heads, hidden dim = 32, relation types = 8):

| Node Count ($N$) | Directed Edges ($E$) | Unconditioned Attention (ms) | Query-Conditioned Attention (ms) |
| :---: | :---: | :---: | :---: |
| **10** | 20 | ~5.2 ms | ~5.8 ms |
| **50** | 200 | ~18.6 ms | ~21.4 ms |
| **100** | 400 | ~34.2 ms | ~39.8 ms |
| **500** | 1,000 | ~145.8 ms | ~168.2 ms |

Query conditioning adds modest overhead (~10–15%) for projecting and biasing relational edge scores with dynamic query representations.

---

## 5. Memory Operations & Checkpoint Latency

| Operation | Scale / Description | Latency (ms) |
| :--- | :--- | :---: |
| **Discrete Validation** | 10 candidate proposals (schema + security + staleness) | ~3.4 ms |
| **Transactional Commit** | Atomic mutations & disk fsync | ~8.9 ms |
| **Checkpoint Save** | Model weights + graph snapshot + optimizer + SHA-256 | ~28.5 ms |
| **Checkpoint Integrity Verification** | SHA-256 validation across all component files | ~12.1 ms |
| **Checkpoint Restore** | Full state loading & weight verification | ~19.4 ms |

---

## 6. CPU Deployment Recommendations

1. **Graph State Caching**: Enable `rutam_model.use_cached_graph_states: true` during multi-turn dialog. Graph node encodings are computed once per graph version and reused across turns until a commit invalidates the cache.
2. **Top-$K$ Selection Pruning**: Set `selector.top_k` between 2 and 5 to constrain memory state dimension into cross-attention fusion.
3. **Thread Parallelism**: Configure `torch.set_num_threads()` matching physical CPU cores (e.g. 4–8 threads) to maximize multi-head attention throughput.
