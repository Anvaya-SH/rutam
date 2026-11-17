# RUTAM Memory Training Objectives & Multi-Task Formulation

The neural memory training system optimizes multiple loss components simultaneously to train the network on memory gating, operation classification, embedding alignment, and graph retrieval.

## Supervised Loss Formulation

Total memory loss $\mathcal{L}_{\text{total}}$ is computed by `RUTAMMemoryLearningObjective`:

$$\mathcal{L}_{\text{total}} = \sum_{k} w_k \mathcal{L}_k$$

Where loss components $\mathcal{L}_k$ include:

1. **Write Decision Loss ($\mathcal{L}_{\text{write}}$)**:
   Binary cross-entropy loss supervised by `expected_write_decision`:
   $$\mathcal{L}_{\text{write}} = - \left[ y \log \sigma(s) + (1 - y) \log (1 - \sigma(s)) \right]$$

2. **Operation Classification Loss ($\mathcal{L}_{\text{op}}$)**:
   Multi-class cross-entropy over memory operations (`CREATE_NODE`, `UPDATE_NODE`, `PROPOSE_LINK`, `NO_OP`, `REQUEST_VALIDATION`):
   $$\mathcal{L}_{\text{op}} = - \log \left( \frac{\exp(z_c)}{\sum_j \exp(z_j)} \right)$$

3. **Embedding Alignment Loss ($\mathcal{L}_{\text{align}}$)**:
   Cosine distance loss pulling proposed candidate representations towards ground-truth reference embeddings:
   $$\mathcal{L}_{\text{align}} = 1 - \cos(\mathbf{e}_{\text{prop}}, \mathbf{e}_{\text{ref}})$$

4. **Query Gating Loss ($\mathcal{L}_{\text{gate}}$)**:
   Supervises the neural query gate to open when memory is relevant and close when no-op is desired:
   $$\mathcal{L}_{\text{gate}} = (g - y_{\text{gate}})^2$$

## Loss Balancing

- **Fixed Balancing**: Static weighting parameters defined in `MemoryLearningConfig`.
- **Uncertainty Balancing**: Homoscedastic task uncertainty weighting following Kendall & Gal (2018):
  $$\mathcal{L} = \sum_k \frac{1}{2 \sigma_k^2} \mathcal{L}_k + \log \sigma_k$$
