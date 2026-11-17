# Training Guide for Parameter-Shared Recurrent Reasoning in RUTAM

Developed by **Anvaya.sh** for the RUTAM graph-memory foundation.

---

## 1. Training Principles

Training deep parameter-shared recurrent architectures introduces specific gradient dynamics that differ
from standard feed-forward networks:
1. **Weight Tying Gradient Accumulation**: Because block weights $W$ are shared across all $N$ loops,
   gradients accumulate across iterations:
   $$\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=0}^{N-1} \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial W}$$
2. **Identity-Preserving Warm-Starts**: With `residual_scale_init = 0.001`, recurrent reasoning layers
   can be added to pre-trained or frozen backbones without disrupting initial language modeling loss.
3. **Adaptive Computation Time (ACT) Regularization**: When dynamic halting is enabled, a ponder cost penalty
   encourages the model to halt early on simple tokens and ponder longer on complex graph reasoning queries.

---

## 2. Loss Formulations

### 2.1 Causal Language Modeling Loss
The primary objective is standard next-token prediction computed from the final decoder output:
$$\mathcal{L}_{\text{LM}} = -\frac{1}{T-1} \sum_{i=1}^{T-1} \log P(x_{i+1} \mid x_{\le i})$$

### 2.2 Ponder Cost Regularization
When `dynamic_halting = True`, to prevent the model from always executing the maximum loop budget,
the cumulative ponder cost is penalized:
$$\mathcal{L}_{\text{ponder}} = \lambda_{\text{ponder}} \cdot \frac{1}{B} \sum_{b=1}^B N_b$$
where $N_b$ is the number of executed loops for batch element $b$, and $\lambda_{\text{ponder}} \in [0.001, 0.01]$.

### 2.3 Total Training Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{ponder}} + \mathcal{L}_{\text{graph}}$$

---

## 3. Recommended Optimization Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | `AdamW` | Decoupled weight decay on linear projections |
| Learning Rate (Recurrent) | `1e-4` to `3e-4` | Moderate rate for shared weights |
| Learning Rate (Residual Scale) | `5e-4` | Allows residual scale $\alpha$ to adapt gradually |
| Weight Decay | `0.01` | Applied to projection weights; omitted on LayerNorm/Embeddings |
| Gradient Clipping | `1.0` norm | Prevents gradient spikes across deep unrolled loops |
| Warmup Steps | `100` to `500` | Smooth initial parameter discovery |
| Min / Max Loops | `min_loops=1, max_loops=6` | Balanced compute budget |

---

## 4. Training Example Script

```python
import torch
from rutam.config import RUTAMConfig
from rutam.model.rutam_model import RUTAMForCausalLM

# 1. Configure RUTAM with parameter-shared recurrence enabled
config = RUTAMConfig().with_overrides({
    "recurrent": {
        "enabled": True,
        "hidden_dim": 128,
        "num_heads": 4,
        "default_loops": 3,
        "max_loops": 6,
        "min_loops": 1,
        "dynamic_halting": False,
        "residual_scale_init": 0.001,
    },
    "rutam_model": {
        "backbone_frozen": True,
    }
})

# 2. Instantiate model
model = RUTAMForCausalLM(config=config)

# 3. Partition parameters into optimizer groups
param_groups = model.get_parameter_groups(include_frozen=False)
optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

# 4. Training loop step
input_ids = torch.randint(0, 1000, (4, 32), dtype=torch.long)
labels = input_ids.clone()

model.train()
optimizer.zero_grad()

outputs = model(input_ids, labels=labels)
loss = outputs.loss

# Backpropagate through decoder layers and all recurrent loops
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

print(f"Loss: {loss.item():.4f}, Executed Loops: {outputs.recurrent_output.num_loops}")
```

---

## 5. Numerical Monitoring Guidelines

1. **Gate Saturation**: Verify `diagnostics["mean_residual_gate"]` stays in $[0.1, 0.9]$. Extreme 0.0 indicates
   bypassed recurrence; 1.0 indicates unbounded residual accumulation.
2. **Ponder Distribution**: Under dynamic halting, inspect `diagnostics["dynamic_halted"]` and `num_loops`
   across batches to ensure diverse pondering depth across simple and complex queries.
3. **Finite Check**: Ensure gradient norm remains finite without Inf/NaN across 8 unrolled loops.
