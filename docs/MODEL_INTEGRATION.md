# RUTAM Model Integration Guide

This guide details how to integrate custom language model backbones, external storage backends, and deployment runtimes with RUTAM v0.1.

---

## 1. Integrating Custom Backbones with `BackboneProtocol`

Any neural language model (HuggingFace, LFM, PyTorch Transformer, or hybrid SSM) can be plugged directly into RUTAM by conforming to `BackboneProtocol`.

### The `BackboneProtocol` Contract

```python
from typing import Protocol, runtime_checkable
import torch
from torch import nn

@runtime_checkable
class BackboneProtocol(Protocol):
    def get_hidden_dim(self) -> int:
        """Return backbone hidden dimension D."""
        ...

    def get_vocab_size(self) -> int:
        """Return vocabulary size V."""
        ...

    def get_num_layers(self) -> int:
        """Return total count of decoder layers."""
        ...

    def get_input_embeddings(self) -> nn.Module:
        """Return embedding module converting token IDs to vectors."""
        ...

    def get_layers(self) -> Sequence[nn.Module]:
        """Return ordered list of decoder layers."""
        ...

    def get_output_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project final hidden states [B, T, D] to unnormalized logits [B, T, V]."""
        ...
```

### Implementing a Custom Backbone Adapter

```python
import torch
from torch import nn
from rutam.model import BackboneProtocol

class CustomTransformerAdapter(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def get_hidden_dim(self) -> int:
        return self.model.config.hidden_size

    def get_vocab_size(self) -> int:
        return self.model.config.vocab_size

    def get_num_layers(self) -> int:
        return len(self.model.model.layers)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.model.embed_tokens

    def get_layers(self) -> list[nn.Module]:
        return list(self.model.model.layers)

    def get_output_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden_states)
```

---

## 2. Attaching Backbone via `RUTAMModelFactory`

Use `RUTAMModelFactory.attach_backbone` to construct an integrated model around your custom backbone:

```python
from rutam import RUTAMConfig, RUTAMModelFactory

config = RUTAMConfig().with_mode("full_rutam")
adapter = CustomTransformerAdapter(my_hf_model)

# Attaches backbone and binds all neural graph and recurrent modules
model = RUTAMModelFactory.attach_backbone(
    backbone=adapter,
    config=config,
    freeze_backbone=True,  # Keeps backbone frozen while training graph/recurrent modules
)
```

---

## 3. Persistent Graph Storage Backends

RUTAM abstracts graph persistence through `GraphStorage`.

### Saving and Reloading Snapshots
```python
from rutam.graph.storage import GraphStorage

# Save graph snapshot
storage = GraphStorage("data/checkpoints/run_01")
storage.save(model.graph)

# Restore snapshot in new model
loaded_graph = storage.load()
reloaded_model = RUTAMModelFactory.create_model(config, graph=loaded_graph)
```

---

## 4. Fine-Tuning Parameter Groups

RUTAM cleanly partitions trainable parameters according to operational policies:

```python
import torch

# Extract optimizer parameter groups
optimizer = torch.optim.AdamW(
    model.get_parameter_groups(include_frozen=False, only_active=True),
    lr=1e-4,
    weight_decay=0.01,
)
```
When `include_frozen=False`, frozen backbone weights are excluded from the optimizer footprint, saving memory.
