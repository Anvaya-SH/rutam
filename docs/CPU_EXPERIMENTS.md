# RUTAM Local CPU Training Experiments

This document records the measured results of reproducible neural training experiments performed on local CPU hardware without GPU acceleration.

## Experiment 1: Synthetic Overfitting Benchmark

- **Target**: Trainable `RUTAMMemoryWriteHead` with frozen `SyntheticBackbone`.
- **Dataset**: `data/memory_training/train.jsonl` (815 examples) and `data/memory_training/validation.jsonl` (100 examples).
- **Batch Size**: 4
- **Optimizer**: AdamW (`lr = 0.005`, `weight_decay = 0.01`, `max_grad_norm = 1.0`)
- **Device**: Intel Core i3 CPU (PyTorch Float32)
- **Gradient Accumulation Steps**: 1

### Measured Results

| Metric | Measured Value |
|---|---|
| Trainable Parameters | 13,528 |
| Frozen Parameters | 37,664 |
| Total Steps | 10 |
| Initial Total Loss | 2.7033 |
| Final Total Loss | 2.6234 |
| Loss Change ($\Delta$) | -0.0799 (Decreasing) |
| Wall-Clock Runtime | 0.23 seconds |
| Validation Loss | 2.7305 |
| Write Decision Precision | 0.3800 |
| Write Decision Recall | 0.3800 |
| Write Decision F1 | 0.3752 |

## Evaluation on Held-Out Test Set (85 Examples)

- **Mean Total Loss**: 2.6906
- **Write Decision Loss**: 0.6747
- **Operation Loss**: 2.0159
- **Write Precision**: 0.4848
- **Write Recall**: 0.4129
- **Write F1**: 0.4301
- **False Positive Rate**: 0.3106
- **Operation Accuracy**: 0.0227
