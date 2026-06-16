# Fine-Tuning and Optimizer Comparison

This document explains the fine-tuning part of the project:

**Optimisation et compression des Small Language Models pour applications Edge AI**

## Goal

The goal is to fine-tune `google/flan-t5-small` on the prepared instruction-tuning dataset and compare optimization algorithms.

The comparison should help identify which optimizer gives the best compromise between:

- model quality
- training speed
- memory consumption
- inference latency
- energy consumption
- future compression and performance

## Dataset

The training data is already prepared in:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

The fine-tuning script uses:

- `source_text` as the model input
- `target_text` as the expected answer

## Model

The model used for fine-tuning is:

```text
google/flan-t5-small
```

## Optimizers

The project compares:

- `sgd`: Stochastic Gradient Descent
- `sgd_momentum`: SGD with momentum
- `adam`: Adam optimizer
- `lbfgs`: Limited-memory BFGS
- `newton_cg`: documented as unsupported for practical Flan-T5 training

L-BFGS is included because this is an optimization module project. In practice, it is usually expensive for transformer models because it needs closure-based optimization steps.

Newton-CG is not implemented because it requires Hessian-vector products and is not available as a standard optimizer for this transformer fine-tuning workflow.

## GPU Requirement

The training script is designed to run on GPU.

By default, it requires CUDA:

```bash
python src/train.py --optimizer adam
```

If no CUDA GPU is available, the script stops before training.

For debugging only, CPU can be allowed with:

```bash
python src/train.py --optimizer adam --allow_cpu
```

## Commands

Train with one optimizer:

```bash
python src/train.py --optimizer sgd
python src/train.py --optimizer sgd_momentum
python src/train.py --optimizer adam
python src/train.py --optimizer lbfgs
```

Run all runnable optimizers:

```bash
python src/train.py --optimizer all
```

## Saved Models

Each optimizer saves a separate model directory:

```text
results/models/flan_t5_sgd/
results/models/flan_t5_sgd_momentum/
results/models/flan_t5_adam/
results/models/flan_t5_lbfgs/
```

## Saved Metrics

Each optimizer saves metrics in:

```text
results/metrics/train_<optimizer>.json
```

The metrics include:

- `final_model_size_mb`
- `training_time_seconds`
- `training_examples_per_second`
- `final_train_loss`
- `final_validation_loss`
- `final_validation_perplexity`
- `memory_usage`
- `inference_time_seconds`
- `latency_ms_per_sample`
- `energy_consumption`
- `compression_performance_tradeoff`

## What to Compare

After running the optimizers, compare:

- **Final model size:** smaller is better for Edge AI.
- **Training speed:** more examples per second is better.
- **Fine-tuning quality:** lower validation loss and perplexity are better.
- **Memory consumption:** lower GPU memory is better for limited hardware.
- **Inference time:** lower total generation time is better.
- **Latency:** lower milliseconds per sample is better.
- **Energy consumption:** lower energy use is better for embedded or edge devices.
- **Compression/performance tradeoff:** compare this fine-tuned baseline later with quantized or compressed versions.
