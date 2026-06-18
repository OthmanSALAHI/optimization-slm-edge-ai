# Optimisation et compression des Small Language Models pour applications Edge AI

This project prepares and fine-tunes `google/flan-t5-small` on a light instruction-tuning dataset, then compares optimization algorithms for an Edge AI context.

## Data preparation

The processed data is stored in:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

The model input column is:

- `source_text`

The target answer column is:

- `target_text`

## Fine-tuning

The fine-tuning code is implemented in:

- `src/train.py`
- `src/train_utils.py`
- `src/optimizers.py`
- `src/tokenizer_utils.py`
- `src/model_utils.py`
- `src/metrics.py`
- `src/training_config.py`

The project compares these optimizers:

- SGD
- SGD with momentum
- Adam
- L-BFGS
- Newton-CG, documented as unsupported for practical Flan-T5 training

Run one optimizer:

```bash
python src/train.py --optimizer sgd
python src/train.py --optimizer sgd_momentum
python src/train.py --optimizer adam
python src/train.py --optimizer lbfgs
```

Run every runnable optimizer:

```bash
python src/train.py --optimizer all
```

Recommended GPU command for a fair small experiment:

```bash
python src/train.py --optimizer all --batch_size 4 --gradient_accumulation_steps 2 --epochs 2 --learning_rate 5e-5
```

By default, the script requires a CUDA GPU. It will stop if no GPU is available.

For CPU debugging only:

```bash
python src/train.py --optimizer adam --allow_cpu
```

## Outputs

Each optimizer saves a separate model:

```text
results/models/flan_t5_sgd/
results/models/flan_t5_sgd_momentum/
results/models/flan_t5_adam/
results/models/flan_t5_lbfgs/
```

Each optimizer saves a metrics file:

```text
results/metrics/train_<optimizer>.json
```

The metrics include:

- final model size
- training speed
- fine-tuning quality using validation loss and perplexity
- memory consumption
- inference time
- latency
- energy consumption estimate when NVIDIA power telemetry is available
- compression/performance baseline for later quantization comparison

The script also updates:

```text
results/metrics/optimizer_summary.csv
```

This file is the easiest file to use for comparing optimizers in a report.

Newton-CG is kept in the optimizer list for theoretical comparison, but it is not implemented because standard PyTorch fine-tuning for transformer models does not provide a practical Newton-CG optimizer.
