Your task is to implement and run the fine-tuning part of the project on a GPU machine.

The data preparation is already done and the processed files are available in:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

Fine-tune:

```text
google/flan-t5-small
```

The model input is stored in:

```text
source_text
```

The expected answer is stored in:

```text
target_text
```

## Optimizers to Compare

Compare the following optimization algorithms:

- SGD
- SGD with momentum
- Adam
- L-BFGS
- Newton-CG, for theoretical discussion only

Newton-CG is documented as unsupported in the code because it is not practical for standard Flan-T5 fine-tuning with PyTorch.

## Training Commands

Run one optimizer:

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

The script requires CUDA by default. Use CPU only for debugging:

```bash
python src/train.py --optimizer adam --allow_cpu
```

## Model Outputs

Each trained model is saved separately:

```text
results/models/flan_t5_sgd/
results/models/flan_t5_sgd_momentum/
results/models/flan_t5_adam/
results/models/flan_t5_lbfgs/
```

## Metrics Outputs

For each optimizer, metrics are saved in:

```text
results/metrics/train_<optimizer>.json
```

The metrics include:

- training loss
- validation loss
- validation perplexity
- training time
- training speed
- memory usage
- final model size
- inference time
- latency
- energy consumption estimate when available
- compression/performance baseline

After this step, the optimizer results can be compared before model compression.
