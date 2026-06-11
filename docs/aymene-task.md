Your task is to implement the fine-tuning part of the project.

The data preparation is already done and the processed files are available in:

* `data/processed/train.csv`
* `data/processed/validation.csv`
* `data/processed/test.csv`

You need to fine-tune the model `google/flan-t5-small` on the prepared instruction-tuning dataset.

The model input is stored in the column:

* `source_text`

The expected answer is stored in the column:

* `target_text`

You must implement the training pipeline in the existing fine-tuning files, especially:

* `src/train.py`
* `src/train_utils.py`
* `src/optimizers.py`
* `src/tokenizer_utils.py`
* `src/model_utils.py`
* `src/metrics.py`
* `src/training_config.py`

The goal is to fine-tune the same model several times using different optimizers:

* AdamW
* Adafactor
* Lion
* LAMB
* SGD

For each optimizer, keep the same dataset, model, batch size, number of epochs, and preprocessing. Only the optimizer should change. This is important to make the comparison fair.

The training command should look like this:

```bash
python src/train.py --optimizer adamw
python src/train.py --optimizer adafactor
python src/train.py --optimizer lion
python src/train.py --optimizer lamb
python src/train.py --optimizer sgd
```

Each trained model must be saved separately:

```text
results/models/flan_t5_adamw/
results/models/flan_t5_adafactor/
results/models/flan_t5_lion/
results/models/flan_t5_lamb/
results/models/flan_t5_sgd/
```

For each optimizer, save the training metrics in:

```text
results/metrics/train_<optimizer>.json
```

The metrics should include:

* training loss
* validation loss
* training time
* memory usage
* final learning rate
* number of epochs
* optimizer name

After this step, we should be able to compare which optimizer gives the best fine-tuning result before compression.
