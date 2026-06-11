# Data Preparation

This document explains the completed data preparation step for the project:

**Optimisation et compression des Small Language Models pour applications Edge AI**

## Goal

The goal of this step is to prepare a small instruction-tuning dataset that can later be used to fine-tune `google/flan-t5-small`.

This step only prepares data. It does not train, evaluate, or quantize a model.

## Dataset

The dataset used is:

```text
yahma/alpaca-cleaned
```

It is downloaded with the Hugging Face `datasets` library.

The configured split is:

```text
train
```

## Configuration

The default configuration is stored in:

```text
config.yaml
```

Current default values:

```yaml
seed: 42

data:
  dataset_name: yahma/alpaca-cleaned
  dataset_split: train
  processed_dir: data/processed
  train_size: 2000
  val_size: 500
  test_size: 500
```

## Script

The data preparation script is:

```text
src/prepare_data.py
```

It can be run with:

```bash
python src/prepare_data.py
```

It also supports custom split sizes:

```bash
python src/prepare_data.py --train_size 1000 --val_size 200 --test_size 200
```

## Processing Steps

The script performs these steps:

1. Loads settings from `config.yaml`.
2. Downloads `yahma/alpaca-cleaned`.
3. Shuffles the dataset using the configured seed.
4. Creates train, validation, and test subsets.
5. Cleans missing or empty `instruction`, `input`, and `output` fields.
6. Converts each example into instruction-tuning format.
7. Saves CSV and JSONL files under `data/processed/`.
8. Computes and saves dataset statistics.

## Instruction-Tuning Format

Each processed row contains:

- `instruction`
- `input`
- `output`
- `source_text`
- `target_text`
- `source_length`
- `target_length`

If the original example has an input, `source_text` is:

```text
Instruction: <instruction>
Input: <input>
Answer:
```

If the original example has no input, `source_text` is:

```text
Instruction: <instruction>
Answer:
```

The `target_text` field is:

```text
<output>
```

## Processed Files

The processed files are stored in:

```text
data/processed/
```

Expected files:

- `train.csv`
- `validation.csv`
- `test.csv`
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `dataset_stats.json`

The CSV files are useful for quick inspection with pandas or spreadsheet tools.

The JSONL files are useful for machine learning pipelines and line-by-line processing.

## Current Dataset Statistics

The current processed dataset statistics are:

```json
{
  "total_original_examples": 51760,
  "train_size": 2000,
  "validation_size": 500,
  "test_size": 500,
  "average_source_length": 17.14,
  "average_target_length": 109.21,
  "max_source_length": 279,
  "max_target_length": 450
}
```

Lengths are counted as word counts.

## Utility Functions

Common helper functions are stored in:

```text
src/utils.py
```

They include:

- `load_config`
- `ensure_dir`
- `save_json`
- `set_seed`
- `clean_text`

These utilities keep the data preparation script simple and easier to read.

## Notebook

An exploratory data analysis notebook is available at:

```text
notebooks/eda_dataset.ipynb
```

It can be used to inspect the raw dataset, text lengths, missing values, and examples of the final prompt format.

## Next Step

The next project step is fine-tuning preparation and training implementation.

The fine-tuning step should use:

- `source_text` as the model input.
- `target_text` as the expected output.
