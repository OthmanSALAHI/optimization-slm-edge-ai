# Data Preparation

This project uses the Hugging Face dataset `yahma/alpaca-cleaned` for light instruction tuning.

The data preparation script downloads the dataset, shuffles it with the configured seed, creates train, validation, and test subsets, and saves processed files under `data/processed/`.

## Processed Files

- `train.csv`: training examples in CSV format.
- `validation.csv`: validation examples in CSV format.
- `test.csv`: test examples in CSV format.
- `train.jsonl`: training examples in JSON Lines format.
- `validation.jsonl`: validation examples in JSON Lines format.
- `test.jsonl`: test examples in JSON Lines format.
- `dataset_stats.json`: summary statistics for the processed dataset.

Each processed row contains:

- `instruction`: cleaned instruction text.
- `input`: cleaned optional input text.
- `output`: cleaned answer text from the original dataset.
- `source_text`: formatted prompt given to the model.
- `target_text`: expected answer for instruction tuning.
- `source_length`: word count of `source_text`.
- `target_length`: word count of `target_text`.

## Text Format

When the original example has an input, `source_text` is:

```text
Instruction: <instruction>
Input: <input>
Answer:
```

When the original example has no input, `source_text` is:

```text
Instruction: <instruction>
Answer:
```

The `target_text` field is the cleaned original `output` field.
