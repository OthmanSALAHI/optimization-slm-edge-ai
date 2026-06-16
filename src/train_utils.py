import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer_utils import preprocess_function


REQUIRED_COLUMNS = {
    "instruction",
    "input",
    "output",
    "source_text",
    "target_text",
    "source_length",
    "target_length",
}


class InstructionTuningDataset(Dataset):
    """Torch dataset for the prepared source_text and target_text columns."""

    def __init__(self, dataframe, tokenizer, max_input_length, max_target_length):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        encoded = preprocess_function(
            {
                "source_text": row["source_text"],
                "target_text": row["target_text"],
            },
            tokenizer=self.tokenizer,
            max_input_length=self.max_input_length,
            max_target_length=self.max_target_length,
        )

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
        }


def load_processed_dataset(file_path, max_examples=None):
    """Load a processed CSV file created by src/prepare_data.py."""
    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset file not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path).fillna("")
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_path} is missing required columns: {sorted(missing_columns)}"
        )

    if max_examples is not None:
        dataframe = dataframe.head(max_examples)

    return dataframe


def tokenize_dataset(dataframe, tokenizer, config):
    """Create a Torch dataset from a processed dataframe."""
    return InstructionTuningDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )


def create_dataloader(dataset, batch_size, shuffle):
    """Create a DataLoader for training or validation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def create_trainer(model, optimizer, train_dataloader, validation_dataloader, config):
    """Collect training objects in one dictionary for a simple custom loop."""
    return {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "config": config,
    }


def save_training_results(metrics, metrics_dir, optimizer_name):
    """Save training metrics as JSON."""
    output_dir = Path(metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"train_{optimizer_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    return metrics_path
