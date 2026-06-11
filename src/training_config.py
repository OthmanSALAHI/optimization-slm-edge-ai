from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Default configuration values for the future fine-tuning step."""

    model_name: str = "google/flan-t5-small"
    train_file: str = "data/processed/train.csv"
    validation_file: str = "data/processed/validation.csv"
    output_dir: str = "results/models"
    optimizer_name: str = "adamw"
    learning_rate: float = 5e-5
    batch_size: int = 4
    epochs: int = 2
    max_input_length: int = 256
    max_target_length: int = 128
    seed: int = 42
