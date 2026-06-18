from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning Flan-T5 Small on the prepared data."""

    model_name: str = "google/flan-t5-small"
    train_file: str = "data/processed/train.csv"
    validation_file: str = "data/processed/validation.csv"
    output_dir: str = "results/models"
    metrics_dir: str = "results/metrics"
    logs_dir: str = "results/logs"
    checkpoints_dir: str = "results/checkpoints"
    optimizer_name: str = "adam"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    momentum: float = 0.9
    batch_size: int = 4
    epochs: int = 2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_input_length: int = 256
    max_target_length: int = 128
    seed: int = 42
    require_gpu: bool = True
    use_amp: bool = True
    num_workers: int = 0
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.06
    max_train_examples: int | None = None
    max_validation_examples: int | None = None
    max_eval_batches: int | None = None
    inference_samples: int = 16
    generation_max_new_tokens: int = 128
    lbfgs_max_iter: int = 4
    lbfgs_history_size: int = 10
    save_prediction_preview: bool = True


SUPPORTED_OPTIMIZERS = (
    "sgd",
    "sgd_momentum",
    "adam",
    "lbfgs",
    "newton_cg",
)


RUNNABLE_OPTIMIZERS = (
    "sgd",
    "sgd_momentum",
    "adam",
    "lbfgs",
)
