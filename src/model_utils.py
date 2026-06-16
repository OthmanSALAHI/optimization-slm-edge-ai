from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM


def get_device(require_gpu=True):
    """Return the training device and require CUDA by default."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_gpu:
        raise RuntimeError(
            "CUDA GPU is required for this fine-tuning script. "
            "Run on a GPU machine, or set --allow_cpu only for debugging."
        )

    return torch.device("cpu")


def load_model(model_name, device):
    """Load Flan-T5 Small and move it to the selected device."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model.to(device)


def save_model(model, tokenizer, output_dir):
    """Save the fine-tuned model and tokenizer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


def count_model_parameters(model):
    """Count total and trainable model parameters."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def get_model_size_mb(model_dir):
    """Measure the size of a saved model directory in megabytes."""
    model_path = Path(model_dir)
    total_bytes = 0

    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            total_bytes += file_path.stat().st_size

    return round(total_bytes / (1024 * 1024), 2)
