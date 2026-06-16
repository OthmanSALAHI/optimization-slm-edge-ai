import json
import os
import random
from pathlib import Path


def load_config(config_path):
    """Load a YAML configuration file."""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return config or {}
    except ImportError:
        return _load_simple_yaml(config_path)


def _load_simple_yaml(config_path):
    """Small fallback parser for the simple config.yaml used in this project."""
    config = {}
    current_section = None

    with open(config_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.endswith(":"):
                current_section = stripped[:-1]
                config[current_section] = {}
                continue

            if ":" not in stripped:
                continue

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = _parse_simple_yaml_value(value.strip())

            if line.startswith(" ") and current_section:
                config[current_section][key] = value
            else:
                config[key] = value
                current_section = None

    return config


def _parse_simple_yaml_value(value):
    if value.isdigit():
        return int(value)

    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, file_path):
    """Save a Python object as a pretty JSON file."""
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        ensure_dir(parent_dir)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def set_seed(seed):
    """Set random seeds for reproducible processing and training."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean_text(value):
    """Safely convert missing values to clean strings."""
    if value is None:
        return ""

    text = str(value)
    return " ".join(text.strip().split())
