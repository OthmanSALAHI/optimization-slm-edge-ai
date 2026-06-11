import argparse
import csv
import json
import os

from utils import clean_text, ensure_dir, load_config, save_json, set_seed


DEFAULT_CONFIG_PATH = "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare yahma/alpaca-cleaned for light instruction tuning."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--train_size", type=int, default=None, help="Number of training examples")
    parser.add_argument("--val_size", type=int, default=None, help="Number of validation examples")
    parser.add_argument("--test_size", type=int, default=None, help="Number of test examples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling")
    return parser.parse_args()


def get_nested(config, keys, default):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_source_text(instruction, input_text):
    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}\nAnswer:"
    return f"Instruction: {instruction}\nAnswer:"


def convert_example(example):
    instruction = clean_text(example.get("instruction", ""))
    input_text = clean_text(example.get("input", ""))
    output = clean_text(example.get("output", ""))

    source_text = build_source_text(instruction, input_text)
    target_text = output

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "source_text": source_text,
        "target_text": target_text,
        "source_length": len(source_text.split()),
        "target_length": len(target_text.split()),
    }


def save_csv(rows, file_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_jsonl(rows, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hugging_face_dataset(dataset_name, dataset_split):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The Hugging Face datasets package is required. "
            "Install it with: pip install datasets"
        ) from exc

    return load_dataset(dataset_name, split=dataset_split)


def save_split(rows, split_name, output_dir):
    csv_path = os.path.join(output_dir, f"{split_name}.csv")
    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")

    print(f"Saving {split_name} split to {csv_path}")
    save_csv(rows, csv_path)

    print(f"Saving {split_name} split to {jsonl_path}")
    save_jsonl(rows, jsonl_path)


def calculate_stats(total_original_examples, splits):
    all_rows = []
    for rows in splits.values():
        all_rows.extend(rows)

    source_lengths = [row["source_length"] for row in all_rows]
    target_lengths = [row["target_length"] for row in all_rows]

    return {
        "total_original_examples": total_original_examples,
        "train_size": len(splits["train"]),
        "validation_size": len(splits["validation"]),
        "test_size": len(splits["test"]),
        "average_source_length": round(sum(source_lengths) / len(source_lengths), 2),
        "average_target_length": round(sum(target_lengths) / len(target_lengths), 2),
        "max_source_length": max(source_lengths),
        "max_target_length": max(target_lengths),
    }


def print_stats(stats):
    print("\nDataset statistics:")
    print(f"Total original examples: {stats['total_original_examples']}")
    print(f"Train size: {stats['train_size']}")
    print(f"Validation size: {stats['validation_size']}")
    print(f"Test size: {stats['test_size']}")
    print(f"Average source length: {stats['average_source_length']}")
    print(f"Average target length: {stats['average_target_length']}")
    print(f"Max source length: {stats['max_source_length']}")
    print(f"Max target length: {stats['max_target_length']}")


def main():
    args = parse_args()

    print("Starting data preparation.")
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    dataset_name = get_nested(config, ["data", "dataset_name"], "yahma/alpaca-cleaned")
    dataset_split = get_nested(config, ["data", "dataset_split"], "train")
    output_dir = get_nested(config, ["data", "processed_dir"], "data/processed")
    seed = args.seed if args.seed is not None else get_nested(config, ["seed"], 42)
    train_size = args.train_size if args.train_size is not None else get_nested(config, ["data", "train_size"], 2000)
    val_size = args.val_size if args.val_size is not None else get_nested(config, ["data", "val_size"], 500)
    test_size = args.test_size if args.test_size is not None else get_nested(config, ["data", "test_size"], 500)

    print(f"Using dataset: {dataset_name}")
    print(f"Using split: {dataset_split}")
    print(f"Using seed: {seed}")
    print(f"Requested sizes: train={train_size}, validation={val_size}, test={test_size}")

    set_seed(seed)
    ensure_dir(output_dir)

    print("Downloading dataset from Hugging Face datasets.")
    dataset = load_hugging_face_dataset(dataset_name, dataset_split)
    total_original_examples = len(dataset)
    print(f"Loaded {total_original_examples} original examples.")

    total_requested = train_size + val_size + test_size
    if total_requested > total_original_examples:
        raise ValueError(
            f"Requested {total_requested} examples, but the dataset only has "
            f"{total_original_examples} examples."
        )

    print("Shuffling dataset.")
    shuffled_dataset = dataset.shuffle(seed=seed)

    print("Creating train, validation, and test subsets.")
    train_dataset = shuffled_dataset.select(range(0, train_size))
    validation_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
    test_dataset = shuffled_dataset.select(
        range(train_size + val_size, train_size + val_size + test_size)
    )

    print("Converting examples to instruction tuning format.")
    splits = {
        "train": [convert_example(example) for example in train_dataset],
        "validation": [convert_example(example) for example in validation_dataset],
        "test": [convert_example(example) for example in test_dataset],
    }

    print("Saving processed files.")
    for split_name, rows in splits.items():
        save_split(rows, split_name, output_dir)

    print("Calculating dataset statistics.")
    stats = calculate_stats(total_original_examples, splits)
    print_stats(stats)

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    print(f"Saving dataset statistics to {stats_path}")
    save_json(stats, stats_path)

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()
