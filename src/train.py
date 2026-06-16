import argparse
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from metrics import (
    GpuEnergyTracker,
    compute_compression_performance_tradeoff,
    compute_inference_metrics,
    compute_loss,
    compute_memory_usage,
    compute_perplexity,
    compute_training_time,
)
from model_utils import (
    count_model_parameters,
    get_device,
    get_model_size_mb,
    load_model,
    save_model,
)
from optimizers import get_optimizer
from tokenizer_utils import decode_predictions, load_tokenizer
from training_config import RUNNABLE_OPTIMIZERS, SUPPORTED_OPTIMIZERS, TrainingConfig
from train_utils import (
    create_dataloader,
    create_trainer,
    load_processed_dataset,
    save_training_results,
    tokenize_dataset,
)
from utils import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune google/flan-t5-small and compare optimizers."
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=[*SUPPORTED_OPTIMIZERS, "all"],
        help="Optimizer to use. Use 'all' to run every runnable optimizer.",
    )
    parser.add_argument("--model_name", default=TrainingConfig.model_name)
    parser.add_argument("--train_file", default=TrainingConfig.train_file)
    parser.add_argument("--validation_file", default=TrainingConfig.validation_file)
    parser.add_argument("--output_dir", default=TrainingConfig.output_dir)
    parser.add_argument("--metrics_dir", default=TrainingConfig.metrics_dir)
    parser.add_argument("--logs_dir", default=TrainingConfig.logs_dir)
    parser.add_argument("--checkpoints_dir", default=TrainingConfig.checkpoints_dir)
    parser.add_argument("--learning_rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=TrainingConfig.momentum)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--max_input_length", type=int, default=TrainingConfig.max_input_length)
    parser.add_argument("--max_target_length", type=int, default=TrainingConfig.max_target_length)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_validation_examples", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--inference_samples", type=int, default=TrainingConfig.inference_samples)
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=TrainingConfig.generation_max_new_tokens,
    )
    parser.add_argument("--lbfgs_max_iter", type=int, default=TrainingConfig.lbfgs_max_iter)
    parser.add_argument(
        "--lbfgs_history_size",
        type=int,
        default=TrainingConfig.lbfgs_history_size,
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU execution for debugging. By default this script requires CUDA.",
    )
    return parser.parse_args()


def build_config(args, optimizer_name):
    return TrainingConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        logs_dir=args.logs_dir,
        checkpoints_dir=args.checkpoints_dir,
        optimizer_name=optimizer_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        seed=args.seed,
        require_gpu=not args.allow_cpu,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
        max_eval_batches=args.max_eval_batches,
        inference_samples=args.inference_samples,
        generation_max_new_tokens=args.generation_max_new_tokens,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history_size,
    )


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, optimizer, dataloader, device, optimizer_name):
    model.train()
    batch_losses = []

    progress_bar = tqdm(dataloader, desc=f"Training ({optimizer_name})")
    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)

        if optimizer_name == "lbfgs":

            def closure():
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
        else:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

        batch_losses.append(loss_value)
        progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

    return compute_loss(batch_losses)


@torch.no_grad()
def evaluate_model(model, dataloader, device, max_eval_batches=None):
    model.eval()
    losses = []

    for batch_index, batch in enumerate(tqdm(dataloader, desc="Validation")):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break

        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss.item())

    return compute_loss(losses)


@torch.no_grad()
def measure_inference(model, tokenizer, dataframe, config, device):
    model.eval()
    samples = dataframe.head(config.inference_samples)
    if samples.empty:
        return compute_inference_metrics(0, 0), []

    prompts = samples["source_text"].tolist()
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_input_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=config.generation_max_new_tokens,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    predictions = decode_predictions(generated_ids, tokenizer)

    preview = []
    for source_text, prediction, target_text in zip(
        prompts,
        predictions,
        samples["target_text"].tolist(),
    ):
        preview.append(
            {
                "source_text": source_text,
                "prediction": prediction,
                "target_text": target_text,
            }
        )

    inference_metrics = compute_inference_metrics(
        total_time_seconds=end_time - start_time,
        number_of_samples=len(samples),
    )
    return inference_metrics, preview


def prepare_directories(config):
    ensure_dir(config.output_dir)
    ensure_dir(config.metrics_dir)
    ensure_dir(config.logs_dir)
    ensure_dir(config.checkpoints_dir)


def train_with_optimizer(config):
    print(f"\nStarting fine-tuning with optimizer: {config.optimizer_name}")
    print("This run uses the prepared source_text and target_text columns.")

    if config.optimizer_name == "newton_cg":
        raise NotImplementedError(
            "Newton-CG is listed for theoretical comparison, but this project "
            "does not implement it for Flan-T5 because it is not practical for "
            "large transformer fine-tuning with standard PyTorch optimizers."
        )

    prepare_directories(config)
    set_seed(config.seed)

    device = get_device(require_gpu=config.require_gpu)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    print("Loading tokenizer.")
    tokenizer = load_tokenizer(config.model_name)

    print("Loading model.")
    model = load_model(config.model_name, device)
    parameter_counts = count_model_parameters(model)

    print("Loading processed train and validation data.")
    train_dataframe = load_processed_dataset(
        config.train_file,
        max_examples=config.max_train_examples,
    )
    validation_dataframe = load_processed_dataset(
        config.validation_file,
        max_examples=config.max_validation_examples,
    )

    print("Tokenizing datasets.")
    train_dataset = tokenize_dataset(train_dataframe, tokenizer, config)
    validation_dataset = tokenize_dataset(validation_dataframe, tokenizer, config)

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_dataloader = create_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    print("Creating optimizer.")
    optimizer = get_optimizer(model.parameters(), config)
    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=config,
    )

    energy_tracker = GpuEnergyTracker()
    energy_tracker.start()
    training_start_time = time.perf_counter()
    epoch_metrics = []

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        train_loss = train_one_epoch(
            trainer["model"],
            trainer["optimizer"],
            trainer["train_dataloader"],
            device,
            config.optimizer_name,
        )
        validation_loss = evaluate_model(
            trainer["model"],
            trainer["validation_dataloader"],
            device,
            max_eval_batches=config.max_eval_batches,
        )

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": compute_perplexity(validation_loss),
            }
        )

    training_end_time = time.perf_counter()
    energy_tracker.stop()

    training_time_seconds = compute_training_time(
        training_start_time,
        training_end_time,
    )
    train_examples_seen = len(train_dataframe) * config.epochs
    training_examples_per_second = (
        train_examples_seen / training_time_seconds if training_time_seconds > 0 else None
    )

    print("Measuring inference latency on validation examples.")
    inference_metrics, prediction_preview = measure_inference(
        model=model,
        tokenizer=tokenizer,
        dataframe=validation_dataframe,
        config=config,
        device=device,
    )

    model_output_dir = Path(config.output_dir) / f"flan_t5_{config.optimizer_name}"
    print(f"Saving model to {model_output_dir}")
    save_model(model, tokenizer, model_output_dir)
    final_model_size_mb = get_model_size_mb(model_output_dir)

    final_train_loss = epoch_metrics[-1]["train_loss"] if epoch_metrics else None
    final_validation_loss = epoch_metrics[-1]["validation_loss"] if epoch_metrics else None
    final_validation_perplexity = (
        epoch_metrics[-1]["validation_perplexity"] if epoch_metrics else None
    )

    metrics = {
        "model_name": config.model_name,
        "optimizer_name": config.optimizer_name,
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "momentum": config.momentum if config.optimizer_name == "sgd_momentum" else None,
        "train_examples": len(train_dataframe),
        "validation_examples": len(validation_dataframe),
        "parameter_counts": parameter_counts,
        "epoch_metrics": epoch_metrics,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "final_validation_perplexity": final_validation_perplexity,
        "training_time_seconds": training_time_seconds,
        "training_examples_per_second": training_examples_per_second,
        "memory_usage": compute_memory_usage(device),
        "final_model_size_mb": final_model_size_mb,
        "inference": inference_metrics,
        "energy_consumption": energy_tracker.summary(),
        "compression_performance_tradeoff": compute_compression_performance_tradeoff(
            final_model_size_mb=final_model_size_mb,
            validation_loss=final_validation_loss,
            validation_perplexity=final_validation_perplexity,
        ),
        "prediction_preview": prediction_preview[:3],
    }

    metrics_path = save_training_results(
        metrics=metrics,
        metrics_dir=config.metrics_dir,
        optimizer_name=config.optimizer_name,
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Completed fine-tuning with optimizer: {config.optimizer_name}")

    return metrics


def main():
    args = parse_args()

    if args.optimizer == "all":
        optimizer_names = RUNNABLE_OPTIMIZERS
    else:
        optimizer_names = (args.optimizer,)

    for optimizer_name in optimizer_names:
        config = build_config(args, optimizer_name)
        train_with_optimizer(config)


if __name__ == "__main__":
    main()
