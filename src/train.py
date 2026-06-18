import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import get_scheduler

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
    save_prediction_preview,
    save_training_results,
    tokenize_dataset,
)
from utils import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune google/flan-t5-small and compare optimizers fairly."
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
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=TrainingConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--max_grad_norm", type=float, default=TrainingConfig.max_grad_norm)
    parser.add_argument("--max_input_length", type=int, default=TrainingConfig.max_input_length)
    parser.add_argument("--max_target_length", type=int, default=TrainingConfig.max_target_length)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--scheduler_type", default=TrainingConfig.scheduler_type)
    parser.add_argument("--warmup_ratio", type=float, default=TrainingConfig.warmup_ratio)
    parser.add_argument("--num_workers", type=int, default=TrainingConfig.num_workers)
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
        "--disable_amp",
        action="store_true",
        help="Disable mixed precision. AMP is enabled by default on CUDA.",
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        seed=args.seed,
        require_gpu=not args.allow_cpu,
        use_amp=not args.disable_amp,
        num_workers=args.num_workers,
        scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
        max_eval_batches=args.max_eval_batches,
        inference_samples=args.inference_samples,
        generation_max_new_tokens=args.generation_max_new_tokens,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history_size,
    )


def move_batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def count_tokens(batch):
    return int(batch["attention_mask"].sum().item())


def create_lr_scheduler(optimizer, config, steps_per_epoch):
    if config.optimizer_name == "lbfgs":
        return None

    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    return get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    config,
    scaler,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_losses = []
    step_latencies = []
    processed_tokens = 0
    optimizer_steps = 0

    progress_bar = tqdm(dataloader, desc=f"Training ({config.optimizer_name})")
    for batch_index, batch in enumerate(progress_bar):
        batch = move_batch_to_device(batch, device)
        processed_tokens += count_tokens(batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        if config.optimizer_name == "lbfgs":

            def closure():
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
            optimizer_steps += 1
        else:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.use_amp and device.type == "cuda",
            ):
                outputs = model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or (batch_index + 1) == len(dataloader)
            )
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            loss_value = loss.item() * config.gradient_accumulation_steps

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_end = time.perf_counter()

        batch_losses.append(loss_value)
        step_latencies.append(step_end - step_start)
        progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

    epoch_time = sum(step_latencies)
    return {
        "train_loss": compute_loss(batch_losses),
        "avg_step_ms": (epoch_time / len(step_latencies)) * 1000 if step_latencies else None,
        "tokens_per_second": processed_tokens / epoch_time if epoch_time > 0 else None,
        "optimizer_steps": optimizer_steps,
    }


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

    validation_loss = compute_loss(losses)
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": compute_perplexity(validation_loss),
    }


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
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}

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

    return (
        compute_inference_metrics(end_time - start_time, len(samples)),
        preview,
    )


def prepare_directories(config):
    ensure_dir(config.output_dir)
    ensure_dir(config.metrics_dir)
    ensure_dir(config.logs_dir)
    ensure_dir(config.checkpoints_dir)


def summarize_metrics(metrics_dir):
    """Create a CSV summary from all train_<optimizer>.json files."""
    metrics_path = Path(metrics_dir)
    rows = []

    for file_path in sorted(metrics_path.glob("train_*.json")):
        with open(file_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        inference = metrics.get("inference", {})
        memory = metrics.get("memory_usage", {})
        rows.append(
            {
                "optimizer": metrics.get("optimizer_name"),
                "final_train_loss": metrics.get("final_train_loss"),
                "final_validation_loss": metrics.get("final_validation_loss"),
                "final_validation_perplexity": metrics.get("final_validation_perplexity"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "training_examples_per_second": metrics.get("training_examples_per_second"),
                "avg_step_ms": metrics.get("average_step_ms"),
                "tokens_per_second": metrics.get("average_tokens_per_second"),
                "gpu_max_allocated_mb": memory.get("gpu_max_allocated_mb"),
                "gpu_max_reserved_mb": memory.get("gpu_max_reserved_mb"),
                "final_model_size_mb": metrics.get("final_model_size_mb"),
                "inference_time_seconds": inference.get("inference_time_seconds"),
                "latency_ms_per_sample": inference.get("latency_ms_per_sample"),
            }
        )

    if not rows:
        return None

    summary_file = metrics_path / "optimizer_summary.csv"
    with open(summary_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return summary_file


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
    model.config.use_cache = False
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

    pin_memory = device.type == "cuda"
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        seed=config.seed,
    )
    validation_dataloader = create_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        seed=config.seed,
    )

    print("Creating optimizer and scheduler.")
    optimizer = get_optimizer(model.parameters(), config)
    steps_per_epoch = max(
        1,
        math.ceil(len(train_dataloader) / config.gradient_accumulation_steps),
    )
    scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch)
    amp_enabled = (
        config.use_amp
        and device.type == "cuda"
        and config.optimizer_name != "lbfgs"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

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
        train_metrics = train_one_epoch(
            model=trainer["model"],
            optimizer=trainer["optimizer"],
            scheduler=scheduler,
            dataloader=trainer["train_dataloader"],
            device=device,
            config=config,
            scaler=scaler,
        )
        validation_metrics = evaluate_model(
            model=trainer["model"],
            dataloader=trainer["validation_dataloader"],
            device=device,
            max_eval_batches=config.max_eval_batches,
        )

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                **train_metrics,
                **validation_metrics,
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

    final_epoch = epoch_metrics[-1] if epoch_metrics else {}
    final_train_loss = final_epoch.get("train_loss")
    final_validation_loss = final_epoch.get("validation_loss")
    final_validation_perplexity = final_epoch.get("validation_perplexity")

    metrics = {
        "model_name": config.model_name,
        "optimizer_name": config.optimizer_name,
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "momentum": config.momentum if config.optimizer_name == "sgd_momentum" else None,
        "scheduler_type": config.scheduler_type if scheduler is not None else None,
        "warmup_ratio": config.warmup_ratio if scheduler is not None else None,
        "max_grad_norm": config.max_grad_norm,
        "use_amp": amp_enabled,
        "train_examples": len(train_dataframe),
        "validation_examples": len(validation_dataframe),
        "parameter_counts": parameter_counts,
        "epoch_metrics": epoch_metrics,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "final_validation_perplexity": final_validation_perplexity,
        "average_step_ms": compute_loss(
            [item["avg_step_ms"] for item in epoch_metrics if item.get("avg_step_ms") is not None]
        ),
        "average_tokens_per_second": compute_loss(
            [
                item["tokens_per_second"]
                for item in epoch_metrics
                if item.get("tokens_per_second") is not None
            ]
        ),
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

    if config.save_prediction_preview:
        save_prediction_preview(
            predictions=prediction_preview,
            metrics_dir=config.metrics_dir,
            optimizer_name=config.optimizer_name,
        )

    summary_path = summarize_metrics(config.metrics_dir)
    if summary_path is not None:
        print(f"Updated optimizer summary: {summary_path}")

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
