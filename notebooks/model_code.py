"""
Multi-Optimizer Deep Learning Benchmark Suite for Flan-T5-Small
===============================================================
Benchmarks six optimizer configurations on real JSONL data from
data/processed/{train,validation,test}.jsonl, reports training loss
convergence, validation loss, step latency, inference latency,
throughput (tok/s), and peak VRAM/RAM deltas.

Outputs:
  results/metrics/benchmark_summary.json
  results/metrics/optimizer_comparison_matrix.png
  outputs/mini_chatbot_model/  (final model weights)
"""

import os
import sys
import time
import gc
import json
import math
import csv
import torch
import psutil
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless / SSH environments
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Performance Flags ─────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True   # cuDNN auto-tuner
_HAS_COMPILE = hasattr(torch, "compile")

# ══════════════════════════════════════════════════════════════════════════════
# TrainingConfig — mirrors src/training_config.py so the notebook stays
# self-contained while honouring the project's canonical defaults.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrainingConfig:
    """Benchmark-aware superset of the project's TrainingConfig."""
    model_name: str         = "google/flan-t5-small"
    output_dir: str         = "results/metrics"
    model_save_dir: str     = "outputs/mini_chatbot_model"
    optimizer_name: str     = "adamw"
    learning_rate: float    = 5e-5
    batch_size: int         = 4
    n_steps: int            = 10          # training steps per experiment
    val_eval_batches: int   = 5           # max validation / test batches
    max_input_length: int   = 256
    max_target_length: int  = 128
    seed: int               = 42
    data_dir: str           = "./data/processed"
    vocab_size: int         = 32128

CFG = TrainingConfig()

# ── Derived Constants ─────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR    = Path(CFG.data_dir)

torch.manual_seed(CFG.seed)
if DEVICE == "cpu":
    print("[WARN] CUDA not available -- running on CPU. Timings will be wall-clock.")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & TOKENISATION
# ══════════════════════════════════════════════════════════════════════════════
def load_split(split_name: str) -> Dataset:
    """Parse a .jsonl split, tokenise on-the-fly, return a TensorDataset.

    Dynamically extracts source/target text from varying JSON schemas
    (instruction+input, prompt, text, source_text / target_text, etc.).
    CRITICAL T5 RULE: pad token indices in labels are masked to -100.
    """
    jsonl_path = DATA_DIR / f"{split_name}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing: {jsonl_path.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    input_ids_all, attn_mask_all, labels_all = [], [], []
    print(f"  [DATA] Tokenising {jsonl_path.name} ...")

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)

            # --- Dynamic source extraction ---
            if "source_text" in item:
                source_text = item["source_text"]
            elif "instruction" in item and item.get("input"):
                source_text = f"{item['instruction']}\nInput: {item['input']}"
            else:
                source_text = (
                    item.get("instruction", "")
                    or item.get("prompt", "")
                    or item.get("input", "")
                    or item.get("text", "")
                )

            # --- Dynamic target extraction ---
            target_text = (
                item.get("target_text", "")
                or item.get("output", "")
                or item.get("response", "")
                or item.get("target", "")
            )

            if not source_text or not target_text:
                continue

            src_enc = tokenizer(
                source_text,
                max_length=CFG.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tgt_enc = tokenizer(
                text_target=target_text,
                max_length=CFG.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            ids   = src_enc["input_ids"].squeeze(0)
            mask  = src_enc["attention_mask"].squeeze(0)
            lbls  = tgt_enc["input_ids"].squeeze(0)

            # ── CRITICAL: mask padding in labels to -100 for cross-entropy ──
            lbls[lbls == tokenizer.pad_token_id] = -100

            input_ids_all.append(ids)
            attn_mask_all.append(mask)
            labels_all.append(lbls)

    if not input_ids_all:
        raise ValueError(f"No valid records in {jsonl_path.name}")

    return TensorDataset(
        torch.stack(input_ids_all),
        torch.stack(attn_mask_all),
        torch.stack(labels_all),
    )


def collate_fn(batch):
    """Handles both dict-style and tuple-style (TensorDataset) batches."""
    if isinstance(batch[0], dict):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    tensors = [torch.stack(t) for t in zip(*batch)]
    return {"input_ids": tensors[0], "attention_mask": tensors[1], "labels": tensors[2]}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def make_model():
    """Instantiate a fresh Flan-T5-Small from config (no pretrained weights)."""
    cfg = T5Config(
        d_ff=1024, d_kv=64, d_model=512,
        decoder_start_token_id=0,
        dense_act_fn="gelu_new", eos_token_id=1,
        feed_forward_proj="gated-gelu",
        num_heads=8, num_layers=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        vocab_size=CFG.vocab_size,
    )
    return T5ForConditionalGeneration(cfg).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROFILERS
# ══════════════════════════════════════════════════════════════════════════════
_proc = psutil.Process(os.getpid())

def ram_mb() -> float:
    return _proc.memory_info().rss / (1024 ** 2)

def peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    return 0.0

def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

def safe_float(x):
    """Replace NaN/Inf/None with None so JSON serialisation never crashes."""
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(float(x), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX-FREE NEWTON-CG OPTIMISER STEP
#  Uses double-backpropagation to compute Hessian-vector products without
#  ever instantiating an explicit Hessian matrix.
# ══════════════════════════════════════════════════════════════════════════════
def newton_cg_step(model, batch, lr=0.1, cg_max_iter=3) -> float:
    params = [p for p in model.parameters() if p.requires_grad]

    # 1. Forward + first-order gradients (keep graph alive for HVPs)
    out = model(
        input_ids=batch["input_ids"].to(DEVICE),
        attention_mask=batch["attention_mask"].to(DEVICE),
        labels=batch["labels"].to(DEVICE),
    )
    loss = out.loss
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    loss_val = loss.item()

    # 2. CG initialisation — solve H d = -g
    b_vec = -flat_grad.detach()
    d_vec = torch.zeros_like(b_vec)
    r_vec = b_vec.clone()
    p_vec = r_vec.clone()
    rs_old = torch.dot(r_vec, r_vec)

    # 3. CG loop — Hessian-vector products via double backprop
    for _ in range(cg_max_iter):
        gv = torch.dot(flat_grad, p_vec.detach())           # scalar, differentiable
        hvp = torch.autograd.grad(gv, params, retain_graph=True)
        Ap = torch.cat([g.contiguous().reshape(-1) for g in hvp]).detach()

        denom = torch.dot(p_vec, Ap) + 1e-8
        alpha = rs_old / denom
        d_vec = d_vec + alpha * p_vec
        r_vec = r_vec - alpha * Ap
        rs_new = torch.dot(r_vec, r_vec)
        if torch.sqrt(rs_new) < 1e-4:
            break
        p_vec = r_vec + (rs_new / rs_old) * p_vec
        rs_old = rs_new

    # 4. Release compute graph before parameter update
    del out, loss, grads, flat_grad, gv, hvp, Ap, b_vec, r_vec, p_vec

    # 5. Apply Newton direction
    with torch.no_grad():
        idx = 0
        for param in params:
            n = param.numel()
            param.add_(d_vec[idx : idx + n].view_as(param), alpha=lr)
            idx += n

    del d_vec
    return loss_val


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(
    name: str,
    opt_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    use_lbfgs: bool = False,
    use_amp: bool = False,
    use_scheduler: bool = False,
    grad_clip: bool = False,
    use_compile: bool = False,
) -> dict:
    """Run a single optimizer benchmark and return structured telemetry."""
    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"{'='*70}")

    torch.manual_seed(CFG.seed)
    model = make_model()

    # Optional torch.compile (PyTorch 2+)
    if use_compile and _HAS_COMPILE:
        try:
            model = torch.compile(model)
            print("  [COMPILE] torch.compile activated")
        except Exception as e:
            print(f"  [COMPILE] Skipped — {e}")

    reset_vram_stats()
    ram_before = ram_mb()

    # ── Optimiser setup ───────────────────────────────────────────────────────
    optimizer = None
    scheduler = None
    scaler = GradScaler("cuda") if (use_amp and DEVICE == "cuda") else None

    if opt_type == "__sgd__":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    elif opt_type == "__sgdm__":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    elif opt_type == "__adam__":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    elif opt_type == "__adamw__":
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=1e-2)
    elif opt_type == "__lbfgs__":
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=0.1, max_iter=4,
            history_size=10, line_search_fn="strong_wolfe",
        )

    if use_scheduler and optimizer is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.n_steps)

    # ── 1. TRAINING ───────────────────────────────────────────────────────────
    model.train()
    train_losses = []
    step_times   = []
    tokens_processed = 0

    # Pre-fetch N_STEPS batches for strict reproducibility across runs
    prefetched = []
    it = iter(train_loader)
    for _ in range(CFG.n_steps):
        try:
            prefetched.append(next(it))
        except StopIteration:
            it = iter(train_loader)
            prefetched.append(next(it))

    # CUDA events give sub-millisecond precision on GPU
    ev_start = torch.cuda.Event(enable_timing=True) if DEVICE == "cuda" else None
    ev_end   = torch.cuda.Event(enable_timing=True) if DEVICE == "cuda" else None

    for step, batch in enumerate(prefetched):
        t0 = time.perf_counter()
        if DEVICE == "cuda":
            ev_start.record()

        batch_tokens = batch["input_ids"].numel() + batch["labels"].numel()
        tokens_processed += batch_tokens

        # ── Branch: inference-only baseline ───────────────────────────────
        if opt_type == "__baseline__":
            with torch.no_grad():
                with autocast("cuda", enabled=use_amp and DEVICE == "cuda"):
                    out = model(
                        input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                        labels=batch["labels"].to(DEVICE),
                    )
            ls = out.loss.item()

        # ── Branch: Newton-CG (custom second-order) ──────────────────────
        elif opt_type == "__newton_cg__":
            ls = newton_cg_step(model, batch, lr=0.1, cg_max_iter=3)

        # ── Branch: L-BFGS (closure-based) ───────────────────────────────
        elif use_lbfgs:
            _last = [None]
            def closure():
                optimizer.zero_grad()
                with autocast("cuda", enabled=use_amp and DEVICE == "cuda"):
                    out = model(
                        input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                        labels=batch["labels"].to(DEVICE),
                    )
                _last[0] = out.loss
                if scaler:
                    scaler.scale(out.loss).backward()
                else:
                    out.loss.backward()
                return out.loss

            if scaler:
                scaler.step(optimizer, closure)
                scaler.update()
            else:
                optimizer.step(closure)
            ls = _last[0].item()

        # ── Branch: standard first-order optimisers ──────────────────────
        else:
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp and DEVICE == "cuda"):
                out = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
            loss = out.loss
            if scaler:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            ls = loss.item()

            if scheduler is not None:
                scheduler.step()

        # ── Timing ────────────────────────────────────────────────────────
        if DEVICE == "cuda":
            ev_end.record()
            torch.cuda.synchronize()
            dt = ev_start.elapsed_time(ev_end)          # milliseconds
        else:
            dt = (time.perf_counter() - t0) * 1000.0

        step_times.append(dt)
        train_losses.append(ls)
        print(f"    Step {step+1:>2}/{CFG.n_steps} | Loss {ls:.4f} | {dt:.1f} ms")

    # ── 2. VALIDATION ─────────────────────────────────────────────────────────
    model.eval()
    val_losses = []
    print(f"  [VAL] Evaluating up to {CFG.val_eval_batches} batches ...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= CFG.val_eval_batches:
                break
            with autocast("cuda", enabled=use_amp and DEVICE == "cuda"):
                out = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
            val_losses.append(out.loss.item())
    avg_val = sum(val_losses) / len(val_losses) if val_losses else 0.0

    # ── 3. INFERENCE LATENCY ──────────────────────────────────────────────────
    inf_latencies = []
    print(f"  [INF] Profiling up to {CFG.val_eval_batches} batches ...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= CFG.val_eval_batches:
                break
            if DEVICE == "cuda":
                ev_start.record()
            else:
                t0 = time.perf_counter()

            with autocast("cuda", enabled=use_amp and DEVICE == "cuda"):
                _ = model.generate(
                    input_ids=batch["input_ids"].to(DEVICE),
                    max_new_tokens=16,
                )

            if DEVICE == "cuda":
                ev_end.record()
                torch.cuda.synchronize()
                inf_latencies.append(ev_start.elapsed_time(ev_end))
            else:
                inf_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_inf = sum(inf_latencies) / len(inf_latencies) if inf_latencies else 0.0

    # ── 4. RESOURCE DELTAS ────────────────────────────────────────────────────
    vram_peak = peak_vram_mb()
    ram_delta = max(ram_mb() - ram_before, 0.0)

    # Save last model weights
    os.makedirs(CFG.model_save_dir, exist_ok=True)
    model.save_pretrained(CFG.model_save_dir)
    print(f"  [SAVE] Weights -> {CFG.model_save_dir}")

    # ── 5. CLEANUP (guard against OOM on next experiment) ─────────────────────
    del model
    if optimizer is not None:
        del optimizer
    if scheduler is not None:
        del scheduler
    if scaler is not None:
        del scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 6. THROUGHPUT ─────────────────────────────────────────────────────────
    total_time_s = sum(step_times) / 1000.0
    throughput   = tokens_processed / total_time_s if total_time_s > 0 else 0.0

    return {
        "optimizer":          name,
        "train_losses":       [safe_float(l) for l in train_losses],
        "final_train_loss":   safe_float(train_losses[-1]) if train_losses else None,
        "final_val_loss":     safe_float(avg_val),
        "avg_step_ms":        safe_float(sum(step_times) / len(step_times)),
        "avg_inf_latency_ms": safe_float(avg_inf),
        "throughput_tok_s":   safe_float(throughput),
        "vram_peak_mb":       safe_float(vram_peak),
        "ram_delta_mb":       safe_float(ram_delta),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main_pipeline(train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn)

    # ── Experiment Matrix ─────────────────────────────────────────────────────
    # Each tuple: (name, opt_type, use_lbfgs, use_amp, use_scheduler, grad_clip, use_compile)
    experiments = [
        ("Inference Only (baseline)", "__baseline__",   False, False, False, False, False),
        ("SGD",                       "__sgd__",        False, False, False, False, False),
        ("SGD + Momentum",            "__sgdm__",       False, False, False, False, False),
        ("Adam",                      "__adam__",       False, False, False, False, False),
        ("AdamW + Clip + Sched",      "__adamw__",      False, False, True,  True,  False),
        ("L-BFGS",                    "__lbfgs__",      True,  False, False, False, False),
        ("Newton-CG",                 "__newton_cg__",  False, False, False, False, False),
    ]

    results = []
    for name, opt_type, use_lbfgs, use_amp, use_sched, grad_clip, use_compile in experiments:
        r = run_experiment(
            name, opt_type, train_loader, val_loader, test_loader,
            use_lbfgs=use_lbfgs,
            use_amp=use_amp,
            use_scheduler=use_sched,
            grad_clip=grad_clip,
            use_compile=use_compile,
        )
        results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    #  CONSOLE REPORT
    # ══════════════════════════════════════════════════════════════════════════
    hdr = (
        f"{'Optimizer':<28}|{'Train Loss':>12}|{'Val Loss':>10}"
        f"|{'ms/step':>10}|{'Tok/s':>10}|{'Inf ms':>10}|{'VRAM MB':>10}"
    )
    sep = "-" * len(hdr)
    print(f"\n\n{'='*len(hdr)}")
    print(f"  OPTIMIZER BENCHMARK REPORT  |  {CFG.n_steps} steps x batch {CFG.batch_size}")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print(sep)
    for r in results:
        tl = r["final_train_loss"] if r["final_train_loss"] is not None else float("nan")
        vl = r["final_val_loss"]   if r["final_val_loss"]   is not None else float("nan")
        sm = r["avg_step_ms"]      if r["avg_step_ms"]      is not None else float("nan")
        tp = r["throughput_tok_s"] if r["throughput_tok_s"] is not None else float("nan")
        il = r["avg_inf_latency_ms"] if r["avg_inf_latency_ms"] is not None else float("nan")
        vm = r["vram_peak_mb"]     if r["vram_peak_mb"]     is not None else 0.0
        print(
            f"{r['optimizer']:<28}|{tl:>12.4f}|{vl:>10.4f}"
            f"|{sm:>10.1f}|{tp:>10.0f}|{il:>10.1f}|{vm:>10.1f}"
        )
    print("=" * len(hdr))

    # ══════════════════════════════════════════════════════════════════════════
    #  EXPORT RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    os.makedirs(CFG.output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(CFG.output_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [EXPORT] JSON  -> {json_path}")

    # CSV
    csv_path = os.path.join(CFG.output_dir, "benchmark_summary.csv")
    flat_keys = [k for k in results[0] if k != "train_losses"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in flat_keys}
            w.writerow(row)
    print(f"  [EXPORT] CSV   -> {csv_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  6-PANEL DIAGNOSTIC DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    colors = plt.cm.tab10.colors
    opt_names = [r["optimizer"] for r in results]
    n_opt = len(results)

    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    fig.suptitle(
        f"Flan-T5-Small Optimizer Benchmark  ({CFG.n_steps} steps, batch {CFG.batch_size})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Panel 1: Training Loss Convergence Curves
    ax = axes[0, 0]
    for i, r in enumerate(results):
        ax.plot(
            range(1, CFG.n_steps + 1), r["train_losses"],
            label=r["optimizer"], color=colors[i % len(colors)], marker="o", markersize=4,
        )
    ax.set_title("Training Loss Convergence", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    # Panel 2: Validation Loss (Generalization Profile)
    ax = axes[0, 1]
    vals = [r["final_val_loss"] or 0.0 for r in results]
    bars = ax.bar(range(n_opt), vals, color=colors[:n_opt], edgecolor="black", alpha=0.85)
    ax.set_title(f"Validation Loss (max {CFG.val_eval_batches * CFG.batch_size} samples)", fontweight="bold")
    ax.set_ylabel("Loss")
    ax.set_xticks(range(n_opt))
    ax.set_xticklabels(opt_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # Panel 3: Average Step Latency (ms / step)
    ax = axes[0, 2]
    sms = [r["avg_step_ms"] or 0.0 for r in results]
    bars = ax.bar(range(n_opt), sms, color=colors[:n_opt], edgecolor="black", alpha=0.85)
    ax.set_title("Training Step Latency", fontweight="bold")
    ax.set_ylabel("ms / step")
    ax.set_xticks(range(n_opt))
    ax.set_xticklabels(opt_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, sms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    # Panel 4: Peak VRAM Allocation (MB)
    ax = axes[1, 0]
    vrams = [r["vram_peak_mb"] or 0.0 for r in results]
    bars = ax.bar(range(n_opt), vrams, color="#9467bd", edgecolor="black", alpha=0.85)
    ax.set_title("Peak VRAM per Optimizer", fontweight="bold")
    ax.set_ylabel("MB")
    ax.set_xticks(range(n_opt))
    ax.set_xticklabels(opt_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 5: Throughput (tokens / sec)
    ax = axes[1, 1]
    tps = [r["throughput_tok_s"] or 0.0 for r in results]
    bars = ax.bar(range(n_opt), tps, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.set_title("Training Throughput", fontweight="bold")
    ax.set_ylabel("Tokens / sec")
    ax.set_xticks(range(n_opt))
    ax.set_xticklabels(opt_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 6: Inference Latency (ms)
    ax = axes[1, 2]
    infs = [r["avg_inf_latency_ms"] or 0.0 for r in results]
    bars = ax.bar(range(n_opt), infs, color="#d62728", edgecolor="black", alpha=0.85)
    ax.set_title("Inference Latency", fontweight="bold")
    ax.set_ylabel("ms / batch")
    ax.set_xticks(range(n_opt))
    ax.set_xticklabels(opt_names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(CFG.output_dir, "optimizer_comparison_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [EXPORT] Plot  -> {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[INFO] Device:    {DEVICE}")
    print(f"[INFO] Data dir:  {DATA_DIR.resolve()}")
    print(f"[INFO] Steps:     {CFG.n_steps}  |  Batch: {CFG.batch_size}")
    print(f"[INFO] Src len:   {CFG.max_input_length}  |  Tgt len: {CFG.max_target_length}")
    print()

    try:
        train_ds = load_split("train")
        val_ds   = load_split("validation")
        test_ds  = load_split("test")
        print(
            f"\n  Splits loaded: train={len(train_ds)}, "
            f"val={len(val_ds)}, test={len(test_ds)}\n"
        )
        main_pipeline(train_ds, val_ds, test_ds)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)