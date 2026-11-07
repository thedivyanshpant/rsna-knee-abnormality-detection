"""
Phase C2: Main training & evaluation script.

Single configurable pipeline: select representation via --pipeline argument.
Trains the same SmallResNet3D model with identical hyperparameters, optimizer,
loss function, and evaluation — only the input representation differs.

Usage:
    python train.py --pipeline raw
    python train.py --pipeline pipeline_a
    python train.py --pipeline pipeline_b
    python train.py --pipeline pipeline_c
    python train.py --smoke-test          # Quick end-to-end verification
"""
import argparse
import json
import os
import sys
import time
import random
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, accuracy_score,
)

from config import (
    LABELS_CSV, TRAIN_SERIES_CSV, TARGETS, PIPELINE_NAMES,
    SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_WORKERS, DEVICE, PIN_MEMORY, MIXED_PRECISION,
    LOGS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, CONFIGS_DIR, MODEL_INPUT_SHAPE,
)
from dataset import KneeC2Dataset
from model import SmallResNet3D, count_parameters
from split import create_split, load_split


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_bce_loss(logits, targets, weights):
    """Phase B confidence-weighted BCE loss.

    L = mean(W * BCE(sigmoid(logit), Y))
    where Y = label_probability (soft target), W = sample_weight
    Abstained samples have W=0 and contribute nothing to the loss.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weighted = bce * weights
    # Average over non-zero weights to avoid dilution by abstentions
    denom = weights.sum()
    if denom > 0:
        return weighted.sum() / denom
    return weighted.mean()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (volumes, y, w, _) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=MIXED_PRECISION):
            logits = model(volumes)
            loss = weighted_bce_loss(logits, y, w)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation set.

    Returns dict with:
        val_loss, macro_f1, macro_auc, per_target_f1, per_target_precision,
        per_target_recall, exact_study_accuracy, all_logits, all_targets, all_weights
    """
    model.eval()
    all_logits = []
    all_y = []
    all_w = []
    all_uids = []
    total_loss = 0.0
    n_batches = 0

    for volumes, y, w, uids in loader:
        volumes = volumes.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        w_dev = w.to(device, non_blocking=True)

        with autocast('cuda', enabled=MIXED_PRECISION):
            logits = model(volumes)
            loss = weighted_bce_loss(logits, y_dev, w_dev)

        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu().float())
        all_y.append(y)
        all_w.append(w)
        all_uids.extend(uids)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_y = torch.cat(all_y, dim=0).numpy()
    all_w = torch.cat(all_w, dim=0).numpy()

    # Predictions
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    all_preds = (all_probs >= 0.5).astype(int)
    # Binarize soft targets for evaluation
    all_y_binary = (all_y >= 0.5).astype(int)

    val_loss = total_loss / max(n_batches, 1)

    # Per-target metrics
    per_target_f1 = {}
    per_target_precision = {}
    per_target_recall = {}
    per_target_auc = {}

    for i, target in enumerate(TARGETS):
        y_t = all_y_binary[:, i]
        pred_t = all_preds[:, i]
        prob_t = all_probs[:, i]

        per_target_f1[target] = float(f1_score(y_t, pred_t, zero_division=0))
        per_target_precision[target] = float(precision_score(y_t, pred_t, zero_division=0))
        per_target_recall[target] = float(recall_score(y_t, pred_t, zero_division=0))

        # AUC requires both classes present
        if len(np.unique(y_t)) > 1:
            per_target_auc[target] = float(roc_auc_score(y_t, prob_t))
        else:
            per_target_auc[target] = float("nan")

    # Macro metrics
    macro_f1 = float(np.nanmean(list(per_target_f1.values())))
    valid_aucs = [v for v in per_target_auc.values() if not np.isnan(v)]
    macro_auc = float(np.nanmean(valid_aucs)) if valid_aucs else float("nan")

    # Exact study accuracy: a study is correct if ALL 12 targets match
    n_studies = len(all_uids)
    exact_correct = 0
    for i in range(n_studies):
        if np.array_equal(all_preds[i], all_y_binary[i]):
            exact_correct += 1
    exact_study_acc = exact_correct / max(n_studies, 1)

    return {
        "val_loss": val_loss,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "exact_study_accuracy": exact_study_acc,
        "per_target_f1": per_target_f1,
        "per_target_precision": per_target_precision,
        "per_target_recall": per_target_recall,
        "per_target_auc": per_target_auc,
        "study_uids": all_uids,
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
        "ground_truth": all_y_binary.tolist(),
    }


def run_experiment(pipeline: str, epochs: int = EPOCHS, smoke_test: bool = False):
    """Run the complete training experiment for one pipeline representation."""
    print(f"\n{'='*70}")
    print(f"  PHASE C2 EXPERIMENT: {pipeline.upper()}")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{pipeline}_{timestamp}"

    # 0. Set seeds
    set_seed(SEED)

    # 1. Load split
    split_path = os.path.join(CONFIGS_DIR, "train_val_split.json")
    if not os.path.exists(split_path):
        print("Creating train/val split...")
        train_uids, val_uids = create_split()
    else:
        train_uids, val_uids = load_split()
    print(f"Train: {len(train_uids)} studies, Val: {len(val_uids)} studies")

    # 2. Load labels and series metadata
    labels_df = pd.read_csv(LABELS_CSV)
    series_meta = pd.read_csv(TRAIN_SERIES_CSV)

    # 3. Create datasets
    if smoke_test:
        # Use 4 train + 2 val for smoke test
        train_uids = train_uids[:4]
        val_uids = val_uids[:2]
        epochs = 2
        print(f"SMOKE TEST: {len(train_uids)} train, {len(val_uids)} val, {epochs} epochs")

    train_ds = KneeC2Dataset(train_uids, pipeline, labels_df, series_meta)
    val_ds = KneeC2Dataset(val_uids, pipeline, labels_df, series_meta)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    # 4. Create model
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = SmallResNet3D().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: SmallResNet3D, {total_params:,} params ({trainable_params:,} trainable)")

    # 5. Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda', enabled=MIXED_PRECISION)

    # 6. Training loop
    best_val_f1 = -1.0
    best_epoch = -1
    history = []
    peak_vram_mb = 0.0

    torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        # Track VRAM
        current_peak = torch.cuda.max_memory_allocated(device) / 1e6
        peak_vram_mb = max(peak_vram_mb, current_peak)

        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "macro_f1": val_metrics["macro_f1"],
            "macro_auc": val_metrics["macro_auc"],
            "exact_study_accuracy": val_metrics["exact_study_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
            "peak_vram_mb": current_peak,
        }
        history.append(epoch_record)

        # Checkpoint best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics.copy()

            ckpt_dir = os.path.join(CHECKPOINTS_DIR, pipeline)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": {
                    k: v for k, v in val_metrics.items()
                    if k not in ("study_uids", "predictions", "probabilities", "ground_truth")
                },
                "pipeline": pipeline,
                "seed": SEED,
            }, ckpt_path)

        print(
            f"  Epoch {epoch:02d}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['val_loss']:.4f}, "
            f"macro_f1={val_metrics['macro_f1']:.4f}, macro_auc={val_metrics['macro_auc']:.4f}, "
            f"study_acc={val_metrics['exact_study_accuracy']:.4f}, "
            f"VRAM={current_peak:.0f}MB, time={epoch_time:.1f}s"
            f"{'  ★ BEST' if epoch == best_epoch else ''}"
        )

    total_time = time.time() - t_start
    print(f"\nTraining complete: {total_time:.1f}s total, best epoch {best_epoch}")

    # 7. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Training history
    history_path = os.path.join(LOGS_DIR, f"{pipeline}_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Best validation predictions (for later analysis)
    preds_path = os.path.join(RESULTS_DIR, f"{pipeline}_val_predictions.json")
    with open(preds_path, "w") as f:
        json.dump({
            "study_uids": best_metrics["study_uids"],
            "predictions": best_metrics["predictions"],
            "probabilities": best_metrics["probabilities"],
            "ground_truth": best_metrics["ground_truth"],
            "targets": TARGETS,
        }, f, indent=2)

    # Experiment summary
    summary = {
        "pipeline": pipeline,
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "seed": SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "model": "SmallResNet3D",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_input_shape": list(MODEL_INPUT_SHAPE),
        "mixed_precision": MIXED_PRECISION,
        "device": DEVICE,
        "train_studies": len(train_uids),
        "val_studies": len(val_uids),
        "total_training_time_sec": total_time,
        "peak_vram_mb": peak_vram_mb,
        "best_epoch": best_epoch,
        "best_val_loss": best_metrics["val_loss"],
        "best_macro_f1": best_val_f1,
        "best_macro_auc": best_metrics["macro_auc"],
        "best_exact_study_accuracy": best_metrics["exact_study_accuracy"],
        "per_target_f1": best_metrics["per_target_f1"],
        "per_target_precision": best_metrics["per_target_precision"],
        "per_target_recall": best_metrics["per_target_recall"],
        "per_target_auc": best_metrics["per_target_auc"],
    }

    summary_path = os.path.join(RESULTS_DIR, f"{pipeline}_experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved:")
    print(f"  History: {history_path}")
    print(f"  Predictions: {preds_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Checkpoint: {os.path.join(CHECKPOINTS_DIR, pipeline, 'best_model.pt')}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase C2: Controlled preprocessing experiment")
    parser.add_argument(
        "--pipeline", type=str, default=None,
        choices=PIPELINE_NAMES,
        help="Which representation pipeline to train on",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 4 experiments sequentially",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick end-to-end smoke test (4 train + 2 val, 2 epochs)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    args = parser.parse_args()

    if args.smoke_test:
        print("\n=== SMOKE TEST MODE ===")
        for p in PIPELINE_NAMES:
            run_experiment(p, smoke_test=True)
        print("\n✓ All 4 pipelines smoke test passed!")
        return

    epochs = args.epochs if args.epochs is not None else EPOCHS

    if args.all:
        summaries = {}
        for p in PIPELINE_NAMES:
            summaries[p] = run_experiment(p, epochs=epochs)

        # Save combined comparison
        comparison_path = os.path.join(RESULTS_DIR, "c2_pipeline_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nCombined comparison saved: {comparison_path}")
        return

    if args.pipeline:
        run_experiment(args.pipeline, epochs=epochs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
