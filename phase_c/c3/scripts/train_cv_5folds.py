#!/usr/bin/env python3
"""
Phase C3 Stage 11: Production Full 5-Fold Cross-Validation Training Pipeline
RSNA Knee Abnormality Detection

Architecture: MultiSeriesKneeModel (Shared 3D ResNet + Masked Pool + Linear)
Hardware: DistributedDataParallel on 2 x NVIDIA A100 80GB (NCCL)
Loss: Confidence-weighted soft BCE (Phase B remediated labels)
Cross-Validation: Frozen 5-fold disjoint partition (configs/cv_splits.json)
"""

import os
import sys
import time
import json
import random
import argparse
import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from sklearn.metrics import roc_auc_score, f1_score
except ImportError:
    def roc_auc_score(y_true, y_score):
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("Only one class present in y_true")
        diff = pos[:, None] - neg[None, :]
        return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))

    def f1_score(y_true, y_pred, zero_division=0):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else float(zero_division)

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase_c", "c3", "scripts"))

from model import MultiSeriesKneeModel, count_parameters, TARGETS, NUM_SLOTS
from loss import weighted_soft_bce_loss
from dataset import MultiSeriesKneeDataset, multi_series_collate_fn


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        is_distributed = True
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        is_distributed = False
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    return local_rank, global_rank, world_size, is_distributed


def cleanup_ddp(is_distributed: bool):
    if is_distributed:
        dist.destroy_process_group()


def set_seed(seed: int = 42, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(all_preds: np.ndarray, all_targets: np.ndarray, all_weights: np.ndarray) -> Dict[str, Any]:
    """Compute per-target and macro validation metrics."""
    metrics = {}
    aucs = []
    f1s = []

    # Binarize targets at 0.5 for classification metrics comparison
    binary_targets = (all_targets >= 0.5).astype(int)
    binary_preds = (all_preds >= 0.5).astype(int)

    for i, target_name in enumerate(TARGETS):
        y_true = binary_targets[:, i]
        y_score = all_preds[:, i]
        y_pred = binary_preds[:, i]

        target_metrics = {}
        if len(np.unique(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_score))
                target_metrics["roc_auc"] = auc
                aucs.append(auc)
            except Exception:
                target_metrics["roc_auc"] = None
        else:
            target_metrics["roc_auc"] = None

        try:
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            target_metrics["f1"] = f1
            f1s.append(f1)
        except Exception:
            target_metrics["f1"] = None

        metrics[target_name] = target_metrics

    metrics["macro_roc_auc"] = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
    metrics["macro_f1"] = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    return metrics


def train_fold(
    fold_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    aggregation: str,
    local_rank: int,
    global_rank: int,
    world_size: int,
    is_distributed: bool,
):
    set_seed(42, rank=global_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Directories
    checkpoints_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "checkpoints", f"fold_{fold_idx}")
    results_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "results")
    logs_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "logs")

    if global_rank == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        print(f"\n{'='*80}\nSTARTING TRAINING: FOLD {fold_idx} (Epochs: {epochs}, Batch/GPU: {batch_size}, LR: {lr})\n{'='*80}")

    # Load frozen CV split
    cv_path = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
    with open(cv_path) as f:
        cv_data = json.load(f)

    fold_key = f"fold_{fold_idx}"
    train_uids = cv_data["folds"][fold_key]["train_studies"]
    val_uids = cv_data["folds"][fold_key]["val_studies"]

    # Verify Gold-58 isolation
    gold_df = pd.read_csv(os.path.join(REPO_ROOT, "csv", "gold_studies.csv"))
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    assert len(set(train_uids) & gold_uids) == 0, f"CRITICAL: Gold leakage in Fold {fold_idx} train!"
    assert len(set(val_uids) & gold_uids) == 0, f"CRITICAL: Gold leakage in Fold {fold_idx} val!"

    if global_rank == 0:
        print(f"[Fold {fold_idx}] Train studies: {len(train_uids):,} | Val studies: {len(val_uids):,}")

    # Load labels & metadata
    labels_df = pd.read_csv(os.path.join(REPO_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv"))
    meta_df = pd.read_csv(os.path.join(REPO_ROOT, "csv", "train_series.csv"))
    processed_dir = os.path.join(REPO_ROOT, "phase_c", "c1", "processed")

    train_ds = MultiSeriesKneeDataset(
        study_uids=train_uids,
        processed_dir=processed_dir,
        labels_df=labels_df,
        series_meta_df=meta_df,
        gold_uids=gold_uids,
    )
    val_ds = MultiSeriesKneeDataset(
        study_uids=val_uids,
        processed_dir=processed_dir,
        labels_df=labels_df,
        series_meta_df=meta_df,
        gold_uids=gold_uids,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank, shuffle=True, seed=42) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=global_rank, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, collate_fn=multi_series_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=multi_series_collate_fn
    )

    # Model
    raw_model = MultiSeriesKneeModel(
        embed_dim=128,
        num_targets=len(TARGETS),
        num_slots=NUM_SLOTS,
        aggregation=aggregation,
        dropout=0.3,
    ).to(device)

    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = raw_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss = float("inf")
    history = []
    fold_start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_weight = 0.0

        for batch in train_loader:
            volumes = batch["volumes"].to(device, non_blocking=True)
            slot_mask = batch["slot_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(volumes, slot_mask)
                loss = weighted_soft_bce_loss(logits, targets, weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            w_sum = max(weights.sum().item(), 1.0)
            train_loss += loss.item() * w_sum
            train_weight += w_sum

        scheduler.step()

        # Reduce train loss across ranks
        if is_distributed:
            t_tensor = torch.tensor([train_loss, train_weight], device=device)
            dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)
            global_train_loss = (t_tensor[0] / t_tensor[1].clamp(min=1.0)).item()
        else:
            global_train_loss = train_loss / max(train_weight, 1.0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_weight = 0.0
        local_val_preds = []
        local_val_targets = []
        local_val_weights = []

        with torch.no_grad():
            for batch in val_loader:
                volumes = batch["volumes"].to(device)
                slot_mask = batch["slot_mask"].to(device)
                targets = batch["targets"].to(device)
                weights = batch["weights"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(volumes, slot_mask)
                    loss = weighted_soft_bce_loss(logits, targets, weights)
                    probs = torch.sigmoid(logits)

                w_sum = max(weights.sum().item(), 1.0)
                val_loss += loss.item() * w_sum
                val_weight += w_sum

                local_val_preds.append(probs.float().cpu().numpy())
                local_val_targets.append(targets.float().cpu().numpy())
                local_val_weights.append(weights.float().cpu().numpy())

        if is_distributed:
            v_tensor = torch.tensor([val_loss, val_weight], device=device)
            dist.all_reduce(v_tensor, op=dist.ReduceOp.SUM)
            global_val_loss = (v_tensor[0] / v_tensor[1].clamp(min=1.0)).item()
        else:
            global_val_loss = val_loss / max(val_weight, 1.0)

        ep_duration = time.time() - ep_start
        throughput = len(train_uids) / max(ep_duration, 0.01)

        # Gather predictions on rank 0 for metric computation
        if len(local_val_preds) > 0:
            local_preds_np = np.concatenate(local_val_preds, axis=0)
            local_targets_np = np.concatenate(local_val_targets, axis=0)
            local_weights_np = np.concatenate(local_val_weights, axis=0)
        else:
            local_preds_np = np.empty((0, 12))
            local_targets_np = np.empty((0, 12))
            local_weights_np = np.empty((0, 12))

        if is_distributed:
            gathered_preds = [None for _ in range(world_size)]
            gathered_targets = [None for _ in range(world_size)]
            gathered_weights = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_preds, local_preds_np)
            dist.all_gather_object(gathered_targets, local_targets_np)
            dist.all_gather_object(gathered_weights, local_weights_np)
            if global_rank == 0:
                all_preds = np.concatenate(gathered_preds, axis=0)[:len(val_uids)]
                all_targets = np.concatenate(gathered_targets, axis=0)[:len(val_uids)]
                all_weights = np.concatenate(gathered_weights, axis=0)[:len(val_uids)]
        else:
            all_preds = local_preds_np
            all_targets = local_targets_np
            all_weights = local_weights_np

        if global_rank == 0:
            eval_metrics = compute_metrics(all_preds, all_targets, all_weights)
            macro_auc = eval_metrics["macro_roc_auc"]
            macro_f1 = eval_metrics["macro_f1"]

            is_best = global_val_loss < best_val_loss
            if is_best:
                best_val_loss = global_val_loss
                torch.save({
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "macro_roc_auc": macro_auc,
                    "macro_f1": macro_f1,
                    "metrics": eval_metrics,
                }, os.path.join(checkpoints_dir, "best_model.pt"))

            print(
                f"[Fold {fold_idx}] Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {global_train_loss:.4f} | Val Loss: {global_val_loss:.4f} (Best: {best_val_loss:.4f}) | "
                f"Macro AUC: {macro_auc:.4f} | Macro F1: {macro_f1:.4f} | Time: {ep_duration:.1f}s ({throughput:.1f} st/s)"
            )

            history.append({
                "fold": fold_idx,
                "epoch": epoch,
                "train_loss": global_train_loss,
                "val_loss": global_val_loss,
                "macro_roc_auc": macro_auc,
                "macro_f1": macro_f1,
                "duration_sec": ep_duration,
                "throughput_studies_per_sec": throughput,
            })

    if global_rank == 0:
        total_time = time.time() - fold_start_time
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(logs_dir, f"fold_{fold_idx}_training_history.csv"), index=False)

        summary = {
            "fold": fold_idx,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "epochs": epochs,
            "batch_size_per_gpu": batch_size,
            "world_size": world_size,
            "best_val_loss": float(best_val_loss),
            "final_val_loss": float(history[-1]["val_loss"]),
            "best_macro_auc": float(history_df["macro_roc_auc"].max()),
            "best_macro_f1": float(history_df["macro_f1"].max()),
            "total_runtime_sec": float(total_time),
            "checkpoint_path": os.path.join(checkpoints_dir, "best_model.pt"),
            "status": "COMPLETED",
        }
        with open(os.path.join(results_dir, f"fold_{fold_idx}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Fold {fold_idx} COMPLETE] Best Val Loss: {best_val_loss:.4f} | Total Runtime: {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Phase C3 Full 5-Fold Cross-Validation Training")
    parser.add_argument("--fold", type=str, default="0", help="Fold index (0-4) or 'all'")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per fold")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--aggregation", type=str, default="masked_pool", choices=["masked_pool", "masked_attention"])
    args = parser.parse_args()

    local_rank, global_rank, world_size, is_distributed = setup_ddp()

    folds_to_run = [int(args.fold)] if args.fold != "all" else list(range(5))

    for f_idx in folds_to_run:
        train_fold(
            fold_idx=f_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            aggregation=args.aggregation,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
