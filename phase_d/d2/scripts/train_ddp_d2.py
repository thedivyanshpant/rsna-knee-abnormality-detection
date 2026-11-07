#!/usr/bin/env python3
"""
Phase D2: Production DDP Training Engine for RadImageNet ResNet-50 + SlotHead.
RSNA Knee Abnormality Detection

Architecture: RadImageNet ResNet-50 (lower layers frozen, layers 3-4 trainable) + SlotHead (12 pathology queries + 0.55 priors)
Hardware: 2 x NVIDIA A100 PCIe 80GB (NCCL DistributedDataParallel)
Optimizer: AdamW (Backbone LR: 1e-4, SlotHead LR: 1e-3, Weight Decay: 0.02)
Scheduler: OneCycleLR (pct_start: 0.15)
Effective Batch Size: 8 studies (2 studies/GPU x 2 GPUs x 2 grad accum steps)
Validation: Held-out validation split evaluated on rank 0
Gold-58: Strictly excluded (0 leakage assertion)
"""

import os
import sys
import time
import json
import random
import argparse
import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from phase_d.d2.scripts.model_d2_cnn import RadImageNetResNet50SlotHead
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset, TARGETS
from phase_d.d1.scripts.loss_d1 import weighted_soft_bce_loss

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/manifests/d1_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/checkpoints")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/predictions/val")
METRICS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d2/reports")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")


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


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy Mann-Whitney U / Pairwise difference ROC-AUC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def compute_macro_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray
) -> Tuple[float, float, Dict[str, Optional[float]]]:
    """Compute unweighted and sample-weighted macro ROC-AUC across 12 pathologies."""
    per_target = {}
    valid_aucs = []
    weighted_aucs = []
    weights_sum = []

    for c in range(12):
        t_name = TARGETS[c]
        y_c = y_true[:, c]
        pred_c = y_pred[:, c]
        w_c = weights[:, c]

        bin_y = (y_c >= 0.5).astype(int)
        if len(np.unique(bin_y)) > 1:
            auc = roc_auc_score_np(bin_y, pred_c)
            per_target[t_name] = auc
            if auc is not None:
                valid_aucs.append(auc)
                total_w = float(np.sum(w_c))
                if total_w > 0:
                    weighted_aucs.append(auc * total_w)
                    weights_sum.append(total_w)
        else:
            per_target[t_name] = None

    macro_unweighted = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    macro_weighted = float(np.sum(weighted_aucs) / np.sum(weights_sum)) if weights_sum else macro_unweighted
    return macro_unweighted, macro_weighted, per_target


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 2,
    epoch: int = 0,
    global_rank: int = 0
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        slot_mask = batch["slot_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        weights = batch["weights"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(images, slot_mask)
            loss = weighted_soft_bce_loss(logits, targets, weights)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, Dict[str, Optional[float]], np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_preds = []
    all_targets = []
    all_weights = []
    all_uids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            slot_mask = batch["slot_mask"].to(device, non_blocking=True)
            uids = batch["study_uid"]
            targets = batch["targets"].numpy()
            weights = batch["weights"].numpy()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(images, slot_mask)
                probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets)
            all_weights.append(weights)
            all_uids.extend(uids)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    w_true = np.concatenate(all_weights, axis=0)

    macro_unweighted, macro_weighted, per_target = compute_macro_roc_auc(y_true, y_pred, w_true)
    return macro_unweighted, macro_weighted, per_target, y_pred, y_true, all_uids


def main():
    parser = argparse.ArgumentParser(description="Phase D2 RadImageNet ResNet50 DDP Training Engine")
    parser.add_argument("--fold", type=int, default=0, help="CV fold index (0..4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Microbatch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="ResNet backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="SlotHead learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay")
    args = parser.parse_args()

    local_rank, global_rank, world_size, is_distributed = setup_ddp()
    set_seed(42, global_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    fold_ckpt_dir = os.path.join(CHECKPOINTS_DIR, f"fold_{args.fold}")
    if global_rank == 0:
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        print("=" * 80)
        print(f"  PHASE D2 RADIMAGENET RESNET-50 DDP TRAINING: FOLD {args.fold}")
        print("=" * 80)
        print(f"World Size: {world_size} GPUs | Device: {device} | Microbatch: {args.batch_size} | Grad Accum: {args.grad_accum}")
        print(f"Backbone LR: {args.backbone_lr} | Head LR: {args.head_lr} | Weight Decay: {args.weight_decay}")

    # Load CV splits
    with open(CONFIG_CV) as f:
        cv_splits = json.load(f)
    fold_data = cv_splits["folds"][f"fold_{args.fold}"]
    train_uids = fold_data["train_studies"]
    val_uids = fold_data["val_studies"]

    # Gold-58 zero leakage assertion
    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    assert len(set(train_uids) & gold_uids) == 0, "CRITICAL LEAKAGE: Gold in train set!"
    assert len(set(val_uids) & gold_uids) == 0, "CRITICAL LEAKAGE: Gold in val set!"

    # Load slot manifest & labels
    slot_df = pd.read_csv(SLOT_MANIFEST_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    train_dataset = PhaseD1KneeDataset(
        study_uids=train_uids,
        slot_manifest_df=slot_df,
        labels_df=labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
    )
    val_dataset = PhaseD1KneeDataset(
        study_uids=val_uids,
        slot_manifest_df=slot_df,
        labels_df=labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Instantiate model
    model = RadImageNetResNet50SlotHead(freeze_lower_blocks=True).to(device)

    # Differential learning rates
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(prefix in name for prefix in ["layer3", "layer4"]):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = ((len(train_loader) + args.grad_accum - 1) // args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.backbone_lr, args.head_lr],
        total_steps=total_steps,
        pct_start=0.15,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda")
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    best_val_auc = 0.0
    best_weighted_auc = 0.0
    best_predictions = None
    best_uids = None

    for epoch in range(args.epochs):
        t0 = time.time()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            grad_accum_steps=args.grad_accum, epoch=epoch, global_rank=global_rank
        )

        val_auc_unw, val_auc_w, per_target, y_pred, y_true, uids = evaluate(
            model.module if is_distributed else model, val_loader, device
        )

        elapsed = time.time() - t0
        is_best = val_auc_unw > best_val_auc

        if global_rank == 0:
            print(f"[Fold {args.fold}] Epoch {epoch+1:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Unw AUC: {val_auc_unw:.4f} | Val Wtd AUC: {val_auc_w:.4f} | Time: {elapsed:.1f}s {'*** BEST ***' if is_best else ''}")
            if is_best:
                best_val_auc = val_auc_unw
                best_weighted_auc = val_auc_w
                best_predictions = y_pred
                best_uids = uids

                # Save best model checkpoint
                ckpt_path = os.path.join(fold_ckpt_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": (model.module if is_distributed else model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_macro_roc_auc_unweighted": val_auc_unw,
                        "val_macro_roc_auc_weighted": val_auc_w,
                        "per_target_auc": per_target,
                        "fold": args.fold,
                    },
                    ckpt_path,
                )

                # Save validation predictions
                pred_df = pd.DataFrame(best_predictions, columns=TARGETS)
                pred_df.insert(0, "StudyInstanceUID", best_uids)
                pred_path = os.path.join(PREDICTIONS_DIR, f"fold_{args.fold}_val_predictions.csv")
                pred_df.to_csv(pred_path, index=False)

    if global_rank == 0:
        print(f"\n[Fold {args.fold} COMPLETE] Best Val Macro ROC-AUC: {best_val_auc:.4f} (Unweighted) | {best_weighted_auc:.4f} (Weighted)")

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
