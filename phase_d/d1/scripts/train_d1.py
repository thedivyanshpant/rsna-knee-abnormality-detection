"""
Phase D1: Training Engine for DINOv2 / SlotHead Architecture (D1.1).

Implements Sections 22, 24, 25, 26, 27, 28, 29, 30, 36, 37:
- Model: DINOv2 ViT-S/14 (blocks 0-5 frozen, 6-11 trainable) + SlotHead (12 pathology queries + 0.55 priors).
- Optimizer: AdamW with differential learning rates:
    * Backbone LR: 8e-6
    * SlotHead LR: 1e-3
    * Weight decay: 0.02
- Loss: Weighted soft-BCE using Phase B continuous targets and sample weights.
- Scheduler: OneCycleLR or CosineAnnealingLR.
- Validation: Per-target ROC-AUC and primary Macro ROC-AUC across 12 targets.
- Checkpoints: Saves best model to phase_d/d1/checkpoints/fold_<fold_idx>/best_model.pt.
"""

import sys
import os
import time
import json
import argparse
import datetime

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase_d.d1.scripts.model_d1 import PhaseD1Model, PATHOLOGY_NAMES
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset
from phase_d.d1.scripts.loss_d1 import weighted_soft_bce_loss

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/manifests/d1_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/checkpoints")
METRICS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/metrics")
LOGS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/logs")


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
) -> Tuple[float, Dict[str, Optional[float]]]:
    """Compute per-target and macro ROC-AUC across 12 pathologies."""
    per_target = {}
    valid_aucs = []

    for i, name in enumerate(PATHOLOGY_NAMES):
        t_true = y_true[:, i]
        t_pred = y_pred[:, i]
        w = weights[:, i]

        # Filter out abstentions (w == 0)
        active_mask = (w > 0.0)
        if active_mask.sum() > 0:
            act_true = t_true[active_mask]
            act_pred = t_pred[active_mask]
            bin_true = (act_true >= 0.5).astype(int)
            if len(np.unique(bin_true)) >= 2:
                auc = roc_auc_score_np(bin_true, act_pred)
                if auc is not None:
                    per_target[name] = round(float(auc), 4)
                    valid_aucs.append(auc)
                else:
                    per_target[name] = None
            else:
                per_target[name] = None
        else:
            per_target[name] = None

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return round(macro_auc, 4), per_target


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    grad_accum_steps: int = 2,
    max_batches: Optional[int] = None
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device, non_blocking=True)       # (B, 6, 12, 3, 224, 224)
        slot_mask = batch["slot_mask"].to(device, non_blocking=True) # (B, 6)
        targets = batch["targets"].to(device, non_blocking=True)     # (B, 12)
        weights = batch["weights"].to(device, non_blocking=True)     # (B, 12)

        with torch.amp.autocast("cuda"):
            logits, attn = model(images, slot_mask)
            loss = weighted_soft_bce_loss(logits, targets, weights)
            loss_scaled = loss / grad_accum_steps

        loss_scaled.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * targets.size(0)
        total_weight += targets.size(0)

    return total_loss / max(total_weight, 1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0

    all_preds = []
    all_targets = []
    all_weights = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device, non_blocking=True)
        slot_mask = batch["slot_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        weights = batch["weights"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits, attn = model(images, slot_mask)
            loss = weighted_soft_bce_loss(logits, targets, weights)
            probs = torch.sigmoid(logits)

        total_loss += loss.item() * targets.size(0)
        total_weight += targets.size(0)

        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_weights.append(weights.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 12))
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 12))
    y_w = np.concatenate(all_weights, axis=0) if all_weights else np.zeros((0, 12))

    val_loss = total_loss / max(total_weight, 1.0)
    macro_auc, per_target_auc = compute_macro_roc_auc(y_true, y_pred, y_w)

    return val_loss, macro_auc, per_target_auc, y_pred, y_true


def run_training_fold(
    fold: int = 0,
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    backbone_lr: float = 8e-6,
    head_lr: float = 1e-3,
    weight_decay: float = 0.02,
    n_groups: int = 12,
    num_workers: int = 4,
    sanity_check: bool = False,
    device_id: int = 0
) -> Dict[str, Any]:
    """Execute training for one fold."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase D1.1 Training: Fold {fold} on {device} ===")

    # 1. Load CV Splits
    with open(CONFIG_CV) as f:
        cv_data = json.load(f)

    fold_key = f"fold_{fold}"
    train_studies = cv_data["folds"][fold_key]["train_studies"]
    val_studies = cv_data["folds"][fold_key]["val_studies"]

    # 2. Load Manifest & Labels
    slot_manifest_df = pd.read_csv(SLOT_MANIFEST_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    if sanity_check:
        print("[SANITY CHECK MODE] Restricting to 8 train studies and 8 val studies.")
        train_studies = train_studies[:8]
        val_studies = val_studies[:8]
        epochs = 2

    # 3. Create Datasets
    train_ds = PhaseD1KneeDataset(
        study_uids=train_studies,
        slot_manifest_df=slot_manifest_df,
        labels_df=labels_df,
        processed_dir=PROCESSED_DIR,
        n_groups=n_groups
    )
    val_ds = PhaseD1KneeDataset(
        study_uids=val_studies,
        slot_manifest_df=slot_manifest_df,
        labels_df=labels_df,
        processed_dir=PROCESSED_DIR,
        n_groups=n_groups
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 4. Instantiate Model & Parameter Groups (Section 28)
    model = PhaseD1Model().to(device)
    param_summary = model.get_parameter_summary()
    print("Parameter Architecture Summary:")
    print(f"  Total: {param_summary['total_parameters']:,}")
    print(f"  Trainable: {param_summary['trainable_parameters']:,} ({param_summary['trainable_percentage']}%)")
    print(f"  Frozen: {param_summary['frozen_parameters']:,}")

    # Differential learning rates: backbone LR = 8e-6, SlotHead LR = 1e-3
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-7
    )

    # Output paths
    fold_ckpt_dir = os.path.join(CHECKPOINTS_DIR, f"fold_{fold}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(fold_ckpt_dir, "best_model.pt")

    history = []
    best_macro_auc = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, macro_auc, per_target_auc, y_pred, y_true = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )
        ep_sec = time.time() - ep_t0

        is_best = macro_auc > best_macro_auc
        if is_best:
            best_macro_auc = macro_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_auc": macro_auc,
                "per_target_auc": per_target_auc,
                "fold": fold,
                "config": {
                    "backbone_lr": backbone_lr,
                    "head_lr": head_lr,
                    "batch_size": batch_size * grad_accum_steps,
                    "epochs": epochs
                }
            }, best_ckpt_path)

        ep_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "macro_roc_auc": macro_auc,
            "best_macro_roc_auc": best_macro_auc,
            "elapsed_sec": round(ep_sec, 2),
            "per_target_auc": per_target_auc
        }
        history.append(ep_record)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {macro_auc:.4f} | Time: {ep_sec:.1f}s | Best: {is_best}")

    total_wall_sec = round(time.time() - start_time, 2)
    summary = {
        "experiment_id": "D1.1-DINOV2-SLOTHEAD",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fold": fold,
        "epochs_completed": epochs,
        "best_val_macro_auc": best_macro_auc,
        "total_wall_sec": total_wall_sec,
        "parameter_summary": param_summary,
        "best_checkpoint": best_ckpt_path,
        "history": history,
        "sanity_check": sanity_check
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    summary_path = os.path.join(METRICS_DIR, f"d1_fold_{fold}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase D1.1 Training Engine")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    summary = run_training_fold(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        sanity_check=args.sanity_check
    )
    print("\n[SUCCESS] Fold Execution Finished.")
    print(f"Best Val Macro AUC: {summary['best_val_macro_auc']}")
    print(f"Summary saved to: {os.path.join(METRICS_DIR, f'd1_fold_{args.fold}_summary.json')}")
