#!/usr/bin/env python3
"""
Phase D1.1: Production DDP Training Engine for Fold 0.
RSNA Knee Abnormality Detection

Architecture: DINOv2 ViT-S/14 (Blocks 0-5 frozen, 6-11 trainable) + SlotHead (12 pathology queries + 0.55 priors)
Hardware: 2 x NVIDIA A100 PCIe 80GB (NCCL DistributedDataParallel)
Optimizer: AdamW (Backbone LR: 8e-6, SlotHead LR: 1e-3, Weight Decay: 0.02)
Scheduler: OneCycleLR (pct_start: 0.15)
Effective Batch Size: 8 studies (2 studies/GPU x 2 GPUs x 2 grad accum steps)
Validation: Exact 872 held-out validation studies evaluated on rank 0
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

from phase_d.d1.scripts.model_d1 import PhaseD1Model, PATHOLOGY_NAMES
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset
from phase_d.d1.scripts.loss_d1 import weighted_soft_bce_loss

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/manifests/d1_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/checkpoints")
METRICS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/metrics")
LOGS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d1/logs")
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
) -> Tuple[float, Dict[str, Optional[float]]]:
    """Compute per-target and macro ROC-AUC across 12 pathologies."""
    per_target = {}
    valid_aucs = []

    for i, name in enumerate(PATHOLOGY_NAMES):
        t_true = y_true[:, i]
        t_pred = y_pred[:, i]
        w = weights[:, i]

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


def run_ddp_training(
    fold: int = 0,
    epochs: int = 10,
    batch_size: int = 2,
    grad_accum_steps: int = 2,
    backbone_lr: float = 8e-6,
    head_lr: float = 1e-3,
    weight_decay: float = 0.02,
    pct_start: float = 0.15,
    num_workers: int = 4
):
    local_rank, global_rank, world_size, is_distributed = setup_ddp()
    set_seed(42, rank=global_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print("=" * 80)
        print(f"  PHASE D1.1 PRODUCTION TRAINING: FOLD {fold}")
        print(f"  Architecture: DINOv2 ViT-S/14 + SlotHead (12 queries + 0.55 priors)")
        print(f"  GPUs: {world_size} x NVIDIA A100 80GB (NCCL DDP)")
        print(f"  Epochs: {epochs} | Microbatch: {batch_size}/GPU | Grad Accum: {grad_accum_steps}")
        print(f"  Effective Batch Size: {batch_size * world_size * grad_accum_steps} studies")
        print(f"  Backbone LR: {backbone_lr:.2e} | SlotHead LR: {head_lr:.2e} | Weight Decay: {weight_decay}")
        print("=" * 80)

    # 1. Load CV Splits and strictly verify Gold-58 isolation
    with open(CONFIG_CV) as f:
        cv_data = json.load(f)

    fold_key = f"fold_{fold}"
    train_uids = cv_data["folds"][fold_key]["train_studies"]
    val_uids = cv_data["folds"][fold_key]["val_studies"]

    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    assert len(set(train_uids) & gold_uids) == 0, f"CRITICAL: Gold leakage in Fold {fold} train!"
    assert len(set(val_uids) & gold_uids) == 0, f"CRITICAL: Gold leakage in Fold {fold} val!"

    if global_rank == 0:
        print(f"[DATA] Train studies: {len(train_uids):,} | Val studies: {len(val_uids):,}")
        print(f"[GOVERNANCE] Gold-58 overlap: 0 (Strictly isolated)")

    # 2. Datasets
    slot_manifest_df = pd.read_csv(SLOT_MANIFEST_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    train_ds = PhaseD1KneeDataset(
        study_uids=train_uids,
        slot_manifest_df=slot_manifest_df,
        labels_df=labels_df,
        processed_dir=PROCESSED_DIR,
        n_groups=12,
        gold_uids=gold_uids
    )

    val_ds = PhaseD1KneeDataset(
        study_uids=val_uids,
        slot_manifest_df=slot_manifest_df,
        labels_df=labels_df,
        processed_dir=PROCESSED_DIR,
        n_groups=12,
        gold_uids=gold_uids
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=global_rank, shuffle=True, seed=42
    ) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 3. Model & Parameter Freezing
    raw_model = PhaseD1Model().to(device)

    if global_rank == 0:
        pinfo = raw_model.get_parameter_summary()
        print(f"[MODEL] Total Parameters: {pinfo['total_parameters']:,}")
        print(f"[MODEL] Trainable Parameters: {pinfo['trainable_parameters']:,} ({pinfo['trainable_percentage']}%)")
        print(f"[MODEL] Frozen Parameters: {pinfo['frozen_parameters']:,}")

    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = raw_model

    # 4. Differential Learning Rates Optimizer & Scheduler
    backbone_params = [p for p in raw_model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in raw_model.head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    steps_per_epoch = (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
    total_steps = steps_per_epoch * epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[backbone_lr, head_lr],
        total_steps=max(total_steps, 1),
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # Output directory
    fold_ckpt_dir = os.path.join(CHECKPOINTS_DIR, f"fold_{fold}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(fold_ckpt_dir, "best_model.pt")

    best_macro_auc = -1.0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Training Phase ----
        model.train()
        train_loss = 0.0
        train_weight = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            slot_mask = batch["slot_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits, _ = model(images, slot_mask)
                loss = weighted_soft_bce_loss(logits, targets, weights)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            w_sum = max(weights.sum().item(), 1.0)
            train_loss += loss.item() * w_sum
            train_weight += w_sum

        # Reduce train loss across ranks
        if is_distributed:
            t_tensor = torch.tensor([train_loss, train_weight], device=device)
            dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)
            global_train_loss = (t_tensor[0] / t_tensor[1].clamp(min=1.0)).item()
        else:
            global_train_loss = train_loss / max(train_weight, 1.0)

        # ---- Validation Phase (Rank 0 Evaluates Exact Held-Out Val Set) ----
        if global_rank == 0:
            raw_model.eval()
            val_loss = 0.0
            val_weight = 0.0
            val_preds_list = []
            val_targets_list = []
            val_weights_list = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["images"].to(device, non_blocking=True)
                    slot_mask = batch["slot_mask"].to(device, non_blocking=True)
                    targets = batch["targets"].to(device, non_blocking=True)
                    weights = batch["weights"].to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):
                        logits, _ = raw_model(images, slot_mask)
                        loss = weighted_soft_bce_loss(logits, targets, weights)
                        probs = torch.sigmoid(logits)

                    w_sum = max(weights.sum().item(), 1.0)
                    val_loss += loss.item() * w_sum
                    val_weight += w_sum

                    val_preds_list.append(probs.cpu().numpy())
                    val_targets_list.append(targets.cpu().numpy())
                    val_weights_list.append(weights.cpu().numpy())

            val_preds_np = np.concatenate(val_preds_list, axis=0)
            val_targets_np = np.concatenate(val_targets_list, axis=0)
            val_weights_np = np.concatenate(val_weights_list, axis=0)

            global_val_loss = val_loss / max(val_weight, 1.0)
            macro_auc, per_target_auc = compute_macro_roc_auc(
                val_targets_np, val_preds_np, val_weights_np
            )

            ep_time = time.time() - ep_t0
            is_best = macro_auc > best_macro_auc

            if is_best:
                best_macro_auc = macro_auc
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": global_val_loss,
                    "macro_roc_auc": macro_auc,
                    "per_target_auc": per_target_auc,
                    "config": {
                        "backbone_lr": backbone_lr,
                        "head_lr": head_lr,
                        "effective_batch_size": batch_size * world_size * grad_accum_steps,
                        "epochs": epochs
                    }
                }, best_ckpt_path)

                # Save best validation predictions to CSV
                val_pred_dict = {"StudyInstanceUID": val_uids, "fold": fold}
                for t_idx, t_name in enumerate(PATHOLOGY_NAMES):
                    val_pred_dict[f"pred_{t_name}"] = val_preds_np[:, t_idx]
                    val_pred_dict[f"true_{t_name}"] = val_targets_np[:, t_idx]
                    val_pred_dict[f"weight_{t_name}"] = val_weights_np[:, t_idx]
                val_df = pd.DataFrame(val_pred_dict)
                val_res_dir = os.path.join(PROJECT_ROOT, "phase_d", "d1", "results", "val")
                os.makedirs(val_res_dir, exist_ok=True)
                val_df.to_csv(os.path.join(val_res_dir, f"fold_{fold}_val_predictions.csv"), index=False)

            ep_rec = {
                "epoch": epoch,
                "train_loss": round(global_train_loss, 4),
                "val_loss": round(global_val_loss, 4),
                "val_macro_roc_auc": macro_auc,
                "best_val_macro_roc_auc": best_macro_auc,
                "elapsed_sec": round(ep_time, 1),
                "is_best": is_best,
                "per_target_auc": per_target_auc
            }
            history.append(ep_rec)

            print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss: {global_train_loss:.4f} | Val Loss: {global_val_loss:.4f} | Val Macro AUC: {macro_auc:.4f} (Best: {best_macro_auc:.4f}) | Time: {ep_time:.1f}s | Saved: {is_best}")

        if is_distributed:
            dist.barrier()

    # Save summary on rank 0
    if global_rank == 0:
        c3_baselines = {0: 0.6146, 1: 0.6400, 2: 0.6603, 3: 0.6647, 4: 0.6122}
        fold_c3_base = c3_baselines.get(fold, 0.6198)

        total_time = round(time.time() - start_time, 2)
        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETE: FOLD {fold}")
        print(f"Total Wall Time: {total_time:.1f}s ({total_time / 60.0:.2f} min)")
        print(f"Best Val Macro ROC-AUC: {best_macro_auc:.4f} (Phase C Fold {fold} Baseline: {fold_c3_base:.4f})")
        print(f"Best Checkpoint: {best_ckpt_path}")
        print("=" * 80)

        summary = {
            "experiment_id": f"D1.1-FOLD-{fold}-PRODUCTION",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "fold": fold,
            "epochs": epochs,
            "world_size": world_size,
            "effective_batch_size": batch_size * world_size * grad_accum_steps,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "weight_decay": weight_decay,
            "best_val_macro_auc": best_macro_auc,
            "phase_c_baseline_fold_auc": fold_c3_base,
            "delta_vs_phase_c": round(best_macro_auc - fold_c3_base, 4),
            "total_runtime_sec": total_time,
            "checkpoint_path": best_ckpt_path,
            "history": history,
            "status": "COMPLETED"
        }

        os.makedirs(METRICS_DIR, exist_ok=True)
        summary_path = os.path.join(METRICS_DIR, f"d1_fold_{fold}_production_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase D1.1 DDP Training Engine")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--backbone_lr", type=float, default=8e-6)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_ddp_training(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers
    )
