#!/usr/bin/env python3
"""
Phase D4: Multi-Node DDP Training Engine for RadImageNet ResNet-50 on Remediated Supervision (D3).
RSNA Knee Abnormality Detection

Architecture: RadImageNet ResNet-50 (lower layers frozen, layers 3-4 trainable) + Anatomical SlotHead (12 pathology queries + 0.55 priors)
Supervision: Phase D3 Remediated Labels (phase_d3_candidate_label_corpus.csv)
Cluster Topology: 3 nodes x 2 GPUs = 6 x NVIDIA A100 80GB (NCCL DistributedDataParallel)
Optimizer: AdamW (Backbone LR: 1e-4, SlotHead LR: 1e-3, Weight Decay: 0.02)
Scheduler: OneCycleLR (pct_start: 0.15, cosine annealing, 10 epochs)
Effective Batch Size: 6 studies (1 study/GPU x 6 GPUs x 1 grad accum step)
Dual Evaluation: Validates against D3 Remediated Labels (selection criterion) & Legacy Phase B Labels (continuity)
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

from phase_d.d4.scripts.model_d4_cnn import RadImageNetResNet50SlotHead
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset, TARGETS
from phase_d.d1.scripts.loss_d1 import weighted_soft_bce_loss

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
PROD_LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/manifests/d1_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/checkpoints")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/predictions/val")
METRICS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d4/reports")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "eth2"
        os.environ["GLOO_SOCKET_IFNAME"] = "eth2"
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


def pr_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy PR-AUC using trapezoidal rule over sorted precision/recall curve."""
    pos = np.sum(y_true == 1)
    if pos == 0 or len(y_true) == 0:
        return None
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    recalls = tps / pos
    precisions = tps / (tps + fps)
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([1.0], precisions))
    return float(np.trapz(precisions, recalls))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray
) -> Tuple[float, float, float, Dict[str, Dict[str, Optional[float]]]]:
    """Compute unweighted and sample-weighted macro ROC-AUC and PR-AUC across 12 pathologies."""
    valid_aucs_unw = []
    valid_aucs_wtd = []
    valid_praucs = []
    weights_sum = []
    per_target = {}

    for c in range(12):
        t_name = TARGETS[c]
        y_c = y_true[:, c]
        pred_c = y_pred[:, c]
        w_c = weights[:, c]

        bin_y = (y_c >= 0.5).astype(int)
        pos_count = int(np.sum(bin_y == 1))
        neg_count = int(np.sum(bin_y == 0))

        if pos_count > 0 and neg_count > 0:
            auc = roc_auc_score_np(bin_y, pred_c)
            prauc = pr_auc_score_np(bin_y, pred_c)
            total_w = float(np.sum(w_c))
            per_target[t_name] = {
                "pos_count": pos_count,
                "neg_count": neg_count,
                "roc_auc_unweighted": round(auc, 4) if auc is not None else None,
                "pr_auc": round(prauc, 4) if prauc is not None else None,
            }
            if auc is not None:
                valid_aucs_unw.append(auc)
                if total_w > 0:
                    valid_aucs_wtd.append(auc * total_w)
                    weights_sum.append(total_w)
            if prauc is not None:
                valid_praucs.append(prauc)
        else:
            per_target[t_name] = {
                "pos_count": pos_count,
                "neg_count": neg_count,
                "roc_auc_unweighted": None,
                "pr_auc": None,
            }

    macro_unweighted = float(np.mean(valid_aucs_unw)) if valid_aucs_unw else 0.0
    macro_weighted = float(np.sum(valid_aucs_wtd) / np.sum(weights_sum)) if weights_sum else macro_unweighted
    macro_prauc = float(np.mean(valid_praucs)) if valid_praucs else 0.0
    return macro_unweighted, macro_weighted, macro_prauc, per_target


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    epoch: int = 0,
    global_rank: int = 0,
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
    device: torch.device,
    prod_targets_map: Optional[Dict[str, np.ndarray]] = None,
    prod_weights_map: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets_d3 = []
    all_weights_d3 = []
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
                probs = torch.sigmoid(logits).float().cpu().numpy()

            all_preds.append(probs)
            all_targets_d3.append(targets)
            all_weights_d3.append(weights)
            all_uids.extend(uids)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true_d3 = np.concatenate(all_targets_d3, axis=0)
    w_true_d3 = np.concatenate(all_weights_d3, axis=0)

    # 1. Evaluation against D3 Candidate Remediated Labels
    unw_d3, wtd_d3, pr_d3, per_target_d3 = compute_metrics(y_true_d3, y_pred, w_true_d3)

    # 2. Evaluation against Original Production Labels (if provided)
    unw_prod, wtd_prod, pr_prod, per_target_prod = 0.0, 0.0, 0.0, {}
    if prod_targets_map is not None and prod_weights_map is not None:
        y_true_prod = np.array([prod_targets_map[u] for u in all_uids])
        w_true_prod = np.array([prod_weights_map[u] for u in all_uids])
        unw_prod, wtd_prod, pr_prod, per_target_prod = compute_metrics(y_true_prod, y_pred, w_true_prod)

    return {
        "d3": {
            "unweighted_auc": unw_d3,
            "weighted_auc": wtd_d3,
            "pr_auc": pr_d3,
            "per_target": per_target_d3,
        },
        "prod": {
            "unweighted_auc": unw_prod,
            "weighted_auc": wtd_prod,
            "pr_auc": pr_prod,
            "per_target": per_target_prod,
        },
        "predictions": y_pred,
        "uids": all_uids,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase D4 RadImageNet ResNet50 Multi-Node DDP Training Engine")
    parser.add_argument("--fold", type=int, default=0, help="CV fold index (0..4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Microbatch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
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
        print(f"  PHASE D4 RADIMAGENET RESNET-50 + REMEDIATED LABELS DDP: FOLD {args.fold}")
        print("=" * 80)
        print(f"World Size: {world_size} GPUs | Device: {device} | Microbatch: {args.batch_size} | Grad Accum: {args.grad_accum}")
        print(f"Effective Batch: {args.batch_size * world_size * args.grad_accum} studies")
        print(f"Backbone LR: {args.backbone_lr} | Head LR: {args.head_lr} | Weight Decay: {args.weight_decay}")
        sys.stdout.flush()

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

    # Load slot manifest & candidate D3 remediated labels
    slot_df = pd.read_csv(SLOT_MANIFEST_CSV)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)

    # Pre-build legacy production labels mapping for dual evaluation
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)
    prod_targets_map = {}
    prod_weights_map = {}
    for uid, grp in prod_labels_df.groupby("StudyInstanceUID"):
        grp_indexed = grp.set_index("target")
        t_arr = np.array([grp_indexed.loc[t, "label_probability"] if t in grp_indexed.index else 0.20 for t in TARGETS], dtype=np.float32)
        w_arr = np.array([grp_indexed.loc[t, "sample_weight"] if t in grp_indexed.index else 0.85 for t in TARGETS], dtype=np.float32)
        prod_targets_map[uid] = t_arr
        prod_weights_map[uid] = w_arr

    train_dataset = PhaseD1KneeDataset(
        study_uids=train_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
    )
    val_dataset = PhaseD1KneeDataset(
        study_uids=val_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
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
    raw_model = RadImageNetResNet50SlotHead(freeze_lower_blocks=True).to(device)

    # Differential learning rates
    backbone_params = [
        p for name, p in raw_model.named_parameters()
        if p.requires_grad and not any(k in name for k in ["slot_norm", "slot_proj", "pathology_queries", "q_proj", "k_proj", "v_proj", "classifier"])
    ]
    head_params = [
        p for name, p in raw_model.named_parameters()
        if p.requires_grad and any(k in name for k in ["slot_norm", "slot_proj", "pathology_queries", "q_proj", "k_proj", "v_proj", "classifier"])
    ]

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
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = raw_model

    best_val_auc_d3 = 0.0
    best_results = None

    for epoch in range(args.epochs):
        t0 = time.time()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            grad_accum_steps=args.grad_accum, epoch=epoch, global_rank=global_rank
        )

        eval_res = evaluate(
            raw_model, val_loader, device,
            prod_targets_map=prod_targets_map, prod_weights_map=prod_weights_map
        )

        elapsed = time.time() - t0
        auc_d3 = eval_res["d3"]["unweighted_auc"]
        wtd_d3 = eval_res["d3"]["weighted_auc"]
        pr_d3 = eval_res["d3"]["pr_auc"]
        auc_prod = eval_res["prod"]["unweighted_auc"]
        is_best = auc_d3 > best_val_auc_d3

        if global_rank == 0:
            print(
                f"[Fold {args.fold}] Ep {epoch+1:2d}/{args.epochs:2d} | "
                f"Loss: {train_loss:.4f} | "
                f"D3 AUC: {auc_d3:.4f} (Wtd: {wtd_d3:.4f} | PR: {pr_d3:.4f}) | "
                f"Prod AUC: {auc_prod:.4f} | Time: {elapsed:.1f}s "
                f"{'*** BEST ***' if is_best else ''}",
                flush=True,
            )
            if is_best:
                best_val_auc_d3 = auc_d3
                best_results = eval_res

                # Save best checkpoint
                ckpt_path = os.path.join(fold_ckpt_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval_metrics_d3": eval_res["d3"],
                        "eval_metrics_prod": eval_res["prod"],
                        "fold": args.fold,
                    },
                    ckpt_path,
                )

                # Save validation predictions
                pred_df = pd.DataFrame(eval_res["predictions"], columns=TARGETS)
                pred_df.insert(0, "StudyInstanceUID", eval_res["uids"])
                pred_path = os.path.join(PREDICTIONS_DIR, f"fold_{args.fold}_val_predictions.csv")
                pred_df.to_csv(pred_path, index=False)

    if global_rank == 0:
        print(f"\n[Fold {args.fold} COMPLETE] Best Val Macro ROC-AUC (D3 Labels): {best_val_auc_d3:.4f}", flush=True)

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
