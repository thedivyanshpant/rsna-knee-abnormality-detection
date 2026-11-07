#!/usr/bin/env python3
"""
Phase D9-A Production DDP Training Engine: Full-Depth 32-Slice Hybrid ConvNet.
RSNA Knee Abnormality Detection

Architecture: DINOv2 ViT-S/14 + Local Depthwise Conv Stem + SlotHead
Input: 6 Canonical MRI Slots, 12 groups of 3 slices across FULL 32-slice depth (0.0, 1.0)
Hardware: 2 x NVIDIA A100 per fold node
Optimizer: AdamW (Backbone LR: 8e-6, Head LR: 1e-3, Weight Decay: 0.02)
Scheduler: OneCycleLR (10 epochs, pct_start: 0.15)
Effective Batch Size: 8 studies (2/GPU x 2 GPUs x 2 grad accum)
Precision: BF16 Mixed Precision
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase_r_b", "d9a"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase_d", "d1", "scripts"))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from loss_d1 import weighted_soft_bce_loss
from model_d9a import PhaseD9AModel_FullDepthHybrid, PATHOLOGY_NAMES
from dataset_d9a import PhaseD9AKneeDataset, TARGETS

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
PROD_LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d6/resolver/d6_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
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
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def pr_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2 or np.sum(y_true == 1) == 0:
        return None
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true == 1)
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) + float(precision[0] * recall[0])
    return ap


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    epoch: int,
    global_rank: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

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

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    mean_loss = total_loss / max(num_batches, 1)
    if dist.is_initialized():
        loss_tensor = torch.tensor([mean_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        mean_loss = loss_tensor.item() / dist.get_world_size()

    return mean_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prod_targets_map: Dict[str, np.ndarray],
    prod_weights_map: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    model.eval()
    all_study_uids = []
    all_preds = []
    all_d3_targets = []
    all_d3_weights = []

    for batch in loader:
        uids = batch["study_uid"]
        images = batch["images"].to(device, non_blocking=True)
        slot_mask = batch["slot_mask"].to(device, non_blocking=True)
        targets = batch["targets"]
        weights = batch["weights"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(images, slot_mask)
            probs = torch.sigmoid(logits).float().cpu()

        all_study_uids.extend(uids)
        all_preds.append(probs.numpy())
        all_d3_targets.append(targets.numpy())
        all_d3_weights.append(weights.numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    d3_targets_np = np.concatenate(all_d3_targets, axis=0)
    d3_weights_np = np.concatenate(all_d3_weights, axis=0)

    prod_targets_list = []
    prod_weights_list = []
    for uid in all_study_uids:
        prod_targets_list.append(prod_targets_map.get(uid, np.full(len(TARGETS), 0.20, dtype=np.float32)))
        prod_weights_list.append(prod_weights_map.get(uid, np.full(len(TARGETS), 0.85, dtype=np.float32)))
    prod_targets_np = np.array(prod_targets_list, dtype=np.float32)
    prod_weights_np = np.array(prod_weights_list, dtype=np.float32)

    def calc_metrics(y_true, y_score, weights):
        unw_aucs = []
        wtd_aucs = []
        pr_aucs = []
        per_target = {}
        for c, t_name in enumerate(TARGETS):
            yc = (y_true[:, c] >= 0.5).astype(int)
            sc = y_score[:, c]
            wc = weights[:, c]
            auc_u = roc_auc_score_np(yc, sc)
            pr_u = pr_auc_score_np(yc, sc)
            if auc_u is not None:
                unw_aucs.append(auc_u)
                wtd_aucs.append(auc_u * float(np.mean(wc)))
            if pr_u is not None:
                pr_aucs.append(pr_u)
            per_target[t_name] = {
                "auc": auc_u,
                "pr_auc": pr_u,
                "prevalence": float(np.mean(yc)),
            }
        mean_unw = float(np.mean(unw_aucs)) if unw_aucs else 0.0
        mean_wtd = float(np.mean(wtd_aucs)) if wtd_aucs else 0.0
        mean_pr = float(np.mean(pr_aucs)) if pr_aucs else 0.0
        return {
            "unweighted_auc": mean_unw,
            "weighted_auc": mean_wtd,
            "prauc": mean_pr,
            "per_target": per_target,
        }

    d3_metrics = calc_metrics(d3_targets_np, preds_np, d3_weights_np)
    prod_metrics = calc_metrics(prod_targets_np, preds_np, prod_weights_np)

    pred_df = pd.DataFrame({"StudyInstanceUID": all_study_uids})
    for c, t_name in enumerate(TARGETS):
        pred_df[f"pred_{t_name}"] = preds_np[:, c]
        pred_df[f"true_{t_name}"] = d3_targets_np[:, c]
        pred_df[f"weight_{t_name}"] = d3_weights_np[:, c]

    return {
        "d3": d3_metrics,
        "prod": prod_metrics,
        "pred_df": pred_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase D9-A DDP Training Engine")
    parser.add_argument("--fold", type=int, required=True, help="CV fold index (0-4)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Microbatch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--backbone_lr", type=float, default=8e-6, help="Backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="SlotHead learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay")
    args = parser.parse_args()

    local_rank, global_rank, world_size, is_distributed = setup_ddp()
    set_seed(42, global_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_ckpt_dir = os.path.join(PROJECT_ROOT, f"phase_r_b/d9a/checkpoints/fold_{args.fold}")
    out_pred_dir = os.path.join(PROJECT_ROOT, "phase_r_b/d9a/predictions/val")
    if global_rank == 0:
        os.makedirs(out_ckpt_dir, exist_ok=True)
        os.makedirs(out_pred_dir, exist_ok=True)

    if global_rank == 0:
        print("=" * 80)
        print(f"  PHASE D9-A TRAINING: FULL-DEPTH HYBRID CONVNET — FOLD {args.fold}")
        print("=" * 80)
        print(f"World Size: {world_size} GPUs | Device: {device}")
        print(f"Microbatch: {args.batch_size} | Grad Accum: {args.grad_accum} | Effective Batch: {args.batch_size * world_size * args.grad_accum}")
        print(f"Backbone LR: {args.backbone_lr} | Head LR: {args.head_lr} | Weight Decay: {args.weight_decay}")
        print(f"Full-Depth Slices: [0.0, 1.0] (Slices 0 to 31) | Groups: 12")
        sys.stdout.flush()

    with open(CONFIG_CV) as f:
        cv_data = json.load(f)
    fold_key = f"fold_{args.fold}"
    train_uids = cv_data["folds"][fold_key]["train_studies"]
    val_uids = cv_data["folds"][fold_key]["val_studies"]

    slot_df = pd.read_csv(SLOT_MANIFEST_CSV)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)
    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    prod_targets_map = {}
    prod_weights_map = {}
    for uid, grp in prod_labels_df.groupby("StudyInstanceUID"):
        grp_indexed = grp.set_index("target")
        t_arr = np.array([grp_indexed.loc[t, "label_probability"] if t in grp_indexed.index else 0.20 for t in TARGETS], dtype=np.float32)
        w_arr = np.array([grp_indexed.loc[t, "sample_weight"] if t in grp_indexed.index else 0.85 for t in TARGETS], dtype=np.float32)
        prod_targets_map[uid] = t_arr
        prod_weights_map[uid] = w_arr

    train_dataset = PhaseD9AKneeDataset(
        study_uids=train_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
        n_groups=12,
        slice_band=(0.0, 1.0),
        target_resolution=(224, 224),
        target_fov_mm=130.0,
    )

    val_dataset = PhaseD9AKneeDataset(
        study_uids=val_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
        n_groups=12,
        slice_band=(0.0, 1.0),
        target_resolution=(224, 224),
        target_fov_mm=130.0,
    )

    if global_rank == 0:
        print(f"Dataset Loaded: {len(train_dataset)} train studies | {len(val_dataset)} val studies")
        sys.stdout.flush()

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=42) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    raw_model = PhaseD9AModel_FullDepthHybrid().to(device)

    if global_rank == 0:
        param_summary = raw_model.get_parameter_summary()
        print(f"Model Summary: {param_summary}")
        sys.stdout.flush()

    backbone_params = []
    head_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name or "conv_stem" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])

    total_steps = ((len(train_loader) + args.grad_accum - 1) // args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.backbone_lr, args.head_lr],
        total_steps=total_steps,
        pct_start=0.15,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    scaler = torch.amp.GradScaler("cuda")
    best_val_auc_d3 = -1.0
    best_eval_dict = {}
    history = []

    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = raw_model

    for epoch in range(args.epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t_epoch_start = time.time()
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum_steps=args.grad_accum,
            epoch=epoch,
            global_rank=global_rank,
        )
        t_train = time.time() - t_epoch_start

        if global_rank == 0:
            t_val_start = time.time()
            eval_res = evaluate(
                model=raw_model,
                loader=val_loader,
                device=device,
                prod_targets_map=prod_targets_map,
                prod_weights_map=prod_weights_map,
            )
            t_val = time.time() - t_val_start

            d3_unw = eval_res["d3"]["unweighted_auc"]
            d3_wtd = eval_res["d3"]["weighted_auc"]
            d3_prauc = eval_res["d3"]["prauc"]
            prod_unw = eval_res["prod"]["unweighted_auc"]
            prod_prauc = eval_res["prod"]["prauc"]

            print(
                f"[D9-A Fold {args.fold} | Epoch {epoch+1:02d}/{args.epochs:02d}] "
                f"Loss: {train_loss:.4f} | D3 AUC (Unw/Wtd): {d3_unw:.4f} / {d3_wtd:.4f} | "
                f"PR-AUC: {d3_prauc:.4f} | Prod AUC: {prod_unw:.4f} (PR: {prod_prauc:.4f}) | "
                f"Time: {t_train:.1f}s+{t_val:.1f}s",
                flush=True,
            )

            history.append({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "d3_unweighted_auc": round(d3_unw, 4),
                "d3_weighted_auc": round(d3_wtd, 4),
                "d3_prauc": round(d3_prauc, 4),
                "prod_unweighted_auc": round(prod_unw, 4),
                "prod_prauc": round(prod_prauc, 4),
            })

            if d3_unw > best_val_auc_d3:
                best_val_auc_d3 = d3_unw
                best_eval_dict = eval_res
                ckpt_path = os.path.join(out_ckpt_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc_d3": best_val_auc_d3,
                    "metrics": eval_res["d3"],
                    "config": vars(args),
                }, ckpt_path)

                pred_csv_path = os.path.join(out_pred_dir, f"fold_{args.fold}_val_predictions.csv")
                best_eval_dict["pred_df"]["fold"] = args.fold
                best_eval_dict["pred_df"].to_csv(pred_csv_path, index=False)
                print(f"  >>> Best Checkpoint Saved: Remediated Macro AUC = {best_val_auc_d3:.4f} (PR-AUC: {d3_prauc:.4f})", flush=True)

        if is_distributed:
            dist.barrier()

    if global_rank == 0:
        metrics_file = os.path.join(out_ckpt_dir, "training_history.json")
        with open(metrics_file, "w") as f:
            json.dump({
                "fold": args.fold,
                "best_d3_auc": best_val_auc_d3,
                "history": history,
                "best_metrics": best_eval_dict.get("d3", {}),
            }, f, indent=2)

        final_pred_path = os.path.join(out_pred_dir, f"fold_{args.fold}_val_predictions.csv")
        print(f"\n[D9-A FOLD {args.fold} COMPLETE] Best Remediated AUC: {best_val_auc_d3:.4f} | Predictions: {final_pred_path}", flush=True)

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
