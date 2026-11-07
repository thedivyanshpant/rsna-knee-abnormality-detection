#!/usr/bin/env python3
"""
Phase D8 Production DDP Training Engine for Experiments 8A, 8B, 8C, and 8D.
RSNA Knee Abnormality Detection

Models:
- D8-A: Pathology-Conditioned Multi-Head Cross-Attention (ViT-S/14, 12 groups)
- D8-B: High-Budget Multi-Window Representation (ViT-S/14, 24 groups)
- D8-C: Hybrid Convolution + Attention Architecture (CoAtNet-Style, 12 groups)
- D8-D: Full Combined Configuration (Hybrid Conv+Attn + 24 groups + Pathology Cross-Attention)

Protocol:
- 5-Fold Stratified CV
- Effective Batch Size: 8 (2/GPU x 2 GPUs x 2 grad accum)
- OneCycleLR (10 epochs, max LR: 8e-6 backbone, 1e-3 head)
- BF16 Autocast
- Validation on rank 0 with barrier synchronization
- Zero Gold-58 exposure
"""

import os
import sys
import time
import io
import json
import random
import argparse
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

from phase_d.d1.scripts.loss_d1 import weighted_soft_bce_loss
from phase_d.d8.models.model_d8 import (
    PhaseD8Model_PathologyCrossAttention,
    PhaseD8Model_HighBudgetDepth,
    PhaseD8Model_HybridConvAttn,
    PhaseD8Model_FullCombination,
    PATHOLOGY_NAMES
)
from phase_d.d8.dataset_d8 import PhaseD8KneeDataset, TARGETS

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
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    prec = tp / (tp + fp)
    rec = tp / np.sum(y_true == 1)
    rec_diff = np.diff(np.concatenate(([0.0], rec)))
    return float(np.sum(rec_diff * prec))


def compute_macro_roc_auc(
    targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray
) -> Tuple[float, float, float, Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    valid_aucs = []
    valid_praucs = []
    weighted_aucs = []
    weights_sum = []
    per_target = {}
    per_target_prauc = {}

    for c, t_name in enumerate(TARGETS):
        y_c = targets[:, c]
        pred_c = predictions[:, c]
        w_c = weights[:, c]

        bin_y = (y_c >= 0.5).astype(int)
        if len(np.unique(bin_y)) > 1:
            auc = roc_auc_score_np(bin_y, pred_c)
            prauc = pr_auc_score_np(bin_y, pred_c)
            per_target[t_name] = auc
            per_target_prauc[t_name] = prauc
            if auc is not None:
                valid_aucs.append(auc)
                total_w = float(np.sum(w_c))
                if total_w > 0:
                    weighted_aucs.append(auc * total_w)
                    weights_sum.append(total_w)
            if prauc is not None:
                valid_praucs.append(prauc)
        else:
            per_target[t_name] = None
            per_target_prauc[t_name] = None

    macro_unweighted = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    macro_weighted = float(np.sum(weighted_aucs) / np.sum(weights_sum)) if weights_sum else macro_unweighted
    macro_prauc = float(np.mean(valid_praucs)) if valid_praucs else 0.0
    return macro_unweighted, macro_weighted, macro_prauc, per_target, per_target_prauc


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
            try:
                scheduler.step()
            except ValueError:
                pass

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prod_targets_map: Optional[Dict[str, np.ndarray]] = None,
    prod_weights_map: Optional[Dict[str, np.ndarray]] = None
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

    unw_d3, wtd_d3, prauc_d3, per_target_d3, per_target_prauc_d3 = compute_macro_roc_auc(y_true_d3, y_pred, w_true_d3)

    unw_prod, wtd_prod, prauc_prod, per_target_prod, per_target_prauc_prod = 0.0, 0.0, 0.0, {}, {}
    if prod_targets_map is not None and prod_weights_map is not None:
        y_true_prod = np.array([prod_targets_map[u] for u in all_uids])
        w_true_prod = np.array([prod_weights_map[u] for u in all_uids])
        unw_prod, wtd_prod, prauc_prod, per_target_prod, per_target_prauc_prod = compute_macro_roc_auc(y_true_prod, y_pred, w_true_prod)

    return {
        "d3": {
            "unweighted_auc": unw_d3,
            "weighted_auc": wtd_d3,
            "prauc": prauc_d3,
            "per_target": per_target_d3,
            "per_target_prauc": per_target_prauc_d3
        },
        "prod": {
            "unweighted_auc": unw_prod,
            "weighted_auc": wtd_prod,
            "prauc": prauc_prod,
            "per_target": per_target_prod,
            "per_target_prauc": per_target_prauc_prod
        },
        "predictions": y_pred,
        "targets_d3": y_true_d3,
        "weights_d3": w_true_d3,
        "uids": all_uids,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase D8 Production DDP Training Engine")
    parser.add_argument("--model_type", type=str, required=True, choices=["d8a", "d8b", "d8c", "d8d"],
                        help="d8a: Cross-Attn | d8b: High-Budget Depth | d8c: Hybrid Conv+Attn | d8d: Full Combined")
    parser.add_argument("--fold", type=int, default=0, help="CV fold index (0..4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Microbatch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--backbone_lr", type=float, default=8e-6, help="Backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay")
    args = parser.parse_args()

    local_rank, global_rank, world_size, is_distributed = setup_ddp()
    set_seed(42, global_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    base_out_dir = os.path.join(PROJECT_ROOT, "phase_d", "d8", args.model_type)
    fold_ckpt_dir = os.path.join(base_out_dir, "checkpoints", f"fold_{args.fold}")
    pred_val_dir = os.path.join(base_out_dir, "predictions", "val")
    reports_dir = os.path.join(base_out_dir, "reports")

    if global_rank == 0:
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        os.makedirs(pred_val_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        print("=" * 80)
        print(f"  PHASE D8 TRAINING: EXPERIMENT [{args.model_type.upper()}] — FOLD {args.fold}")
        print("=" * 80)
        print(f"Model Type: {args.model_type.upper()} | World Size: {world_size} GPUs | Device: {device}")
        print(f"Microbatch: {args.batch_size} | Grad Accum: {args.grad_accum} | Effective Batch: {args.batch_size * world_size * args.grad_accum}")
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

    slot_df = pd.read_csv(SLOT_MANIFEST_CSV)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)

    # Pre-build production labels mapping for dual evaluation
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)
    prod_targets_map = {}
    prod_weights_map = {}
    for uid, grp in prod_labels_df.groupby("StudyInstanceUID"):
        grp_indexed = grp.set_index("target")
        t_arr = np.array([grp_indexed.loc[t, "label_probability"] if t in grp_indexed.index else 0.20 for t in TARGETS], dtype=np.float32)
        w_arr = np.array([grp_indexed.loc[t, "sample_weight"] if t in grp_indexed.index else 0.85 for t in TARGETS], dtype=np.float32)
        prod_targets_map[uid] = t_arr
        prod_weights_map[uid] = w_arr

    # High-budget flag: 24 groups for D8-B and D8-D; 12 groups for D8-A and D8-C
    use_high_budget = (args.model_type in ["d8b", "d8d"])

    train_dataset = PhaseD8KneeDataset(
        study_uids=train_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
        high_budget=use_high_budget,
        target_resolution=(224, 224),
        target_fov_mm=130.0
    )

    val_dataset = PhaseD8KneeDataset(
        study_uids=val_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
        high_budget=use_high_budget,
        target_resolution=(224, 224),
        target_fov_mm=130.0
    )

    if global_rank == 0:
        print(f"Dataset Loaded: {len(train_dataset)} train studies | {len(val_dataset)} val studies | HighBudget: {use_high_budget} ({train_dataset.num_groups} groups)")
        sys.stdout.flush()

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=42) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Instantiate Model
    if args.model_type == "d8a":
        raw_model = PhaseD8Model_PathologyCrossAttention().to(device)
    elif args.model_type == "d8b":
        raw_model = PhaseD8Model_HighBudgetDepth().to(device)
    elif args.model_type == "d8c":
        raw_model = PhaseD8Model_HybridConvAttn().to(device)
    elif args.model_type == "d8d":
        raw_model = PhaseD8Model_FullCombination().to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

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
            global_rank=global_rank
        )
        t_train = time.time() - t_epoch_start

        # Validation from rank 0
        if global_rank == 0:
            t_val_start = time.time()
            eval_res = evaluate(
                model=raw_model,
                loader=val_loader,
                device=device,
                prod_targets_map=prod_targets_map,
                prod_weights_map=prod_weights_map
            )
            t_val = time.time() - t_val_start

            d3_unw = eval_res["d3"]["unweighted_auc"]
            d3_wtd = eval_res["d3"]["weighted_auc"]
            d3_prauc = eval_res["d3"]["prauc"]
            prod_unw = eval_res["prod"]["unweighted_auc"]
            prod_prauc = eval_res["prod"]["prauc"]

            print(
                f"[{args.model_type.upper()} Fold {args.fold} | Epoch {epoch+1:02d}/{args.epochs:02d}] "
                f"Loss: {train_loss:.4f} | D3 AUC (Unw/Wtd): {d3_unw:.4f} / {d3_wtd:.4f} | "
                f"PR-AUC: {d3_prauc:.4f} | Prod AUC: {prod_unw:.4f} (PR: {prod_prauc:.4f}) | "
                f"Time: {t_train:.1f}s+{t_val:.1f}s",
                flush=True
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

            # Checkpoint on best Remediated D3 AUC
            if d3_unw > best_val_auc_d3:
                best_val_auc_d3 = d3_unw
                best_eval_dict = eval_res
                ckpt_path = os.path.join(fold_ckpt_dir, "best_model.pt")
                ckpt_payload = {
                    "epoch": epoch + 1,
                    "fold": args.fold,
                    "model_type": args.model_type,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc_d3": d3_unw,
                    "val_prauc_d3": d3_prauc,
                    "val_auc_prod": prod_unw,
                    "val_prauc_prod": prod_prauc,
                    "per_target_d3": eval_res["d3"]["per_target"],
                    "per_target_prauc_d3": eval_res["d3"]["per_target_prauc"],
                    "history": history,
                }
                save_ok = False
                for attempt in range(5):
                    tmp_ckpt = f"{ckpt_path}.tmp_{os.getpid()}_{attempt}_{int(time.time())}"
                    try:
                        buf = io.BytesIO()
                        torch.save(ckpt_payload, buf)
                        data = buf.getvalue()
                        with open(tmp_ckpt, "wb") as f:
                            f.write(data)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_ckpt, ckpt_path)
                        save_ok = True
                        break
                    except Exception as e:
                        print(f"  [WARN] Checkpoint save attempt {attempt+1} failed: {e}. Retrying...", flush=True)
                        if os.path.exists(tmp_ckpt):
                            try: os.remove(tmp_ckpt)
                            except OSError: pass
                        time.sleep(2)
                if not save_ok:
                    raise RuntimeError(f"Failed to atomically save checkpoint to {ckpt_path} after 5 attempts!")
                print(f"  >>> Best Checkpoint Saved: Remediated Macro AUC = {d3_unw:.4f} (PR-AUC: {d3_prauc:.4f})", flush=True)

        if is_distributed:
            dist.barrier()

    # Save final best validation predictions from rank 0
    if global_rank == 0 and len(best_eval_dict) > 0:
        pred_rows = []
        uids = best_eval_dict["uids"]
        preds = best_eval_dict["predictions"]
        targets_d3 = best_eval_dict["targets_d3"]
        weights_d3 = best_eval_dict["weights_d3"]

        for i, uid in enumerate(uids):
            row_dict = {"StudyInstanceUID": uid, "fold": args.fold}
            for c, t in enumerate(TARGETS):
                row_dict[f"pred_{t}"] = float(preds[i, c])
                row_dict[f"true_{t}"] = float(targets_d3[i, c])
                row_dict[f"weight_{t}"] = float(weights_d3[i, c])
            pred_rows.append(row_dict)

        pred_df = pd.DataFrame(pred_rows)
        val_csv_path = os.path.join(pred_val_dir, f"fold_{args.fold}_val_predictions.csv")
        tmp_csv = f"{val_csv_path}.tmp_{os.getpid()}"
        pred_df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, val_csv_path)

        metrics_json_path = os.path.join(reports_dir, f"fold_{args.fold}_metrics.json")
        tmp_json = f"{metrics_json_path}.tmp_{os.getpid()}"
        with open(tmp_json, "w") as f:
            json.dump({
                "model_type": args.model_type,
                "fold": args.fold,
                "best_d3_unweighted_auc": round(best_val_auc_d3, 4),
                "best_d3_prauc": round(best_eval_dict["d3"]["prauc"], 4),
                "best_prod_unweighted_auc": round(best_eval_dict["prod"]["unweighted_auc"], 4),
                "best_prod_prauc": round(best_eval_dict["prod"]["prauc"], 4),
                "per_target_d3": {k: round(v, 4) if v is not None else None for k, v in best_eval_dict["d3"]["per_target"].items()},
                "per_target_prauc_d3": {k: round(v, 4) if v is not None else None for k, v in best_eval_dict["d3"]["per_target_prauc"].items()},
                "history": history
            }, f, indent=2)
        os.replace(tmp_json, metrics_json_path)

        print(f"\n[{args.model_type.upper()} FOLD {args.fold} COMPLETE] Best Remediated AUC: {best_val_auc_d3:.4f} | Predictions: {val_csv_path}")

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
