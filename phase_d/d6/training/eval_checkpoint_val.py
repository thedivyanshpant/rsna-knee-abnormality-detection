#!/usr/bin/env python3
"""
Evaluate saved Phase D6 best_model.pt on its CV validation split and export predictions CSV and metrics JSON.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_d.d1.scripts.model_d1 import PhaseD1Model, PATHOLOGY_NAMES
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset, TARGETS
from phase_d.d6.training.train_ddp_d6 import evaluate, compute_macro_roc_auc

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
PROD_LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
SLOT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d6/resolver/d6_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d6/checkpoints")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d6/predictions/val")
METRICS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d6/reports")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")


def eval_fold(fold: int, device_id: int = 0):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Evaluating Fold {fold} on device {device}...")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"fold_{fold}", "best_model.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"[CHECKPOINT] Loaded fold {fold} best_model.pt (Epoch {checkpoint['epoch']}, Stored AUC: {checkpoint['val_auc_d3']:.4f})")

    with open(CONFIG_CV) as f:
        cv_splits = json.load(f)
    val_uids = cv_splits["folds"][f"fold_{fold}"]["val_studies"]

    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    assert len(set(val_uids) & gold_uids) == 0, "CRITICAL LEAKAGE: Gold in val set!"

    slot_df = pd.read_csv(SLOT_MANIFEST_CSV)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    prod_labels_df = pd.read_csv(PROD_LABELS_CSV)

    prod_targets_map = {}
    prod_weights_map = {}
    for uid, grp in prod_labels_df.groupby("StudyInstanceUID"):
        grp_indexed = grp.set_index("target")
        t_arr = np.array([grp_indexed.loc[t, "label_probability"] if t in grp_indexed.index else 0.20 for t in TARGETS], dtype=np.float32)
        w_arr = np.array([grp_indexed.loc[t, "sample_weight"] if t in grp_indexed.index else 0.85 for t in TARGETS], dtype=np.float32)
        prod_targets_map[uid] = t_arr
        prod_weights_map[uid] = w_arr

    val_dataset = PhaseD1KneeDataset(
        study_uids=val_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
        n_groups=12,
        slice_band=(0.2, 0.8),
        target_resolution=(224, 224),
        target_fov_mm=130.0,
        transform=None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = PhaseD1Model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_res = evaluate(model, val_loader, device, prod_targets_map, prod_weights_map)

    d3_unw = eval_res["d3"]["unweighted_auc"]
    d3_prauc = eval_res["d3"]["prauc"]
    prod_unw = eval_res["prod"]["unweighted_auc"]
    prod_prauc = eval_res["prod"]["prauc"]

    print(f"[RESULT Fold {fold}] Remediated Macro AUC: {d3_unw:.4f} | PR-AUC: {d3_prauc:.4f} | Prod AUC: {prod_unw:.4f}")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    pred_rows = []
    uids = eval_res["uids"]
    preds = eval_res["predictions"]
    targets_d3 = eval_res["targets_d3"]
    weights_d3 = eval_res["weights_d3"]

    for i, uid in enumerate(uids):
        row_dict = {"StudyInstanceUID": uid, "fold": fold}
        for c, t in enumerate(TARGETS):
            row_dict[f"pred_{t}"] = float(preds[i, c])
            row_dict[f"true_{t}"] = float(targets_d3[i, c])
            row_dict[f"weight_{t}"] = float(weights_d3[i, c])
        pred_rows.append(row_dict)

    pred_df = pd.DataFrame(pred_rows)
    val_csv_path = os.path.join(PREDICTIONS_DIR, f"fold_{fold}_val_predictions.csv")
    pred_df.to_csv(val_csv_path, index=False)

    metrics_json_path = os.path.join(METRICS_DIR, f"fold_{fold}_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump({
            "fold": fold,
            "best_epoch": checkpoint["epoch"],
            "best_d3_unweighted_auc": round(d3_unw, 4),
            "best_d3_prauc": round(d3_prauc, 4),
            "best_prod_unweighted_auc": round(prod_unw, 4),
            "best_prod_prauc": round(prod_prauc, 4),
            "per_target_d3": {k: round(v, 4) if v is not None else None for k, v in eval_res["d3"]["per_target"].items()},
            "per_target_prauc_d3": {k: round(v, 4) if v is not None else None for k, v in eval_res["d3"]["per_target_prauc"].items()},
            "history": checkpoint.get("history", [])
        }, f, indent=2)

    print(f"[SAVED] Saved predictions: {val_csv_path} ({len(pred_df)} rows)")
    print(f"[SAVED] Saved metrics:     {metrics_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    eval_fold(args.fold, args.device_id)
