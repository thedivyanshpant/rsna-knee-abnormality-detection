#!/usr/bin/env python3
"""
Phase C3.12 Stage A: Dedicated Study-Level Out-of-Fold (OOF) Inference Engine
RSNA Knee Abnormality Detection

Strict Provenance Rules:
- For each fold f in 0..4:
    Load frozen checkpoint: phase_c/c3/checkpoints/fold_{f}/best_model.pt
    Predict ONLY on held-out validation cohort for fold_{f} from cv_splits.json
- Strictly 0 Gold-58 leakage (asserted at dataset and inference levels)
- Continuous sigmoid probabilities sigma(z) in (0, 1) across all 12 targets
- Export per-fold predictions to: phase_c/c3/results/oof/fold_{f}_predictions.csv
- Consolidate to: phase_c/c3/results/oof_predictions_5folds.csv (4,349 rows x 13 columns)
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase_c", "c3", "scripts"))

from model import MultiSeriesKneeModel, TARGETS, NUM_SLOTS
from dataset import MultiSeriesKneeDataset, multi_series_collate_fn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_oof_inference():
    print("=" * 80)
    print("  PHASE C3.12 STAGE A: DEDICATED OOF INFERENCE")
    print("=" * 80)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'CPU'})")

    # Paths
    splits_path = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
    labels_path = os.path.join(REPO_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")
    meta_path = os.path.join(REPO_ROOT, "csv", "train_series.csv")
    gold_path = os.path.join(REPO_ROOT, "csv", "gold_studies.csv")
    processed_dir = os.path.join(REPO_ROOT, "phase_c", "c1", "processed")
    oof_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "results", "oof")
    os.makedirs(oof_dir, exist_ok=True)

    # 1. Load inputs
    with open(splits_path) as f:
        cv_data = json.load(f)

    gold_df = pd.read_csv(gold_path)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    labels_df = pd.read_csv(labels_path)
    meta_df = pd.read_csv(meta_path)

    all_fold_dfs = []
    total_start = time.time()

    for fold_idx in range(5):
        fold_key = f"fold_{fold_idx}"
        val_uids = cv_data["folds"][fold_key]["val_studies"]

        # Strict Gold-58 isolation check
        overlap = set(val_uids) & gold_uids
        assert len(overlap) == 0, f"CRITICAL LEAKAGE: Gold study in Fold {fold_idx} val: {overlap}"

        ckpt_path = os.path.join(REPO_ROOT, "phase_c", "c3", "checkpoints", f"fold_{fold_idx}", "best_model.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

        print(f"\n[Fold {fold_idx}] Held-out validation studies: {len(val_uids):,}")
        print(f"[Fold {fold_idx}] Loading checkpoint: {ckpt_path}")

        # Model instantiation
        model = MultiSeriesKneeModel(
            embed_dim=128,
            num_targets=len(TARGETS),
            num_slots=NUM_SLOTS,
            aggregation="masked_pool",
            dropout=0.0,
        ).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        val_ds = MultiSeriesKneeDataset(
            study_uids=val_uids,
            processed_dir=processed_dir,
            labels_df=labels_df,
            series_meta_df=meta_df,
            gold_uids=gold_uids,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=multi_series_collate_fn,
        )

        fold_preds = []
        fold_study_uids = []
        fold_start = time.time()

        with torch.no_grad():
            for batch in val_loader:
                vols = batch["volumes"].to(DEVICE, non_blocking=True)
                mask = batch["slot_mask"].to(DEVICE, non_blocking=True)
                uids = batch["study_uid"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(vols, mask)
                    probs = torch.sigmoid(logits)

                fold_preds.append(probs.float().cpu().numpy())
                fold_study_uids.extend(uids)

        fold_preds_np = np.concatenate(fold_preds, axis=0)  # (N_val, 12)
        fold_time = time.time() - fold_start
        throughput = len(val_uids) / max(fold_time, 0.01)

        print(f"[Fold {fold_idx} COMPLETE] Predicted {len(fold_study_uids):,} studies in {fold_time:.2f}s ({throughput:.1f} st/s)")

        # Create DataFrame for this fold
        fold_df = pd.DataFrame(fold_preds_np, columns=TARGETS)
        fold_df.insert(0, "StudyInstanceUID", fold_study_uids)
        fold_df.insert(1, "cv_fold", fold_idx)

        # Save per-fold predictions
        fold_out_path = os.path.join(oof_dir, f"fold_{fold_idx}_predictions.csv")
        fold_df.to_csv(fold_out_path, index=False)
        print(f"[Fold {fold_idx}] Saved per-fold predictions to: {fold_out_path}")

        all_fold_dfs.append(fold_df)

    total_time = time.time() - total_start
    print(f"\n{'='*80}\nAll 5 folds evaluated in {total_time:.2f}s.")

    # 2. Consolidate into canonical OOF prediction file
    consolidated_df = pd.concat(all_fold_dfs, ignore_index=True)
    
    # Primary canonical file as requested by Section 8:
    # Structure: StudyInstanceUID + 12 targets (4,349 rows x 13 columns)
    canonical_cols = ["StudyInstanceUID"] + TARGETS
    final_oof_df = consolidated_df[canonical_cols]

    canonical_out_path = os.path.join(REPO_ROOT, "phase_c", "c3", "results", "oof_predictions_5folds.csv")
    final_oof_df.to_csv(canonical_out_path, index=False)
    print(f"[SUCCESS] Consolidated OOF saved to: {canonical_out_path}")

    # Also save metadata-enriched version for provenance audit
    provenance_out_path = os.path.join(oof_dir, "oof_predictions_with_fold_provenance.csv")
    consolidated_df.to_csv(provenance_out_path, index=False)

    print(f"Final OOF Shape: {final_oof_df.shape} (Expected: 4349, 13)")
    return canonical_out_path

if __name__ == "__main__":
    run_oof_inference()
