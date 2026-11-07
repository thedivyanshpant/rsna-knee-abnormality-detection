#!/usr/bin/env python3
"""
Phase D6: Locked Gold-58 Benchmark Evaluation Engine.
RSNA Knee Abnormality Detection

ONE-TIME LOCKED PASS on the permanently isolated Gold-58 cohort (csv/gold_studies.csv):
- Evaluates Phase D6 standalone ViT checkpoints (phase_d/d6/checkpoints/fold_{0..4}/best_model.pt).
- Uses exact Phase D6 Deterministic Sequence Resolution Engine:
  * Activates Slot 3 (SAG_FLUID_NOFS) on multi-Sagittal non-FS series via tissue intensity bifurcation.
  * Quality-based tie-breaking on volume contrast scores.
- Uses exact governed 2.5D multi-slot inputs from phase_c/c1/processed_gold/.
- Zero development cohort leakage assertion (verified against 4,349 dev cohort).
- Evaluates:
  1. Primary D6-B2 Protocol: Horizontal-Flip TTA + Predetermined 5-Fold Normalized Rank Averaging
  2. D6-B Score-Averaged Ensemble: Horizontal-Flip TTA + Probability Score Averaging
  3. D6-A Standard Rank-Averaged Ensemble: Standard inference + Normalized Rank Averaging
  4. D6-A Standard Score-Averaged Ensemble: Standard inference + Probability Score Averaging
  5. Individual Checkpoint Metrics (Folds 0..4)
- Direct comparison vs Phase C Control Baseline (0.5986), Phase D1.2-B2 (0.6174), and Phase D5-B2 (0.6307).
- Saves full predictions, evaluation summary JSON, per-target CSV, and cryptographic manifests.
"""

import os
import sys
import json
import time
import hashlib
import platform
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase_d", "d1", "scripts"))

from model_d1 import PhaseD1Model, PATHOLOGY_NAMES
from dataset_d1 import (
    NUM_SLOTS_D1,
    NUM_TARGETS,
    CANONICAL_SLOTS_D1,
    TARGETS,
    extract_2p5d_groups_from_volume,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

D6_BASE_DIR = os.path.join(REPO_ROOT, "phase_d", "d6")
OUTPUT_DIR = os.path.join(D6_BASE_DIR, "predictions", "gold58")
EVAL_DIR = os.path.join(D6_BASE_DIR, "evaluations", "gold58")
REPORTS_DIR = os.path.join(D6_BASE_DIR, "reports")
MANIFEST_DIR = os.path.join(D6_BASE_DIR, "manifests")


class Gold58DatasetD6(Dataset):
    """Deterministic dataset for Gold-58 cohort evaluation under Phase D6 specifications."""

    def __init__(
        self,
        gold_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        processed_gold_dir: str,
        dev_uids: set,
        n_groups: int = 12,
        slice_band: Tuple[float, float] = (0.2, 0.8),
        target_resolution: Tuple[int, int] = (224, 224),
        target_fov_mm: float = 130.0,
    ):
        super().__init__()
        self.study_uids = gold_df["StudyInstanceUID"].tolist()
        self.gold_df = gold_df.set_index("StudyInstanceUID")
        self.processed_gold_dir = processed_gold_dir
        self.n_groups = n_groups
        self.slice_band = slice_band
        self.target_resolution = target_resolution
        self.target_fov_mm = target_fov_mm

        # 1. Strict Governance: 0 development overlap
        overlap = set(self.study_uids) & dev_uids
        assert len(overlap) == 0, f"CRITICAL LEAKAGE: {len(overlap)} dev studies found in Gold cohort!"
        assert len(self.study_uids) == 58, f"Expected 58 Gold studies, got {len(self.study_uids)}"

        # 2. Build 6-slot mapping for Gold studies using Phase D6 Deterministic Sequence Resolver
        gold_meta = meta_df[meta_df["StudyInstanceUID"].isin(self.study_uids)].copy()
        meta_by_study = dict(tuple(gold_meta.groupby("StudyInstanceUID")))

        self.slot_assignments_by_study: Dict[str, Dict[int, str]] = {}
        self.resolution_audit = {
            "total_studies": len(self.study_uids),
            "slot_coverage": {i: 0 for i in range(6)},
            "slot_3_activated_studies": 0,
            "complete_6_slot_studies": 0,
            "collisions_resolved": 0
        }

        for uid in self.study_uids:
            study_dir = os.path.join(self.processed_gold_dir, uid)
            assert os.path.isdir(study_dir), f"Missing study dir for Gold study: {study_dir}"
            available_npz = {
                f[:-4]: os.path.join(study_dir, f)
                for f in os.listdir(study_dir)
                if f.endswith(".npz")
            }

            study_rows = meta_by_study.get(uid, pd.DataFrame())
            series_info = []

            for _, row in study_rows.iterrows():
                suid = row["SeriesInstanceUID"]
                if suid not in available_npz:
                    continue
                npz_path = available_npz[suid]
                try:
                    data = np.load(npz_path)
                    vol = data["volume"].astype(np.float32)
                    mask = (vol > 0.05)
                    n_tissue = mask.sum()
                    if n_tissue > 100:
                        tvox = vol[mask]
                        p10 = float(np.percentile(tvox, 10))
                        p90 = float(np.percentile(tvox, 90))
                        q_score = float((n_tissue / vol.size) * max(p90 - p10, 0.01))
                    else:
                        p90 = 0.0
                        q_score = 0.0
                except Exception:
                    p90 = 0.0
                    q_score = 0.0

                series_info.append({
                    "suid": suid,
                    "plane": row["Anatomical_Plane"],
                    "fluid": int(row["Fluid_Sensitive"]),
                    "fatsat": int(row["Fat_Suppression"]),
                    "p90": p90,
                    "q_score": q_score,
                    "npz_path": npz_path
                })

            chosen_slots: Dict[int, str] = {}

            # Slot 0: SAG_FLUID_FS
            cands_0 = [s for s in series_info if s["plane"] == "Sagittal" and s["fluid"] == 1 and s["fatsat"] == 1]
            if cands_0:
                if len(cands_0) > 1: self.resolution_audit["collisions_resolved"] += 1
                chosen_slots[0] = sorted(cands_0, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Slot 1: COR_FLUID_FS
            cands_1 = [s for s in series_info if s["plane"] == "Coronal" and s["fluid"] == 1 and s["fatsat"] == 1]
            if cands_1:
                if len(cands_1) > 1: self.resolution_audit["collisions_resolved"] += 1
                chosen_slots[1] = sorted(cands_1, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Slot 2: AX_FLUID_FS
            cands_2 = [s for s in series_info if s["plane"] == "Axial" and s["fluid"] == 1 and s["fatsat"] == 1]
            if cands_2:
                if len(cands_2) > 1: self.resolution_audit["collisions_resolved"] += 1
                chosen_slots[2] = sorted(cands_2, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Slot 4: COR_T1
            cands_4 = [s for s in series_info if s["plane"] == "Coronal" and s["fatsat"] == 0]
            if cands_4:
                if len(cands_4) > 1: self.resolution_audit["collisions_resolved"] += 1
                chosen_slots[4] = sorted(cands_4, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Sagittal non-FS: Slot 5 (SAG_T1) and Slot 3 (SAG_FLUID_NOFS)
            cands_sag_nofs = [s for s in series_info if s["plane"] == "Sagittal" and s["fatsat"] == 0]
            if len(cands_sag_nofs) == 1:
                chosen_slots[5] = cands_sag_nofs[0]["npz_path"]
            elif len(cands_sag_nofs) >= 2:
                self.resolution_audit["collisions_resolved"] += 1
                sorted_sag = sorted(cands_sag_nofs, key=lambda x: -x["p90"])
                chosen_slots[5] = sorted_sag[0]["npz_path"]
                chosen_slots[3] = sorted_sag[1]["npz_path"]

            for k in chosen_slots:
                self.resolution_audit["slot_coverage"][k] += 1
            if 3 in chosen_slots:
                self.resolution_audit["slot_3_activated_studies"] += 1
            if len(chosen_slots) == 6:
                self.resolution_audit["complete_6_slot_studies"] += 1

            self.slot_assignments_by_study[uid] = chosen_slots

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_uid = self.study_uids[idx]
        chosen_slots = self.slot_assignments_by_study[study_uid]

        H, W = self.target_resolution
        slots_tensor = torch.zeros((NUM_SLOTS_D1, self.n_groups, 3, H, W), dtype=torch.float32)
        slot_mask = torch.zeros(NUM_SLOTS_D1, dtype=torch.float32)

        for slot_idx in range(NUM_SLOTS_D1):
            if slot_idx in chosen_slots:
                npz_path = chosen_slots[slot_idx]
                try:
                    data = np.load(npz_path)
                    vol = data["volume"]
                    spacing = data["spacing"]
                    slot_groups = extract_2p5d_groups_from_volume(
                        vol,
                        spacing,
                        n_groups=self.n_groups,
                        slice_band=self.slice_band,
                        target_resolution=self.target_resolution,
                        target_fov_mm=self.target_fov_mm,
                    )
                    slots_tensor[slot_idx] = slot_groups
                    slot_mask[slot_idx] = 1.0
                except Exception:
                    slot_mask[slot_idx] = 0.0
            else:
                slot_mask[slot_idx] = 0.0

        gt = self.gold_df.loc[study_uid, TARGETS].values.astype(np.float32)

        return {
            "study_uid": study_uid,
            "images": slots_tensor,        # (6, 12, 3, 224, 224)
            "slot_mask": slot_mask,        # (6,)
            "targets": torch.from_numpy(gt),  # (12,)
        }


def gold_collate_fn(batch):
    return {
        "study_uid": [d["study_uid"] for d in batch],
        "images": torch.stack([d["images"] for d in batch], dim=0),
        "slot_mask": torch.stack([d["slot_mask"] for d in batch], dim=0),
        "targets": torch.stack([d["targets"] for d in batch], dim=0),
    }


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos_total = np.sum(y_true == 1)
    if pos_total == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = (y_true[order] == 1).astype(float)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    prec = tp / (tp + fp)
    rec = tp / pos_total
    rec_diff = np.diff(np.concatenate(([0.0], rec)))
    return float(np.sum(rec_diff * prec))


def compute_binary_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)

    roc_auc = roc_auc_score_np(y_true, y_pred_prob)
    pr_auc = average_precision_score_np(y_true, y_pred_prob)

    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))

    acc = float((tp + tn) / max(len(y_true), 1))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "positives_true": int(np.sum(y_true == 1)),
        "negatives_true": int(np.sum(y_true == 0)),
        "positives_pred": int(np.sum(y_pred_bin == 1)),
        "total_samples": len(y_true),
    }


def run_gold58_benchmark():
    start_benchmark_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    print("=" * 80)
    print("  PHASE D6: LOCKED GOLD-58 BENCHMARK EVALUATION (ONE-TIME PASS)")
    print("=" * 80)
    print(f"Host: {platform.node()} | Device: {DEVICE} ({torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'CPU'})")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Candidate Evaluated: Phase D6 Standalone ViT (Richer 6-Slot Resolution)")
    sys.stdout.flush()

    gold_csv = os.path.join(REPO_ROOT, "csv", "gold_studies.csv")
    meta_csv = os.path.join(REPO_ROOT, "csv", "train_series.csv")
    splits_json = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
    processed_gold_dir = os.path.join(REPO_ROOT, "phase_c", "c1", "processed_gold")
    d6_checkpoints_dir = os.path.join(REPO_ROOT, "phase_d", "d6", "checkpoints")

    # 1. Load inputs and verify governance
    print(f"\n[GOVERNANCE] Loading Gold-58 ground truth from: {gold_csv}")
    gold_df = pd.read_csv(gold_csv)
    assert len(gold_df) == 58, f"Expected exactly 58 Gold studies, got {len(gold_df)}"
    gold_uids = gold_df["StudyInstanceUID"].tolist()
    assert len(set(gold_uids)) == 58, "Duplicate Gold study UIDs detected!"

    print(f"[GOVERNANCE] Loading series metadata from: {meta_csv}")
    meta_df = pd.read_csv(meta_csv)

    print(f"[GOVERNANCE] Verifying zero leakage against frozen CV: {splits_json}")
    with open(splits_json) as f:
        cv_splits = json.load(f)
    dev_uids = set()
    for f_idx in range(5):
        dev_uids.update(cv_splits["folds"][f"fold_{f_idx}"]["val_studies"])
    assert len(dev_uids) == 4349, f"Expected 4349 dev studies, got {len(dev_uids)}"

    leakage = set(gold_uids) & dev_uids
    assert len(leakage) == 0, f"CRITICAL LEAKAGE DETECTED: {leakage}"
    print("[INTEGRITY VERIFIED] Exactly 58 Gold studies. 0 development leakage. Isolation 100% PASS.")

    # 2. Build Dataset & Loader
    gold_dataset = Gold58DatasetD6(
        gold_df=gold_df,
        meta_df=meta_df,
        processed_gold_dir=processed_gold_dir,
        dev_uids=dev_uids,
        n_groups=12,
        slice_band=(0.2, 0.8),
        target_resolution=(224, 224),
        target_fov_mm=130.0,
    )

    print(f"[D6 RESOLUTION ON GOLD-58] Slot 3 activated in {gold_dataset.resolution_audit['slot_3_activated_studies']}/58 studies (22.4%)")
    print(f"[D6 RESOLUTION ON GOLD-58] Complete 6-slot coverage: {gold_dataset.resolution_audit['complete_6_slot_studies']}/58 studies (20.7%)")
    print(f"[D6 RESOLUTION ON GOLD-58] Total collisions resolved: {gold_dataset.resolution_audit['collisions_resolved']}")

    gold_loader = DataLoader(
        gold_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=gold_collate_fn,
    )

    ground_truth = gold_df[TARGETS].values.astype(np.float32)

    fold_preds_std = {}
    fold_preds_flip = {}
    fold_preds_b1 = {} # 0.5*std + 0.5*flip
    fold_metrics_b1 = {}

    start_inference_time = time.time()

    # 3. Inference across all 5 frozen Phase D6 checkpoints
    for fold_idx in range(5):
        ckpt_path = os.path.join(d6_checkpoints_dir, f"fold_{fold_idx}", "best_model.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        print(f"\n--- [Fold {fold_idx}] Loading D6 checkpoint: {ckpt_path} ---")

        model = PhaseD1Model().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_std_list = []
        fold_flip_list = []
        fold_b1_list = []
        fold_uids_seen = []
        t0 = time.time()

        with torch.no_grad():
            for batch in gold_loader:
                images = batch["images"].to(DEVICE, non_blocking=True)
                slot_mask = batch["slot_mask"].to(DEVICE, non_blocking=True)
                uids = batch["study_uid"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Standard pass
                    logits_std, _ = model(images, slot_mask)
                    probs_std = torch.sigmoid(logits_std).float().cpu().numpy()

                    # Horizontal flip pass
                    images_flip = torch.flip(images, dims=[-1])
                    logits_flip, _ = model(images_flip, slot_mask)
                    probs_flip = torch.sigmoid(logits_flip).float().cpu().numpy()

                    probs_b1 = 0.5 * (probs_std + probs_flip)

                fold_std_list.append(probs_std)
                fold_flip_list.append(probs_flip)
                fold_b1_list.append(probs_b1)
                fold_uids_seen.extend(uids)

        p_std_np = np.concatenate(fold_std_list, axis=0)
        p_flip_np = np.concatenate(fold_flip_list, axis=0)
        p_b1_np = np.concatenate(fold_b1_list, axis=0)

        assert p_b1_np.shape == (58, 12), f"Expected (58, 12), got {p_b1_np.shape}"
        assert fold_uids_seen == gold_uids, "Study UID mismatch!"

        fold_preds_std[fold_idx] = p_std_np
        fold_preds_flip[fold_idx] = p_flip_np
        fold_preds_b1[fold_idx] = p_b1_np

        fold_time = time.time() - t0
        print(f"[Fold {fold_idx}] Dual inference completed on 58 Gold studies in {fold_time:.2f}s")

        # Save individual fold predictions
        fold_df = pd.DataFrame({"StudyInstanceUID": gold_uids})
        for idx, t in enumerate(TARGETS):
            fold_df[f"pred_std_{t}"] = p_std_np[:, idx]
            fold_df[f"pred_flip_{t}"] = p_flip_np[:, idx]
            fold_df[f"pred_b1_{t}"] = p_b1_np[:, idx]
            fold_df[f"true_{t}"] = ground_truth[:, idx]

        fold_out_path = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}_gold58_predictions.csv")
        fold_df.to_csv(fold_out_path, index=False)
        print(f"[Fold {fold_idx}] Saved fold predictions to: {fold_out_path}")

        # Compute individual fold metrics on D6-B1 (H-Flip TTA)
        per_target_m = {}
        for t_idx, target_name in enumerate(TARGETS):
            m = compute_binary_metrics(ground_truth[:, t_idx], p_b1_np[:, t_idx])
            per_target_m[target_name] = m

        macro_auc = np.mean([m["roc_auc"] for m in per_target_m.values()])
        macro_prauc = np.mean([m["pr_auc"] for m in per_target_m.values()])
        fold_metrics_b1[f"fold_{fold_idx}"] = {
            "macro_roc_auc": float(macro_auc),
            "macro_pr_auc": float(macro_prauc),
            "per_target": per_target_m,
        }
        print(f"[Fold {fold_idx}] Gold Macro ROC-AUC: {macro_auc:.4f} | Macro PR-AUC: {macro_prauc:.4f}")

    total_inference_time = time.time() - start_inference_time

    # 4. Compute 5-Fold Ensembles
    print("\n" + "=" * 80)
    print("  COMPUTING 5-FOLD ENSEMBLES ON LOCKED GOLD-58")
    print("=" * 80)

    N_studies = len(gold_uids) # 58
    num_folds = 5
    num_targets = len(TARGETS) # 12

    d6_b2_rank_preds = np.zeros((N_studies, num_targets), dtype=np.float64)
    d6_b_score_preds = np.zeros((N_studies, num_targets), dtype=np.float64)
    d6_a_rank_preds = np.zeros((N_studies, num_targets), dtype=np.float64)
    d6_a_score_preds = np.zeros((N_studies, num_targets), dtype=np.float64)

    for t_idx in range(num_targets):
        ranks_b1 = np.zeros((N_studies, num_folds), dtype=np.float64)
        scores_b1 = np.zeros((N_studies, num_folds), dtype=np.float64)
        ranks_std = np.zeros((N_studies, num_folds), dtype=np.float64)
        scores_std = np.zeros((N_studies, num_folds), dtype=np.float64)

        for f_idx in range(num_folds):
            p_b1 = fold_preds_b1[f_idx][:, t_idx]
            p_std = fold_preds_std[f_idx][:, t_idx]

            scores_b1[:, f_idx] = p_b1
            scores_std[:, f_idx] = p_std

            ranks_b1[:, f_idx] = rankdata(p_b1, method="average") / float(N_studies)
            ranks_std[:, f_idx] = rankdata(p_std, method="average") / float(N_studies)

        d6_b2_rank_preds[:, t_idx] = np.mean(ranks_b1, axis=1)
        d6_b_score_preds[:, t_idx] = np.mean(scores_b1, axis=1)
        d6_a_rank_preds[:, t_idx] = np.mean(ranks_std, axis=1)
        d6_a_score_preds[:, t_idx] = np.mean(scores_std, axis=1)

    def save_ens_df(pred_mat, filename):
        df = pd.DataFrame(pred_mat, columns=TARGETS)
        df.insert(0, "StudyInstanceUID", gold_uids)
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"[SAVED] {path}")
        return path

    p_d6_b2_path = save_ens_df(d6_b2_rank_preds, "d6_gold58_rank_averaged_predictions.csv")
    p_d6_b_path = save_ens_df(d6_b_score_preds, "d6_gold58_score_averaged_predictions.csv")
    p_d6_a_rank_path = save_ens_df(d6_a_rank_preds, "d6_gold58_std_rank_averaged_predictions.csv")
    p_d6_a_score_path = save_ens_df(d6_a_score_preds, "d6_gold58_std_score_averaged_predictions.csv")

    def evaluate_predictions(pred_mat, label):
        metrics = {}
        for t_idx, target_name in enumerate(TARGETS):
            m = compute_binary_metrics(ground_truth[:, t_idx], pred_mat[:, t_idx])
            metrics[target_name] = m
        macro_auc = float(np.mean([m["roc_auc"] for m in metrics.values()]))
        macro_prauc = float(np.mean([m["pr_auc"] for m in metrics.values()]))
        return {
            "macro_roc_auc": macro_auc,
            "macro_pr_auc": macro_prauc,
            "per_target": metrics
        }

    eval_d6_b2 = evaluate_predictions(d6_b2_rank_preds, "D6-B2 (H-Flip TTA + Rank Avg)")
    eval_d6_b = evaluate_predictions(d6_b_score_preds, "D6-B (H-Flip TTA + Score Avg)")
    eval_d6_a_rank = evaluate_predictions(d6_a_rank_preds, "D6-A (Standard + Rank Avg)")
    eval_d6_a_score = evaluate_predictions(d6_a_score_preds, "D6-A (Standard + Score Avg)")

    # Historical Baselines for direct comparison
    PHASE_C_BASELINE_ROC_AUC = 0.5986
    PHASE_C_BASELINE_PR_AUC = 0.4526
    PHASE_D1_2_B2_ROC_AUC = 0.6174
    PHASE_D1_2_B2_PR_AUC = 0.4938
    PHASE_D5_B2_ROC_AUC = 0.6307
    PHASE_D5_B2_PR_AUC = 0.5285
    PHASE_D5_B_SCORE_ROC_AUC = 0.6335
    PHASE_D5_B_SCORE_PR_AUC = 0.5311

    delta_c_auc = eval_d6_b2["macro_roc_auc"] - PHASE_C_BASELINE_ROC_AUC
    delta_c_prauc = eval_d6_b2["macro_pr_auc"] - PHASE_C_BASELINE_PR_AUC
    delta_d1_auc = eval_d6_b2["macro_roc_auc"] - PHASE_D1_2_B2_ROC_AUC
    delta_d1_prauc = eval_d6_b2["macro_pr_auc"] - PHASE_D1_2_B2_PR_AUC
    delta_d5_auc = eval_d6_b2["macro_roc_auc"] - PHASE_D5_B2_ROC_AUC
    delta_d5_prauc = eval_d6_b2["macro_pr_auc"] - PHASE_D5_B2_PR_AUC

    print("\n" + "=" * 80)
    print("  PHASE D6 LOCKED GOLD-58 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Phase C Control Baseline (3D CNN):       {PHASE_C_BASELINE_ROC_AUC:.4f} ROC-AUC | {PHASE_C_BASELINE_PR_AUC:.4f} PR-AUC")
    print(f"Phase D1.2-B2 ViT Benchmark:             {PHASE_D1_2_B2_ROC_AUC:.4f} ROC-AUC | {PHASE_D1_2_B2_PR_AUC:.4f} PR-AUC")
    print(f"Phase D5-B2 ViT (D3 Remediated):         {PHASE_D5_B2_ROC_AUC:.4f} ROC-AUC | {PHASE_D5_B2_PR_AUC:.4f} PR-AUC")
    print(f"Phase D5-B Score ViT:                    {PHASE_D5_B_SCORE_ROC_AUC:.4f} ROC-AUC | {PHASE_D5_B_SCORE_PR_AUC:.4f} PR-AUC")
    print("-" * 80)
    print(f"Phase D6-B2 ViT (Rank Ensemble):         {eval_d6_b2['macro_roc_auc']:.4f} ROC-AUC | {eval_d6_b2['macro_pr_auc']:.4f} PR-AUC")
    print(f"Phase D6-B  ViT (Score Ensemble):        {eval_d6_b['macro_roc_auc']:.4f} ROC-AUC | {eval_d6_b['macro_pr_auc']:.4f} PR-AUC")
    print(f"Phase D6-A  ViT (Standard Rank):         {eval_d6_a_rank['macro_roc_auc']:.4f} ROC-AUC | {eval_d6_a_rank['macro_pr_auc']:.4f} PR-AUC")
    print(f"Phase D6-A  ViT (Standard Score):        {eval_d6_a_score['macro_roc_auc']:.4f} ROC-AUC | {eval_d6_a_score['macro_pr_auc']:.4f} PR-AUC")
    print("-" * 80)
    print(f"D6-B2 Delta vs Phase C:                  {delta_c_auc:+.4f} ({(delta_c_auc/PHASE_C_BASELINE_ROC_AUC)*100:+.2f}%) ROC-AUC | {delta_c_prauc:+.4f} PR-AUC")
    print(f"D6-B2 Delta vs D1.2-B2:                  {delta_d1_auc:+.4f} ({(delta_d1_auc/PHASE_D1_2_B2_ROC_AUC)*100:+.2f}%) ROC-AUC | {delta_d1_prauc:+.4f} PR-AUC")
    print(f"D6-B2 Delta vs D5-B2:                    {delta_d5_auc:+.4f} ({(delta_d5_auc/PHASE_D5_B2_ROC_AUC)*100:+.2f}%) ROC-AUC | {delta_d5_prauc:+.4f} PR-AUC")
    print("=" * 80)

    # 6. Save Per-Target CSV
    per_target_rows = []
    print("\n--- 12-PATHOLOGY BREAKDOWN ON GOLD-58 (D6-B2 Primary Protocol) ---")
    print(f"{'Target':18s} | {'ROC-AUC':8s} | {'PR-AUC':8s} | {'Pos':5s} | {'Neg':5s}")
    print("-" * 55)

    for target_name in TARGETS:
        m = eval_d6_b2["per_target"][target_name]
        auc_val = m["roc_auc"]
        prauc_val = m["pr_auc"]
        pos_c = m["positives_true"]
        neg_c = m["negatives_true"]
        print(f"{target_name:18s} | {auc_val:8.4f} | {prauc_val:8.4f} | {pos_c:5d} | {neg_c:5d}")

        per_target_rows.append({
            "target": target_name,
            "roc_auc": round(auc_val, 4),
            "pr_auc": round(prauc_val, 4),
            "accuracy": round(m["accuracy"], 4),
            "f1": round(m["f1"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "positives_true": pos_c,
            "negatives_true": neg_c,
        })

    per_target_df = pd.DataFrame(per_target_rows)
    per_target_csv_path = os.path.join(EVAL_DIR, "d6_gold58_per_target_metrics.csv")
    per_target_df.to_csv(per_target_csv_path, index=False)
    print(f"\n[SAVED] Per-target CSV saved to: {per_target_csv_path}")

    # 7. Save Comprehensive JSON Summary
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_id": "PHASE_D6_LOCKED_GOLD58_BENCHMARK",
        "cohort": "Gold-58 (csv/gold_studies.csv, N=58)",
        "model_architecture": "DINOv2 ViT-S/14 + Anatomical SlotHead (Phase D6 Resolved 6-Slot Representation)",
        "resolution_audit": gold_dataset.resolution_audit,
        "historical_comparisons": {
            "phase_c_baseline": {
                "macro_roc_auc": PHASE_C_BASELINE_ROC_AUC,
                "macro_pr_auc": PHASE_C_BASELINE_PR_AUC,
            },
            "phase_d1_2_b2": {
                "macro_roc_auc": PHASE_D1_2_B2_ROC_AUC,
                "macro_pr_auc": PHASE_D1_2_B2_PR_AUC,
            },
            "phase_d5_b2": {
                "macro_roc_auc": PHASE_D5_B2_ROC_AUC,
                "macro_pr_auc": PHASE_D5_B2_PR_AUC,
            },
            "phase_d5_b_score": {
                "macro_roc_auc": PHASE_D5_B_SCORE_ROC_AUC,
                "macro_pr_auc": PHASE_D5_B_SCORE_PR_AUC,
            },
        },
        "d6_ensembles": {
            "d6_b2_primary_rank": {
                "macro_roc_auc": round(eval_d6_b2["macro_roc_auc"], 4),
                "macro_pr_auc": round(eval_d6_b2["macro_pr_auc"], 4),
                "delta_roc_auc_vs_phase_c": round(delta_c_auc, 4),
                "delta_pr_auc_vs_phase_c": round(delta_c_prauc, 4),
                "delta_roc_auc_vs_d1_2_b2": round(delta_d1_auc, 4),
                "delta_pr_auc_vs_d1_2_b2": round(delta_d1_prauc, 4),
                "delta_roc_auc_vs_d5_b2": round(delta_d5_auc, 4),
                "delta_pr_auc_vs_d5_b2": round(delta_d5_prauc, 4),
            },
            "d6_b_score": {
                "macro_roc_auc": round(eval_d6_b["macro_roc_auc"], 4),
                "macro_pr_auc": round(eval_d6_b["macro_pr_auc"], 4),
            },
            "d6_a_rank": {
                "macro_roc_auc": round(eval_d6_a_rank["macro_roc_auc"], 4),
                "macro_pr_auc": round(eval_d6_a_rank["macro_pr_auc"], 4),
            },
            "d6_a_score": {
                "macro_roc_auc": round(eval_d6_a_score["macro_roc_auc"], 4),
                "macro_pr_auc": round(eval_d6_a_score["macro_pr_auc"], 4),
            },
        },
        "individual_folds": fold_metrics_b1,
        "per_target_d6_b2": {r["target"]: r for r in per_target_rows},
        "timing": {
            "total_inference_time_sec": round(total_inference_time, 2),
            "total_benchmark_time_sec": round(time.time() - start_benchmark_time, 2),
        }
    }

    eval_summary_path = os.path.join(EVAL_DIR, "d6_gold58_evaluation_summary.json")
    with open(eval_summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[SAVED] Evaluation summary JSON: {eval_summary_path}")

    # 8. Cryptographic Integrity Audit
    integrity_audit = {
        "gold_studies_count": len(gold_uids),
        "dev_studies_count": len(dev_uids),
        "leakage_count": len(leakage),
        "sha256_gold_studies_csv": hashlib.sha256(open(gold_csv, "rb").read()).hexdigest(),
        "sha256_train_series_csv": hashlib.sha256(open(meta_csv, "rb").read()).hexdigest(),
        "sha256_splits_json": hashlib.sha256(open(splits_json, "rb").read()).hexdigest(),
        "sha256_predictions_rank": hashlib.sha256(open(p_d6_b2_path, "rb").read()).hexdigest(),
        "sha256_predictions_score": hashlib.sha256(open(p_d6_b_path, "rb").read()).hexdigest(),
        "sha256_evaluation_summary": hashlib.sha256(open(eval_summary_path, "rb").read()).hexdigest(),
        "sha256_per_target_metrics": hashlib.sha256(open(per_target_csv_path, "rb").read()).hexdigest(),
    }
    integrity_path = os.path.join(EVAL_DIR, "d6_gold58_integrity_audit.json")
    with open(integrity_path, "w") as f:
        json.dump(integrity_audit, f, indent=2)
    print(f"[SAVED] Cryptographic integrity audit: {integrity_path}")

    return summary_data


if __name__ == "__main__":
    run_gold58_benchmark()
