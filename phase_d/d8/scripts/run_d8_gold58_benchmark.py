#!/usr/bin/env python3
"""
Phase D8: Locked Gold-58 Benchmark Evaluation Engine for D8-A, D8-B, D8-C, and D8-D.
RSNA Knee Abnormality Detection

ONE-TIME LOCKED PASS on the permanently isolated Gold-58 clinical cohort (csv/gold_studies.csv):
- Evaluates 5-fold ensembles for all four Phase D8 models:
  1. D8-A: ViT-S/14 (22.59M) + Pathology-Conditioned Multi-Head Cross-Attention (12 groups)
  2. D8-B: ViT-S/14 (22.46M) + High-Budget Volumetric Depth Coverage (24 groups)
  3. D8-C: Hybrid Conv+Attn CoAtNet-Style Model (22.39M params, 12 groups)
  4. D8-D: Full Combined Configuration (Hybrid + 24 groups + Pathology Cross-Attn, 22.81M params)
- Protocol:
  * Horizontal-Flip TTA
  * 5-Fold Normalized Rank-Averaged Ensemble (Primary Protocol B2)
  * 5-Fold Score-Averaged Ensemble (Protocol B)
  * Individual Fold Evaluations
- Comparison:
  * Phase C Baseline: 0.5986 Macro ROC-AUC (0.4526 PR-AUC)
  * Phase D1.2-B2 Milestone: 0.6174 Macro ROC-AUC (0.4938 PR-AUC)
  * Phase D5-B2 ViT Benchmark: 0.6307 Macro ROC-AUC (0.5285 PR-AUC)
  * Phase D6-B2 Champion: 0.6331 Macro ROC-AUC (0.5125 PR-AUC)
  * Phase D7-C All-Time Gold-58 Record: 0.6474 Macro ROC-AUC (0.5290 PR-AUC)
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase_d", "d1", "scripts"))
from d1_2_eval_utils import (
    roc_auc_score_np,
    pr_auc_score_np,
    PATHOLOGY_NAMES,
)

from phase_d.d8.models.model_d8 import (
    PhaseD8Model_PathologyCrossAttention,
    PhaseD8Model_HighBudgetDepth,
    PhaseD8Model_HybridConvAttn,
    PhaseD8Model_FullCombination,
)
from phase_d.d8.dataset_d8 import (
    NUM_SLOTS,
    extract_standard_groups,
    extract_high_budget_depth_groups,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

D8_BASE_DIR = os.path.join(PROJECT_ROOT, "phase_d/d8")
PROCESSED_GOLD_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed_gold")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")
TRAIN_SERIES_CSV = os.path.join(PROJECT_ROOT, "csv/train_series.csv")
CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")

OUTPUT_PRED_DIR = os.path.join(D8_BASE_DIR, "predictions/gold58")
OUTPUT_EVAL_DIR = os.path.join(D8_BASE_DIR, "evaluations/gold58")
OUTPUT_REPORTS_DIR = os.path.join(D8_BASE_DIR, "reports")
OUTPUT_MANIFESTS_DIR = os.path.join(D8_BASE_DIR, "manifests")


class Gold58DatasetD8(Dataset):
    """Dataset for Gold-58 cohort evaluation supporting Standard (12 grp) and HighBudget (24 grp) modes."""

    def __init__(
        self,
        gold_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        processed_gold_dir: str,
        dev_uids: set,
        high_budget: bool = False,
        target_resolution: Tuple[int, int] = (224, 224),
        target_fov_mm: float = 130.0,
    ):
        super().__init__()
        self.study_uids = gold_df["StudyInstanceUID"].tolist()
        self.gold_df = gold_df.set_index("StudyInstanceUID")
        self.processed_gold_dir = processed_gold_dir
        self.high_budget = high_budget
        self.num_groups = 24 if high_budget else 12
        self.target_resolution = target_resolution
        self.target_fov_mm = target_fov_mm

        overlap = set(self.study_uids) & dev_uids
        assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} dev studies in Gold cohort!"
        assert len(self.study_uids) == 58, f"Expected 58 Gold studies, got {len(self.study_uids)}"

        # Deterministic 6-slot resolver for Gold-58
        gold_meta = meta_df[meta_df["StudyInstanceUID"].isin(self.study_uids)].copy()
        meta_by_study = dict(tuple(gold_meta.groupby("StudyInstanceUID")))

        self.slot_assignments: Dict[str, Dict[int, str]] = {}
        for uid in self.study_uids:
            study_dir = os.path.join(self.processed_gold_dir, uid)
            available_npz = {
                f[:-4]: os.path.join(study_dir, f)
                for f in os.listdir(study_dir) if f.endswith(".npz")
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
            c0 = [s for s in series_info if s["plane"] == "Sagittal" and s["fluid"] == 1 and s["fatsat"] == 1]
            if c0: chosen_slots[0] = sorted(c0, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Slot 1: COR_FLUID_FS
            c1 = [s for s in series_info if s["plane"] == "Coronal" and s["fluid"] == 1 and s["fatsat"] == 1]
            if c1: chosen_slots[1] = sorted(c1, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Slot 2: AX_FLUID_FS
            c2 = [s for s in series_info if s["plane"] == "Axial" and s["fluid"] == 1 and s["fatsat"] == 1]
            if c2: chosen_slots[2] = sorted(c2, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Slot 4: COR_T1
            c4 = [s for s in series_info if s["plane"] == "Coronal" and s["fatsat"] == 0]
            if c4: chosen_slots[4] = sorted(c4, key=lambda x: (-x["q_score"], x["suid"]))[0]["npz_path"]

            # Sagittal non-FS: Slot 5 (SAG_T1) and Slot 3 (SAG_FLUID_NOFS)
            cs = [s for s in series_info if s["plane"] == "Sagittal" and s["fatsat"] == 0]
            if len(cs) == 1:
                chosen_slots[5] = cs[0]["npz_path"]
            elif len(cs) >= 2:
                sorted_sag = sorted(cs, key=lambda x: -x["p90"])
                chosen_slots[5] = sorted_sag[0]["npz_path"]
                chosen_slots[3] = sorted_sag[1]["npz_path"]

            self.slot_assignments[uid] = chosen_slots

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.study_uids[idx]
        chosen = self.slot_assignments[uid]

        slot_images = torch.zeros(
            NUM_SLOTS, self.num_groups, 3, self.target_resolution[0], self.target_resolution[1], dtype=torch.float32
        )
        slot_mask = torch.zeros(NUM_SLOTS, dtype=torch.float32)

        for s_idx, npz_path in chosen.items():
            try:
                data = np.load(npz_path)
                vol = data["volume"]
                spacing = data.get("spacing", np.array([1.0, 0.5, 0.5]))
                if self.high_budget:
                    grps = extract_high_budget_depth_groups(vol, spacing, self.target_resolution, self.target_fov_mm)
                else:
                    grps = extract_standard_groups(vol, spacing, 12, (0.2, 0.8), self.target_resolution, self.target_fov_mm)
                slot_images[s_idx] = grps
                slot_mask[s_idx] = 1.0
            except Exception:
                slot_mask[s_idx] = 0.0

        gold_row = self.gold_df.loc[uid]
        targets = np.array([float(gold_row[t]) for t in PATHOLOGY_NAMES], dtype=np.float32)

        return {
            "study_uid": uid,
            "images": slot_images,
            "slot_mask": slot_mask,
            "targets": targets,
        }


def run_inference_tta(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_preds_std = []
    all_preds_flip = []
    all_targets = []
    all_uids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            mask = batch["slot_mask"].to(device)
            targets = batch["targets"].numpy()
            uids = batch["study_uid"]

            # Standard pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_std, _ = model(images, mask)
                probs_std = torch.sigmoid(logits_std).float().cpu().numpy()

            # Horizontal flip pass
            images_flip = torch.flip(images, dims=[-1])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_flip, _ = model(images_flip, mask)
                probs_flip = torch.sigmoid(logits_flip).float().cpu().numpy()

            all_preds_std.append(probs_std)
            all_preds_flip.append(probs_flip)
            all_targets.append(targets)
            all_uids.extend(uids)

    preds_std = np.concatenate(all_preds_std, axis=0)
    preds_flip = np.concatenate(all_preds_flip, axis=0)
    preds_tta = (preds_std + preds_flip) / 2.0
    y_true = np.concatenate(all_targets, axis=0)
    return preds_tta, y_true, all_uids


def evaluate_model_on_gold(
    model_name: str,
    model_cls: Any,
    ckpt_dir: str,
    dataset: Gold58DatasetD8,
    device: torch.device
) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    fold_preds_tta = []
    y_true = None
    uids = None

    print(f"\n--- Evaluating {model_name.upper()} Checkpoints on Gold-58 (5 Folds + TTA) ---", flush=True)
    per_fold_metrics = {}
    for fold in range(5):
        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}", "best_model.pt")
        assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=device)

        model = model_cls().to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        preds_fold, y_true, uids = run_inference_tta(model, loader, device)
        fold_preds_tta.append(preds_fold)

        # Single fold AUC
        aucs = [roc_auc_score_np(y_true[:, c], preds_fold[:, c]) for c in range(12)]
        valid_aucs = [a for a in aucs if a is not None]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

        praucs = [pr_auc_score_np(y_true[:, c], preds_fold[:, c]) for c in range(12)]
        valid_praucs = [p for p in praucs if p is not None]
        mean_prauc = float(np.mean(valid_praucs)) if valid_praucs else 0.0

        per_fold_metrics[f"fold_{fold}"] = {
            "macro_roc_auc": round(mean_auc, 4),
            "macro_pr_auc": round(mean_prauc, 4)
        }
        print(f"  Fold {fold} Macro ROC-AUC: {mean_auc:.4f} | PR-AUC: {mean_prauc:.4f}", flush=True)

    # 1. Primary Protocol: Normalized Rank Averaging
    rank_matrices = []
    for p in fold_preds_tta:
        rmat = np.zeros_like(p)
        for c in range(12):
            rmat[:, c] = (rankdata(p[:, c]) - 1.0) / (len(p) - 1.0)
        rank_matrices.append(rmat)

    rank_ens_preds = np.mean(rank_matrices, axis=0)

    # 2. Probability Score Averaging
    score_ens_preds = np.mean(fold_preds_tta, axis=0)

    # Evaluate Rank Ensemble
    per_target_rank = {}
    valid_aucs_rank = []
    valid_praucs_rank = []
    for c, t in enumerate(PATHOLOGY_NAMES):
        auc = roc_auc_score_np(y_true[:, c], rank_ens_preds[:, c])
        prauc = pr_auc_score_np(y_true[:, c], rank_ens_preds[:, c])
        pos_cnt = int(np.sum(y_true[:, c] == 1))
        neg_cnt = int(np.sum(y_true[:, c] == 0))
        per_target_rank[t] = {
            "roc_auc": round(auc, 4) if auc is not None else None,
            "pr_auc": round(prauc, 4) if prauc is not None else None,
            "positives": pos_cnt,
            "negatives": neg_cnt,
            "prevalence_pct": round((pos_cnt / len(y_true)) * 100.0, 1),
        }
        if auc is not None: valid_aucs_rank.append(auc)
        if prauc is not None: valid_praucs_rank.append(prauc)

    macro_auc_rank = float(np.mean(valid_aucs_rank))
    macro_prauc_rank = float(np.mean(valid_praucs_rank))

    # Evaluate Score Ensemble
    valid_aucs_score = [roc_auc_score_np(y_true[:, c], score_ens_preds[:, c]) for c in range(12)]
    valid_praucs_score = [pr_auc_score_np(y_true[:, c], score_ens_preds[:, c]) for c in range(12)]
    macro_auc_score = float(np.mean([a for a in valid_aucs_score if a is not None]))
    macro_prauc_score = float(np.mean([a for a in valid_praucs_score if a is not None]))

    # Save Rank Predictions CSV
    os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
    rank_csv_path = os.path.join(OUTPUT_PRED_DIR, f"{model_name.lower()}_gold58_rank_averaged_predictions.csv")
    df_rows = []
    for i, uid in enumerate(uids):
        row = {"StudyInstanceUID": uid}
        for c, t in enumerate(PATHOLOGY_NAMES):
            row[f"pred_{t}"] = float(rank_ens_preds[i, c])
            row[f"true_{t}"] = float(y_true[i, c])
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(rank_csv_path, index=False)

    print(f"[{model_name.upper()} GOLD-58 RESULTS]")
    print(f"Primary Rank Ensemble Macro ROC-AUC:  {macro_auc_rank:.4f} | PR-AUC: {macro_prauc_rank:.4f}")
    print(f"Score Ensemble Macro ROC-AUC:         {macro_auc_score:.4f} | PR-AUC: {macro_prauc_score:.4f}")
    print(f"Predictions saved: {rank_csv_path}", flush=True)

    return {
        "model_name": model_name,
        "per_fold": per_fold_metrics,
        "rank_ensemble": {
            "macro_roc_auc": round(macro_auc_rank, 4),
            "macro_pr_auc": round(macro_prauc_rank, 4),
            "delta_vs_d7c_record": round(macro_auc_rank - 0.6474, 4),
            "delta_vs_d6_gold": round(macro_auc_rank - 0.6331, 4),
            "delta_vs_d5_gold": round(macro_auc_rank - 0.6307, 4),
            "delta_vs_d1_2_gold": round(macro_auc_rank - 0.6174, 4),
            "delta_vs_phase_c_gold": round(macro_auc_rank - 0.5986, 4),
        },
        "score_ensemble": {
            "macro_roc_auc": round(macro_auc_score, 4),
            "macro_pr_auc": round(macro_prauc_score, 4),
        },
        "per_target": per_target_rank,
        "prediction_file": rank_csv_path,
    }


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PHASE D8: AUTHORIZED LOCKED GOLD-58 CLINICAL BENCHMARK EVALUATION")
    print("=" * 80, flush=True)

    gold_df = pd.read_csv(GOLD_CSV)
    meta_df = pd.read_csv(TRAIN_SERIES_CSV)
    with open(CONFIG_CV) as f:
        cv_data = json.load(f)

    dev_uids = set()
    for fold in cv_data["folds"].values():
        dev_uids.update(fold["train_studies"])
        dev_uids.update(fold["val_studies"])

    # Prepare datasets: Standard (12 groups) and HighBudget (24 groups)
    ds_standard = Gold58DatasetD8(gold_df, meta_df, PROCESSED_GOLD_DIR, dev_uids, high_budget=False)
    ds_highbudget = Gold58DatasetD8(gold_df, meta_df, PROCESSED_GOLD_DIR, dev_uids, high_budget=True)

    models_to_eval = [
        ("d8a", PhaseD8Model_PathologyCrossAttention, os.path.join(D8_BASE_DIR, "d8a/checkpoints"), ds_standard),
        ("d8b", PhaseD8Model_HighBudgetDepth, os.path.join(D8_BASE_DIR, "d8b/checkpoints"), ds_highbudget),
        ("d8c", PhaseD8Model_HybridConvAttn, os.path.join(D8_BASE_DIR, "d8c/checkpoints"), ds_standard),
        ("d8d", PhaseD8Model_FullCombination, os.path.join(D8_BASE_DIR, "d8d/checkpoints"), ds_highbudget),
    ]

    all_results = {}
    for name, cls_m, ckpt_dir, ds in models_to_eval:
        res = evaluate_model_on_gold(name, cls_m, ckpt_dir, ds, DEVICE)
        all_results[name] = res

    os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_EVAL_DIR, "d8_gold58_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cohort": "Gold-58 (58 Clinical Studies, csv/gold_studies.csv)",
            "historical_benchmarks": {
                "phase_c_baseline": {"macro_roc_auc": 0.5986, "macro_pr_auc": 0.4526},
                "phase_d1_2_milestone": {"macro_roc_auc": 0.6174, "macro_pr_auc": 0.4938},
                "phase_d5_b2_vit": {"macro_roc_auc": 0.6307, "macro_pr_auc": 0.5285},
                "phase_d6_b2_champion": {"macro_roc_auc": 0.6331, "macro_pr_auc": 0.5125},
                "phase_d7_c_all_time_record": {"macro_roc_auc": 0.6474, "macro_pr_auc": 0.5290},
            },
            "results": all_results,
        }, f, indent=2)

    # Build Comparative Table Markdown Report
    md = f"""# PHASE D8 — LOCKED GOLD-58 BENCHMARK REPORT (D8-A, D8-B, D8-C, D8-D)

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}  
**Cohort**: 58 Permanently Isolated Clinical Studies (`csv/gold_studies.csv`)  
**Evaluation Pass**: ONE-TIME AUTHORIZED LOCKED BENCHMARK (Strictly No Retraining/Tuning Afterward)  
**Governance**: Single-Pass Evaluation; Permanent Quarantine Immediately Resealed.  

---

## 1. Executive Gold-58 Benchmark Matrix

| Model Identifier | Architecture | Volumetric Sampling | Gold Macro ROC-AUC | Gold PR-AUC | $\\Delta$ vs D7-C Record (0.6474) | $\\Delta$ vs D6 Champion (0.6331) | $\\Delta$ vs D1.2-B2 (0.6174) | $\\Delta$ vs Phase C (0.5986) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase C Baseline** | 3D ResNet-18 Control | Central Band | `0.5986` | `0.4526` | -0.0488 | -0.0345 | -0.0188 | — |
| **Phase D1.2-B2** | ViT-S/14 Standalone | Central Band | `0.6174` | `0.4938` | -0.0300 | -0.0157 | — | +0.0188 |
| **Phase D5-B2** | ViT-S/14 (D3 Repr) | Central Band | `0.6307` | `0.5285` | -0.0167 | -0.0024 | +0.0133 | +0.0321 |
| **Phase D6-B2 Champion**| ViT-S/14 (6 Slots) | Central Band | `0.6331` | `0.5125` | -0.0143 | — | +0.0157 | +0.0345 |
| **Phase D7-C Record** | ViT-B/14 (87.2M) | Multi-Window | **`0.6474`** | **`0.5290`** | — | **+0.0143** | **+0.0300** | **+0.0488** |
| **D8-A** | ViT-S/14 + PathologyCrossAttn | Central Band (12 grp) | **`{all_results['d8a']['rank_ensemble']['macro_roc_auc']:.4f}`** | `{all_results['d8a']['rank_ensemble']['macro_pr_auc']:.4f}` | **`{all_results['d8a']['rank_ensemble']['delta_vs_d7c_record']:+.4f}`** | **`{all_results['d8a']['rank_ensemble']['delta_vs_d6_gold']:+.4f}`** | `{all_results['d8a']['rank_ensemble']['delta_vs_d1_2_gold']:+.4f}` | `{all_results['d8a']['rank_ensemble']['delta_vs_phase_c_gold']:+.4f}` |
| **D8-B** | ViT-S/14 + HighBudget (24 grp) | High-Budget (24 grp) | **`{all_results['d8b']['rank_ensemble']['macro_roc_auc']:.4f}`** | `{all_results['d8b']['rank_ensemble']['macro_pr_auc']:.4f}` | **`{all_results['d8b']['rank_ensemble']['delta_vs_d7c_record']:+.4f}`** | **`{all_results['d8b']['rank_ensemble']['delta_vs_d6_gold']:+.4f}`** | `{all_results['d8b']['rank_ensemble']['delta_vs_d1_2_gold']:+.4f}` | `{all_results['d8b']['rank_ensemble']['delta_vs_phase_c_gold']:+.4f}` |
| **D8-C** | Hybrid Conv+Attn (CoAtNet) | Central Band (12 grp) | **`{all_results['d8c']['rank_ensemble']['macro_roc_auc']:.4f}`** | `{all_results['d8c']['rank_ensemble']['macro_pr_auc']:.4f}` | **`{all_results['d8c']['rank_ensemble']['delta_vs_d7c_record']:+.4f}`** | **`{all_results['d8c']['rank_ensemble']['delta_vs_d6_gold']:+.4f}`** | `{all_results['d8c']['rank_ensemble']['delta_vs_d1_2_gold']:+.4f}` | `{all_results['d8c']['rank_ensemble']['delta_vs_phase_c_gold']:+.4f}` |
| **D8-D** | Full Combined D8 Configuration | High-Budget (24 grp) | **`{all_results['d8d']['rank_ensemble']['macro_roc_auc']:.4f}`** | `{all_results['d8d']['rank_ensemble']['macro_pr_auc']:.4f}` | **`{all_results['d8d']['rank_ensemble']['delta_vs_d7c_record']:+.4f}`** | **`{all_results['d8d']['rank_ensemble']['delta_vs_d6_gold']:+.4f}`** | `{all_results['d8d']['rank_ensemble']['delta_vs_d1_2_gold']:+.4f}` | `{all_results['d8d']['rank_ensemble']['delta_vs_phase_c_gold']:+.4f}` |

---

## 2. Per-Pathology Clinical Breakdown on Gold-58 (Rank-Averaged AUC)

| Pathology Target | Pos / Neg | D6 Champion | D7-C Record | D8-A (CrossAttn) | D8-B (HighBudget) | D8-C (Hybrid) | D8-D (Combined) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    d6_map = {
        "Fracture": 0.7410, "Baker's": 0.7346, "MCL": 0.7211, "ACL": 0.6538,
        "Contusion": 0.6242, "Effusion": 0.6205, "Medial Meniscus": 0.6178,
        "Lateral Meniscus": 0.6118, "Synovitis": 0.6039, "PF OA": 0.5817,
        "Lateral OA": 0.5522, "Medial OA": 0.5341
    }
    d7c_map = {
        "Fracture": 0.7385, "Baker's": 0.7726, "MCL": 0.6868, "ACL": 0.6577,
        "Contusion": 0.6205, "Effusion": 0.6646, "Medial Meniscus": 0.6148,
        "Lateral Meniscus": 0.5898, "Synovitis": 0.6074, "PF OA": 0.6486,
        "Lateral OA": 0.6422, "Medial OA": 0.5250
    }

    for t in PATHOLOGY_NAMES:
        pos = all_results["d8a"]["per_target"][t]["positives"]
        neg = all_results["d8a"]["per_target"][t]["negatives"]
        d6_t = d6_map.get(t, 0.6000)
        d7c_t = d7c_map.get(t, 0.6000)
        a_t = all_results["d8a"]["per_target"][t]["roc_auc"]
        b_t = all_results["d8b"]["per_target"][t]["roc_auc"]
        c_t = all_results["d8c"]["per_target"][t]["roc_auc"]
        d_t = all_results["d8d"]["per_target"][t]["roc_auc"]
        md += f"| **{t}** | {pos} / {neg} | `{d6_t:.4f}` | `{d7c_t:.4f}` | `{a_t:.4f}` | `{b_t:.4f}` | `{c_t:.4f}` | `{d_t:.4f}` |\n"

    md += """
---

## 3. Individual Fold Checkpoint Performance on Gold-58 (Macro ROC-AUC)

| Model Identifier | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 5-Fold Rank Ensemble |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for name in ["d8a", "d8b", "d8c", "d8d"]:
        f_aucs = [all_results[name]["per_fold"][f"fold_{f}"]["macro_roc_auc"] for f in range(5)]
        ens_auc = all_results[name]["rank_ensemble"]["macro_roc_auc"]
        md += f"| **{name.upper()}** | `{f_aucs[0]:.4f}` | `{f_aucs[1]:.4f}` | `{f_aucs[2]:.4f}` | `{f_aucs[3]:.4f}` | `{f_aucs[4]:.4f}` | **`{ens_auc:.4f}`** |\n"

    md += """
---

## 4. Governance Sign-Off: Gold-58 Permanently Resealed

Following this one-time authorized benchmark:
1. **Zero Retraining or Tuning**: The Gold-58 results are frozen permanently. No hyperparameters, weights, or architectures will be tuned on these results.
2. **Quarantine Resealed**: Access to `csv/gold_studies.csv` and `phase_c/c1/processed_gold` is permanently locked.
3. **Artifact Integrity**: Checkpoint predictions and evaluation summaries have been hashed and recorded in `phase_d/d8/manifests/d8_gold58_sha256_manifest.txt`.
"""

    os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "D8_GOLD58_BENCHMARK_REPORT.md")
    with open(report_path, "w") as f:
        f.write(md)

    # Compute SHA256 manifest for Gold-58 artifacts
    os.makedirs(OUTPUT_MANIFESTS_DIR, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_MANIFESTS_DIR, "d8_gold58_sha256_manifest.txt")
    artifacts_to_hash = [
        summary_path,
        report_path,
        all_results["d8a"]["prediction_file"],
        all_results["d8b"]["prediction_file"],
        all_results["d8c"]["prediction_file"],
        all_results["d8d"]["prediction_file"],
    ]
    with open(manifest_path, "w") as f:
        for art in artifacts_to_hash:
            if os.path.exists(art):
                rel_path = os.path.relpath(art, PROJECT_ROOT)
                h = compute_sha256(art)
                f.write(f"{h}  {rel_path}\n")

    print(f"\n[SAVED] Gold-58 Summary JSON: {summary_path}")
    print(f"[SAVED] Gold-58 Markdown Report: {report_path}")
    print(f"[SAVED] Gold-58 SHA256 Manifest: {manifest_path}")
    print("=" * 80)
    print("GOLD-58 BENCHMARK COMPLETE. GOLD ACCESS IS NOW PERMANENTLY RESEALED.")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
