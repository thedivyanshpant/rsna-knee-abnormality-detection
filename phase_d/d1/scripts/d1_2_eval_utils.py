"""
Phase D1.2: Evaluation Utilities and Common Metric Functions.

Implements standard metric evaluation matching aggregate_d1_oof.py:
- Unweighted Macro ROC-AUC (binarized Y >= 0.5)
- Sample-weighted Macro ROC-AUC (active W > 0)
- Macro PR-AUC (Average Precision)
- Per-target metrics across all 12 pathologies
- Per-fold breakdown
- Integrity audit assertions
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")

PATHOLOGY_NAMES = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PHASE_C_GLOBAL_OOF_MACRO_AUC = 0.6198
PHASE_C_PER_TARGET = {
    "ACL": 0.6096,
    "MCL": 0.5126,
    "Medial Meniscus": 0.6839,
    "Lateral Meniscus": 0.5861,
    "Medial OA": 0.5571,
    "Lateral OA": 0.6419,
    "PF OA": 0.4449,
    "Effusion": 0.7666,
    "Synovitis": 0.7398,
    "Baker's": 0.6777,
    "Contusion": 0.6365,
    "Fracture": 0.5811,
}

D1_1_BASELINE_UNWEIGHTED_AUC = 0.7510
D1_1_BASELINE_WEIGHTED_AUC = 0.7573


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy Mann-Whitney U / Pairwise difference ROC-AUC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def pr_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy Average Precision / Precision-Recall AUC."""
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


def evaluate_oof_df(df: pd.DataFrame, experiment_id: str = "D1.2") -> Dict[str, Any]:
    """Perform full statistical evaluation and governance audit on an OOF dataframe."""
    with open(CONFIG_CV) as f:
        cv_data = json.load(f)

    all_expected_studies = set()
    for fold_idx in range(5):
        all_expected_studies.update(cv_data["folds"][f"fold_{fold_idx}"]["val_studies"])

    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    total_rows = len(df)
    unique_studies = df["StudyInstanceUID"].nunique()
    duplicates_count = total_rows - unique_studies
    missing_studies = len(all_expected_studies - set(df["StudyInstanceUID"]))
    gold_leakage = len(set(df["StudyInstanceUID"]) & gold_uids)
    nan_count = int(df.isna().sum().sum())
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number]).values).sum())

    audit_passed = (
        total_rows == 4349 and
        unique_studies == 4349 and
        duplicates_count == 0 and
        missing_studies == 0 and
        gold_leakage == 0 and
        nan_count == 0 and
        inf_count == 0
    )

    y_true_all = df[[f"true_{t}" for t in PATHOLOGY_NAMES]].values
    y_pred_all = df[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values
    y_w_all = df[[f"weight_{t}" for t in PATHOLOGY_NAMES]].values

    per_target = {}
    valid_aucs_unw = []
    valid_aucs_w = []
    valid_praucs = []

    for t_idx, name in enumerate(PATHOLOGY_NAMES):
        t_true_raw = y_true_all[:, t_idx]
        t_pred = y_pred_all[:, t_idx]
        t_w = y_w_all[:, t_idx]

        t_true_bin = (t_true_raw >= 0.5).astype(int)
        pos_count = int(np.sum(t_true_bin == 1))
        neg_count = int(np.sum(t_true_bin == 0))

        auc_unw = roc_auc_score_np(t_true_bin, t_pred) if (pos_count > 0 and neg_count > 0) else None
        prauc = pr_auc_score_np(t_true_bin, t_pred) if (pos_count > 0 and neg_count > 0) else None

        active = (t_w > 0.0)
        if np.sum(t_true_bin[active] == 1) > 0 and np.sum(t_true_bin[active] == 0) > 0:
            auc_w = roc_auc_score_np(t_true_bin[active], t_pred[active])
        else:
            auc_w = None

        if auc_unw is not None:
            valid_aucs_unw.append(auc_unw)
        if auc_w is not None:
            valid_aucs_w.append(auc_w)
        if prauc is not None:
            valid_praucs.append(prauc)

        per_target[name] = {
            "pos_count": pos_count,
            "neg_count": neg_count,
            "roc_auc_unweighted": round(auc_unw, 4) if auc_unw is not None else None,
            "roc_auc_weighted": round(auc_w, 4) if auc_w is not None else None,
            "pr_auc": round(prauc, 4) if prauc is not None else None,
            "delta_vs_d1_1": round(auc_unw - PHASE_C_PER_TARGET.get(name, 0.0), 4) if auc_unw is not None else None,
        }

    global_unw = float(np.mean(valid_aucs_unw)) if valid_aucs_unw else 0.0
    global_w = float(np.mean(valid_aucs_w)) if valid_aucs_w else 0.0
    global_pr = float(np.mean(valid_praucs)) if valid_praucs else 0.0

    # Per-fold breakdown
    per_fold = {}
    for fold_idx in range(5):
        df_f = df[df["fold"] == fold_idx]
        f_targets = df_f[[f"true_{t}" for t in PATHOLOGY_NAMES]].values
        f_preds = df_f[[f"pred_{t}" for t in PATHOLOGY_NAMES]].values
        f_weights = df_f[[f"weight_{t}" for t in PATHOLOGY_NAMES]].values

        f_aucs = []
        f_w_aucs = []
        for t_idx, name in enumerate(PATHOLOGY_NAMES):
            t_true = (f_targets[:, t_idx] >= 0.5).astype(int)
            t_pred = f_preds[:, t_idx]
            t_w = f_weights[:, t_idx]
            if len(np.unique(t_true)) >= 2:
                a = roc_auc_score_np(t_true, t_pred)
                if a is not None:
                    f_aucs.append(a)
                act = (t_w > 0)
                if np.sum(t_true[act] == 1) > 0 and np.sum(t_true[act] == 0) > 0:
                    wa = roc_auc_score_np(t_true[act], t_pred[act])
                    if wa is not None:
                        f_w_aucs.append(wa)

        per_fold[f"fold_{fold_idx}"] = {
            "study_count": len(df_f),
            "macro_roc_auc_unweighted": round(float(np.mean(f_aucs)), 4) if f_aucs else None,
            "macro_roc_auc_weighted": round(float(np.mean(f_w_aucs)), 4) if f_w_aucs else None,
        }

    delta_unw = global_unw - D1_1_BASELINE_UNWEIGHTED_AUC
    delta_w = global_w - D1_1_BASELINE_WEIGHTED_AUC

    return {
        "experiment_id": experiment_id,
        "audit_passed": audit_passed,
        "total_studies": total_rows,
        "unique_studies": unique_studies,
        "gold_leakage": gold_leakage,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "global_macro_roc_auc_unweighted": round(global_unw, 4),
        "global_macro_roc_auc_weighted": round(global_w, 4),
        "global_macro_pr_auc": round(global_pr, 4),
        "delta_unweighted_vs_d1_1": round(delta_unw, 4),
        "delta_weighted_vs_d1_1": round(delta_w, 4),
        "relative_gain_unweighted_pct": round((delta_unw / D1_1_BASELINE_UNWEIGHTED_AUC) * 100.0, 2),
        "per_fold": per_fold,
        "per_target": per_target,
    }


if __name__ == "__main__":
    baseline_csv = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")
    if os.path.exists(baseline_csv):
        df_base = pd.read_csv(baseline_csv)
        res = evaluate_oof_df(df_base, "D1.1_REPRO_CHECK")
        print("Audit Passed:", res["audit_passed"])
        print("Global Macro ROC-AUC Unweighted:", res["global_macro_roc_auc_unweighted"])
        print("Global Macro ROC-AUC Weighted:", res["global_macro_roc_auc_weighted"])
        print("Global Macro PR-AUC:", res["global_macro_pr_auc"])
        assert res["global_macro_roc_auc_unweighted"] == 0.7510, f"Mismatch: {res['global_macro_roc_auc_unweighted']}"
        assert res["global_macro_roc_auc_weighted"] == 0.7573, f"Mismatch: {res['global_macro_roc_auc_weighted']}"
        print("[SUCCESS] D1.1 Baseline reproduction confirmed exactly!")
