#!/usr/bin/env python3
"""
Phase C3.12 Stage A: Out-of-Fold (OOF) Integrity & Global Metric Audit
RSNA Knee Abnormality Detection

Performs:
1. Strict OOF Integrity Verification (Coverage, Duplicates, Gold-58, Provenance, Partitions)
2. Global Study-Level Macro & Per-Target ROC-AUC calculation (4,349 concatenated predictions)
3. Soft label continuous correlation analysis (Spearman rank correlation, Pearson r)
4. Comprehensive Investigation of Target Distribution Anomalies (Medial OA, PF OA, Lateral OA)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase_c", "c3", "scripts"))

from model import TARGETS

def roc_auc_score_np(y_true, y_score):
    """Pure NumPy Mann-Whitney U / Pairwise difference ROC-AUC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))

def audit_oof():
    print("=" * 80)
    print("  PHASE C3.12: OUT-OF-FOLD (OOF) INTEGRITY & GLOBAL METRIC AUDIT")
    print("=" * 80)

    # 1. Load files
    oof_path = os.path.join(REPO_ROOT, "phase_c", "c3", "results", "oof_predictions_5folds.csv")
    prov_path = os.path.join(REPO_ROOT, "phase_c", "c3", "results", "oof", "oof_predictions_with_fold_provenance.csv")
    splits_path = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
    labels_path = os.path.join(REPO_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")
    gold_path = os.path.join(REPO_ROOT, "csv", "gold_studies.csv")
    results_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "results")

    oof_df = pd.read_csv(oof_path)
    prov_df = pd.read_csv(prov_path)
    labels_df = pd.read_csv(labels_path)
    gold_df = pd.read_csv(gold_path)
    with open(splits_path) as f:
        cv_data = json.load(f)

    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    # --- INTEGRITY CHECKS ---
    print("\n--- 1. OOF INTEGRITY AUDIT ---")
    
    # Coverage
    unique_studies = oof_df["StudyInstanceUID"].nunique()
    total_rows = len(oof_df)
    print(f"Total OOF rows: {total_rows:,} (Expected: 4,349) -> {total_rows == 4349}")
    print(f"Unique StudyInstanceUIDs: {unique_studies:,} (Expected: 4,349) -> {unique_studies == 4349}")
    assert total_rows == 4349, f"Row count error: {total_rows}"
    assert unique_studies == 4349, f"Coverage error: {unique_studies}"

    # Duplicates
    dup_count = oof_df["StudyInstanceUID"].duplicated().sum()
    print(f"Duplicate StudyInstanceUIDs: {dup_count} (Expected: 0) -> {dup_count == 0}")
    assert dup_count == 0, f"Duplicate error: {dup_count}"

    # Missing studies
    all_dev_uids = set()
    for f_idx in range(5):
        all_dev_uids.update(cv_data["folds"][f"fold_{f_idx}"]["val_studies"])
    missing_studies = all_dev_uids - set(oof_df["StudyInstanceUID"])
    print(f"Missing Development Studies: {len(missing_studies)} (Expected: 0) -> {len(missing_studies) == 0}")
    assert len(missing_studies) == 0, f"Missing error: {len(missing_studies)}"

    # Gold leakage
    gold_overlap = set(oof_df["StudyInstanceUID"]) & gold_uids
    print(f"Gold-58 Studies in OOF: {len(gold_overlap)} (Expected: 0) -> {len(gold_overlap) == 0}")
    assert len(gold_overlap) == 0, f"CRITICAL LEAKAGE: Gold studies found in OOF: {gold_overlap}"

    # Prediction Completeness & Dimensions
    expected_cols = ["StudyInstanceUID"] + TARGETS
    cols_match = list(oof_df.columns) == expected_cols
    print(f"Column schema match: {cols_match} (Expected: 13 columns)")
    assert cols_match, f"Column mismatch: {oof_df.columns}"

    null_count = oof_df[TARGETS].isnull().sum().sum()
    print(f"Null predictions count: {null_count} (Expected: 0) -> {null_count == 0}")
    assert null_count == 0, f"Null predictions found: {null_count}"

    # Partition integrity & provenance
    partition_counts = prov_df["cv_fold"].value_counts().to_dict()
    print("\nPartition Counts by Checkpoint Fold:")
    expected_partitions = {0: 872, 1: 871, 2: 870, 3: 869, 4: 867}
    for f_idx, exp_cnt in expected_partitions.items():
        act_cnt = partition_counts.get(f_idx, 0)
        match = act_cnt == exp_cnt
        print(f"  Fold {f_idx}: {act_cnt:,} (Expected: {exp_cnt:,}) -> {match}")
        assert match, f"Partition count mismatch for Fold {f_idx}"

    # Provenance match check: Every study was predicted by the exact fold in which it was held out
    provenance_errors = 0
    for _, row in prov_df.iterrows():
        suid = row["StudyInstanceUID"]
        pred_fold = row["cv_fold"]
        # check which fold this suid belongs to in cv_splits.json
        expected_fold = None
        for f_idx in range(5):
            if suid in cv_data["folds"][f"fold_{f_idx}"]["val_studies"]:
                expected_fold = f_idx
                break
        if pred_fold != expected_fold:
            provenance_errors += 1
    print(f"Provenance alignment errors: {provenance_errors} (Expected: 0) -> {provenance_errors == 0}")
    assert provenance_errors == 0, f"Provenance error: {provenance_errors}"

    print("\n  >>> OOF INTEGRITY AUDIT: 100% PASS <<<")

    # --- 2. GLOBAL OOF METRICS ---
    print("\n--- 2. GLOBAL OOF METRIC EVALUATION ---")
    # Reshape Phase B labels into (4349, 12) matrix aligned with OOF predictions
    dev_labels_df = labels_df[labels_df["StudyInstanceUID"].isin(all_dev_uids)].copy()
    
    # Pivot labels: rows = StudyInstanceUID, cols = target
    y_pivot = dev_labels_df.pivot(index="StudyInstanceUID", columns="target", values="label_probability").reindex(TARGETS, axis=1)
    w_pivot = dev_labels_df.pivot(index="StudyInstanceUID", columns="target", values="sample_weight").reindex(TARGETS, axis=1)
    abstain_pivot = dev_labels_df.pivot(index="StudyInstanceUID", columns="target", values="abstain_flag").reindex(TARGETS, axis=1)

    # Force sample weight to 0 for abstentions
    w_pivot[abstain_pivot == 1] = 0.0

    # Align with OOF predictions
    oof_indexed = oof_df.set_index("StudyInstanceUID").reindex(y_pivot.index)

    y_matrix = y_pivot.values.astype(np.float32)  # (4349, 12)
    w_matrix = w_pivot.values.astype(np.float32)  # (4349, 12)
    pred_matrix = oof_indexed.values.astype(np.float32)  # (4349, 12)

    # Binary targets at threshold 0.5 (standard classification threshold)
    binary_y = (y_matrix >= 0.5).astype(int)

    per_target_metrics = {}
    valid_aucs = []

    print(f"{'Target':<18} | {'OOF AUC (bin=0.5)':<18} | {'Spearman rho':<14} | {'Pearson r':<12} | {'Positives (Y>=0.5)':<20} | {'Prevalence'}")
    print("-" * 105)

    for i, target_name in enumerate(TARGETS):
        y_true = binary_y[:, i]
        y_soft = y_matrix[:, i]
        y_pred = pred_matrix[:, i]
        w = w_matrix[:, i]

        pos_cnt = int(np.sum(y_true))
        neg_cnt = int(len(y_true) - pos_cnt)
        prev_pct = float(pos_cnt / len(y_true) * 100.0)

        # Global ROC-AUC on concatenated 4,349 studies
        if pos_cnt > 0 and neg_cnt > 0:
            auc = roc_auc_score_np(y_true, y_pred)
            valid_aucs.append(auc)
            auc_str = f"{auc:.4f}"
        else:
            auc = None
            auc_str = "N/A"

        # Continuous correlations
        # evaluate on non-abstained samples (w > 0)
        active_mask = (w > 0)
        if np.sum(active_mask) > 10:
            rho, _ = spearmanr(y_soft[active_mask], y_pred[active_mask])
            r, _ = pearsonr(y_soft[active_mask], y_pred[active_mask])
        else:
            rho, r = 0.0, 0.0

        per_target_metrics[target_name] = {
            "global_oof_roc_auc": auc,
            "spearman_rho": float(rho),
            "pearson_r": float(r),
            "positives_count": pos_cnt,
            "negatives_count": neg_cnt,
            "prevalence_pct": prev_pct,
            "mean_pred": float(np.mean(y_pred)),
            "std_pred": float(np.std(y_pred)),
            "mean_soft_y": float(np.mean(y_soft)),
            "std_soft_y": float(np.std(y_soft)),
        }

        print(f"{target_name:<18} | {auc_str:<18} | {rho:14.4f} | {r:12.4f} | {pos_cnt:>8} / {len(y_true):<9} | {prev_pct:6.2f}%")

    global_macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.0
    print("-" * 105)
    print(f"GLOBAL STUDY-LEVEL OOF MACRO ROC-AUC: {global_macro_auc:.4f} (across {len(valid_aucs)} evaluable targets)")
    print(f"C3.11 FOLD-LEVEL MEAN MACRO ROC-AUC:   0.6333 ± 0.0326 (historical fold average)")

    # --- 3. ANOMALY AUDIT ---
    print("\n--- 3. ANOMALY INVESTIGATION: Medial OA, PF OA, Lateral OA ---")
    
    # Medial OA breakdown by fold
    print("\nMedial OA Fold-by-Fold Diagnostics:")
    for f_idx in range(5):
        val_uids_f = cv_data["folds"][f"fold_{f_idx}"]["val_studies"]
        sub_oof = oof_indexed.loc[val_uids_f, "Medial OA"].values
        sub_y = y_pivot.loc[val_uids_f, "Medial OA"].values
        sub_bin_y = (sub_y >= 0.5).astype(int)
        f_auc = roc_auc_score_np(sub_bin_y, sub_oof)
        f_rho, _ = spearmanr(sub_y, sub_oof)
        auc_display = f"{f_auc:.4f}" if f_auc is not None else "N/A"
        print(f"  Fold {f_idx}: AUC = {auc_display:<7} | Spearman rho = {f_rho:7.4f} | Positives: {np.sum(sub_bin_y):>3} / {len(sub_bin_y)}")

    # PF OA breakdown across cohort
    print("\nPF OA Cohort Diagnostics:")
    pf_soft = y_pivot["PF OA"].values
    pf_preds = oof_indexed["PF OA"].values
    pf_pos_05 = np.sum(pf_soft >= 0.5)
    pf_pos_03 = np.sum(pf_soft >= 0.3)
    pf_pos_01 = np.sum(pf_soft >= 0.1)
    print(f"  PF OA Soft Target Range: min={pf_soft.min():.4f}, max={pf_soft.max():.4f}, mean={pf_soft.mean():.4f}")
    print(f"  Studies with Y >= 0.5: {pf_pos_05} (Prevalence: {pf_pos_05/4349*100:.2f}%)")
    print(f"  Studies with Y >= 0.3: {pf_pos_03} (Prevalence: {pf_pos_03/4349*100:.2f}%)")
    print(f"  Studies with Y >= 0.1: {pf_pos_01} (Prevalence: {pf_pos_01/4349*100:.2f}%)")
    for f_idx in range(5):
        val_uids_f = cv_data["folds"][f"fold_{f_idx}"]["val_studies"]
        sub_y = y_pivot.loc[val_uids_f, "PF OA"].values
        print(f"    Fold {f_idx} PF OA: max soft Y = {sub_y.max():.4f}, count Y >= 0.5 = {np.sum(sub_y >= 0.5)}")

    # Lateral OA breakdown across cohort
    print("\nLateral OA Cohort Diagnostics:")
    lat_soft = y_pivot["Lateral OA"].values
    lat_preds = oof_indexed["Lateral OA"].values
    lat_pos_05 = np.sum(lat_soft >= 0.5)
    print(f"  Lateral OA Soft Target Range: min={lat_soft.min():.4f}, max={lat_soft.max():.4f}, mean={lat_soft.mean():.4f}")
    print(f"  Studies with Y >= 0.5: {lat_pos_05} (Prevalence: {lat_pos_05/4349*100:.2f}%)")
    for f_idx in range(5):
        val_uids_f = cv_data["folds"][f"fold_{f_idx}"]["val_studies"]
        sub_y = y_pivot.loc[val_uids_f, "Lateral OA"].values
        print(f"    Fold {f_idx} Lateral OA: max soft Y = {sub_y.max():.4f}, count Y >= 0.5 = {np.sum(sub_y >= 0.5)}")

    # Save summary json
    summary = {
        "stage": "C3.12_Stage_A_OOF_Audit",
        "total_studies": total_rows,
        "unique_studies": unique_studies,
        "gold_leakage": len(gold_overlap),
        "global_macro_roc_auc": global_macro_auc,
        "evaluable_targets_count": len(valid_aucs),
        "historical_c311_fold_mean_macro_auc": 0.6333,
        "historical_c311_fold_std_macro_auc": 0.0326,
        "per_target_metrics": per_target_metrics,
        "partition_counts": expected_partitions,
    }

    summary_json_path = os.path.join(results_dir, "oof_audit_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved OOF audit summary to: {summary_json_path}")

    # Save audit markdown
    md_path = os.path.join(results_dir, "oof_audit_report.md")
    with open(md_path, "w") as f:
        f.write("# Phase C3.12 Stage A: Out-of-Fold (OOF) Audit Report\n\n")
        f.write("## 1. OOF Integrity Summary\n\n")
        f.write(f"- **Total Non-Gold Development Studies**: `4,349`\n")
        f.write(f"- **Unique Studies Covered**: `4,349` (`100.0%`)\n")
        f.write(f"- **Duplicate Records**: `0`\n")
        f.write(f"- **Missing Development Studies**: `0`\n")
        f.write(f"- **Gold-58 Leakage**: **`0` (PASS)**\n")
        f.write(f"- **Provenance Alignment**: `100.0%` (Every study predicted strictly by its held-out fold checkpoint)\n")
        f.write(f"- **Consolidated OOF Path**: `phase_c/c3/results/oof_predictions_5folds.csv`\n\n")
        f.write("## 2. Global Study-Level OOF Metrics vs Fold Means\n\n")
        f.write(f"- **Global Study-Level OOF Macro ROC-AUC**: **`{global_macro_auc:.4f}`** (concatenated 4,349 predictions)\n")
        f.write(f"- **Historical C3.11 Fold-Level Mean Macro ROC-AUC**: `0.6333 ± 0.0326`\n\n")
        f.write("### Per-Target Global OOF Performance\n\n")
        f.write("| Target Pathology | Global OOF ROC-AUC | Spearman rho | Pearson r | Positives (Y>=0.5) | Prevalence (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for t_name in TARGETS:
            m = per_target_metrics[t_name]
            auc_v = f"`{m['global_oof_roc_auc']:.4f}`" if m['global_oof_roc_auc'] is not None else "`N/A`"
            f.write(f"| **{t_name}** | {auc_v} | `{m['spearman_rho']:.4f}` | `{m['pearson_r']:.4f}` | `{m['positives_count']:,} / 4,349` | `{m['prevalence_pct']:.2f}%` |\n")
    print(f"Saved OOF audit markdown to: {md_path}")

if __name__ == "__main__":
    audit_oof()
