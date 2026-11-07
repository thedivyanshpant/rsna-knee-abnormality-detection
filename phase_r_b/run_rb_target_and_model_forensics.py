#!/usr/bin/env python3
"""
Phase R/B: Target Bottlenecks, Model Disagreement, and Fold Stability Forensics.
Computes authoritative statistical metrics across all 15 master OOF predictions on HPC.
"""

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase_d", "d1", "scripts"))
from d1_2_eval_utils import roc_auc_score_np, pr_auc_score_np, PATHOLOGY_NAMES

D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")

MODEL_OOF_PATHS = {
    "D1.1": "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv",
    "D1.2": "phase_d/d1/predictions/d1_2_b_hflip/d1_2_b_hflip_global_oof.csv",
    "D2": "phase_d/d2/predictions/oof/d2_global_oof_predictions.csv",
    "D3": "phase_d/d3/predictions/d3_global_oof_predictions.csv",
    "D4": "phase_d/d4/predictions/oof/d4_cnn_oof_predictions.csv",
    "D5": "phase_d/d5/predictions/d5_b2_oof_predictions.csv",
    "D6": "phase_d/d6/oof/d6_global_oof_predictions.csv",
    "D7-A": "phase_d/d7/oof/d7a_global_oof_predictions.csv",
    "D7-B": "phase_d/d7/oof/d7b_global_oof_predictions.csv",
    "D7-C": "phase_d/d7/oof/d7c_global_oof_predictions.csv",
    "D7-D": "phase_d/d7/oof/d7d_global_oof_predictions.csv",
    "D8-A": "phase_d/d8/oof/d8a_global_oof_predictions.csv",
    "D8-B": "phase_d/d8/oof/d8b_global_oof_predictions.csv",
    "D8-C": "phase_d/d8/oof/d8c_global_oof_predictions.csv",
    "D8-D": "phase_d/d8/oof/d8d_global_oof_predictions.csv",
}

def load_and_standardize_oof(name, rel_path, true_labels_df):
    path = os.path.join(PROJECT_ROOT, rel_path)
    df = pd.read_csv(path)
    # Ensure study order
    df = df.sort_values("StudyInstanceUID").reset_index(drop=True)
    
    # Check if true labels exist; if not, merge from true_labels_df
    has_true = any(f"true_{t}" in df.columns for t in PATHOLOGY_NAMES)
    if not has_true:
        # Merge true labels from true_labels_df
        df = pd.merge(df, true_labels_df, on="StudyInstanceUID", how="left")
    
    return df

def spearman_corr_np(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])

def pearson_corr_np(x, y):
    return float(np.corrcoef(x, y)[0, 1])

def main():
    print("Loading D6 reference OOF for authoritative D3 remediated ground truth...")
    d6_ref = pd.read_csv(os.path.join(PROJECT_ROOT, MODEL_OOF_PATHS["D6"]))
    true_labels_df = d6_ref[["StudyInstanceUID"] + [f"true_{t}" for t in PATHOLOGY_NAMES]]

    print("Loading all OOF prediction dataframes...")
    oofs = {}
    for name, rel_path in MODEL_OOF_PATHS.items():
        path = os.path.join(PROJECT_ROOT, rel_path)
        df = pd.read_csv(path)
        df = df.sort_values("StudyInstanceUID").reset_index(drop=True)
        # Drop any existing true_ columns to ensure identical authoritative ground truth across all models
        true_cols = [c for c in df.columns if c.startswith("true_")]
        if true_cols:
            df = df.drop(columns=true_cols)
        df = pd.merge(df, true_labels_df, on="StudyInstanceUID", how="left")
        oofs[name] = df
        print(f"Loaded {name}: {len(df)} rows")

    # Verify identical study order across all dataframes
    studies_ref = oofs["D3"]["StudyInstanceUID"].values
    for name, df in oofs.items():
        assert np.array_equal(df["StudyInstanceUID"].values, studies_ref), f"Study UID mismatch in {name}"
    print("All study alignments verified: 4,349 studies identical across all 15 models.")

    # -------------------------------------------------------------
    # 1. R/B-E: TARGET-WISE ERROR FORENSICS & METRICS
    # -------------------------------------------------------------
    print("\n--- Computing Target-wise Metrics ---")
    target_results = {}
    
    # Let's compute prevalence from D3 ground truth
    prevalence = {}
    for t in PATHOLOGY_NAMES:
        y = (oofs["D3"][f"true_{t}"].values >= 0.5).astype(int)
        pos_cnt = int(np.sum(y == 1))
        neg_cnt = int(np.sum(y == 0))
        prev_pct = float(np.mean(y) * 100.0)
        prevalence[t] = {
            "pos_count": pos_cnt,
            "neg_count": neg_cnt,
            "prevalence_pct": prev_pct
        }

    eval_models = ["D3", "D5", "D6", "D7-A", "D7-B", "D7-C", "D7-D", "D8-A", "D8-B", "D8-C", "D8-D", "D1.1", "D2", "D4"]
    model_target_metrics = {m: {} for m in eval_models}
    
    for m in eval_models:
        df = oofs[m]
        macro_aucs = []
        macro_pr_aucs = []
        for t in PATHOLOGY_NAMES:
            y = (df[f"true_{t}"].values >= 0.5).astype(int)
            score = df[f"pred_{t}"].values
            auc = roc_auc_score_np(y, score)
            pr_auc = pr_auc_score_np(y, score)
            macro_aucs.append(auc)
            macro_pr_aucs.append(pr_auc)
            
            # Per-fold calculations
            fold_aucs = []
            fold_pr_aucs = []
            for f in range(5):
                f_mask = (df["fold"].values == f)
                f_y = y[f_mask]
                f_score = score[f_mask]
                f_auc = roc_auc_score_np(f_y, f_score)
                f_pr = pr_auc_score_np(f_y, f_score)
                fold_aucs.append(f_auc)
                fold_pr_aucs.append(f_pr)
            
            model_target_metrics[m][t] = {
                "global_auc": auc,
                "global_pr_auc": pr_auc,
                "fold_aucs": fold_aucs,
                "fold_mean_auc": float(np.mean(fold_aucs)),
                "fold_std_auc": float(np.std(fold_aucs)),
                "fold_min_auc": float(np.min(fold_aucs)),
                "fold_max_auc": float(np.max(fold_aucs)),
                "fold_best": int(np.argmax(fold_aucs)),
                "fold_worst": int(np.argmin(fold_aucs)),
            }
        model_target_metrics[m]["MACRO"] = {
            "global_auc": float(np.mean(macro_aucs)),
            "global_pr_auc": float(np.mean(macro_pr_aucs))
        }

    # Target classifications based on D6 (Dev Champion) & D8-C (Gold Champion)
    # Thresholds: Strong (AUC >= 0.86), Intermediate (0.78 <= AUC < 0.86), Weak (AUC < 0.78)
    target_classifications = {}
    for t in PATHOLOGY_NAMES:
        d6_auc = model_target_metrics["D6"][t]["global_auc"]
        d3_auc = model_target_metrics["D3"][t]["global_auc"]
        d8c_auc = model_target_metrics["D8-C"][t]["global_auc"]
        avg_auc = (d6_auc + d3_auc + d8c_auc) / 3.0
        
        if avg_auc >= 0.86:
            tier = "Strong"
        elif avg_auc >= 0.78:
            tier = "Intermediate"
        else:
            tier = "Weak"
            
        target_classifications[t] = {
            "tier": tier,
            "avg_auc": avg_auc,
            "d6_auc": d6_auc,
            "d3_auc": d3_auc,
            "d8c_auc": d8c_auc,
            "d6_pr_auc": model_target_metrics["D6"][t]["global_pr_auc"],
            "prevalence": prevalence[t]
        }

    # Save target bottlenecks
    with open(os.path.join(PROJECT_ROOT, "phase_r_b/rb_target_bottlenecks.json"), "w") as f:
        json.dump({
            "prevalence": prevalence,
            "metrics": model_target_metrics,
            "classifications": target_classifications
        }, f, indent=2)
    print("Saved rb_target_bottlenecks.json")

    # -------------------------------------------------------------
    # 2. R/B-F: MODEL DISAGREEMENT & ERROR CORRELATION
    # -------------------------------------------------------------
    print("\n--- Computing Model Disagreement & Pairwise Error Forensics ---")
    compare_models = ["D3", "D5", "D6", "D7-A", "D7-B", "D7-C", "D7-D", "D8-A", "D8-B", "D8-C", "D8-D"]
    pairwise_results = {}
    
    for m1, m2 in itertools.combinations(compare_models, 2):
        pair_key = f"{m1}_vs_{m2}"
        df1 = oofs[m1]
        df2 = oofs[m2]
        
        # All predictions flattened
        p1_all = []
        p2_all = []
        y_all = []
        target_stats = {}
        
        for t in PATHOLOGY_NAMES:
            p1 = df1[f"pred_{t}"].values
            p2 = df2[f"pred_{t}"].values
            y = (df1[f"true_{t}"].values >= 0.5).astype(int)
            
            p1_all.extend(p1)
            p2_all.extend(p2)
            y_all.extend(y)
            
            # Per-target statistics
            p_corr = pearson_corr_np(p1, p2)
            s_corr = spearman_corr_np(p1, p2)
            mae = float(np.mean(np.abs(p1 - p2)))
            
            # Binary decisions at 0.5
            bin1 = (p1 >= 0.5).astype(int)
            bin2 = (p2 >= 0.5).astype(int)
            disagree_rate = float(np.mean(bin1 != bin2) * 100.0)
            
            # Error overlap:
            # err1: m1 wrong, err2: m2 wrong
            err1 = (bin1 != y)
            err2 = (bin2 != y)
            both_err = float(np.sum(err1 & err2))
            only1_err = float(np.sum(err1 & ~err2))
            only2_err = float(np.sum(~err1 & err2))
            total_err_union = float(np.sum(err1 | err2))
            jaccard_err = float(both_err / total_err_union) if total_err_union > 0 else 1.0
            
            target_stats[t] = {
                "pearson": p_corr,
                "spearman": s_corr,
                "mae": mae,
                "disagree_rate_pct": disagree_rate,
                "both_error_count": int(both_err),
                "only_m1_error": int(only1_err),
                "only_m2_error": int(only2_err),
                "error_jaccard": jaccard_err
            }
            
        p1_all = np.array(p1_all)
        p2_all = np.array(p2_all)
        y_all = np.array(y_all)
        
        global_pearson = pearson_corr_np(p1_all, p2_all)
        global_spearman = spearman_corr_np(p1_all, p2_all)
        global_mae = float(np.mean(np.abs(p1_all - p2_all)))
        bin1_all = (p1_all >= 0.5).astype(int)
        bin2_all = (p2_all >= 0.5).astype(int)
        global_disagree = float(np.mean(bin1_all != bin2_all) * 100.0)
        
        # Classification
        # HIGHLY REDUNDANT: Spearman >= 0.93 and Disagreement <= 6.0%
        # STRONGLY COMPLEMENTARY: Spearman < 0.85 or Disagreement > 12.0%
        # MODERATELY DIFFERENT: in between
        if global_spearman >= 0.93 and global_disagree <= 6.0:
            classification = "HIGHLY REDUNDANT"
        elif global_spearman < 0.85 or global_disagree > 12.0:
            classification = "STRONGLY COMPLEMENTARY"
        else:
            classification = "MODERATELY DIFFERENT"
            
        pairwise_results[pair_key] = {
            "m1": m1,
            "m2": m2,
            "global_pearson": global_pearson,
            "global_spearman": global_spearman,
            "global_mae": global_mae,
            "global_disagree_pct": global_disagree,
            "classification": classification,
            "targets": target_stats
        }

    with open(os.path.join(PROJECT_ROOT, "phase_r_b/rb_model_disagreement.json"), "w") as f:
        json.dump(pairwise_results, f, indent=2)
    print("Saved rb_model_disagreement.json")

    # -------------------------------------------------------------
    # 3. R/B-G: FOLD STABILITY FORENSICS
    # -------------------------------------------------------------
    print("\n--- Computing Fold Stability Forensics ---")
    stability_models = ["D3", "D6", "D7-A", "D7-B", "D7-C", "D7-D", "D8-A", "D8-B", "D8-C", "D8-D"]
    stability_results = {}
    
    for m in stability_models:
        df = oofs[m]
        fold_macro_aucs = []
        target_stabilities = {}
        for f in range(5):
            f_mask = (df["fold"].values == f)
            f_aucs = []
            for t in PATHOLOGY_NAMES:
                f_y = (df[f_mask][f"true_{t}"].values >= 0.5).astype(int)
                f_score = df[f_mask][f"pred_{t}"].values
                f_aucs.append(roc_auc_score_np(f_y, f_score))
            fold_macro_aucs.append(float(np.mean(f_aucs)))
            
        for t in PATHOLOGY_NAMES:
            t_f_aucs = model_target_metrics[m][t]["fold_aucs"]
            std = float(np.std(t_f_aucs))
            mean = float(np.mean(t_f_aucs))
            f_min = float(np.min(t_f_aucs))
            f_max = float(np.max(t_f_aucs))
            target_stabilities[t] = {
                "mean": mean,
                "std": std,
                "min": f_min,
                "max": f_max,
                "range": f_max - f_min,
                "cv_coeff_var": (std / mean) if mean > 0 else 0.0
            }
            
        macro_std = float(np.std(fold_macro_aucs))
        macro_mean = float(np.mean(fold_macro_aucs))
        macro_min = float(np.min(fold_macro_aucs))
        macro_max = float(np.max(fold_macro_aucs))
        
        stability_results[m] = {
            "fold_macro_aucs": fold_macro_aucs,
            "macro_mean": macro_mean,
            "macro_std": macro_std,
            "macro_min": macro_min,
            "macro_max": macro_max,
            "macro_range": macro_max - macro_min,
            "targets": target_stabilities
        }

    with open(os.path.join(PROJECT_ROOT, "phase_r_b/rb_fold_stability.json"), "w") as f:
        json.dump(stability_results, f, indent=2)
    print("Saved rb_fold_stability.json")
    print("\nAll forensics computation completed successfully!")

if __name__ == "__main__":
    main()
