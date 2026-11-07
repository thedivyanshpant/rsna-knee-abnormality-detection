"""
AUDIT 3: Gold-58 Development / Validation Separation & Holdout Evaluation.
Performs a deterministic 38 Dev / 20 Holdout split of the Gold-58 dataset.
Evaluates retrospective performance vs true held-out performance,
and analyzes rule-leakage and Gold-study provenance.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from phase_b.config import TARGETS
from phase_b.pipeline import process_study_target
from phase_b.multilingual import COMPILED_LEXICON

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
OUTPUT_DIR = os.environ.get("PHASE_B_OUTPUT_DIR", os.path.join(REPO_ROOT, "results/archive"))
PHASE_A_DIR = os.environ.get("PHASE_A_DIR", os.path.join(REPO_ROOT, "labels/gold58"))
DATA_DIR = os.environ.get("RSNA_DATA_ROOT", os.path.join(REPO_ROOT, "csv"))

def run_audit3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Gold-58 Data & Phase A Predictions
    df_preds_a = pd.read_csv(f"{PHASE_A_DIR}/predictions_gold58.csv")
    df_gold = pd.read_csv(f"{DATA_DIR}/gold_studies.csv").set_index("StudyInstanceUID")
    all_uids = sorted(df_gold.index.tolist())

    # 2. Deterministic 38 / 20 Split using fixed seed (seed=42)
    # Stratify by language distribution if possible or deterministic sorted slice
    rng = np.random.RandomState(42)
    shuffled_uids = list(all_uids)
    rng.shuffle(shuffled_uids)
    
    dev_uids = sorted(shuffled_uids[:38])
    holdout_uids = sorted(shuffled_uids[38:])

    print(f"Total Studies: {len(all_uids)}")
    print(f"Development Set: {len(dev_uids)} studies ({len(dev_uids)*12} target evaluations)")
    print(f"Holdout Validation Set: {len(holdout_uids)} studies ({len(holdout_uids)*12} target evaluations)")

    # 3. Evaluate Step 3 on Dev (38) and Holdout (20)
    records_in = df_preds_a.to_dict(orient="records")
    
    dev_records = []
    holdout_records = []

    for r in records_in:
        uid = r["StudyInstanceUID"]
        res = process_study_target(
            study_uid=uid,
            target=r["target"],
            baseline_pred=int(r["predicted_label"]),
            baseline_conf=float(r["confidence"]),
            evidence=str(r["extracted_evidence"]),
            full_report=str(r["report_text"]),
            gold_label=int(r["gold_label"])
        )
        
        # Determine split
        split_name = "dev" if uid in dev_uids else "holdout"
        res["split"] = split_name
        
        g = int(r["gold_label"])
        p_base = res["baseline_predicted_label"]
        p_s1 = res["step1_predicted_label"]
        p_s2 = res["step2_predicted_label"]
        p_s3 = res["step3_predicted_label"]

        res["base_correct"] = (p_base == g)
        res["s2_correct"] = (p_s2 == g)
        res["s3_correct"] = (p_s3 == g)
        res["is_s3_fix"] = (p_s2 != g and p_s3 == g)
        res["is_s3_regression"] = (p_s2 == g and p_s3 != g)

        if split_name == "dev":
            dev_records.append(res)
        else:
            holdout_records.append(res)

    df_dev = pd.DataFrame(dev_records)
    df_holdout = pd.DataFrame(holdout_records)
    df_all = pd.concat([df_dev, df_holdout], ignore_index=True)

    # Save holdout results CSV
    csv_path = f"{OUTPUT_DIR}/phase_b_step3_holdout_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df_all)} rows)")

    # 4. Calculate Metrics for Dev and Holdout Sets
    def compute_metrics(df_sub, set_name):
        y_true = df_sub["gold_label"].values
        y_pred = df_sub["step3_predicted_label"].values
        y_score = df_sub["step3_probability"].values

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # Exact study match
        unique_uids = df_sub["StudyInstanceUID"].unique()
        exact_study_matches = 0
        for uid in unique_uids:
            s_df = df_sub[df_sub["StudyInstanceUID"] == uid]
            if np.all(s_df["gold_label"].values == s_df["step3_predicted_label"].values):
                exact_study_matches += 1
        exact_study_acc = exact_study_matches / len(unique_uids)

        # Per-target metrics for macro F1
        target_f1s = []
        for t in TARGETS:
            t_sub = df_sub[df_sub["target"] == t]
            yt = t_sub["gold_label"].values
            yp = t_sub["step3_predicted_label"].values
            t_tp = int(np.sum((yt == 1) & (yp == 1)))
            t_fp = int(np.sum((yt == 0) & (yp == 1)))
            t_fn = int(np.sum((yt == 1) & (yp == 0)))
            t_pr = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
            t_rc = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
            t_f1 = 2 * t_pr * t_rc / (t_pr + t_rc) if (t_pr + t_rc) > 0 else 0.0
            target_f1s.append(t_f1)

        macro_f1 = float(np.mean(target_f1s))

        fixes = int(df_sub["is_s3_fix"].sum())
        regs = int(df_sub["is_s3_regression"].sum())

        return {
            "set_name": set_name,
            "total_studies": len(unique_uids),
            "total_evaluations": len(df_sub),
            "accuracy": acc,
            "accuracy_pct": round(acc * 100, 2),
            "macro_f1": round(macro_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "exact_study_matches": exact_study_matches,
            "exact_study_accuracy_pct": round(exact_study_acc * 100, 2),
            "step2_to_step3_fixes": fixes,
            "step2_to_step3_regressions": regs
        }

    dev_metrics = compute_metrics(df_dev, "Development Set (38 studies)")
    holdout_metrics = compute_metrics(df_holdout, "Holdout Set (20 studies)")
    overall_metrics = compute_metrics(df_all, "Full Gold-58 (58 studies)")

    # 5. Rule Provenance Breakdown for Holdout
    # Check which holdout studies had rules fire
    holdout_rule_firings = []
    for _, r in df_holdout[df_holdout["step3_rule_applied"] != "STEP2_PRESERVED"].iterrows():
        holdout_rule_firings.append({
            "StudyInstanceUID": r["StudyInstanceUID"],
            "target": r["target"],
            "gold_label": r["gold_label"],
            "step2_predicted": r["step2_predicted_label"],
            "step3_predicted": r["step3_predicted_label"],
            "step3_rule": r["step3_rule_applied"],
            "step3_concept": r["step3_canonical_concept"],
            "step3_language": r["step3_language"],
            "is_fix": r["is_s3_fix"],
            "is_regression": r["is_s3_regression"],
            "evidence": r["extracted_evidence"][:100]
        })

    # Summary JSON
    summary_data = {
        "audit_name": "AUDIT 3: Gold-58 Development / Validation Separation",
        "split_method": "Deterministic RandomState(seed=42)",
        "dev_studies_count": len(dev_uids),
        "holdout_studies_count": len(holdout_uids),
        "dev_study_uids": dev_uids,
        "holdout_study_uids": holdout_uids,
        "development_metrics": dev_metrics,
        "holdout_metrics": holdout_metrics,
        "overall_gold58_metrics": overall_metrics,
        "holdout_active_rules_fired": holdout_rule_firings
    }

    json_path = f"{OUTPUT_DIR}/phase_b_step3_holdout_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved {json_path}")

    print("\n" + "="*80)
    print("DEVELOPMENT (38) vs HOLDOUT (20) PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Dev Set (38 Studies):    Accuracy = {dev_metrics['accuracy_pct']}% | Macro F1 = {dev_metrics['macro_f1']} | Fixes = {dev_metrics['step2_to_step3_fixes']} | Regressions = {dev_metrics['step2_to_step3_regressions']}")
    print(f"Holdout Set (20 Studies): Accuracy = {holdout_metrics['accuracy_pct']}% | Macro F1 = {holdout_metrics['macro_f1']} | Fixes = {holdout_metrics['step2_to_step3_fixes']} | Regressions = {holdout_metrics['step2_to_step3_regressions']}")
    print(f"Full Gold-58:            Accuracy = {overall_metrics['accuracy_pct']}% | Macro F1 = {overall_metrics['macro_f1']} | Fixes = {overall_metrics['step2_to_step3_fixes']} | Regressions = {overall_metrics['step2_to_step3_regressions']}")
    print("="*80)

    return summary_data

if __name__ == "__main__":
    run_audit3()
