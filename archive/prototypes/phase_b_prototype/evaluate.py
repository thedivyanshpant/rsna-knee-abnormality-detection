"""
Evaluation and Comparison Engine for Phase B Step 3 vs Phase B Step 2 vs Phase B Step 1 vs Phase A Baseline.
Evaluates on Gold-58 verification dataset and generates all comparison artifacts and transition audits.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phase_b.config import TARGETS
from phase_b.pipeline import run_phase_b_pipeline_batch

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
OUTPUT_DIR = os.environ.get("PHASE_B_OUTPUT_DIR", os.path.join(REPO_ROOT, "results/archive"))
PHASE_A_DIR = os.environ.get("PHASE_A_DIR", os.path.join(REPO_ROOT, "labels/gold58"))
DATA_DIR = os.environ.get("RSNA_DATA_ROOT", os.path.join(REPO_ROOT, "csv"))


def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Phase A baseline predictions and reference data
    df_preds_a = pd.read_csv(f"{PHASE_A_DIR}/predictions_gold58.csv")
    df_gold = pd.read_csv(f"{DATA_DIR}/gold_studies.csv").set_index("StudyInstanceUID")
    df_reports = pd.read_csv(f"{DATA_DIR}/golden_reports.csv").set_index("StudyInstanceUID")

    records_in = df_preds_a.to_dict(orient="records")
    
    # 2. Run Phase B Pipeline (Step 1 + Step 2 + Step 3)
    records_out = run_phase_b_pipeline_batch(records_in)
    df_preds_b = pd.DataFrame(records_out)
    
    # Save Step 3 predictions
    df_preds_b.to_csv(f"{OUTPUT_DIR}/phase_b_step3_predictions_gold58.csv", index=False)
    print(f"Saved {OUTPUT_DIR}/phase_b_step3_predictions_gold58.csv ({len(df_preds_b)} records)")

    # 3. Identify Error Cases in Phase B Step 3
    error_cases_b = []
    for _, r in df_preds_b.iterrows():
        g = int(r["gold_label"])
        p = int(r["step3_predicted_label"])
        if g != p:
            err_rec = dict(r)
            err_rec["is_false_positive"] = (g == 0 and p == 1)
            err_rec["is_false_negative"] = (g == 1 and p == 0)
            error_cases_b.append(err_rec)
            
    df_errors_b = pd.DataFrame(error_cases_b)
    df_errors_b.to_csv(f"{OUTPUT_DIR}/phase_b_step3_error_cases.csv", index=False)
    print(f"Saved {OUTPUT_DIR}/phase_b_step3_error_cases.csv ({len(df_errors_b)} errors)")

    # 4. Compute Per-Target Metrics for Step 3
    metrics_list = []
    cm_list = []
    
    for t in TARGETS:
        sub = df_preds_b[df_preds_b["target"] == t]
        y_true = sub["gold_label"].values
        y_pred = sub["step3_predicted_label"].values
        y_score = sub["step3_probability"].values
        
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        try:
            roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
        except Exception:
            roc_auc = np.nan
            
        metrics_list.append({
            "target": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": prec, "Recall": rec, "Specificity": spec,
            "F1": f1, "Accuracy": acc, "ROC_AUC": roc_auc
        })
        cm_list.append({"target": t, "TN": tn, "FP": fp, "FN": fn, "TP": tp})

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(f"{OUTPUT_DIR}/phase_b_step3_metrics.csv", index=False)
    print(f"Saved {OUTPUT_DIR}/phase_b_step3_metrics.csv")

    df_cm = pd.DataFrame(cm_list)
    df_cm.to_csv(f"{OUTPUT_DIR}/phase_b_step3_confusion_matrices.csv", index=False)
    print(f"Saved {OUTPUT_DIR}/phase_b_step3_confusion_matrices.csv")

    # 5. Build Transition Audit (Baseline -> Step 1 -> Step 2 -> Step 3)
    transition_rows = []
    for _, r in df_preds_b.iterrows():
        uid = r["StudyInstanceUID"]
        t = r["target"]
        g = int(r["gold_label"])
        p_base = int(r["baseline_predicted_label"])
        p_step1 = int(r["step1_predicted_label"])
        p_step2 = int(r["step2_predicted_label"])
        p_step3 = int(r["step3_predicted_label"])
        
        # Determine transition state from Step 2 -> Step 3
        base_correct = (p_base == g)
        step1_correct = (p_step1 == g)
        step2_correct = (p_step2 == g)
        step3_correct = (p_step3 == g)
        
        if not step2_correct and step3_correct:
            trans_class = "Corrected error"
        elif step2_correct and not step3_correct:
            trans_class = "Correct -> incorrect regression"
        elif not step2_correct and not step3_correct:
            trans_class = "Unchanged incorrect"
        else: # step2_correct and step3_correct
            trans_class = "Unchanged correct"
            
        transition_rows.append({
            "StudyInstanceUID": uid,
            "target": t,
            "gold_label": g,
            "baseline_predicted": p_base,
            "step1_predicted": p_step1,
            "step2_predicted": p_step2,
            "step3_predicted": p_step3,
            "step1_rule": r["step1_rule_applied"],
            "step2_rule": r["step2_rule_applied"],
            "step3_rule": r["step3_rule_applied"],
            "step3_concept": r["step3_canonical_concept"],
            "step3_language": r["step3_language"],
            "step3_reason": r["step3_reason"],
            "transition_class": trans_class,
            "evidence": r["extracted_evidence"]
        })
        
    df_transitions = pd.DataFrame(transition_rows)
    df_transitions.to_csv(f"{OUTPUT_DIR}/phase_b_step3_transitions.csv", index=False)
    print(f"Saved {OUTPUT_DIR}/phase_b_step3_transitions.csv ({len(df_transitions)} rows)")

    # 6. Overall Summary
    total_preds = len(df_preds_b)
    total_correct = int(np.sum(df_preds_b["gold_label"] == df_preds_b["step3_predicted_label"]))
    overall_acc = total_correct / total_preds
    
    exact_matches = 0
    for uid in df_gold.index:
        sub = df_preds_b[df_preds_b["StudyInstanceUID"] == uid]
        if np.all(sub["gold_label"].values == sub["step3_predicted_label"].values):
            exact_matches += 1
            
    exact_study_acc = exact_matches / len(df_gold)
    
    summary = {
        "experiment_name": "Phase B Step 3: Multilingual Clinical Concept Expansion & Normalization",
        "dataset": "RSNA Knee Abnormality Detection (Gold-58 subset)",
        "total_studies": len(df_gold),
        "total_predictions": total_preds,
        "total_correct": total_correct,
        "overall_accuracy": overall_acc,
        "overall_accuracy_pct": round(overall_acc * 100, 2),
        "exact_study_matches": exact_matches,
        "exact_study_accuracy": exact_study_acc,
        "exact_study_accuracy_pct": round(exact_study_acc * 100, 2),
        "total_errors": len(df_errors_b),
        "total_false_positives": int(df_errors_b["is_false_positive"].sum()),
        "total_false_negatives": int(df_errors_b["is_false_negative"].sum()),
        "macro_f1": float(df_metrics["F1"].mean()),
        "macro_precision": float(df_metrics["Precision"].mean()),
        "macro_recall": float(df_metrics["Recall"].mean()),
        "macro_specificity": float(df_metrics["Specificity"].mean()),
        "transitions_from_step2": {
            "corrected_errors": int(np.sum(df_transitions["transition_class"] == "Corrected error")),
            "regressions": int(np.sum(df_transitions["transition_class"] == "Correct -> incorrect regression")),
            "unchanged_correct": int(np.sum(df_transitions["transition_class"] == "Unchanged correct")),
            "unchanged_incorrect": int(np.sum(df_transitions["transition_class"] == "Unchanged incorrect")),
        },
        "per_target_metrics": metrics_list
    }

    with open(f"{OUTPUT_DIR}/phase_b_step3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/phase_b_step3_summary.json")

    # 7. Print Comparative Table (Phase A vs Step 1 vs Step 2 vs Step 3)
    df_metrics_a = pd.read_csv(f"{PHASE_A_DIR}/metrics_per_target.csv").set_index("target")
    df_metrics_step1 = pd.read_csv(f"{OUTPUT_DIR}/phase_b_step1_metrics.csv").set_index("target")
    df_metrics_step2 = pd.read_csv(f"{OUTPUT_DIR}/phase_b_step2_metrics.csv").set_index("target")
    df_metrics_step3 = df_metrics.set_index("target")

    comp_rows = []
    for t in TARGETS:
        acc_a = df_metrics_a.loc[t, "Accuracy"]
        acc_s1 = df_metrics_step1.loc[t, "Accuracy"]
        acc_s2 = df_metrics_step2.loc[t, "Accuracy"]
        acc_s3 = df_metrics_step3.loc[t, "Accuracy"]
        
        f1_a = df_metrics_a.loc[t, "F1"]
        f1_s1 = df_metrics_step1.loc[t, "F1"]
        f1_s2 = df_metrics_step2.loc[t, "F1"]
        f1_s3 = df_metrics_step3.loc[t, "F1"]
        
        fp_s3 = df_metrics_step3.loc[t, "FP"]
        fn_s3 = df_metrics_step3.loc[t, "FN"]
        
        comp_rows.append({
            "Target": t,
            "Phase A Acc": acc_a,
            "Step 1 Acc": acc_s1,
            "Step 2 Acc": acc_s2,
            "Step 3 Acc": acc_s3,
            "Δ Acc (S3-S2)": acc_s3 - acc_s2,
            "Step 3 F1": f1_s3,
            "Step 3 FP": fp_s3,
            "Step 3 FN": fn_s3
        })
        
    df_comp = pd.DataFrame(comp_rows)
    print("\n===================================================================================================")
    print("PHASE A vs PHASE B STEP 1 vs STEP 2 vs STEP 3 ACCURACY & F1")
    print("===================================================================================================")
    print(df_comp.to_string(index=False))

    return summary, df_comp, df_transitions


if __name__ == "__main__":
    run_evaluation()
