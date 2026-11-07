"""
Evaluation and Comparison Engine for Phase B Step 3 Remediation.
Evaluates on Gold-58 verification dataset and generates all comparison artifacts and transition audits.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phase_b.config import TARGETS
from phase_b.pipeline import run_remediated_pipeline_batch

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
OUTPUT_DIR = os.environ.get("PHASE_B_OUTPUT_DIR", os.path.join(REPO_ROOT, "labels", "gold58"))
PHASE_A_DIR = os.environ.get("PHASE_A_DIR", os.path.join(REPO_ROOT, "labels", "gold58"))
DATA_DIR = os.environ.get("RSNA_DATA_ROOT", os.path.join(REPO_ROOT, "csv"))


def run_remediation_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Phase A baseline predictions and reference data
    pred_a_path = os.path.join(PHASE_A_DIR, "phase_a_baseline_predictions_gold58.csv")
    if not os.path.exists(pred_a_path):
        pred_a_path = os.path.join(PHASE_A_DIR, "predictions_gold58.csv")
    df_preds_a = pd.read_csv(pred_a_path)
    df_gold = pd.read_csv(f"{DATA_DIR}/gold_studies.csv").set_index("StudyInstanceUID")

    records_in = df_preds_a.to_dict(orient="records")

    # 2. Run Remediated Phase B Pipeline
    records_out = run_remediated_pipeline_batch(records_in)
    df_preds_b = pd.DataFrame(records_out)

    # Save Remediated Step 3 predictions
    csv_preds = f"{OUTPUT_DIR}/phase_b_step3_remediated_predictions_gold58.csv"
    df_preds_b.to_csv(csv_preds, index=False)
    print(f"Saved {csv_preds} ({len(df_preds_b)} records)")

    # 3. Identify Error Cases in Remediated Step 3
    error_cases_b = []
    for _, r in df_preds_b.iterrows():
        g = int(r["gold_label"])
        p = int(r["final_predicted_label"])
        if g != p:
            err_rec = dict(r)
            err_rec["is_false_positive"] = (g == 0 and p == 1)
            err_rec["is_false_negative"] = (g == 1 and p == 0)
            error_cases_b.append(err_rec)

    df_errors_b = pd.DataFrame(error_cases_b)
    csv_errors = f"{OUTPUT_DIR}/phase_b_step3_remediated_error_cases.csv"
    df_errors_b.to_csv(csv_errors, index=False)
    print(f"Saved {csv_errors} ({len(df_errors_b)} errors)")

    # 4. Compute Per-Target Metrics
    metrics_list = []
    cm_list = []

    for t in TARGETS:
        sub = df_preds_b[df_preds_b["target"] == t]
        y_true = sub["gold_label"].values
        y_pred = sub["final_predicted_label"].values
        y_score = sub["label_probability"].values

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
    df_metrics.to_csv(f"{OUTPUT_DIR}/phase_b_step3_remediated_metrics.csv", index=False)

    df_cm = pd.DataFrame(cm_list)
    df_cm.to_csv(f"{OUTPUT_DIR}/phase_b_step3_remediated_confusion_matrices.csv", index=False)

    # 5. Build Transition Audit (Baseline -> Step 1 -> Step 2 -> Remediated Step 3)
    transition_rows = []
    for _, r in df_preds_b.iterrows():
        uid = r["StudyInstanceUID"]
        t = r["target"]
        g = int(r["gold_label"])
        p_base = int(r["baseline_predicted_label"])
        p_step1 = int(r["step1_predicted_label"])
        p_step2 = int(r["step2_predicted_label"])
        p_rem = int(r["final_predicted_label"])

        step2_correct = (p_step2 == g)
        rem_correct = (p_rem == g)

        if not step2_correct and rem_correct:
            trans_class = "Corrected error"
        elif step2_correct and not rem_correct:
            trans_class = "Correct -> incorrect regression"
        elif not step2_correct and not rem_correct:
            trans_class = "Unchanged incorrect"
        else:
            trans_class = "Unchanged correct"

        transition_rows.append({
            "StudyInstanceUID": uid,
            "target": t,
            "gold_label": g,
            "baseline_predicted": p_base,
            "step1_predicted": p_step1,
            "step2_predicted": p_step2,
            "remediated_predicted": p_rem,
            "step1_rule": r["step1_rule_applied"],
            "step2_rule": r["step2_rule_applied"],
            "step3_rule": r["step3_rule_applied"],
            "matched_concept": r["matched_canonical_concept"],
            "detected_language": r["detected_language"],
            "transition_class": trans_class,
            "evidence": r["evidence_text"]
        })

    df_transitions = pd.DataFrame(transition_rows)
    df_transitions.to_csv(f"{OUTPUT_DIR}/phase_b_step3_remediated_transitions.csv", index=False)

    # 6. Overall Summary
    total_preds = len(df_preds_b)
    total_correct = int(np.sum(df_preds_b["gold_label"] == df_preds_b["final_predicted_label"]))
    overall_acc = total_correct / total_preds

    exact_matches = 0
    for uid in df_gold.index:
        sub = df_preds_b[df_preds_b["StudyInstanceUID"] == uid]
        if np.all(sub["gold_label"].values == sub["final_predicted_label"].values):
            exact_matches += 1

    exact_study_acc = exact_matches / len(df_gold)

    summary = {
        "experiment_name": "Phase B Step 3 Remediation: Non-Destructive Scoped Parsing, Multilingual Severity Gating & Decoupled Soft Targets",
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

    with open(f"{OUTPUT_DIR}/phase_b_step3_remediated_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/phase_b_step3_remediated_summary.json")

    # 7. Print Comparative Table
    df_metrics_a = pd.read_csv(f"{PHASE_A_DIR}/metrics_per_target.csv").set_index("target")
    df_metrics_step1 = pd.read_csv(f"{OUTPUT_DIR}/phase_b_step1_metrics.csv").set_index("target")
    df_metrics_step2 = pd.read_csv(f"{OUTPUT_DIR}/phase_b_step2_metrics.csv").set_index("target")
    df_metrics_rem = df_metrics.set_index("target")

    comp_rows = []
    for t in TARGETS:
        acc_a = df_metrics_a.loc[t, "Accuracy"]
        acc_s1 = df_metrics_step1.loc[t, "Accuracy"]
        acc_s2 = df_metrics_step2.loc[t, "Accuracy"]
        acc_rem = df_metrics_rem.loc[t, "Accuracy"]

        f1_a = df_metrics_a.loc[t, "F1"]
        f1_s1 = df_metrics_step1.loc[t, "F1"]
        f1_s2 = df_metrics_step2.loc[t, "F1"]
        f1_rem = df_metrics_rem.loc[t, "F1"]

        fp_rem = df_metrics_rem.loc[t, "FP"]
        fn_rem = df_metrics_rem.loc[t, "FN"]

        comp_rows.append({
            "Target": t,
            "Phase A Acc": acc_a,
            "Step 1 Acc": acc_s1,
            "Step 2 Acc": acc_s2,
            "Remediated Acc": acc_rem,
            "Δ Acc (Rem-S2)": acc_rem - acc_s2,
            "Remediated F1": f1_rem,
            "Rem FP": fp_rem,
            "Rem FN": fn_rem
        })

    df_comp = pd.DataFrame(comp_rows)
    print("\n===================================================================================================")
    print("PHASE A vs PHASE B STEP 1 vs STEP 2 vs REMEDIATED STEP 3 ACCURACY & F1")
    print("===================================================================================================")
    print(df_comp.to_string(index=False))

    return summary, df_comp, df_transitions


if __name__ == "__main__":
    run_remediation_evaluation()
