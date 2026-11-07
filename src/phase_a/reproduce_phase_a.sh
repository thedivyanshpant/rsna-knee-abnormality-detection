#!/bin/bash
# Phase A - Baseline Extractor Reproduction Script
# RSNA Knee Abnormality Detection Project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${RSNA_DATA_ROOT:-${REPO_ROOT}/csv}"
OUT_DIR="${PHASE_A_OUTPUT_DIR:-${REPO_ROOT}/results/phase_a_baseline}"

mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo "Phase A Reference Competitor Extractor Evaluation"
echo "=========================================================="

python3 - << 'INNER'
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

repo_root = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.getcwd())))
out_dir = os.environ.get("PHASE_A_OUTPUT_DIR", os.path.join(repo_root, "results", "phase_a_baseline"))
data_dir = os.environ.get("RSNA_DATA_ROOT", os.path.join(repo_root, "csv"))
os.makedirs(out_dir, exist_ok=True)

gold_df = pd.read_csv(f"{data_dir}/gold_studies.csv").set_index("StudyInstanceUID")
reports_df = pd.read_csv(f"{data_dir}/golden_reports.csv").set_index("StudyInstanceUID")

targets = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

with open(f"{data_dir}/llm_results/gold58_predictions.jsonl") as f:
    preds = [json.loads(l) for l in f if l.strip()]

print(f"Loaded {len(preds)} prediction records and {len(gold_df)} gold studies.")

records = []
error_cases = []

for item in preds:
    uid = item["StudyInstanceUID"]
    report_text = reports_df.loc[uid, "Report"] if uid in reports_df.index else ""
    p_dict = item["prediction"]
    labels = p_dict["labels"]
    confs = p_dict.get("confidence", {})
    evids = p_dict.get("evidence", {})
    
    for t in targets:
        gold_val = int(gold_df.loc[uid, t])
        pred_val = int(labels[t])
        conf_val = float(confs.get(t, 0.0))
        evid_val = str(evids.get(t, "")).strip()
        prob_score = conf_val if pred_val == 1 else (1.0 - conf_val)
        
        rec = {
            "StudyInstanceUID": uid,
            "target": t,
            "gold_label": gold_val,
            "predicted_label": pred_val,
            "prediction_probability": prob_score,
            "confidence": conf_val,
            "extractor": "reference_baseline_extractor",
            "extracted_evidence": evid_val,
            "report_text": report_text
        }
        records.append(rec)
        
        if gold_val != pred_val:
            err_rec = dict(rec)
            err_rec["is_false_positive"] = (gold_val == 0 and pred_val == 1)
            err_rec["is_false_negative"] = (gold_val == 1 and pred_val == 0)
            error_cases.append(err_rec)

df_preds = pd.DataFrame(records)
df_preds.to_csv(f"{out_dir}/predictions_gold58.csv", index=False)

metrics_list = []
cm_list = []
for t in targets:
    sub = df_preds[df_preds["target"] == t]
    y_true = sub["gold_label"].values
    y_pred = sub["predicted_label"].values
    y_score = sub["prediction_probability"].values
    
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
        "F1": f1, "Accuracy": acc, "ROC_AUC": roc_auc,
        "ROC_AUC_Note": "Discrete heuristic scores (not calibrated)"
    })
    cm_list.append({"target": t, "TN": tn, "FP": fp, "FN": fn, "TP": tp})

pd.DataFrame(metrics_list).to_csv(f"{out_dir}/metrics_per_target.csv", index=False)
pd.DataFrame(cm_list).to_csv(f"{out_dir}/confusion_matrices.csv", index=False)

print(f"Saved all output files in {out_dir}")
INNER
