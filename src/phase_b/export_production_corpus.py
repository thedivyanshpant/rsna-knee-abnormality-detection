"""
Production Corpus Export Script for Phase B Step 3 Remediation.
Processes all 4,407 training reports across 12 targets (52,884 total target evaluations).
Emits complete 20-field provenance schema, decoupled soft targets, confidence scores,
and sample weights, along with corpus-level diagnostic summaries.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import os
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd

from phase_b.config import TARGETS
from phase_b.lexicon import extract_multilingual_concepts_non_destructive
from phase_b.pipeline import process_remediated_study_target

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA_DIR = os.environ.get("RSNA_DATA_ROOT", os.path.join(REPO_ROOT, "csv"))
OUTPUT_DIR = os.environ.get("PHASE_B_OUTPUT_DIR", os.path.join(REPO_ROOT, "labels", "production"))
TRAIN_CSV_PATH = os.environ.get("TRAIN_CSV_PATH", os.path.join(DATA_DIR, "train.csv"))
GOLD_CSV_PATH = os.environ.get("GOLD_CSV_PATH", os.path.join(DATA_DIR, "gold_studies.csv"))
OUTPUT_CORPUS_CSV = os.path.join(OUTPUT_DIR, "phase_b_step3_remediated_training_corpus.csv")
OUTPUT_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "phase_b_step3_remediated_training_corpus_summary.json")


def export_production_corpus():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading training data from {TRAIN_CSV_PATH}...")
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    total_studies = len(df_train)
    print(f"Loaded {total_studies} studies from training dataset.")

    print(f"Loading gold ground truth from {GOLD_CSV_PATH}...")
    df_gold = pd.read_csv(GOLD_CSV_PATH)
    gold_map = {}
    for _, row in df_gold.iterrows():
        uid = str(row["StudyInstanceUID"])
        for t in TARGETS:
            if t in row and pd.notna(row[t]):
                gold_map[(uid, t)] = int(row[t])
    print(f"Loaded {len(gold_map)} verified gold study-target ground truth labels.")

    records = []
    
    # Statistical counters
    target_stats = defaultdict(lambda: {
        "baseline_positives": 0,
        "step1_positives": 0,
        "step2_positives": 0,
        "step3_positives": 0,
        "final_positives": 0,
        "sum_prob": 0.0,
        "sum_conf": 0.0,
        "sum_weight": 0.0,
        "abstentions": 0,
        "total": 0
    })

    lang_counts = defaultdict(int)
    rule_counts = defaultdict(int)
    concept_counts = defaultdict(int)

    print("Processing all 4,407 studies across 12 targets (52,884 predictions)...")
    
    for idx, row in df_train.iterrows():
        if (idx + 1) % 500 == 0 or idx == total_studies - 1:
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{total_studies} studies ({(idx + 1) * 12}/52,884 predictions) [{elapsed:.1f}s]...")

        uid = str(row["StudyInstanceUID"])
        report = str(row["Report"]) if pd.notna(row["Report"]) else ""
        precomputed = extract_multilingual_concepts_non_destructive(report) if report else []

        for target in TARGETS:
            base_pred = int(row[target]) if pd.notna(row[target]) else 0
            gold_val = gold_map.get((uid, target), None)

            rec = process_remediated_study_target(
                study_uid=uid,
                target=target,
                baseline_pred=base_pred,
                baseline_conf=0.80,
                evidence=report[:250],
                full_report=report,
                gold_label=gold_val,
                precomputed_concepts=precomputed
            )

            records.append(rec)


            # Update stats
            ts = target_stats[target]
            ts["total"] += 1
            if rec["baseline_predicted_label"] == 1: ts["baseline_positives"] += 1
            if rec["step1_predicted_label"] == 1: ts["step1_positives"] += 1
            if rec["step2_predicted_label"] == 1: ts["step2_positives"] += 1
            if rec["step3_predicted_label"] == 1: ts["step3_positives"] += 1
            if rec["final_predicted_label"] == 1: ts["final_positives"] += 1
            ts["sum_prob"] += rec["label_probability"]
            ts["sum_conf"] += rec["confidence_score"]
            ts["sum_weight"] += rec["sample_weight"]
            if rec["abstain_flag"] == 1: ts["abstentions"] += 1

            lang_counts[rec["detected_language"]] += 1
            rule_counts[rec["step3_rule_applied"]] += 1
            if rec["matched_canonical_concept"] != "None":
                concept_counts[rec["matched_canonical_concept"]] += 1

    df_out = pd.DataFrame(records)
    print(f"Writing production corpus table to {OUTPUT_CORPUS_CSV}...")
    df_out.to_csv(OUTPUT_CORPUS_CSV, index=False)
    print(f"Successfully saved {len(df_out)} records to {OUTPUT_CORPUS_CSV}.")


    # Assemble summary JSON
    summary_data = {
        "metadata": {
            "pipeline": "Phase B Step 3 Remediation Production Export",
            "source_corpus": TRAIN_CSV_PATH,
            "total_studies": total_studies,
            "total_target_predictions": len(df_out),
            "total_gold_evaluations": len(gold_map),
            "output_csv": OUTPUT_CORPUS_CSV,
            "execution_time_seconds": round(time.time() - start_time, 2)
        },
        "target_level_statistics": {},
        "language_distribution": dict(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)),
        "rule_activation_top20": dict(sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
        "matched_concept_top20": dict(sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
        "aggregate_metrics": {
            "mean_label_probability": float(df_out["label_probability"].mean()),
            "mean_confidence_score": float(df_out["confidence_score"].mean()),
            "mean_sample_weight": float(df_out["sample_weight"].mean()),
            "total_abstentions": int((df_out["abstain_flag"] == 1).sum()),
            "overall_abstention_rate": float((df_out["abstain_flag"] == 1).mean())
        }
    }

    for target in TARGETS:
        ts = target_stats[target]
        n = ts["total"]
        summary_data["target_level_statistics"][target] = {
            "total_samples": n,
            "baseline_prevalence": round(ts["baseline_positives"] / n, 4),
            "step1_prevalence": round(ts["step1_positives"] / n, 4),
            "step2_prevalence": round(ts["step2_positives"] / n, 4),
            "step3_prevalence": round(ts["step3_positives"] / n, 4),
            "final_prevalence": round(ts["final_positives"] / n, 4),
            "mean_probability": round(ts["sum_prob"] / n, 4),
            "mean_confidence": round(ts["sum_conf"] / n, 4),
            "mean_sample_weight": round(ts["sum_weight"] / n, 4),
            "abstention_count": ts["abstentions"],
            "abstention_rate": round(ts["abstentions"] / n, 4)
        }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary statistics written to {OUTPUT_SUMMARY_JSON}.")

    print("\n" + "=" * 90)
    print("PRODUCTION CORPUS EXPORT SUMMARY (4,407 STUDIES x 12 TARGETS)")
    print("=" * 90)

    print(f"{'Target':<18} | {'Base Prev':<10} | {'S1 Prev':<10} | {'S2 Prev':<10} | {'Rem Prev':<10} | {'Mean Prob':<10} | {'Mean Weight':<10}")
    print("-" * 90)
    for target in TARGETS:
        stats = summary_data["target_level_statistics"][target]
        print(f"{target:<18} | {stats['baseline_prevalence']:<10.4f} | {stats['step1_prevalence']:<10.4f} | {stats['step2_prevalence']:<10.4f} | {stats['final_prevalence']:<10.4f} | {stats['mean_probability']:<10.4f} | {stats['mean_sample_weight']:<10.4f}")
    print("=" * 90)


if __name__ == "__main__":
    export_production_corpus()
