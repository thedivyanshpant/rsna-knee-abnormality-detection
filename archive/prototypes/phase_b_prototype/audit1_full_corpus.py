"""
AUDIT 1: Full 4,407-Report Activation Analysis for Phase B Step 3.
Evaluates all 4,407 radiology reports in the RSNA Knee training dataset.
Analyzes concept activations, rule firing frequencies, linguistic distributions,
and target prevalence shift across Phase A Baseline -> Step 1 -> Step 2 -> Step 3.
"""

import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd

from phase_b.config import TARGETS
from phase_b.multilingual import (
    extract_multilingual_concepts,
    get_multilingual_evidence_for_target,
    COMPILED_LEXICON,
    ConceptRecord
)
from phase_b.assertion import check_scoped_negative_assertion
from phase_b.anatomy import evaluate_compartment_scoping
from phase_b.severity import apply_clinical_severity_gates
from phase_b.pipeline import process_study_target

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
OUTPUT_DIR = os.environ.get("PHASE_B_OUTPUT_DIR", os.path.join(REPO_ROOT, "results/archive"))
DATA_PATH = os.environ.get("TRAIN_CSV_PATH", os.path.join(REPO_ROOT, "csv/train.csv"))

def run_audit1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading full corpus from {DATA_PATH}...")
    df_train = pd.read_csv(DATA_PATH)
    total_reports = len(df_train)
    print(f"Total reports loaded: {total_reports}")

    # 1. Concept-Level Activation Tracking
    # Track statistics for each canonical concept and language in the lexicon
    concept_stats = defaultdict(lambda: {
        "activated_reports": 0,
        "unique_studies": set(),
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "example_spans": [],
        "targets": set()
    })

    study_concepts_count = defaultdict(int)

    # 2. Pipeline-Level Predictions across 4,407 studies x 12 targets = 52,884 predictions
    pipeline_records = []
    
    # Track target positive counts
    target_positives_base = defaultdict(int)
    target_positives_s1 = defaultdict(int)
    target_positives_s2 = defaultdict(int)
    target_positives_s3 = defaultdict(int)

    # Rule activation counts
    step3_rules_activated = defaultdict(lambda: {
        "count": 0,
        "unique_studies": set(),
        "target": "",
        "language": "",
        "examples": []
    })

    print("Running concept extraction and 12-target pipeline over all 4,407 reports...")
    
    for idx, row in df_train.iterrows():
        uid = str(row["StudyInstanceUID"])
        report = str(row["Report"]) if pd.notna(row["Report"]) else ""

        # Extract all concepts in report
        concepts = extract_multilingual_concepts(report)
        seen_concepts_in_study = set()

        for c in concepts:
            key = (c.canonical_concept, c.language)
            concept_stats[key]["activated_reports"] += 1
            concept_stats[key]["unique_studies"].add(uid)
            if c.polarity == "positive":
                concept_stats[key]["positive_count"] += 1
            elif c.polarity == "negative":
                concept_stats[key]["negative_count"] += 1
            else:
                concept_stats[key]["neutral_count"] += 1

            for t in c.target_candidates:
                concept_stats[key]["targets"].add(t)

            if len(concept_stats[key]["example_spans"]) < 5:
                concept_stats[key]["example_spans"].append(c.evidence_text[:120])
            
            seen_concepts_in_study.add(c.canonical_concept)

        study_concepts_count[len(seen_concepts_in_study)] += 1

        # Run 12-target pipeline
        for target in TARGETS:
            base_pred = int(row[target]) if pd.notna(row[target]) else 0
            
            # Run through process_study_target
            res = process_study_target(
                study_uid=uid,
                target=target,
                baseline_pred=base_pred,
                baseline_conf=0.8,
                evidence=report[:200],  # using report as available evidence
                full_report=report,
                gold_label=None
            )

            p_base = res["baseline_predicted_label"]
            p_s1 = res["step1_predicted_label"]
            p_s2 = res["step2_predicted_label"]
            p_s3 = res["step3_predicted_label"]

            if p_base == 1: target_positives_base[target] += 1
            if p_s1 == 1: target_positives_s1[target] += 1
            if p_s2 == 1: target_positives_s2[target] += 1
            if p_s3 == 1: target_positives_s3[target] += 1

            # Track Step 3 active rule firing
            s3_rule = res["step3_rule_applied"]
            if s3_rule != "STEP2_PRESERVED":
                step3_rules_activated[s3_rule]["count"] += 1
                step3_rules_activated[s3_rule]["unique_studies"].add(uid)
                step3_rules_activated[s3_rule]["target"] = target
                step3_rules_activated[s3_rule]["language"] = res["step3_language"]
                if len(step3_rules_activated[s3_rule]["examples"]) < 3:
                    step3_rules_activated[s3_rule]["examples"].append(res["extracted_evidence"][:120])

    print("Concept extraction and pipeline run completed.")

    # 3. Format Output Audit CSV
    audit_rows = []
    for (canonical, lang), data in concept_stats.items():
        n_act = data["activated_reports"]
        n_studies = len(data["unique_studies"])
        pct = (n_studies / total_reports) * 100.0
        
        pol_desc = "mixed"
        if data["positive_count"] > 0 and data["negative_count"] == 0:
            pol_desc = "positive"
        elif data["negative_count"] > 0 and data["positive_count"] == 0:
            pol_desc = "negative"
        elif data["neutral_count"] > 0 and data["positive_count"] == 0 and data["negative_count"] == 0:
            pol_desc = "neutral"

        audit_rows.append({
            "concept_id": canonical,
            "language": lang,
            "target_candidates": "; ".join(sorted(list(data["targets"]))),
            "activated_reports_count": n_act,
            "unique_studies_count": n_studies,
            "pct_of_corpus": round(pct, 3),
            "polarity_distribution": f"pos={data['positive_count']}, neg={data['negative_count']}, neu={data['neutral_count']}",
            "dominant_polarity": pol_desc,
            "example_evidence_spans": " || ".join(data["example_spans"][:3])
        })

    df_audit = pd.DataFrame(audit_rows).sort_values(by="unique_studies_count", ascending=False)
    csv_path = f"{OUTPUT_DIR}/phase_b_step3_full_corpus_activation_audit.csv"
    df_audit.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df_audit)} concept-language pairs)")

    # 4. Target Prevalence Comparison (Phase A -> Step 1 -> Step 2 -> Step 3)
    target_comp = []
    substantial_shifts = []
    for t in TARGETS:
        cnt_base = target_positives_base[t]
        cnt_s1 = target_positives_s1[t]
        cnt_s2 = target_positives_s2[t]
        cnt_s3 = target_positives_s3[t]

        pct_base = (cnt_base / total_reports) * 100.0
        pct_s1 = (cnt_s1 / total_reports) * 100.0
        pct_s2 = (cnt_s2 / total_reports) * 100.0
        pct_s3 = (cnt_s3 / total_reports) * 100.0

        delta_s3_base = pct_s3 - pct_base
        delta_s3_s2 = pct_s3 - pct_s2

        # Flag changes > 3.0 percentage points
        is_flagged = abs(delta_s3_s2) >= 3.0 or abs(delta_s3_base) >= 5.0
        if is_flagged:
            substantial_shifts.append({
                "target": t,
                "base_pct": round(pct_base, 2),
                "s2_pct": round(pct_s2, 2),
                "s3_pct": round(pct_s3, 2),
                "delta_s3_s2": round(delta_s3_s2, 2),
                "delta_s3_base": round(delta_s3_base, 2)
            })

        target_comp.append({
            "target": t,
            "base_positive_count": cnt_base,
            "base_positive_pct": round(pct_base, 2),
            "step1_positive_count": cnt_s1,
            "step1_positive_pct": round(pct_s1, 2),
            "step2_positive_count": cnt_s2,
            "step2_positive_pct": round(pct_s2, 2),
            "step3_positive_count": cnt_s3,
            "step3_positive_pct": round(pct_s3, 2),
            "delta_s3_s2_pct": round(delta_s3_s2, 2),
            "delta_s3_base_pct": round(delta_s3_base, 2),
            "flagged_substantial_shift": is_flagged
        })

    df_target_comp = pd.DataFrame(target_comp)

    # 5. Format Step 3 Rule Firing Summary
    rules_summary = []
    for rule_name, rdata in step3_rules_activated.items():
        rules_summary.append({
            "rule_name": rule_name,
            "target": rdata["target"],
            "language": rdata["language"],
            "firing_count": rdata["count"],
            "unique_studies": len(rdata["unique_studies"]),
            "pct_of_corpus": round((len(rdata["unique_studies"]) / total_reports) * 100, 3),
            "example_spans": rdata["examples"]
        })
    rules_summary.sort(key=lambda x: x["firing_count"], reverse=True)

    # 6. Overall Summary JSON
    summary_data = {
        "audit_name": "AUDIT 1: Full 4,407-Report Activation Analysis",
        "dataset_path": DATA_PATH,
        "total_reports_evaluated": total_reports,
        "total_evaluations_run": total_reports * 12,
        "unique_concepts_in_lexicon": len(COMPILED_LEXICON),
        "active_concept_language_pairs": len(df_audit),
        "multi_concept_reports_histogram": {
            f"{k}_concepts": v for k, v in sorted(study_concepts_count.items())
        },
        "target_prevalence_comparison": target_comp,
        "substantial_shifts_flagged": substantial_shifts,
        "step3_active_rules_summary": rules_summary
    }

    json_path = f"{OUTPUT_DIR}/phase_b_step3_full_corpus_activation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved {json_path}")

    # Print summary table
    print("\n" + "="*110)
    print("FULL 4,407 CORPUS TARGET PREVALENCE SHIFT (Phase A Baseline -> Step 1 -> Step 2 -> Step 3)")
    print("="*110)
    print(df_target_comp[["target", "base_positive_pct", "step1_positive_pct", "step2_positive_pct", "step3_positive_pct", "delta_s3_s2_pct", "delta_s3_base_pct", "flagged_substantial_shift"]].to_string(index=False))

    return summary_data, df_audit, df_target_comp

if __name__ == "__main__":
    run_audit1()
