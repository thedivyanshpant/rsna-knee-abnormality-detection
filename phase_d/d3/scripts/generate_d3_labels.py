#!/usr/bin/env python3
"""
Phase D3: Experimental Label Remediation Engine.
RSNA Knee Abnormality Detection

Addresses the severe weak supervision sparsity identified in Phase B Step 3
for osteoarthritis targets:
- Medial OA (original: 15 positives / 4,349, 0.34%)
- Lateral OA (original: 3 positives / 4,349, 0.07%)
- PF OA (original: 1 positive / 4,349, 0.02%)

Root Cause Remediation:
1. Resolves Spanish word order mismatch: In Latin American/Spanish clinical reports,
   radiologists write compartment qualifiers post-positively ("artrosis femorotibial lateral",
   "artrosis patelofemoral").
2. Expands rigid vocabulary: Includes "condropatía" (the standard Latin American MRI term for
   cartilage loss/chondromalacia), "platillo tibial lateral", "rótula/patelar/troclear".
3. Resolves global/tricompartmental OA distribution: "gonartrosis tricompartimental" and
   "artrosis tricompartimental" propagate to Medial, Lateral, and PF compartments.
4. Preserves clinical rigor & severity gating:
   - Explicit severe / advanced OA (Grade 3-4, bone edema, full-thickness loss): Soft target Y=0.95, weight=1.0.
   - Moderate chondral wear (Grade 2, unclassified chondropathy): Soft target Y=0.80, weight=0.85.
   - Explicit negative / preserved cartilage: Soft target Y=0.05, weight=1.0.
   - Unmentioned / neutral: Preserves Phase B baseline Y=0.20, weight=0.85.
5. Preserves all 9 non-OA targets identically from Phase B Step 3.
6. Zero touch to frozen Phase B production corpus. Zero leakage from Gold-58.
"""

import os
import sys
import re
import json
import hashlib
import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
BASE_PHASE_B_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "csv/train.csv")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")
CV_SPLITS_JSON = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")

D3_LABELS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/labels")
D3_REPORTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/reports")
OUTPUT_LABELS_CSV = os.path.join(D3_LABELS_DIR, "phase_d3_candidate_label_corpus.csv")
OUTPUT_MANIFEST_JSON = os.path.join(D3_LABELS_DIR, "phase_d3_candidate_manifest.json")

# Negation and qualification regexes
NEGATION_PATTERNS = [
    r"\bsin\b", r"\bno\s+se\s+observa\b", r"\bno\s+hay\b", r"\bintacto\b", r"\bintacta\b",
    r"\bnormal(?:es)?\b", r"\bconservado\b", r"\bconservada\b", r"\bíntegro\b", r"\bíntegra\b",
    r"\bsin\s+alteraciones\b", r"\bsin\s+signos\s+de\b", r"\bsin\s+evidencia\s+de\b",
    r"\bno\b", r"\bnone\b", r"\bwithout\b", r"\bnegative\b", r"\bunremarkable\b",
    r"\bkein(?:e|en|er|es)?\b", r"\bohne\b", r"\bunauffällig\b", r"\byok\b", r"\bdoğal\b"
]
COMPILED_NEG = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE | re.UNICODE)

# Anatomical compartment regexes
MEDIAL_PATTERNS = [
    r"femorotibial\s+medial", r"c[óo]ndilo\s+femoral\s+medial", r"platillo\s+tibial\s+medial",
    r"compartimento\s+medial", r"compartimiento\s+medial", r"artrosis\s+medial", r"gonartrosis\s+medial",
    r"medial\s+compartment\s+osteoarthr", r"mediale\s+gonarthrose"
]
LATERAL_PATTERNS = [
    r"femorotibial\s+lateral", r"platillo\s+tibial\s+lateral", r"c[óo]ndilo\s+femoral\s+lateral",
    r"compartimento\s+lateral", r"compartimiento\s+lateral", r"artrosis\s+lateral", r"gonartrosis\s+lateral",
    r"lateral\s+compartment\s+osteoarthr", r"laterale\s+gonarthrose"
]
PF_PATTERNS = [
    r"patelofemoral", r"femoropatelar", r"r[óo]tula", r"rotulian[ao]", r"patelar",
    r"tr[óo]clea\s+femoral", r"troclear", r"retropatelar", r"retropatell"
]
GLOBAL_OA_PATTERNS = [
    r"tricompartiment", r"gonartrosis\s+(?:global|avanzada|severa)", r"artrosis\s+severa"
]

COMPILED_MED = re.compile("|".join(MEDIAL_PATTERNS), re.IGNORECASE | re.UNICODE)
COMPILED_LAT = re.compile("|".join(LATERAL_PATTERNS), re.IGNORECASE | re.UNICODE)
COMPILED_PF = re.compile("|".join(PF_PATTERNS), re.IGNORECASE | re.UNICODE)
COMPILED_GLOBAL = re.compile("|".join(GLOBAL_OA_PATTERNS), re.IGNORECASE | re.UNICODE)

# OA / cartilage pathology indicators
OA_INDICATORS = [
    r"\bartrosis\b", r"\bgonartrosis\b", r"\bosteoarthr", r"\barthrose\b",
    r"\bcondromalacia\b", r"\bcondropat[íi]a\b", r"\bpinzamiento\b", r"\bosteofito",
    r"\bp[ée]rdida\s+(?:completa\s+)?del\s+cart[íi]lago\b", r"\bdesgaste\s+articular\b"
]
COMPILED_OA = re.compile("|".join(OA_INDICATORS), re.IGNORECASE | re.UNICODE)

# Severity grading indicators
SEV_ADVANCED_PATTERNS = [
    r"grado\s+[34]", r"grado\s+iv\b", r"grado\s+iii\b", r"grade\s+[34]",
    r"completa", r"full-thickness", r"sever[ao]", r"avanzad[ao]", r"edema\s+[óo]seo",
    r"esclerosis\s+subcondral", r"bone-on-bone", r"eburnation"
]
SEV_MODERATE_PATTERNS = [
    r"grado\s+2\s+y\s+3", r"grado\s+2-3", r"grado\s+2", r"grade\s+2", r"moderada"
]
COMPILED_ADVANCED = re.compile("|".join(SEV_ADVANCED_PATTERNS), re.IGNORECASE | re.UNICODE)
COMPILED_MODERATE = re.compile("|".join(SEV_MODERATE_PATTERNS), re.IGNORECASE | re.UNICODE)


def extract_study_oa_findings(report_text: str) -> Dict[str, Dict[str, Any]]:
    """Extract compartment-specific OA findings from clinical report.
    
    Returns mapping for 'Medial OA', 'Lateral OA', 'PF OA' with:
    - category: 'advanced_positive', 'moderate_positive', 'explicit_negative', 'unmentioned'
    - evidence_text: text snippet
    """
    findings = {
        "Medial OA": {"category": "unmentioned", "evidence": None, "rule": "PHASE_B_BASELINE_PRESERVED"},
        "Lateral OA": {"category": "unmentioned", "evidence": None, "rule": "PHASE_B_BASELINE_PRESERVED"},
        "PF OA": {"category": "unmentioned", "evidence": None, "rule": "PHASE_B_BASELINE_PRESERVED"},
    }
    if not isinstance(report_text, str) or not report_text.strip():
        return findings

    # Split report into clauses
    clauses = re.split(r"[.;\n]+", report_text)
    
    for cl in clauses:
        cl_clean = cl.strip()
        if not cl_clean or len(cl_clean) < 5:
            continue
        # Exclude historical clinical indication / antecedents to avoid past-history false positives
        if cl_clean.lower().startswith(("antecedentes", "indicaci", "motivo", "clinical indication", "historia")):
            continue

        has_oa = COMPILED_OA.search(cl_clean) is not None
        if not has_oa:
            continue

        is_negated = COMPILED_NEG.search(cl_clean) is not None
        is_advanced = COMPILED_ADVANCED.search(cl_clean) is not None
        is_moderate = COMPILED_MODERATE.search(cl_clean) is not None

        if is_negated:
            cat = "explicit_negative"
            rule = "RULE_D3_EXPLICIT_NEGATIVE"
        elif is_advanced:
            cat = "advanced_positive"
            rule = "RULE_D3_EXPLICIT_ADVANCED_OA"
        elif is_moderate:
            cat = "moderate_positive"
            rule = "RULE_D3_MODERATE_CHONDRAL_WEAR"
        else:
            cat = "moderate_positive"
            rule = "RULE_D3_CHONDRAL_PATHOLOGY"

        targets_in_clause = []
        if COMPILED_MED.search(cl_clean):
            targets_in_clause.append("Medial OA")
        if COMPILED_LAT.search(cl_clean):
            targets_in_clause.append("Lateral OA")
        if COMPILED_PF.search(cl_clean):
            targets_in_clause.append("PF OA")
        if COMPILED_GLOBAL.search(cl_clean):
            targets_in_clause.extend(["Medial OA", "Lateral OA", "PF OA"])

        # Prioritization order: advanced > moderate > negative > unmentioned
        priority = {"advanced_positive": 4, "moderate_positive": 3, "explicit_negative": 2, "unmentioned": 1}

        for t in set(targets_in_clause):
            current_cat = findings[t]["category"]
            if priority[cat] > priority[current_cat]:
                findings[t] = {
                    "category": cat,
                    "evidence": cl_clean,
                    "rule": rule
                }

    return findings


def generate_d3_candidate_corpus():
    os.makedirs(D3_LABELS_DIR, exist_ok=True)
    os.makedirs(D3_REPORTS_DIR, exist_ok=True)

    print("=" * 80)
    print("  PHASE D3: EXPERIMENTAL LABEL REMEDIATION GENERATOR")
    print("=" * 80)

    # 1. Load Base Phase B Production Corpus (READ ONLY)
    print(f"Loading Phase B Base Corpus: {BASE_PHASE_B_CSV}")
    base_df = pd.read_csv(BASE_PHASE_B_CSV)
    print(f"Loaded {len(base_df)} rows across {base_df['StudyInstanceUID'].nunique()} studies.")

    # 2. Load Train CSV with full text reports
    print(f"Loading Raw Reports: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID")
    report_dict = train_df["Report"].to_dict()

    # 3. Load CV splits and isolate the 4,349 development studies
    with open(CV_SPLITS_JSON) as f:
        cv_splits = json.load(f)
    dev_uids = set()
    for f in range(5):
        dev_uids.update(cv_splits["folds"][f"fold_{f}"]["val_studies"])
    assert len(dev_uids) == 4349, f"Expected exactly 4,349 dev studies, got {len(dev_uids)}"

    # Gold-58 zero leakage assertion
    gold_df = pd.read_csv(GOLD_CSV)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    assert len(dev_uids & gold_uids) == 0, "CRITICAL LEAKAGE: Gold UID in development set!"
    print(f"Verified 0 overlap between {len(dev_uids)} development studies and {len(gold_uids)} locked Gold studies.")

    # Filter base corpus to strictly the 4,349 development cohort (Quarantine Gold completely)
    candidate_df = base_df[base_df["StudyInstanceUID"].isin(dev_uids)].copy()
    print(f"Isolated {len(candidate_df)} rows for the 4,349 development cohort.")
    oa_targets = {"Medial OA", "Lateral OA", "PF OA"}

    remediated_counts = {
        "Medial OA": {"base_pos": 0, "advanced_pos": 0, "moderate_pos": 0, "explicit_neg": 0, "total_new_pos": 0},
        "Lateral OA": {"base_pos": 0, "advanced_pos": 0, "moderate_pos": 0, "explicit_neg": 0, "total_new_pos": 0},
        "PF OA": {"base_pos": 0, "advanced_pos": 0, "moderate_pos": 0, "explicit_neg": 0, "total_new_pos": 0},
    }

    # Group by StudyInstanceUID to extract once per study
    study_findings = {}
    for uid in dev_uids:
        rep = report_dict.get(uid, "")
        study_findings[uid] = extract_study_oa_findings(rep)

    # Apply remediation to candidate dataframe
    for idx, row in candidate_df.iterrows():
        t = row["target"]
        if t not in oa_targets:
            continue

        uid = row["StudyInstanceUID"]
        base_prob = row["label_probability"]
        if base_prob >= 0.5:
            remediated_counts[t]["base_pos"] += 1

        finding = study_findings[uid][t]
        cat = finding["category"]
        evidence = finding["evidence"]
        rule = finding["rule"]

        if cat == "advanced_positive":
            candidate_df.at[idx, "final_predicted_label"] = 1
            candidate_df.at[idx, "label_probability"] = 0.95
            candidate_df.at[idx, "confidence_score"] = 0.95
            candidate_df.at[idx, "sample_weight"] = 1.00
            candidate_df.at[idx, "step3_rule_applied"] = rule
            if evidence:
                candidate_df.at[idx, "evidence_text"] = evidence
            remediated_counts[t]["advanced_pos"] += 1
            remediated_counts[t]["total_new_pos"] += 1

        elif cat == "moderate_positive":
            candidate_df.at[idx, "final_predicted_label"] = 1
            candidate_df.at[idx, "label_probability"] = 0.80
            candidate_df.at[idx, "confidence_score"] = 0.85
            candidate_df.at[idx, "sample_weight"] = 0.85
            candidate_df.at[idx, "step3_rule_applied"] = rule
            if evidence:
                candidate_df.at[idx, "evidence_text"] = evidence
            remediated_counts[t]["moderate_pos"] += 1
            remediated_counts[t]["total_new_pos"] += 1

        elif cat == "explicit_negative":
            candidate_df.at[idx, "final_predicted_label"] = 0
            candidate_df.at[idx, "label_probability"] = 0.05
            candidate_df.at[idx, "confidence_score"] = 0.95
            candidate_df.at[idx, "sample_weight"] = 1.00
            candidate_df.at[idx, "step3_rule_applied"] = rule
            if evidence:
                candidate_df.at[idx, "evidence_text"] = evidence
            remediated_counts[t]["explicit_neg"] += 1

        else:
            # Keep base Phase B probability
            pass

    # Save candidate corpus
    candidate_df.to_csv(OUTPUT_LABELS_CSV, index=False)
    print(f"\n[SAVED] D3 Candidate Label Corpus to: {OUTPUT_LABELS_CSV}")

    # Compute SHA256 hash
    with open(OUTPUT_LABELS_CSV, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # Generate metadata manifest
    N_studies = candidate_df["StudyInstanceUID"].nunique()
    manifest = {
        "version_id": "PHASE_D3_CANDIDATE_V1",
        "created_at": datetime.datetime.now().isoformat(),
        "sha256": file_hash,
        "base_corpus": BASE_PHASE_B_CSV,
        "total_studies": N_studies,
        "total_rows": len(candidate_df),
        "remediation_statistics": {},
    }

    for t in sorted(list(oa_targets)):
        stats = remediated_counts[t]
        new_pos = stats["total_new_pos"]
        prev_pct = (new_pos / N_studies) * 100.0
        manifest["remediation_statistics"][t] = {
            "original_phase_b_positives": stats["base_pos"],
            "remediated_advanced_positives": stats["advanced_pos"],
            "remediated_moderate_positives": stats["moderate_pos"],
            "remediated_total_positives": new_pos,
            "remediated_explicit_negatives": stats["explicit_neg"],
            "original_prevalence_pct": (stats["base_pos"] / N_studies) * 100.0,
            "remediated_prevalence_pct": prev_pct,
            "net_gain_positives": new_pos - stats["base_pos"],
        }

    with open(OUTPUT_MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] D3 Candidate Manifest to: {OUTPUT_MANIFEST_JSON}")

    print("\n" + "=" * 80)
    print("  PHASE D3 REMEDIATION SUMMARY (4,349 DEVELOPMENT COHORT)")
    print("=" * 80)
    for t, st in manifest["remediation_statistics"].items():
        print(f"Target: {t:<15s} | Orig Pos: {st['original_phase_b_positives']:3d} ({st['original_prevalence_pct']:.2f}%) -> New Pos: {st['remediated_total_positives']:3d} ({st['remediated_prevalence_pct']:.2f}%) | Net: +{st['net_gain_positives']:3d}")

    return manifest


if __name__ == "__main__":
    generate_d3_candidate_corpus()
