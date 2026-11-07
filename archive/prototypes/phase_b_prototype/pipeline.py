"""
Phase B Pipeline Orchestrator.
Chains Phase B Step 1 (Assertion + Anatomy), Step 2 (Clinical Severity Gates),
and Step 3 (Multilingual Clinical Concept Expansion & Normalization).
Preserves complete transition audit trail for every prediction across all phases.
"""

from typing import Dict, List, Any, Optional
from phase_b.config import TARGETS
from phase_b.assertion import check_scoped_negative_assertion
from phase_b.anatomy import evaluate_compartment_scoping
from phase_b.severity import apply_clinical_severity_gates
from phase_b.multilingual import (
    extract_multilingual_concepts,
    get_multilingual_evidence_for_target,
    ConceptRecord
)


def process_study_target(
    study_uid: str,
    target: str,
    baseline_pred: int,
    baseline_conf: float,
    evidence: str,
    full_report: str = "",
    gold_label: int = None
) -> Dict[str, Any]:
    """
    Process a single (Study, Target) prediction record through Phase B Step 1, Step 2, and Step 3.
    """
    # ----------------------------------------------------
    # PHASE B STEP 1: Assertion + Anatomical Scoping
    # ----------------------------------------------------
    step1_pred = baseline_pred
    step1_conf = baseline_conf
    step1_rule = "BASELINE_UNCHANGED"
    step1_reason = None
    assertion_phrase = None

    # Step 1.1: Scoped Assertion / Polarity Filter
    is_negative, match_phrase, rule_id = check_scoped_negative_assertion(target, evidence, full_report)
    if is_negative:
        if step1_pred == 1:
            step1_pred = 0
            step1_conf = 0.95
            step1_rule = rule_id
            assertion_phrase = match_phrase
            step1_reason = f"Explicit negative assertion ('{match_phrase}') scoped to {target}; inverted polarity from 1 to 0."
        else:
            step1_conf = max(step1_conf, 0.95)
            step1_rule = rule_id
            assertion_phrase = match_phrase
            step1_reason = f"Confirmed negative assertion ('{match_phrase}') scoped to {target}."

    # Step 1.2: Anatomical Compartment Scoping (OA targets)
    if target in ["Medial OA", "Lateral OA", "PF OA"] and not is_negative:
        updated_pred, anat_rule, anat_reason = evaluate_compartment_scoping(
            target=target,
            baseline_pred=step1_pred,
            evidence=evidence,
            full_report=full_report
        )
        if updated_pred != step1_pred or anat_rule != "PASS_THROUGH":
            step1_pred = updated_pred
            step1_conf = 0.90
            step1_rule = anat_rule
            step1_reason = anat_reason

    # ----------------------------------------------------
    # PHASE B STEP 2: Clinical Severity & Conflation Gates
    # ----------------------------------------------------
    step2_pred, step2_conf, step2_rule, step2_reason = apply_clinical_severity_gates(
        target=target,
        step1_pred=step1_pred,
        step1_conf=step1_conf,
        evidence=evidence,
        full_report=full_report
    )

    # ----------------------------------------------------
    # PHASE B STEP 3: Multilingual Concept Expansion
    # ----------------------------------------------------
    step3_pred = step2_pred
    step3_conf = step2_conf
    step3_rule = "STEP2_PRESERVED"
    step3_concept = None
    step3_lang = "EN"
    step3_reason = None
    active_evidence = evidence

    # Multilingual concept recovery for non-English findings
    concept: Optional[ConceptRecord] = get_multilingual_evidence_for_target(target, full_report, evidence)
    if concept and concept.language != "EN":
        step3_concept = concept.canonical_concept
        step3_lang = concept.language

        # Recovery of positive non-English pathology when Step 2 predicted 0
        if step2_pred == 0 and concept.polarity == "positive":
            # Run candidate evidence through Step 1 Assertion
            c_neg, c_match, _ = check_scoped_negative_assertion(target, concept.evidence_text, full_report)
            if not c_neg:
                cand_s1_pred = 1
                cand_s1_conf = concept.confidence
                # Run candidate through Step 1 Compartment Scoping
                if target in ["Medial OA", "Lateral OA", "PF OA"]:
                    cand_s1_pred, _, _ = evaluate_compartment_scoping(
                        target, cand_s1_pred, concept.evidence_text, full_report
                    )
                # Run candidate through Step 2 Clinical Severity Gates
                cand_s2_pred, cand_s2_conf, cand_s2_rule, cand_s2_reason = apply_clinical_severity_gates(
                    target, cand_s1_pred, cand_s1_conf, concept.evidence_text, full_report
                )
                if cand_s2_pred == 1:
                    step3_pred = 1
                    step3_conf = cand_s2_conf
                    step3_rule = f"RULE_MULTILINGUAL_{concept.canonical_concept}"
                    step3_reason = f"Recovered {concept.canonical_concept} from {concept.language} phrasing: '{concept.surface_forms[0]}'"
                    active_evidence = concept.evidence_text

        # Confirmation of explicit intact / normal structures or negated pathology
        elif step2_pred == 1 and (concept.canonical_concept.endswith("_INTACT") or concept.canonical_concept.endswith("_NEGATIVE")):
            step3_pred = 0
            step3_conf = 0.95
            step3_rule = f"RULE_MULTILINGUAL_NEGATIVE_{concept.canonical_concept}"
            step3_reason = f"Confirmed negative {concept.canonical_concept} from {concept.language} phrasing: '{concept.surface_forms[0]}'"
            active_evidence = concept.evidence_text



    # Derived probability score
    prob_score = step3_conf if step3_pred == 1 else (1.0 - step3_conf)

    # Resolve active rule
    if step3_rule != "STEP2_PRESERVED":
        active_rule = step3_rule
    elif step2_rule != "STEP1_UNCHANGED":
        active_rule = step2_rule
    else:
        active_rule = step1_rule

    return {
        "StudyInstanceUID": study_uid,
        "target": target,
        "gold_label": gold_label,
        "baseline_predicted_label": baseline_pred,
        "step1_predicted_label": step1_pred,
        "step1_rule_applied": step1_rule,
        "step1_reason": step1_reason,
        "step2_predicted_label": step2_pred,
        "step2_confidence": step2_conf,
        "step2_rule_applied": step2_rule,
        "step2_reason": step2_reason,
        "step3_predicted_label": step3_pred,
        "step3_confidence": step3_conf,
        "step3_probability": prob_score,
        "step3_rule_applied": step3_rule,
        "step3_canonical_concept": step3_concept,
        "step3_language": step3_lang,
        "step3_reason": step3_reason,
        "final_predicted_label": step3_pred,
        "phase_b_predicted_label": step3_pred,
        "rule_applied": active_rule,
        "assertion_phrase": assertion_phrase,
        "extracted_evidence": active_evidence,
        "report_text": full_report
    }


def run_phase_b_pipeline_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of study-target records through the complete Phase B pipeline."""
    results = []
    for r in records:
        res = process_study_target(
            study_uid=r["StudyInstanceUID"],
            target=r["target"],
            baseline_pred=int(r.get("baseline_predicted_label", r.get("predicted_label", 0))),
            baseline_conf=float(r.get("confidence", 0.5)),
            evidence=str(r.get("extracted_evidence", "")),
            full_report=str(r.get("report_text", "")),
            gold_label=int(r["gold_label"]) if "gold_label" in r and r["gold_label"] is not None else None
        )
        results.append(res)
    return results
