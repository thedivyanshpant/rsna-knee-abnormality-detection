"""
Phase B Step 3 Remediation: Unified Pipeline Orchestrator & Provenance Engine (Step 3D).

Integrates:
- Step 3A: Non-Destructive Scoped Parser (span preservation, relational modifiers)
- Step 3B: Multilingual Clinical Lexicon & Severity Gating Engine
- Step 3C: Decoupled Soft Target & Confidence Architecture
- Step 3D: Complete 20-Field Audit Provenance Emitter
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from typing import Dict, List, Optional, Tuple, Any
import re

from phase_b.config import TARGETS
from phase_b.assertion import check_scoped_negative_assertion
from phase_b.anatomy import evaluate_compartment_scoping
from phase_b.severity import apply_clinical_severity_gates

try:
    from phase_b.parser import (
        SpanOffset,
        ScopedClause,
        split_text_into_scoped_clauses,
        analyze_fracture_displacement_relations
    )
    from phase_b.lexicon import (
        ConceptRecord,
        get_remediated_multilingual_evidence,
        extract_multilingual_concepts_non_destructive
    )
    from phase_b.soft_targets import compute_soft_targets, SoftTargetResult
except ImportError:
    from phase_b.parser import (
        SpanOffset,
        ScopedClause,
        split_text_into_scoped_clauses,
        analyze_fracture_displacement_relations
    )
    from phase_b.lexicon import (
        ConceptRecord,
        get_remediated_multilingual_evidence,
        extract_multilingual_concepts_non_destructive
    )
    from phase_b.soft_targets import compute_soft_targets, SoftTargetResult


TARGET_KEYWORD_MAP = {
    "ACL": [r"\bacl\b", r"\banterior\s+cruciate\b", r"\bvkb\b", r"\bvorderes\s+kreuzband\b", r"\bön\s+çapraz\b", r"\bπρόσθι(?:ου|ος)\s+χιαστ\b"],
    "MCL": [r"\bmcl\b", r"\bmedial\s+collateral\b", r"\binnenband\b", r"\bikb\b", r"\bmedial\s+kollateral\b", r"\bέσω\s+πλάγι\b"],
    "Medial Meniscus": [r"\bmedial\s+menisc", r"\binnenmeniskus\b", r"\bmenisco\s+(?:medial|interno)\b", r"\bmedial\s+menisküs\b", r"\bέσω\s+μηνίσκ\b"],
    "Lateral Meniscus": [r"\blateral\s+menisc", r"\baussenmeniskus\b", r"\baußenmeniskus\b", r"\bmenisco\s+(?:lateral|externo)\b", r"\blateral\s+menisküs\b", r"\bέξω\s+μηνίσκ\b"],
    "Fracture": [r"\bfracture\b", r"\bfraktur\b", r"\bfissur\b", r"\bfractura\b", r"\bkırık\b", r"\bκάταγμα\b", r"\bфрактура\b"],
    "Baker's": [r"\bbaker", r"\bpopliteal\s+cyst\b", r"\bbursa\s+gastrocnemio\b", r"\bpoplíteo\b", r"\bpoplitea\b"],
    "Effusion": [r"\beffusion\b", r"\bgelenkerguss\b", r"\bderrame\b", r"\bjoint\s+fluid\b", r"\bsıvı\b"],
    "Synovitis": [r"\bsynovitis\b", r"\bsynovial\s+thickening\b", r"\bsinovitis\b", r"\bsinovyal\b", r"\bsynovialis\b"],
    "Contusion": [r"\bcontusion\b", r"\bbone\s+bruise\b", r"\bknochenmarködem\b", r"\bkemik\s+iliği\s+ödemi\b", r"\bedema\s+óseo\b", r"\bbotoedeem\b"],
    "Medial OA": [r"\bmedial\s+(?:compartment\s+)?(?:oa|osteoarthritis|arthrose|artrosis)\b", r"\bgonartrosi\b"],
    "Lateral OA": [r"\blateral\s+(?:compartment\s+)?(?:oa|osteoarthritis|arthrose|artrosis)\b"],
    "PF OA": [r"\bpatellofemoral\s+(?:oa|osteoarthritis|arthrose|artrosis)\b", r"\bpf\s+oa\b", r"\bretropatellare\s+arthrose\b"],
}
COMPILED_TARGET_KWS = {t: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in pats] for t, pats in TARGET_KEYWORD_MAP.items()}

HEDGE_TERMS = [
    r"\bcannot\s+(?:rule\s+out|exclude)\b",
    r"\bquestionable\b",
    r"\bequivocal\b",
    r"\bsuspicious\s+for\b",
    r"\bnicht\s+sicher\s+auszuschließ[a-zäöüß]*\b",
    r"\bdifferenzialdiagnostisch\b",
    r"\bno\s+se\s+puede\s+descartar\b",
    r"\bşüpheli\b",
    r"\bdeğerlendirilemed?i\b",
    r"\bevalue\s+edilemedi\b"
]
COMPILED_HEDGES = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in HEDGE_TERMS]

POSITIVE_PATHOLOGY_TERMS = [
    r"\bdisruption\b", r"\btear\b", r"\bruptur\b", r"\brupture\b", r"\briss\b",
    r"\bläsion\b", r"\brotura\b", r"\byırtık\b", r"\bρήξη\b", r"\bскъсване\b"
]
COMPILED_POS_PATHOLOGY = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in POSITIVE_PATHOLOGY_TERMS]


def check_target_contradiction_and_equivocation(
    target: str,
    report_text: str,
    all_concepts: Optional[List[ConceptRecord]] = None
) -> Tuple[bool, str]:
    """Check for explicit hedge/equivocal phrasing or intra-report polarity collisions."""
    if not report_text:
        return False, "NO_CONTRADICTION"

    # 1. Bidirectional clause hedge check
    clauses = split_text_into_scoped_clauses(report_text)
    t_pats = COMPILED_TARGET_KWS.get(target, [])

    for cl in clauses:
        has_h = any(hp.search(cl.text) for hp in COMPILED_HEDGES)
        has_t = any(tp.search(cl.text) for tp in t_pats)
        if has_h and has_t:
            return True, "HEDGE_EQUIVOCAL"

    # 2. Polar contradiction check
    concepts = all_concepts if all_concepts is not None else extract_multilingual_concepts_non_destructive(report_text)
    t_pos = [c for c in concepts if target in c.target_candidates and c.polarity == "positive" and not c.evidence_text.lower().startswith("indication")]
    t_neg = [c for c in concepts if target in c.target_candidates and (c.polarity == "negative" or c.canonical_concept.endswith("_INTACT")) and not c.evidence_text.lower().startswith("indication")]
    
    # Also check cross-clause affirmative vs negative mentions in English/multilingual
    has_pos_mention = False
    has_neg_mention = False
    for cl in clauses:
        if cl.text.lower().startswith("indication"):
            continue
        if any(tp.search(cl.text) for tp in t_pats):
            if any(pp.search(cl.text) for pp in COMPILED_POS_PATHOLOGY) and not any(w in cl.text.lower() for w in ["no ", "without ", "kein", "sin ", "free of"]):
                has_pos_mention = True
            if any(w in cl.text.lower() for w in ["intact", "unremarkable", "regelrecht", "intakt", "normal", "doğaldır", "no tear", "without tear"]):
                has_neg_mention = True

    if (t_pos and t_neg) or (has_pos_mention and has_neg_mention):
        return True, "POLAR_CONTRADICTION"

    return False, "NO_CONTRADICTION"




def process_remediated_study_target(
    study_uid: str,
    target: str,
    baseline_pred: int,
    baseline_conf: float,
    evidence: str,
    full_report: str = "",
    gold_label: Optional[int] = None,
    precomputed_concepts: Optional[List[ConceptRecord]] = None
) -> Dict[str, Any]:
    """
    Processes a single study-target pair through the remediated Phase B pipeline
    and returns a structured 20-field provenance record.
    """
    report_text = full_report if full_report else evidence

    # --------------------------------------------------------------------------
    # 1. STEP 1 BASELINE: Scoped Negative Assertion & Anatomical Scoping
    # --------------------------------------------------------------------------
    step1_pred = baseline_pred
    step1_conf = baseline_conf
    step1_rule = "BASELINE_UNCHANGED"

    is_negative, match_phrase, rule_id = check_scoped_negative_assertion(target, evidence, report_text)
    if is_negative:
        if step1_pred == 1:
            step1_pred = 0
            step1_conf = 0.95
            step1_rule = rule_id
        else:
            step1_conf = max(step1_conf, 0.95)
            step1_rule = rule_id

    if target in ["Medial OA", "Lateral OA", "PF OA"] and not is_negative:
        up_pred, anat_rule, _ = evaluate_compartment_scoping(
            target=target,
            baseline_pred=step1_pred,
            evidence=evidence,
            full_report=report_text
        )
        if up_pred != step1_pred or anat_rule != "PASS_THROUGH":
            step1_pred = up_pred
            step1_conf = 0.90
            step1_rule = anat_rule

    # --------------------------------------------------------------------------
    # 2. STEP 2 BASELINE: Clinical Severity & Condition-Conflation Gating
    # --------------------------------------------------------------------------
    step2_pred, step2_conf, step2_rule, _ = apply_clinical_severity_gates(
        target=target,
        step1_pred=step1_pred,
        step1_conf=step1_conf,
        evidence=evidence,
        full_report=report_text
    )

    # --------------------------------------------------------------------------
    # 3. STEP 3 REMEDIATION: Non-Destructive Parsing & Multilingual Concept Engine
    # --------------------------------------------------------------------------
    step3_pred = step2_pred
    step3_conf = step2_conf
    step3_rule = "STEP2_PRESERVED"
    matched_concept_name: Optional[str] = None
    detected_lang = "EN"
    evidence_span_str = f"[{0}, {len(report_text)}]"
    active_evidence_text = evidence if evidence else report_text
    is_explicit_concept = False

    # Check for contradictions / equivocal hedging
    has_contradiction, contradiction_reason = check_target_contradiction_and_equivocation(
        target, report_text, precomputed_concepts
    )
    if has_contradiction:
        step3_rule = f"RULE_ABSTAIN_{contradiction_reason}"


    # Extract target multilingual concept non-destructively
    concept: Optional[ConceptRecord] = get_remediated_multilingual_evidence(
        target, report_text, evidence, precomputed_concepts=precomputed_concepts
    )

    if concept and concept.language != "EN":
        matched_concept_name = concept.canonical_concept
        detected_lang = concept.language
        if concept.evidence_span:
            evidence_span_str = f"[{concept.evidence_span.start_char}, {concept.evidence_span.end_char}]"

        # Case 1: Recovery of positive non-English pathology when Step 2 predicted 0
        if step2_pred == 0 and concept.polarity == "positive":
            c_neg, _, _ = check_scoped_negative_assertion(target, concept.evidence_text, report_text)
            if not c_neg:
                cand_s1_pred = 1
                cand_s1_conf = concept.confidence
                
                # Check compartment scoping
                if target in ["Medial OA", "Lateral OA", "PF OA"]:
                    cand_s1_pred, _, _ = evaluate_compartment_scoping(
                        target, cand_s1_pred, concept.evidence_text, report_text
                    )
                    
                # Check clinical severity gates
                cand_s2_pred, cand_s2_conf, cand_s2_rule, _ = apply_clinical_severity_gates(
                    target, cand_s1_pred, cand_s1_conf, concept.evidence_text, report_text
                )
                
                if cand_s2_pred == 1:
                    step3_pred = 1
                    step3_conf = cand_s2_conf
                    step3_rule = f"RULE_REMEDIATED_{concept.canonical_concept}"
                    active_evidence_text = concept.evidence_text
                    is_explicit_concept = True

        # Case 2: Confirmation of explicit intact / normal structures or negated pathology
        elif step2_pred == 1 and (concept.canonical_concept.endswith("_INTACT") or concept.canonical_concept.endswith("_NEGATIVE") or concept.polarity == "negative"):
            step3_pred = 0
            step3_conf = 0.95
            step3_rule = f"RULE_REMEDIATED_NEGATIVE_{concept.canonical_concept}"
            active_evidence_text = concept.evidence_text
            is_explicit_concept = True

    if has_contradiction:
        step3_rule = f"RULE_ABSTAIN_{contradiction_reason}"

    # --------------------------------------------------------------------------
    # 4. STEP 3C: Soft Targets & Confidence Calibration
    # --------------------------------------------------------------------------

    soft_res: SoftTargetResult = compute_soft_targets(
        binary_label=step3_pred,
        rule_applied=step3_rule,
        base_confidence=step3_conf,
        has_contradiction=has_contradiction,
        is_mild_gated=("GATE_" in step3_rule),
        is_explicit_concept=is_explicit_concept
    )

    final_label = soft_res.predicted_label

    # --------------------------------------------------------------------------
    # 5. STEP 3D: Assemble Complete 20-Field Audit Record
    # --------------------------------------------------------------------------
    return {
        "StudyInstanceUID": str(study_uid),
        "target": str(target),
        "gold_label": int(gold_label) if gold_label is not None else None,
        "baseline_predicted_label": int(baseline_pred),
        "baseline_confidence": float(round(baseline_conf, 4)),
        "step1_predicted_label": int(step1_pred),
        "step1_rule_applied": str(step1_rule),
        "step2_predicted_label": int(step2_pred),
        "step2_rule_applied": str(step2_rule),
        "step3_predicted_label": int(step3_pred),
        "step3_rule_applied": str(step3_rule),
        "final_predicted_label": int(final_label),
        "label_probability": float(soft_res.label_probability),
        "confidence_score": float(soft_res.confidence_score),
        "sample_weight": float(soft_res.sample_weight),
        "abstain_flag": int(soft_res.abstain_flag),
        "matched_canonical_concept": str(matched_concept_name) if matched_concept_name else "None",
        "detected_language": str(detected_lang),
        "evidence_char_span": str(evidence_span_str),
        "evidence_text": str(active_evidence_text)
    }


def run_remediated_pipeline_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of records through the complete remediated pipeline."""
    results = []
    for r in records:
        uid = r.get("StudyInstanceUID", "")
        target = r.get("target", "")
        b_pred = int(r.get("baseline_predicted_label", r.get("predicted_label", 0)))
        b_conf = float(r.get("baseline_confidence", r.get("confidence", 0.5)))
        evidence = str(r.get("extracted_evidence", r.get("evidence_text", "")))
        report = str(r.get("report_text", r.get("full_report", evidence)))
        gold = int(r["gold_label"]) if "gold_label" in r and r["gold_label"] is not None else None

        res = process_remediated_study_target(
            study_uid=uid,
            target=target,
            baseline_pred=b_pred,
            baseline_conf=b_conf,
            evidence=evidence,
            full_report=report,
            gold_label=gold
        )
        results.append(res)
    return results
