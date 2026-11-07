"""
Anatomical Compartment Scoping Engine for RSNA Knee Extraction.
Isolates Patellofemoral anatomy from Tibiofemoral (Medial/Lateral) compartments,
with full-report contextual verification for multicompartmental pathology.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

# 1. Patellofemoral Landmarks (Must map strictly to PF OA)
PATELLOFEMORAL_PATTERNS = [
    r"\bpatell(?:a|ar|ofemoral)\b",
    r"\bpatel(?:a|ar|ofemoral)\b",
    r"\btrochle(?:a|ar|e|am)\b",
    r"\btrocle(?:a|ar)\b",
    r"\btróclea\b",
    r"\brótula\b",
    r"\brotulian(?:o|a)\b",
    r"\bfasete?\s+patele\b",
    r"\bfacet\b",
    r"\bretropatell(?:ar|air)\b",
    r"\bkniescheibe\b",
    r"\barticulatio\s+patellofemoralis\b",
    r"\bpatellapees\b",
    r"\bpatellofemorale\b",
    r"\bpatellar\s+facet\b",
    r"\bmedial\s+patellar\s+facet\b",
    r"\blateral\s+patellar\s+facet\b",
    r"\bmedial\s+trochle(?:a|ar)\b",
    r"\blateral\s+trochle(?:a|ar)\b",
    r"\bvertiente\s+medial\s+de\s+la\s+tróclea\b",
    r"\bvertiente\s+lateral\s+de\s+la\s+tróclea\b",
    r"\bтрохле(?:я|а)\b",
    r"\bпател(?:а|арна)\b",
    r"\bεπιγονατίδ(?:α|ας)\b",
    r"\bτροχιλί(?:α|ας)\b",
    r"\bfp\s+zgloba\b",
]

# 2. Medial Tibiofemoral Landmarks (Must map strictly to Medial OA)
MEDIAL_TF_PATTERNS = [
    r"\bmedial\s+femoral\s+condyle\b",
    r"\bmedial\s+tibial\s+plateau\b",
    r"\bmedial\s+femorotibial\b",
    r"\bmedial\s+tibiofemoral\b",
    r"\bmedial\s+joint\s+compartment\b",
    r"\bmedial\s+compartment\s+(?:cartilage|chondrosis|osteoarthritis|oa|narrowing|space|load)\b",
    r"\bcóndilo\s+femoral\s+medial\b",
    r"\bplatillo\s+tibial\s+medial\b",
    r"\bcompartimento\s+femorotibial\s+medial\b",
    r"\bartrosis\s+femorotibial\s+medial\b",
    r"\bmediales\s+tibiaplateau\b",
    r"\bmedialer\s+femurkondyl\b",
    r"\bmedialen\s+femorotibialen\b",
    r"\bmedijalnog\s+kondila\s+femura\b",
    r"\bmedijalnog\s+tibijalnog\s+platoa\b",
    r"\bmedijalnog\s+kompartmenta\s+ft\b",
    r"\bмедиално\s+тибиално\s+плато\b",
    r"\bмедиален\s+кондил\s+на\s+фемура\b",
    r"\bέσω\s+μηριαίος\s+κόνδυλος\b",
    r"\bέσω\s+κνημιαίο\s+plateau\b",
    r"\bέσω\s+μεσάρθριο\b",
    r"\bmediaal\s+tibiaplateau\b",
    r"\bmediale\s+femurcondyl\b",
    r"\bmediale\s+femorotibiale\b",
]

# 3. Lateral Tibiofemoral Landmarks (Must map strictly to Lateral OA)
LATERAL_TF_PATTERNS = [
    r"\blateral\s+femoral\s+condyle\b",
    r"\blateral\s+tibial\s+plateau\b",
    r"\blateral\s+femorotibial\b",
    r"\blateral\s+tibiofemoral\b",
    r"\blateral\s+joint\s+compartment\b",
    r"\blateral\s+compartment\s+(?:cartilage|chondrosis|osteoarthritis|oa|narrowing|space|load)\b",
    r"\bcóndilo\s+femoral\s+lateral\b",
    r"\bplatillo\s+tibial\s+lateral\b",
    r"\bcompartimento\s+femorotibial\s+lateral\b",
    r"\bartrosis\s+femorotibial\s+lateral\b",
    r"\blaterales\s+tibiaplateau\b",
    r"\blateraler\s+femurkondyl\b",
    r"\blateralen\s+femorotibialen\b",
    r"\blateralnog\s+kondila\s+femura\b",
    r"\blateralnog\s+tibijalnog\s+platoa\b",
    r"\blateralnog\s+kompartmenta\b",
    r"\bлатерално\s+тибиално\s+плато\b",
    r"\bлатерален\s+кондил\s+на\s+фемура\b",
    r"\bέξω\s+μηριαίος\s+κόνδυλος\b",
    r"\bέξω\s+κνημιαίο\s+plateau\b",
    r"\bέξω\s+μεσάρθριο\b",
    r"\blateraal\s+tibiaplateau\b",
    r"\blaterale\s+femurcondyl\b",
    r"\blaterale\s+femorotibiale\b",
]

COMPILED_PF = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in PATELLOFEMORAL_PATTERNS]
COMPILED_MEDIAL_TF = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in MEDIAL_TF_PATTERNS]
COMPILED_LATERAL_TF = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in LATERAL_TF_PATTERNS]


def extract_anatomical_compartments(text: str) -> Set[str]:
    """
    Extract all distinct anatomical compartments identified in a clause or text snippet.
    Returns a set containing any of: {"PF", "MEDIAL_TF", "LATERAL_TF"}.
    """
    compartments = set()
    
    for pat in COMPILED_PF:
        if pat.search(text):
            compartments.add("PF")
            break
            
    for pat in COMPILED_MEDIAL_TF:
        if pat.search(text):
            compartments.add("MEDIAL_TF")
            break
            
    for pat in COMPILED_LATERAL_TF:
        if pat.search(text):
            compartments.add("LATERAL_TF")
            break
            
    return compartments


def has_cartilage_pathology_in_compartment(compartment: str, text: str) -> bool:
    """Check if the text contains genuine cartilage loss/OA terms associated with the compartment."""
    text_lower = text.lower()
    pathology_cues = [
        "chondro", "condro", "cartilage", "artrosis", "arthrosis", "osteoarthr",
        "ulcer", "úlcera", "defect", "loss", "denudacija", "fisure", "erozije",
        "knorpel", "narrowing", "osteophyte", "osteofit"
    ]
    if not any(cue in text_lower for cue in pathology_cues):
        return False
        
    comps = extract_anatomical_compartments(text)
    return compartment in comps


def evaluate_compartment_scoping(
    target: str,
    baseline_pred: int,
    evidence: str,
    full_report: str = ""
) -> Tuple[int, str, Optional[str]]:
    """
    Evaluate anatomical scoping for OA targets (Medial OA, Lateral OA, PF OA).
    """
    if target not in ["Medial OA", "Lateral OA", "PF OA"]:
        return baseline_pred, "PASS_THROUGH", None
        
    if not evidence or evidence.strip().lower() in ["not mentioned in the report.", "nan", ""]:
        return baseline_pred, "PASS_THROUGH", None

    evidence_compartments = extract_anatomical_compartments(evidence)
    report_compartments = extract_anatomical_compartments(full_report) if full_report else evidence_compartments
    
    # 1. Medial OA Scoping
    if target == "Medial OA":
        if baseline_pred == 1:
            # If evidence is exclusively Patellofemoral and neither evidence nor full report has medial TF OA:
            if "PF" in evidence_compartments and "MEDIAL_TF" not in evidence_compartments:
                if not has_cartilage_pathology_in_compartment("MEDIAL_TF", full_report):
                    return 0, "RULE_ANATOMY_PF_ISOLATION", "Patellofemoral anatomy (trochlea/patellar facet) isolated from Medial OA (no medial TF OA in report)."
        return baseline_pred, "PASS_THROUGH", None

    # 2. Lateral OA Scoping
    if target == "Lateral OA":
        if baseline_pred == 1:
            # If evidence is exclusively Patellofemoral and neither evidence nor full report has lateral TF OA:
            if "PF" in evidence_compartments and "LATERAL_TF" not in evidence_compartments:
                if not has_cartilage_pathology_in_compartment("LATERAL_TF", full_report):
                    return 0, "RULE_ANATOMY_PF_ISOLATION", "Patellofemoral anatomy (trochlea/patellar facet) isolated from Lateral OA (no lateral TF OA in report)."
        return baseline_pred, "PASS_THROUGH", None

    # 3. PF OA Scoping
    if target == "PF OA":
        if baseline_pred == 1:
            # If evidence is exclusively Tibiofemoral and full report contains NO Patellofemoral OA:
            if ("MEDIAL_TF" in evidence_compartments or "LATERAL_TF" in evidence_compartments) and "PF" not in evidence_compartments:
                if not has_cartilage_pathology_in_compartment("PF", full_report):
                    return 0, "RULE_ANATOMY_MISMATCH_SUPPRESSION", "Tibiofemoral compartment finding isolated from Patellofemoral OA (no PF OA in report)."
        return baseline_pred, "PASS_THROUGH", None

    return baseline_pred, "PASS_THROUGH", None
