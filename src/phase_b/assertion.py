"""
Deterministic Assertion and Polarity Scoping Engine for RSNA Knee Extraction.
Handles multilingual negative assertions with target-scoped clause boundaries.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import re
from typing import Dict, List, Optional, Tuple

# Multilingual negative assertion patterns
NEGATIVE_ASSERTION_PATTERNS = [
    # English
    r"\bintact\b",
    r"\bnormal\b",
    r"\bpreserved\b",
    r"\bunremarkable\b",
    r"\bno\s+(?:evidence\s+of\s+)?(?:a\s+)?(?:acute\s+)?(?:meniscal\s+)?tear\b",
    r"\bwithout\s+(?:signs?\s+of\s+)?(?:a\s+)?tear\b",
    r"\bno\s+fracture\b",
    r"\bno\s+(?:joint\s+)?effusion\b",
    r"\bno\s+abnormality\b",
    r"\bnot\s+torn\b",
    r"\bfree\s+of\s+tear\b",
    r"\bno\s+signs?\s+of\b",
    r"\bwithin\s+normal\s+limits\b",
    r"\bnormal\s+in\s+morphology\s+and\s+signal\b",
    r"\bnormal\s+morphology\b",
    r"\bno\s+cartilage\s+defects?\b",
    
    # German
    r"\bintakt(?:e|en|er|es)?\b",
    r"\bunauffällig(?:e|en|er|es)?\b",
    r"\bkein\s+nachweis\b",
    r"\bohne\s+nachweis\b",
    r"\bkeine\s+ruptur\b",
    r"\bkein\s+riss\b",
    r"\bkeine\s+läsion\b",
    r"\bohne\s+fassbare\b",
    r"\bregelrecht(?:e|en|er|es)?\b",
    r"\bphysiologisch\s+signalgebend\b",
    
    # Spanish
    r"\bintacto\b",
    r"\bintacta\b",
    r"\bnormal(?:es)?\b",
    r"\bsin\s+signos\s+de\s+desgarro\b",
    r"\bsin\s+signos\s+de\s+rotura\b",
    r"\bsin\s+evidencia\s+de\b",
    r"\bno\s+se\s+observa\b",
    r"\bsin\s+alteraciones\b",
    r"\bsin\s+alteración\b",
    r"\bmorfología\s+y\s+señal\s+conservada\b",
    r"\bmorfologia\s+y\s+senal\s+conservada\b",
    r"\bmorfología\s+conservada\b",
    r"\bdentro\s+de\s+límites\s+normales\b",
    r"\bdentro\s+de\s+limites\s+normales\b",
    r"\bsin\s+rotura\b",
    r"\bno\s+hay\s+quistes\b",
    
    # Turkish
    r"\bintakt\b",
    r"\bnormal\b",
    r"\byırtık\s+yok\b",
    r"\byirtik\s+yok\b",
    r"\bsaptanmadı\b",
    r"\bsaptanmadi\b",
    r"\bizlenmedi\b",
    r"\bdoğal\b",
    r"\bdogal\b",
    
    # Croatian / Serbian
    r"\bintaktan\b",
    r"\bintakta\b",
    r"\bintaktno\b",
    r"\bintakt\b",
    r"\bbez\s+znakova\b",
    r"\bnema\b",
    r"\bnije\s+nađen\b",
    r"\bnije\s+nadjeno\b",
    r"\buredan\b",
    r"\buredna\b",
    r"\bočuvan\b",
    r"\bocuvan\b",
    r"\bbez\s+rupture\b",
    r"\bbez\s+znakova\s+rupture\b",
    r"\bbez\s+znakova\s+degeneracije\b",
    
    # Bulgarian
    r"\bбез\s+данни\s+за\b",
    r"\bбез\s+разкъсване\b",
    r"\bинтактна\b",
    r"\bинтактен\b",
    r"\bинтактно\b",
    r"\bинтактни\b",
    r"\bняма\s+данни\s+за\b",
    
    # Greek
    r"\bχωρίς\b",
    r"\bχωρίς\s+ρήξη\b",
    r"\bάθικτος\b",
    r"\bάθικτη\b",
    r"\bάθικτο\b",
    r"\bφυσιολογικός\b",
    
    # Dutch
    r"\bintact\b",
    r"\bgeen\s+scheur\b",
    r"\bgeen\s+aanwijzingen\s+voor\b",
    r"\bnormaal\b",
    r"\bzonder\s+afwijkingen\b",
]

COMPILED_NEGATIVE_PATTERNS = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NEGATIVE_ASSERTION_PATTERNS]

TARGET_KEYWORDS: Dict[str, List[str]] = {
    "ACL": ["acl", "vca", "lca", "vkb", "anterior cruciate", "ligamento cruzado anterior", "ön çapraz", "prednja ukrštena", "предна кръстна", "πρόσθιος χιαστός"],
    "MCL": ["mcl", "medial collateral", "ligamento colateral medial", "ligamento colateral interno", "iç yan", "medijalni kolateralni", "медиален колатерален", "έσω πλάγιος"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "menisco interno", "innenmeniskus", "medial menisküs", "medijalni menisk", "медиален менискус", "έσω μηνίσκος", "meniscus medialis", "medial menisci", "meniscus", "meniscal", "menisk"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisco externo", "aussenmeniskus", "außenmeniskus", "lateral menisküs", "lateralni menisk", "латерален менискус", "έξω μηνίσκος", "meniscus lateralis", "lateral menisci", "meniscus", "meniscal", "menisk"],
    "Medial OA": ["medial", "medial femorotibial", "medial compartment", "compartimento medial", "femorotibial medial"],
    "Lateral OA": ["lateral", "lateral femorotibial", "lateral compartment", "compartimento lateral", "femorotibial lateral"],
    "PF OA": ["patellofemoral", "patella", "trochlea", "retropatellar", "patelar", "tróclea", "faseta patele", "rótula"],
    "Effusion": ["effusion", "derrame", "fluid", "gelenkerguss", "erguss", "sıvı", "izljev", "течност", "υγρό"],
    "Synovitis": ["synovitis", "sinovitis", "synovia", "sinovija", "synovial"],
    "Baker's": ["baker", "popliteal cyst", "quiste de baker", "baker-zyste", "baker kisti", "baker cista", "бейкърова киста"],
    "Contusion": ["contusion", "contus", "bone bruise", "edema óseo", "knochenödem", "kemik iliği ödemi", "koštani edem"],
    "Fracture": ["fracture", "fractura", "fraktur", "kırık", "prijelom", "фрактура", "κάταγμα"],
}


def split_into_scoped_clauses(text: str) -> List[str]:
    """
    Split complex multi-target sentences into distinct semantic clauses
    based on punctuation and contrastive conjunctions.
    """
    text = text.replace("\n", " ").strip()
    split_pattern = r"(?:;|(?<=[\w\)])\.\s+|\b(?:but|however|although|while|whereas|aber|jedoch|pero|sin\s+embargo|več|ali|doch)\b)"
    raw_clauses = re.split(split_pattern, text, flags=re.IGNORECASE | re.UNICODE)
    
    clauses = []
    for c in raw_clauses:
        c_clean = c.strip(" ,.:;-")
        if c_clean:
            clauses.append(c_clean)
            
    return clauses if clauses else [text]



def target_mentioned_in_clause(target: str, clause: str) -> bool:
    """Check if the specific target (or general class like 'meniscus/ligaments') is referenced in the clause."""
    clause_lower = clause.lower()
    
    # 1. Meniscus generic keywords
    if target in ["Medial Meniscus", "Lateral Meniscus"]:
        # If clause explicitly specifies the OTHER meniscus compartment, don't match
        if target == "Medial Meniscus" and ("lateral meniscus" in clause_lower or "menisco lateral" in clause_lower or "aussenmeniskus" in clause_lower or "außenmeniskus" in clause_lower):
            if "medial" not in clause_lower and "interno" not in clause_lower:
                return False
        if target == "Lateral Meniscus" and ("medial meniscus" in clause_lower or "menisco medial" in clause_lower or "innenmeniskus" in clause_lower or "menisco interno" in clause_lower):
            if "lateral" not in clause_lower and "externo" not in clause_lower:
                return False
                
        if any(w in clause_lower for w in ["menisci", "meniscos", "meniskler", "менискуси", "μηνίσκοι", "meniscal", "meniscus", "menisk"]):
            return True
            
    # 2. Ligament generic keywords
    if target in ["ACL", "MCL"]:
        if target == "ACL" and ("mcl" in clause_lower or "medial collateral" in clause_lower or "colateral medial" in clause_lower):
            if "acl" not in clause_lower and "cruciate" not in clause_lower and "cruzado" not in clause_lower:
                return False
        if target == "MCL" and ("acl" in clause_lower or "anterior cruciate" in clause_lower or "cruzado anterior" in clause_lower):
            if "mcl" not in clause_lower and "collateral" not in clause_lower and "colateral" not in clause_lower:
                return False
                
        if any(w in clause_lower for w in ["ligaments", "ligamentos", "bänder", "bağlar", "ligamenti", "cruciate", "cruzados", "collateral", "colaterales"]):
            return True

    keywords = TARGET_KEYWORDS.get(target, [target.lower()])
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", clause_lower):
            return True
            
    return False


def check_scoped_negative_assertion(target: str, evidence: str, full_report: str = "") -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Evaluate whether the target has an unambiguous negative assertion in its scoped clause/evidence.
    """
    if not evidence or evidence.strip().lower() in ["not mentioned in the report.", "nan", ""]:
        return False, None, None

    clauses = split_into_scoped_clauses(evidence)
    target_clauses = [c for c in clauses if target_mentioned_in_clause(target, c)]
    if not target_clauses:
        target_clauses = clauses

    for clause in target_clauses:
        for pat in COMPILED_NEGATIVE_PATTERNS:
            match = pat.search(clause)
            if match:
                matched_phrase = match.group(0)
                clause_lower = clause.lower()
                positive_tear_cues = ["complete tear", "full-thickness tear", "radial tear", "horizontal tear", "ruptur", "rotura", "yırtık", "ruptura", "torn", "is torn"]
                has_positive_for_target = False
                for pos in positive_tear_cues:
                    if pos in clause_lower:
                        # Ensure the positive cue is NOT part of a negated phrase like "no tear" or "not torn"
                        if not any(neg in clause_lower for neg in ["no ", "sin ", "kein", "keine", "ohne", "bez ", "няма ", "χωρίς ", "geen ", "not "]):
                            has_positive_for_target = True
                            
                if not has_positive_for_target:
                    return True, matched_phrase, "RULE_ASSERTION_SCOPED_NEGATIVE"

    return False, None, None
