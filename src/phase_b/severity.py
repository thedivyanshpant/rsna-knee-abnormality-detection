"""
Phase B Step 2: Clinical Severity and Condition-Conflation Gating Engine.
Implements five core clinical decision gates operating downstream of Step 1.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import re
from typing import Dict, List, Optional, Tuple

# 1. Gate 1: OA / Chondrosis Severity Patterns
MILD_OA_PATTERNS = [
    r"\bmild\s+chondrosis\b",
    r"\blow-grade\s+(?:articular\s+)?cartilage\s+loss\b",
    r"\bhondromalacične\s+promjene\s+1\.\s+stupnja\b",
    r"\bhondromalacija\s+i°\b",
    r"\bgrade\s+1\s+chondromalacia\b",
    r"\bcondropatía\s+superficial\b",
    r"\bcondropatia\s+superficial\b",
    r"\bfina\s+úlcera\s+condral\b",
    r"\bfina\s+ulcera\s+condral\b",
    r"\bscattered\s+foci\s+of\s+mild\b",
    r"\bminimal\s+chondral\b",
]

SEVERE_OA_PATTERNS = [
    r"\badvanced\b",
    r"\bsevere\b",
    r"\bosteoarthritis\b",
    r"\bgonarthr\b",
    r"\bdenudation\b",
    r"\bdenudacija\b",
    r"\bfull-thickness\b",
    r"\bespesor\s+total\b",
    r"\bgrade\s+[34]\b",
    r"\bgrad\s+[34]\b",
    r"\bhondromalacija\s+iii°\b",
    r"\bhondromalacija\s+iv°\b",
    r"\bosteophytes?\b",
    r"\bosteofit\b",
    r"\bjoint\s+space\s+(?:loss|narrowing)\b",
    r"\bartrosis\s+femorotibial\b",
    r"\boa\s+femorotibial\b",
]

# 2. Gate 2: Effusion / Synovitis Patterns
PHYSIOLOGICAL_EFFUSION_PATTERNS = [
    r"\bphysiologische\s+flüssigkeitsmenge\b",
    r"\bphysiological\s+fluid\b",
    r"\bminimal\s+amount\s+of\s+right\s+knee\s+effusion\s+is\s+1\b",
]

NON_INFLAMMATORY_SYNOVITIS_PATTERNS = [
    r"\bentezopatske\s+promjene\s+hvatišta\s+tetive\b",
    r"\bprepatellar\s+düzeyde\s+cilt\s+altı\s+yağlı\s+dokuda\s+ödem\b",
    r"\bmedial\s+collateral\s+ligament\s+periphery\s+has\s+fluid\b",
]

# 3. Gate 3: Meniscus Degeneration Patterns
NON_TEAR_MENISCUS_PATTERNS = [
    r"\bgrade\s+i\s+dejenerasyon\b",
    r"\bgrade\s+ii\s+dejenerasyon\b",
    r"\bgrade\s+1\s+dejenerasyon\b",
    r"\bgrade\s+2\s+dejenerasyon\b",
    r"\bgrade\s+i\s+degeneration\b",
    r"\bgrade\s+ii\s+degeneration\b",
    r"\bmucoid\s+degeneration\b",
    r"\bmyxoid\s+degeneration\b",
    r"\bintrasubstance\s+signal\b",
    r"\bblaže\s+degenerativno\b",
    r"\bекспулсиран\s+встрани\b",
    r"\bmeniscocapsular\s+separation\b",
    r"\bsuspicious\s+small\s+longitudinal\b",
]

TRUE_MENISCUS_TEAR_PATTERNS = [
    r"\bkova\s+sapı\b",
    r"\bbucket-handle\b",
    r"\bradial\s+tear\b",
    r"\bhorizontal\s+tear\b",
    r"\bvertical\s+tear\b",
    r"\bcomplex\s+tear\b",
    r"\broot\s+tear\b",
    r"\brotura\b",
    r"\byırtık\b",
    r"\byirtik\b",
    r"\bruptur\b",
    r"\bruptura\b",
    r"\briss\b",
    r"\bscheur\b",
    r"\bразкъсване\b",
    r"\bρήξη\b",
    r"\bgrade\s+iii\b",
    r"\bgrade\s+3\b",
    r"\bgrade\s+iv\b",
    r"\bgrade\s+4\b",
    r"\barticular\s+surface\b",
]

# 4. Gate 4: Bone Contusion vs Fracture Patterns
MICRO_FRACTURE_PATTERNS = [
    r"\bpequeñas\s+fracturas\s+subcorticales\b",
    r"\bimpactiefractuur\s+op\s+centraal\s+dragende\b",
]

MACRO_FRACTURE_PATTERNS = [
    r"\bsubchondral\s+insufficiency\s+fracture\b",
    r"\bsubchondral\s+fracture\s+extending\b",
    r"\bcortical\s+step-off\b",
    r"\bdisplaced\b",
    r"\bavulsion\b",
    r"\bcomminuted\b",
    r"\bcomplete\s+fracture\b",
]

# 5. Gate 5: Baker's Cyst Anatomical Patterns
NON_POPLITEAL_BAKER_PATTERNS = [
    r"\bsuprapatellar\s+bursa\b",
    r"\bsuprapatellar\s+bursada\b",
    r"\bπροεπιγονατιδικ\b",
    r"\bprepatellar\b",
    r"\binfrapatellar\b",
    r"\bинфрапател\b",
    r"\bставен\s+излив\b",
    r"\bjoint\s+effusion\b",
]

TRUE_POPLITEAL_PATTERNS = [
    r"\bpopliteal\b",
    r"\bpoplíteo\b",
    r"\bpoplitea\b",
    r"\bпоплитеал\b",
    r"\bgastrocnemius\b",
    r"\bsemimembranos\b",
    r"\bbaker\b",
]

# 6. Gate 6: Ligament Sprain Patterns
ACL_SPRAIN_PATTERNS = [
    r"\binterstitial\s+hyperintense\s+signal\b",
    r"\bin\s+favor\s+of\s+grade\s+1\s+injury\b",
    r"\bin\s+favor\s+of\s+grade\s+2\s+injury\b",
]

ACL_TEAR_PATTERNS = [
    r"\bcomplete\s+tear\b",
    r"\bfull-thickness\b",
    r"\bdisruption\b",
    r"\brotura\s+completa\b",
    r"\bpotpuna\s+ruptura\b",
]

# Compile patterns
COMPILED_MILD_OA = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in MILD_OA_PATTERNS]
COMPILED_SEVERE_OA = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in SEVERE_OA_PATTERNS]
COMPILED_PHYS_EFF = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in PHYSIOLOGICAL_EFFUSION_PATTERNS]
COMPILED_NON_INF_SYN = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NON_INFLAMMATORY_SYNOVITIS_PATTERNS]
COMPILED_NON_TEAR_MEN = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NON_TEAR_MENISCUS_PATTERNS]
COMPILED_TRUE_TEAR_MEN = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRUE_MENISCUS_TEAR_PATTERNS]
COMPILED_MICRO_FX = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in MICRO_FRACTURE_PATTERNS]
COMPILED_MACRO_FX = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in MACRO_FRACTURE_PATTERNS]
COMPILED_NON_POP_BAKER = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NON_POPLITEAL_BAKER_PATTERNS]
COMPILED_TRUE_POP_BAKER = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRUE_POPLITEAL_PATTERNS]
COMPILED_ACL_SPRAIN = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in ACL_SPRAIN_PATTERNS]
COMPILED_ACL_TEAR = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in ACL_TEAR_PATTERNS]


def apply_clinical_severity_gates(
    target: str,
    step1_pred: int,
    step1_conf: float,
    evidence: str,
    full_report: str = ""
) -> Tuple[int, float, str, Optional[str]]:
    """
    Apply Step 2 Clinical Severity and Condition-Conflation Gates downstream of Step 1.
    
    Returns:
        (step2_pred: int, step2_conf: float, rule_applied: str, reason: Optional[str])
    """
    if step1_pred == 0:
        return 0, step1_conf, "STEP1_UNCHANGED", None

    evid_lower = str(evidence).lower()
    rep_lower = str(full_report).lower() if full_report else evid_lower

    # Gate 1: OA Severity Gate
    if target in ["Medial OA", "Lateral OA", "PF OA"]:
        has_mild = any(p.search(evid_lower) for p in COMPILED_MILD_OA)
        has_severe_evid = any(p.search(evid_lower) for p in COMPILED_SEVERE_OA)
        has_severe_rep = any(p.search(rep_lower) for p in COMPILED_SEVERE_OA)
        if has_mild and not has_severe_evid and not has_severe_rep:
            return 0, 0.90, "RULE_OA_SEVERITY_GATE", "Mild/low-grade chondrosis below clinical osteoarthritis threshold."

    # Gate 2A: Effusion Physiological Gate
    if target == "Effusion":
        if any(p.search(evid_lower) for p in COMPILED_PHYS_EFF):
            return 0, 0.90, "RULE_EFFUSION_SEVERITY_GATE", "Physiological fluid or extraction syntax artifact below clinical effusion threshold."

    # Gate 2B: Synovitis Non-Inflammatory Gate
    if target == "Synovitis":
        has_non_syn = any(p.search(evid_lower) for p in COMPILED_NON_INF_SYN)
        has_syn_keyword = any(k in evid_lower for k in ["sinov", "synov", "υμεν", "synovium"])
        if has_non_syn and not has_syn_keyword:
            return 0, 0.90, "RULE_SYNOVITIS_PROLIFERATION_GATE", "Isolated enthesopathy/periligamentous fluid without synovial membrane involvement."

    # Gate 3: Meniscus Degeneration Gate
    if target in ["Medial Meniscus", "Lateral Meniscus"]:
        has_non_tear = any(p.search(evid_lower) for p in COMPILED_NON_TEAR_MEN)
        has_tear_evid = any(p.search(evid_lower) for p in COMPILED_TRUE_TEAR_MEN)
        if has_non_tear and not has_tear_evid:
            comp_keyword = "medial" if target == "Medial Meniscus" else "lateral"
            has_tear_rep = any(p.search(rep_lower) for p in COMPILED_TRUE_TEAR_MEN if comp_keyword in rep_lower)
            if not has_tear_rep:
                return 0, 0.90, "RULE_MENISCUS_TEAR_GATE", "Intrameniscal grade 1/2 signal, mucoid degeneration, or extrusion without true tear."

    # Gate 4: Bone Contusion vs Fracture Gate
    if target == "Fracture":
        has_micro_fx = any(p.search(evid_lower) for p in COMPILED_MICRO_FX)
        has_macro_fx = any(p.search(evid_lower) for p in COMPILED_MACRO_FX)
        if has_micro_fx and not has_macro_fx:
            return 0, 0.90, "RULE_BONE_FRACTURE_GATE", "Subcortical micro-impaction/contusion bone bruise without macroscopic cortical fracture."

    # Gate 5: Baker's Cyst Anatomical Gate
    if target == "Baker's":
        has_non_pop = any(p.search(evid_lower) for p in COMPILED_NON_POP_BAKER)
        has_true_pop = any(p.search(evid_lower) for p in COMPILED_TRUE_POP_BAKER)
        if has_non_pop and not has_true_pop:
            return 0, 0.90, "RULE_BAKER_ANATOMICAL_GATE", "Anterior/suprapatellar fluid collection or effusion isolated from popliteal Baker's cyst."

    # Gate 6: Ligament Sprain vs Tear Gate (ACL)
    if target == "ACL":
        has_sprain = any(p.search(evid_lower) for p in COMPILED_ACL_SPRAIN)
        has_tear = any(p.search(evid_lower) for p in COMPILED_ACL_TEAR)
        if has_sprain and not has_tear:
            return 0, 0.90, "RULE_LIGAMENT_SPRAIN_GATE", "Low-grade interstitial signal/sprain without mechanical ligament tear."

    return step1_pred, step1_conf, "STEP1_UNCHANGED", None
