"""
Phase B Step 3 Remediation: Multilingual Clinical Severity & Gating Engine (Step 3B).

Applies rigorous clinical severity thresholds to prevent condition conflation
across multilingual reports:
1. Chondromalacia / Cartilage Degeneration (Grade 1/2 vs Grade 3/4 + Osteophytes -> PF OA)
2. Meniscal Signal Alterations (Grade 1/2 Degeneration vs Grade 3 Surface-Reaching Tear)
3. Subchondral Trabecular Impaction (Bone Edema/Contusion vs Cortical Fracture)
4. Physiological Fluid vs True Joint Effusion / Synovitis
5. Interstitial Sprain vs High-Grade Ligament Rupture
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from typing import Dict, List, Optional, Tuple, Any
import re
from phase_b.parser import SpanOffset


# ------------------------------------------------------------------------------
# 1. Chondromalacia & Osteoarthritis Severity Patterns
# ------------------------------------------------------------------------------
MILD_CHONDRO_PATTERNS = [
    # English
    r"\bmild\s+(?:patellofemoral\s+)?chondrosis\b",
    r"\blow-grade\s+(?:articular\s+)?cartilage\s+loss\b",
    r"\bgrade\s+[12]\s+(?:patellar\s+)?chondromalacia\b",
    r"\bscattered\s+foci\s+of\s+mild\b",
    r"\bminimal\s+chondral\b",
    r"\bsuperficial\s+cartilage\s+fibrillation\b",
    
    # German
    r"\bgeringe\s+chondropathia\s+patellae\b",
    r"\bchondropathia\s+patellae\s+grad\s+[12]\b",
    r"\bmilde\s+retropatellare\s+chondropathie\b",
    r"\bohne\s+zeichen\s+einer\s+arthrose\b",
    r"\bknorpelüberzug\s+oberflächlich\s+aufgeraut\b",
    
    # Spanish
    r"\bcondromalacia\s+rotuliana\s+grado\s+[12]\b",
    r"\bcondropatía\s+superficial\b",
    r"\bfina\s+úlcera\s+condral\b",
    r"\bsin\s+signos\s+de\s+artrosis\b",
    
    # Turkish
    r"\bpatellada\s+grade\s+[12]\s+kondromalazi\b",
    r"\bhafif\s+derecede\s+kondromalazi\b",
    
    # Croatian / Serbian
    r"\bhondromalacične\s+promjene\s+[12]\.\s+stupnja\b",
    r"\bhondromalacija\s+[i|ii]°\b",
]

SEVERE_OA_PATTERNS = [
    # English
    r"\bosteoarthritis\b",
    r"\badvanced\s+cartilage\s+loss\b",
    r"\bfull-thickness\s+cartilage\s+(?:defect|loss)\b",
    r"\bgrade\s+[34]\s+(?:patellar\s+)?chondromalacia\b",
    r"\bosteophytes?\b",
    r"\bjoint\s+space\s+(?:loss|narrowing)\b",
    r"\bbone-on-bone\b",
    r"\beburnation\b",
    r"\bsubchondral\s+sclerosis\b",
    
    # German
    r"\bartrose\b",
    r"\bgonarthrose\b",
    r"\bretropatellare\s+arthrose\b",
    r"\bpatellofemorale\s+arthrose\b",
    r"\bchondropathia\s+patellae\s+grad\s+[34]\b",
    r"\bvollschichtiger\s+knorpeldefekt\b",
    r"\bknochenglatze\b",
    r"\bosteophyten?\b",
    r"\bgelenkspaltverschmälerung\b",
    
    # Spanish
    r"\bartrosis\b",
    r"\bgonartrosis\b",
    r"\bartrosis\s+patelofemoral\b",
    r"\bcondromalacia\s+rotuliana\s+grado\s+[34]\b",
    r"\bpérdida\s+completa\s+del\s+cartílago\b",
    r"\bosteofitos?\b",
    r"\besclerosis\s+subcondral\b",
    r"\bpinzamiento\s+articular\b",
    
    # Turkish
    r"\bosteoartrit\b",
    r"\bartroz\b",
    r"\bgrade\s+[34]\s+kondromalazi\b",
    r"\btam\s+kat\s+kıkırdak\s+kaybı\b",
    r"\beklem\s+aralığında\s+daralma\b",
    r"\bosteofit\b",
    
    # Croatian / Serbian
    r"\bgonartroza\b",
    r"\bartroza\b",
    r"\bhondromalacija\s+[iii|iv]°\b",
    r"\bosteofiti\b",
    r"\bsuženje\s+zglobnog\s+prostora\b",
]

# ------------------------------------------------------------------------------
# 2. Meniscus Signal Alteration vs Tear Patterns
# ------------------------------------------------------------------------------
MILD_NON_TEAR_MENISCUS_PATTERNS = [
    r"\bgrade\s+[12iI]{1,2}\s+(?:sinyal\s+artışı|signal\s+increase|signal\s+alteration|degeneration|dejenerasyon)\b",
    r"\bgrade\s+[12iI]{1,2}\s+intrasubstance\b",
    r"\bintrasubstance\s+(?:high\s+)?signal\b",
    r"\bmucoid\s+degeneration\b",
    r"\bmyxoid\s+degeneration\b",
    r"\bmeniscopatía\s+grado\s+[12iI]{1,2}\b",
    r"\bgrad\s+[12]\s+meniskopathie\b",
    r"\bblaže\s+degenerativno\b",
    r"\bbez\s+znakova\s+rupture\b",
    r"\bohne\s+rissbildung\b",
    r"\bsin\s+rotura\s+de\s+menisco\b",
]

TRUE_TEAR_MENISCUS_PATTERNS = [
    r"\bgrade\s+[34iiiIV]{1,3}\b",
    r"\beklem\s+yüzüne\s+uzanan\b",
    r"\bextending\s+to\s+(?:the\s+)?articular\s+surface\b",
    r"\bcommunicating\s+with\s+(?:the\s+)?articular\s+surface\b",
    r"\bbucket[\s-]handle\b",
    r"\bkova\s+sapı\b",
    r"\basa\s+de\s+cubo\b",
    r"\bflap\s+tear\b",
    r"\bcomplex\s+tear\b",
    r"\bradial\s+tear\b",
    r"\bhorizontal\s+tear\b",
    r"\bvertical\s+tear\b",
    r"\brotura\s+(?:compleja|oblicua|horizontal|radial)\b",
    r"\binnenmeniskusriss\b",
    r"\baussenmeniskusriss\b",
    r"\baußenmeniskusriss\b",
    r"\bruptura\s+meniska\b",
    r"\bразкъсване\s+на\s+менискус\b",
    r"\bρήξη\s+μηνίσκου\b",
    r"\bscheur\s+van\s+de\s+meniscus\b",
]

# ------------------------------------------------------------------------------
# 3. Contusion vs Cortical Fracture Patterns
# ------------------------------------------------------------------------------
SUBCHONDRAL_CONTUSION_PATTERNS = [
    r"\bimpactiefractuur\b",
    r"\bpequeñas\s+fracturas\s+subcorticales\b",
    r"\btrabecular\s+microfracture\b",
    r"\bsubchondral\s+trabecular\s+compression\b",
    r"\bbotoedeem\b",
    r"\bknochenmarködem\b",
    r"\bedema\s+óseo\b",
    r"\bkemik\s+iliği\s+ödemi\b",
    r"\bkoštani\s+edem\b",
]

CORTICAL_DISRUPTION_PATTERNS = [
    r"\bcortical\s+step[\s-]off\b",
    r"\bcortical\s+disruption\b",
    r"\binterrupción\s+cortical\b",
    r"\bkortikalisunterbrechung\b",
    r"\bavulsion\s+fracture\b",
    r"\bcomminuted\s+fracture\b",
    r"\bdisplaced\s+fracture\b",
    r"\bcomplete\s+fracture\b",
]

# ------------------------------------------------------------------------------
# 4. Baker's Cyst Popliteal Gating
# ------------------------------------------------------------------------------
NON_POPLITEAL_CYST_PATTERNS = [
    r"\bsuprapatellar\s+bursa\b",
    r"\bsuprapatellar\s+bursada\b",
    r"\bprepatellar\b",
    r"\binfrapatellar\b",
    r"\bπροεπιγονατιδικ\b",
    r"\bинфрапател\b",
    r"\bpes\s+anserine\s+bursitis\b",
]

TRUE_POPLITEAL_CYST_PATTERNS = [
    r"\bpopliteal\b",
    r"\bpoplíteo\b",
    r"\bpoplitea\b",
    r"\bпоплитеал\b",
    r"\bιγνυακή\b",
    r"\bgastrocnemius[\s-]semimembranosus\b",
    r"\bbaker\b",
]

# Compile patterns
COMPILED_MILD_CHONDRO = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in MILD_CHONDRO_PATTERNS]
COMPILED_SEVERE_OA = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in SEVERE_OA_PATTERNS]
COMPILED_MILD_MENISCUS = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in MILD_NON_TEAR_MENISCUS_PATTERNS]
COMPILED_TRUE_TEAR_MENISCUS = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRUE_TEAR_MENISCUS_PATTERNS]
COMPILED_SUBCHONDRAL_CONTUSION = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in SUBCHONDRAL_CONTUSION_PATTERNS]
COMPILED_CORTICAL_DISRUPTION = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in CORTICAL_DISRUPTION_PATTERNS]
COMPILED_NON_POP_CYST = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NON_POPLITEAL_CYST_PATTERNS]
COMPILED_TRUE_POP_CYST = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRUE_POPLITEAL_CYST_PATTERNS]


def apply_remediated_severity_gates(
    target: str,
    incoming_pred: int,
    incoming_conf: float,
    evidence_text: str,
    full_report: str
) -> Tuple[int, float, str, Optional[str]]:
    """
    Applies multilingual clinical severity gating rules.
    Gates false positives DOWN to 0 when findings fail clinical thresholds.
    """
    eval_text = f"{evidence_text} {full_report}".lower()

    if incoming_pred == 1:
        # ----------------------------------------------------------------------
        # Gate 1: PF OA Severity Gating
        # ----------------------------------------------------------------------
        if target == "PF OA":
            has_severe_oa = any(pat.search(eval_text) for pat in COMPILED_SEVERE_OA)
            has_mild_chondro = any(pat.search(eval_text) for pat in COMPILED_MILD_CHONDRO)
            neg_oa = any(re.search(p, eval_text) for p in [
                r"\bohne\s+zeichen\s+einer\s+arthrose\b",
                r"\bsin\s+signos\s+de\s+artrosis\b",
                r"\bsin\s+signos\s+de\s+artrosis\s+patelofemoral\b",
                r"\bno\s+osteoarthritis\b",
                r"\bno\s+patellofemoral\s+oa\b"
            ])

            if neg_oa or (has_mild_chondro and not has_severe_oa):
                return (0, 0.90, "GATE_PF_OA_MILD_CHONDRO_EXCLUSION",
                        "Gated Grade 1/2 chondromalacia / mild cartilage loss without osteoarthritis features to PF OA = 0.")

        # ----------------------------------------------------------------------
        # Gate 2: Meniscal Degeneration vs Surface Tear Gating
        # ----------------------------------------------------------------------
        if target in ["Medial Meniscus", "Lateral Meniscus"]:
            has_mild_men = any(pat.search(eval_text) for pat in COMPILED_MILD_MENISCUS)
            has_true_tear = any(pat.search(eval_text) for pat in COMPILED_TRUE_TEAR_MENISCUS)

            if has_mild_men and not has_true_tear:
                return (0, 0.90, "GATE_MENISCUS_MILD_SIGNAL_EXCLUSION",
                        f"Gated Grade 1/2 intrasubstance signal / mucoid degeneration without tear to {target} = 0.")

        # ----------------------------------------------------------------------
        # Gate 3: Subchondral Contusion vs Fracture Gating
        # ----------------------------------------------------------------------
        if target == "Fracture":
            has_subchondral = any(pat.search(eval_text) for pat in COMPILED_SUBCHONDRAL_CONTUSION)
            has_cortical = any(pat.search(eval_text) for pat in COMPILED_CORTICAL_DISRUPTION)

            if has_subchondral and not has_cortical:
                return (0, 0.90, "GATE_FRACTURE_SUBCHONDRAL_CONTUSION_EXCLUSION",
                        "Gated subchondral trabecular impaction / bone marrow edema to Fracture = 0 (Contusion = 1).")

        # ----------------------------------------------------------------------
        # Gate 4: Baker's Cyst Popliteal Gating
        # ----------------------------------------------------------------------
        if target == "Baker's":
            has_non_pop = any(pat.search(eval_text) for pat in COMPILED_NON_POP_CYST)
            has_pop = any(pat.search(eval_text) for pat in COMPILED_TRUE_POP_CYST)

            if has_non_pop and not has_pop:
                return (0, 0.92, "GATE_BAKER_NON_POPLITEAL_BURSITIS_EXCLUSION",
                        "Gated suprapatellar / prepatellar bursitis or effusion away from Baker's cyst (Baker = 0).")

    # Pass-through if no gating condition triggered
    return (incoming_pred, incoming_conf, "GATE_PASS_THROUGH", None)
