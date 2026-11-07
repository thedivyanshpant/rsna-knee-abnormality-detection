"""
Phase B Step 3: Multilingual Clinical Concept Expansion & Normalization Engine.
Provides structured mappings for 8 language groups (EN, DE, ES, TR, HR/SR, GR, BG, NL)
normalizing free-text medical phrasing into canonical clinical concepts.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from phase_b.config import TARGETS
from phase_b.assertion import split_into_scoped_clauses


@dataclass
class ConceptRecord:
    surface_forms: List[str]
    canonical_concept: str
    target_candidates: List[str]
    clinical_type: str
    language: str
    polarity: str  # "positive", "negative", "uncertain", "neutral"
    severity_hint: Optional[str] = None  # "mild", "moderate", "severe", "complete", "partial", "grade_1_2", "grade_3_4"
    anatomical_hint: Optional[str] = None  # "MEDIAL", "LATERAL", "PATELLOFEMORAL", "POPLITEAL", "GLOBAL"
    confidence: float = 0.90
    evidence_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_multilingual_text(text: str) -> str:
    """Normalize text across unicode encodings (e.g. Greek micro sign, Turkish i, ligatures)."""
    if not text:
        return ""
    # Normalize Greek micro sign (\u00b5 -> \u03bc)
    text = text.replace("\u00b5", "\u03bc")
    # Normalize curly apostrophes / quotes
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text


NON_DISPLACEMENT_PATTERNS = [
    r"\bwithout\s+(?:evidence\s+of\s+)?(?:significant\s+|major\s+)?dislocation\b",
    r"\bwithout\s+(?:evidence\s+of\s+)?(?:significant\s+|major\s+)?displacement\b",
    r"\bsin\s+(?:signos\s+de\s+)?(?:significativo\s+)?desplazamiento\b",
    r"\bohne\s+(?:nachweis\s+einer\s+)?(?:wesentliche\s+)?dislokation\b",
    r"\bohne\s+(?:wesentliche\s+)?verschiebung\b",
    r"\bбез\s+(?:данни\s+за\s+)?(?:значима\s+)?дислокация\b",
    r"\bбез\s+разместване\b",
    r"\bnondeplase\b",
    r"\bnon-displaced\b",
    r"\bniet[\s-]*(?:verplaatste|verplaatst)\b"
]
COMPILED_NON_DISPLACEMENT = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NON_DISPLACEMENT_PATTERNS]


NEGATIVE_MODIFIERS = [
    r"\bno\b", r"\bnone\b", r"\bwithout\b", r"\bnegative\b", r"\bintact\b", r"\bunremarkable\b",
    r"\bfree\s+of\b", r"\bsin\b", r"\bkein(?:e|en|er|es)?\b", r"\bohne\b", r"\bnema\b", r"\bbez\b",
    r"\bбез\b", r"\bняма\b", r"\bχωρίς\b", r"\byok\b", r"\bsaptanmadı\b", r"\bsaptanmadi\b",
    r"\bizlenmedi\b", r"\bgeen\b", r"\bniet\b"
]
COMPILED_NEG = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NEGATIVE_MODIFIERS]


def is_clause_truly_negative(clause: str) -> bool:
    """Check if a clause is negative, carefully ignoring non-displacement fracture descriptors."""
    clean = clause
    for p in COMPILED_NON_DISPLACEMENT:
        clean = p.sub("", clean)
    for p in COMPILED_NEG:
        if p.search(clean):
            return True
    return False


# ==============================================================================
# STRUCTURED MULTILINGUAL CLINICAL LEXICON
# ==============================================================================

MULTILINGUAL_LEXICON: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # 1. LIGAMENTS (ACL, MCL)
    # --------------------------------------------------------------------------
    # ACL Tear / Rupture
    {
        "canonical_concept": "ACL_TEAR",
        "target_candidates": ["ACL"],
        "clinical_type": "TEAR",
        "polarity": "positive",
        "severity_hint": "complete",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\bvkb[\s-]*(?:ruptur|riss|läsion|teilruptur|anriss)\b",
                r"\bvorderes?\s+kreuzband(?:ruptur|riss|läsion)?\b",
                r"\bruptur\s+des\s+vorderen\s+kreuzbandes\b",
                r"\bkreuzbandruptur\b",
            ],
            "ES": [
                r"\brotura\s+(?:completa\s+|parcial\s+|de\s+espesor\s+total\s+)?del\s+(?:lca|ligamento\s+cruzado\s+anterior)\b",
                r"\b(?:lca|ligamento\s+cruzado\s+anterior)\s+(?:con\s+)?(?:rotura|desgarro|desinserción)\b",
                r"\bdesgarro\s+del\s+(?:lca|ligamento\s+cruzado\s+anterior)\b",
            ],
            "TR": [
                r"\bön\s+çapraz\s+bağ.*?(?:yırtık|rüptür|devamlılığı\s+seçilememiştir|bütünlük\s+kaybı)\b",
                r"\banterior\s+çapraz\s+bağ.*?(?:komplet\s+|parsiyel\s+)?(?:rüptür|yırtık|rüptüre|bütünlük\s+kaybı)\b",
                r"\böçb.*?(?:komplet\s+|parsiyel\s+)?(?:yırtık|rüptür)\b",
            ],
            "HR/SR": [
                r"\bruptura\s+(?:prednjeg\s+križnog\s+ligamenta|pkl|acl)\b",
                r"\b(?:prednji\s+križni\s+ligament|prednja\s+ukrštena\s+veza|pkl)\s+u\s+cijelosti\s+prekinut\b",
                r"\b(?:prednji\s+križni\s+ligament|pkl).*?(?:ruptur|lezij|prekid)\b",
            ],
            "GR": [
                r"\bπρόσθι(?:ος|ου|ο)\s+χιαστ(?:ός|ού|ο)\s+συνδέσμ(?:ος|ου|ο).*?ρήξ(?:η|ης|εως)\b",
                r"\bπροσθιου\s+χιαστου.*?ρηξ\b",
                r"\bασυνέχεια\s+των\s+ινών\s+του\s+πρόσθιου\s+χιαστού\b",
            ],
            "BG": [
                r"\bпредна\s+кръстна\s+връзка.*?(?:руптур|разкъсван|прекъсван)\b",
                r"\bпкв.*?(?:руптур|разкъсван)\b",
            ],
            "NL": [
                r"\b(?:volledige\s+|partiële\s+)?scheur\s+van\s+de\s+voorste\s+kruisband\b",
                r"\bvoorste\s+kruisband.*?(?:scheur|ruptuur|afgescheurd)\b",
                r"\bvkb.*?(?:scheur|ruptuur)\b",
            ],
            "EN": [
                r"\b(?:complete\s+|partial\s+|high-grade\s+)?(?:tear|rupture)\s+of\s+the\s+(?:acl|anterior\s+cruciate\s+ligament)\b",
                r"\b(?:acl|anterior\s+cruciate\s+ligament)\s+is\s+(?:torn|ruptured)\b",
            ]
        }
    },
    # ACL Sprain (Grade 1 / Interstitial)
    {
        "canonical_concept": "ACL_SPRAIN",
        "target_candidates": ["ACL"],
        "clinical_type": "SPRAIN",
        "polarity": "neutral",
        "severity_hint": "grade_1_2",
        "confidence": 0.90,
        "patterns": {
            "EN": [
                r"\b(?:grade\s+1\s+injury|grade\s+1\s+sprain|interstitial\s+sprain)\s+of\s+(?:the\s+)?acl\b",
                r"\binterstitial\s+hyperintense\s+signal\s+of\s+acl,\s+in\s+favor\s+of\s+grade\s+1\b",
            ],
            "TR": [
                r"\bön\s+çapraz\s+bağda\s+grade\s+[i1]\s+(?:sprain|gerilme|zorlanma)\b",
            ]
        }
    },
    # ACL Intact
    {
        "canonical_concept": "ACL_INTACT",
        "target_candidates": ["ACL"],
        "clinical_type": "NORMAL",
        "polarity": "negative",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\b(?:vkb|vorderes\s+kreuzband)\s+intakt\b"],
            "ES": [r"\b(?:lca|ligamento\s+cruzado\s+anterior)\s+(?:intacto|íntegro|conservado|normal)\b", r"\bsin\s+rotura\s+del\s+lca\b"],
            "TR": [r"\b(?:ön\s+çapraz\s+bağ|öçb)\s+(?:normal|doğal|intakt|salim)\b"],
            "HR/SR": [r"\b(?:prednji\s+križni|pkl|acl)\s+(?:očuvan|intaktan|uredan|b\.o\.)\b"],
            "GR": [r"\bπρόσθιος\s+χιαστός\s+(?:φυσιολογικός|ακέραιος)\b"],
            "BG": [r"\b(?:предна\s+кръстна\s+връзка|пкв)\s+(?:е\s+)?интактн\b"],
            "NL": [r"\b(?:voorste\s+kruisband|vkb)\s+(?:is\s+)?intact\b"],
            "EN": [r"\b(?:acl|anterior\s+cruciate\s+ligament)\s+(?:is\s+)?(?:intact|normal|unremarkable)\b"]
        }
    },
    # MCL Tear / Rupture
    {
        "canonical_concept": "MCL_TEAR",
        "target_candidates": ["MCL"],
        "clinical_type": "TEAR",
        "polarity": "positive",
        "confidence": 0.90,
        "patterns": {
            "DE": [
                r"\binnenband(?:riss|ruptur|läsion)\b",
                r"\bmcl[\s-]*(?:ruptur|riss|läsion|anriss)\b",
                r"\bmediales?\s+kollateralband.*?(?:ruptur|riss|läsion)\b",
            ],
            "ES": [
                r"\brotura\s+(?:completa\s+|parcial\s+)?del\s+(?:lcm|ligamento\s+colateral\s+medial)\b",
                r"\b(?:lcm|ligamento\s+colateral\s+medial)\s+(?:con\s+)?(?:rotura|desgarro)\b",
            ],
            "TR": [
                r"\bmedial\s+kollateral\s+ligaman.*?(?:komplet\s+)?(?:yırtık|rüptür|rüptüre|belirgin\s+ödem\s+ve\s+bütünlük\s+kaybı)\b",
                r"\biç\s+yan\s+bağ.*?(?:yırtık|rüptür)\b",
            ],
            "HR/SR": [
                r"\bruptura\s+(?:medijalnog\s+kolateralnog\s+ligamenta|mcl)\b",
                r"\b(?:medijalni\s+kolateralni\s+ligament|mcl).*?(?:ruptur|lezij|parcijalna\s+lezija)\b",
            ],
            "GR": [
                r"\bέσω\s+πλαγί(?:ος|ου)\s+συνδέσμ(?:ος|ου).*?ρήξ(?:η|ης)\b",
                r"\bεσω\s+πλαγιου\s+συνδεσμου.*?ρηξ\b",
            ],
            "BG": [
                r"\bмедиална\s+колатерална\s+връзка.*?(?:руптур|разкъсван)\b",
            ],
            "NL": [
                r"\b(?:scheur|ruptuur)\s+van\s+de\s+mediale\s+band\b",
                r"\bmediale\s+band.*?(?:scheur|ruptuur)\b",
                r"\bmcl.*?(?:scheur|ruptuur)\b",
            ],
            "EN": [
                r"\b(?:complete\s+|high-grade\s+)?(?:tear|rupture)\s+of\s+the\s+(?:mcl|medial\s+collateral\s+ligament)\b",
                r"\b(?:mcl|medial\s+collateral\s+ligament)\s+is\s+(?:torn|ruptured)\b",
            ]
        }
    },

    # --------------------------------------------------------------------------
    # 2. MENISCI (Medial Meniscus, Lateral Meniscus)
    # --------------------------------------------------------------------------
    # Medial Meniscus Tear
    {
        "canonical_concept": "MEDIAL_MENISCUS_TEAR",
        "target_candidates": ["Medial Meniscus"],
        "clinical_type": "TEAR",
        "polarity": "positive",
        "anatomical_hint": "MEDIAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\binnenmeniskus\w*(?:riss|ruptur|läsion)\b",
                r"\bläsion\s+des\s+innenmeniskus\b",
                r"\briss\s+des\s+innenmeniskus\b",
                r"\bhinterhornriss\s+des\s+innenmeniskus\b",
            ],
            "ES": [
                r"\b(?:rotura|desgarro).*?menisco\s+interno\b",
                r"\brotura\s+(?:del\s+)?menisco\s+interno\b",
                r"\bdesgarro\s+(?:del\s+)?menisco\s+interno\b",
                r"\bmenisco\s+interno.*?(?:rotura|desgarro|compleja|radial|horizontal)\b",
                r"\bcuerno\s+posterior\s+del\s+menisco\s+interno.*?(?:rotura|desgarro)\b",
            ],
            "TR": [
                r"\bmedial\s+menisküs.*?(?:radial\s+|kova\s+sapı\s+|horizontal\s+|kompleks\s+|vertikal\s+)?yırtık\b",
                r"\bmedial\s+menisküs.*?(?:rüptür|yırtığı|yırtıkla\s+uyumlu)\b",
                r"\bmedial\s+menisküs\s+posterior\s+hornda\s+grade\s+iii\b",
            ],
            "HR/SR": [
                r"\bruptura\s+medijalnog\s+meniska\b",
                r"\bmedijalni\s+menisk.*?(?:ruptur|lezij|pukotin|kompleksna\s+ruptura)\b",
                r"\brascjep\s+medijalnog\s+meniska\b",
            ],
            "GR": [
                r"\bρήξη.*?(?:του\s+)?έσω\s+μηνίσκου\b",
                r"\bέσω\s+μηνίσκος.*?(?:ρήξη|ρήξης)\b",
            ],
            "BG": [
                r"\bруптура\s+на\s+медиалния\s+менискус\b",
                r"\bмедиален\s+менискус.*?(?:руптур|разкъсван)\b",
            ],
            "NL": [
                r"\bscheur\s+(?:van\s+de\s+)?mediale\s+meniscus\b",
                r"\bmediale\s+meniscus.*?(?:scheur|ruptuur)\b",
            ],
            "EN": [
                r"\b(?:radial|horizontal|vertical|complex|bucket-handle|root)\s+tear\s+of\s+the\s+medial\s+meniscus\b",
                r"\bmedial\s+menisc(?:us|al)\s+(?:tear|rupture)\b",
            ]
        }
    },
    # Lateral Meniscus Tear
    {
        "canonical_concept": "LATERAL_MENISCUS_TEAR",
        "target_candidates": ["Lateral Meniscus"],
        "clinical_type": "TEAR",
        "polarity": "positive",
        "anatomical_hint": "LATERAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\baußenmeniskus\w*(?:riss|ruptur|läsion)\b",
                r"\bläsion\s+des\s+außenmeniskus\b",
                r"\briss\s+des\s+außenmeniskus\b",
                r"\bhinterhornriss\s+des\s+außenmeniskus\b",
            ],
            "ES": [
                r"\b(?:rotura|desgarro).*?menisco\s+externo\b",
                r"\brotura\s+(?:del\s+)?menisco\s+externo\b",
                r"\bdesgarro\s+(?:del\s+)?menisco\s+externo\b",
                r"\bmenisco\s+externo.*?(?:rotura|desgarro|compleja|radial|horizontal)\b",
                r"\bcuerno\s+posterior\s+del\s+menisco\s+externo.*?(?:rotura|desgarro)\b",
            ],
            "TR": [
                r"\blateral\s+menisküs.*?(?:radial\s+|kova\s+sapı\s+|horizontal\s+|kompleks\s+|vertikal\s+)?yırtık\b",
                r"\blateral\s+menisküs.*?(?:rüptür|yırtığı|yırtıkla\s+uyumlu)\b",
                r"\blateral\s+menisküs\s+posterior\s+hornda\s+grade\s+iii\b",
                r"\bkova\s+sapı\s+yırtığı\b",
            ],
            "HR/SR": [
                r"\bruptura\s+lateralnog\s+meniska\b",
                r"\blateralni\s+menisk.*?(?:ruptur|lezij|pukotin|kompleksna\s+ruptura)\b",
                r"\brascjep\s+lateralnog\s+meniska\b",
            ],
            "GR": [
                r"\bρήξη.*?(?:του\s+)?έξω\s+μηνίσκου\b",
                r"\bέξω\s+μηνίσκος.*?(?:ρήξη|ρήξης)\b",
            ],
            "BG": [
                r"\bруптура\s+на\s+латералния\s+менискус\b",
                r"\bлатерален\s+менискус.*?(?:руптур|разкъсван)\b",
            ],
            "NL": [
                r"\bscheur\s+(?:van\s+de\s+)?laterale\s+meniscus\b",
                r"\blaterale\s+meniscus.*?(?:scheur|ruptuur)\b",
            ],
            "EN": [
                r"\b(?:radial|horizontal|vertical|complex|bucket-handle|root)\s+tear\s+of\s+the\s+lateral\s+meniscus\b",
                r"\blateral\s+menisc(?:us|al)\s+(?:tear|rupture)\b",
            ]
        }
    },
    # Meniscus Degeneration (Grade 1/2, mucoid, non-tear)
    {
        "canonical_concept": "MENISCUS_DEGENERATION",
        "target_candidates": ["Medial Meniscus", "Lateral Meniscus"],
        "clinical_type": "DEGENERATION",
        "polarity": "neutral",
        "severity_hint": "grade_1_2",
        "confidence": 0.90,
        "patterns": {
            "DE": [r"\bmeniskusdegeneration\b", r"\bmyxoide\s+degeneration\b", r"\bintrasubstanzsignal\b"],
            "ES": [r"\bdegeneración\s+meniscal\b", r"\bmeniscopatía\s+grado\s+[12iI]+\b", r"\baumento\s+de\s+señal\s+central\b"],
            "TR": [r"\bgrade\s+[12iI]+\s+dejenerasyon\b", r"\bgrade\s+[12iI]+\s+meniskopati\b", r"\bdejenerasyon\s+izlenmiştir\b"],
            "HR/SR": [r"\bdegeneracija\s+meniska\b", r"\bdegenerativne\s+promjene\s+meniska\b", r"\bblaže\s+degenerativno\b"],
            "GR": [r"\bεκφύλιση.*?(?:του\s+)?(?:έσω|έξω)?\s+μηνίσκου\b"],
            "BG": [r"\bдегенерация\s+на\s+менискус\b", r"\bдегенеративни\s+промени\b"],
            "NL": [r"\bdegeneratie\s+van\s+de\s+meniscus\b", r"\bdegeneratieve\s+veranderingen\b"],
            "EN": [r"\bmucoid\s+degeneration\b", r"\bgrade\s+[12iI]+\s+degeneration\b", r"\bintrasubstance\s+signal\b"]
        }
    },

    # --------------------------------------------------------------------------
    # 3. OSTEOARTHRITIS / CARTILAGE WEAR (Medial OA, Lateral OA, PF OA)
    # --------------------------------------------------------------------------
    # Medial Tibiofemoral OA / Cartilage Loss
    {
        "canonical_concept": "MEDIAL_TF_OA",
        "target_candidates": ["Medial OA"],
        "clinical_type": "CARTILAGE_WEAR",
        "polarity": "positive",
        "anatomical_hint": "MEDIAL",
        "confidence": 0.90,
        "patterns": {
            "DE": [
                r"\bmediale\s+gonarthrose\b",
                r"\bknorpelschaden\s+im\s+medialen\s+femorotibialen\b",
                r"\bmedialen\s+femurkondyl.*?knorpelverlust\b",
            ],
            "ES": [
                r"\b(?:artrosis|gonartrosis)\s+femorotibial\s+medial\b",
                r"\boa\s+femorotibial\s+medial\b",
                r"\bcondropatía\s+(?:de\s+espesor\s+total\s+|avanzada\s+)?del\s+compartimento\s+femorotibial\s+medial\b",
            ],
            "TR": [
                r"\bmedial\s+kompartmanda\s+osteoartrit\b",
                r"\bmedial\s+femorotibial\s+eklem\s+kıkırdak\s+kaybı\b",
                r"\bmedial\s+femoral\s+kondilde\s+evre\s+[34]\s+kıkırdak\b",
            ],
            "HR/SR": [
                r"\bmedijalna\s+gonartroza\b",
                r"\bdenudacija\s+hrskavice\s+medijalnog\s+kompartmenta\b",
                r"\bhondromalacija\s+(?:iii|iv|[34])\b.*?medijaln",
            ],
            "GR": [
                r"\bοστεοαρθρίτιδα.*?έσω\s+διαμερίσματος\b",
                r"\bλέπτυνση\s+και\s+ανωμαλία\s+παρυφής\s+του\s+αρθρικού\s+χόνδρου.*?έσω\b",
            ],
            "BG": [
                r"\bгонартроза\s+медиално\b",
                r"\bизтъняване\s+на\s+хрущяла\s+на\s+медиалния\s+кондил\b",
            ],
            "NL": [
                r"\bmediale\s+gonartrose\b",
                r"\bkraakbeenverlies\s+mediale\s+compartiment\b",
            ],
            "EN": [
                r"\bmedial\s+compartment\s+osteoarthritis\b",
                r"\bmedial\s+femoral\s+condyle\s+(?:full-thickness\s+)?cartilage\s+loss\b",
            ]
        }
    },
    # Lateral Tibiofemoral OA / Cartilage Loss
    {
        "canonical_concept": "LATERAL_TF_OA",
        "target_candidates": ["Lateral OA"],
        "clinical_type": "CARTILAGE_WEAR",
        "polarity": "positive",
        "anatomical_hint": "LATERAL",
        "confidence": 0.90,
        "patterns": {
            "DE": [
                r"\blaterale\s+gonarthrose\b",
                r"\bknorpelschaden\s+im\s+lateralen\s+femorotibialen\b",
            ],
            "ES": [
                r"\b(?:artrosis|gonartrosis)\s+femorotibial\s+lateral\b",
                r"\boa\s+femorotibial\s+lateral\b",
                r"\bcondropatía\s+(?:de\s+espesor\s+total\s+|avanzada\s+)?del\s+compartimento\s+femorotibial\s+lateral\b",
            ],
            "TR": [
                r"\blateral\s+kompartmanda\s+osteoartrit\b",
                r"\blateral\s+femorotibial\s+eklem\s+kıkırdak\s+kaybı\b",
            ],
            "HR/SR": [
                r"\blateralna\s+gonartroza\b",
                r"\bdenudacija\s+hrskavice\s+lateralnog\s+kompartmenta\b",
                r"\bhondromalacija\s+(?:iii|iv|[34])\b.*?lateraln",
            ],
            "GR": [
                r"\bοστεοαρθρίτιδα.*?έξω\s+διαμερίσματος\b",
                r"\bλέπτυνση\s+και\s+ανωμαλία\s+παρυφής\s+του\s+αρθρικού\s+χόνδρου.*?έξω\b",
            ],
            "BG": [
                r"\bгонартроза\s+латерално\b",
                r"\bизтъняване\s+на\s+хрущяла\s+на\s+латералния\s+кондил\b",
            ],
            "NL": [
                r"\blateraal\s+kraakbeenlijden\b",
                r"\bvolledig\s+kraakbeenverlies.*?lateraal\b",
            ],
            "EN": [
                r"\blateral\s+compartment\s+osteoarthritis\b",
                r"\blateral\s+femoral\s+condyle\s+(?:full-thickness\s+)?cartilage\s+loss\b",
            ]
        }
    },
    # Patellofemoral OA / Chondropathy
    {
        "canonical_concept": "PF_OA",
        "target_candidates": ["PF OA"],
        "clinical_type": "CARTILAGE_WEAR",
        "polarity": "positive",
        "anatomical_hint": "PATELLOFEMORAL",
        "confidence": 0.90,
        "patterns": {
            "DE": [
                r"\bretropatellare\s+chondropathie\b",
                r"\bfemoropatellare\s+gonarthrose\b",
                r"\bpatellofemorale\s+arthrose\b",
            ],
            "ES": [
                r"\bcondropatía\s+(?:rotuliana|patelar)\b",
                r"\bartrosis\s+patelofemoral\b",
                r"\bcondropatía\s+de\s+la\s+tróclea\b",
                r"\bcondromalacia\s+rotuliana\b",
            ],
            "TR": [
                r"\bpatellofemoral\s+eklem\s+kıkırdaklarında\s+incelme\b",
                r"\bpatellofemoral\s+osteoartrit\b",
                r"\bretropatellar\s+kondromalazi\b",
                r"\bpatellar\s+kartilajda\s+kayıp\b",
            ],
            "HR/SR": [
                r"\bpf\s+artrotske\s+promjene\b",
                r"\bhondromalacija\s+(?:iii|iv|[34])\b.*?patele\b",
                r"\bhondromalacijom\s+iv\s+stupnja\s+hrskavice\s+interfasetarno\b",
                r"\bhondromalacija\s+iii°\s+obje\s+fasete\s+patele\b",
            ],
            "GR": [
                r"\bεπιγονατιδομηριαία\s+οστεοαρθρίτιδα\b",
                r"\bχονδροπάθεια\s+επιγονατίδας\b",
                r"\bχονδροπάθεια.*?τροχιλίας\b",
            ],
            "BG": [
                r"\bпателофеморална\s+артроза\b",
                r"\bхондромалация\s+на\s+пателата\b",
            ],
            "NL": [
                r"\bpatellofemorale\s+artrose\b",
                r"\bkraakbeendefect\s+patella\b",
            ],
            "EN": [
                r"\bpatellofemoral\s+(?:compartment\s+)?osteoarthritis\b",
                r"\bpatellar\s+(?:facet\s+)?chondromalacia\s+grade\s+[34]\b",
                r"\btrochlear\s+(?:full-thickness\s+)?cartilage\s+loss\b",
            ]
        }
    },

    # --------------------------------------------------------------------------
    # 4. FLUID / SYNOVIUM (Effusion, Synovitis)
    # --------------------------------------------------------------------------
    # Significant Effusion
    {
        "canonical_concept": "SIGNIFICANT_EFFUSION",
        "target_candidates": ["Effusion"],
        "clinical_type": "EFFUSION",
        "polarity": "positive",
        "confidence": 0.90,
        "patterns": {
            "DE": [r"\b(?:deutlicher|ausgeprägter|großer|relevanter)\s+gelenkerguss\b", r"\berguss\s+im\s+kniegelenk\b"],
            "ES": [r"\bderrame\s+articular\b", r"\blíquido\s+articular\s+abundante\b", r"\babundante\s+derrame\b"],
            "TR": [r"\beklem\s+içi\s+sıvı\s+artışı\b", r"\bdiz\s+eklem\s+aralığında\s+sıvı\b", r"\bartmış\s+efüzyon\b", r"\byaygın\s+sıvı\s+artışı\b"],
            "HR/SR": [r"\bopsežan\s+zglobni\s+izljev\b", r"\bzglobni\s+izljev\b", r"\bveća\s+količina\s+izljeva\b"],
            "GR": [r"\bσυλλογή\s+υγρού\b", r"\bυδράρθρου\b", r"\bσημαντική\s+συλλογή\s+υγρού\b"],
            "BG": [r"\bставен\s+излив\b", r"\bзначителен\s+ставен\s+излив\b"],
            "NL": [r"\buitgesproken\s+gewrichtsuitstorting\b", r"\bgewrichtsvocht\b", r"\bhydrops\b"],
            "EN": [r"\b(?:moderate|large|significant)\s+joint\s+effusion\b", r"\bjoint\s+effusion\b"]
        }
    },
    # Synovial Proliferation / Synovitis
    {
        "canonical_concept": "SYNOVIAL_PROLIFERATION",
        "target_candidates": ["Synovitis"],
        "clinical_type": "SYNOVITIS",
        "polarity": "positive",
        "confidence": 0.90,
        "patterns": {
            "DE": [r"\bsynovialisproliferation\b", r"\bsynovialitis\b", r"\bsynoviale\s+verdickung\b", r"\bsynovitis\b"],
            "ES": [r"\bsinovitis\s+proliferativa\b", r"\bengrosamiento\s+sinovial\b", r"\bhipertrofia\s+sinovial\b", r"\bsinovitis\b"],
            "TR": [r"\bsinovyal\s+kalınlaşma\b", r"\bproliferatif\s+sinovit\b", r"\bsinovitis\b", r"\bsinovit\b"],
            "HR/SR": [r"\bproliferacija\s+sinovije\b", r"\bumjerenu\s+proliferaciju\s+sinovije\b", r"\bupalne\s+promjene\s+sinovije\b", r"\bsinovitis\b"],
            "GR": [r"\bυμενίτιδα\b", r"\bπάχυνση\s+του\s+υμένα\b"],
            "BG": [r"\bсиновит\b", r"\bхипертрофия\s+на\s+синовията\b"],
            "NL": [r"\bverdikkingen\s+van\s+het\s+synovium\b", r"\bsynovitis\b", r"\bsynoviale\s+proliferatie\b"],
            "EN": [r"\bsynovial\s+(?:thickening|hypertrophy|proliferation)\b", r"\bprominent\s+synovitis\b"]
        }
    },

    # --------------------------------------------------------------------------
    # 5. BONE (Contusion, Fracture)
    # --------------------------------------------------------------------------
    # Bone Marrow Edema / Contusion
    {
        "canonical_concept": "BONE_CONTUSION",
        "target_candidates": ["Contusion"],
        "clinical_type": "EDEMA",
        "polarity": "positive",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\bknochenödem\b", r"\bknochenmarködem\b"],
            "ES": [r"\bcontusión\s+ósea\b", r"\bedema\s+óseo\s+postraumático\b"],
            "TR": [r"\bkissing\s+kontüzyon\b", r"\bkontüzyonel\s+kemik\s+iliği\s+ödem\b", r"\bkemik\s+kontüzyonu\b"],
            "HR/SR": [r"\bkoštana\s+kontuzija\b", r"\btraumatski\s+koštani\s+edem\b"],
            "GR": [r"\bοίδημα\s+του\s+οστικού\s+μυελού\b", r"\bοστικό\s+μώλωπα\b", r"\bοστικό\s+οίδημα\b"],
            "BG": [r"\bкостно-мозъчен\s+едем\b", r"\bкостна\s+контузия\b"],
            "NL": [r"\bbotoedeem\b", r"\bbotcontusie\b", r"\bbeenmergoedeem\b"],
            "EN": [r"\bbone\s+marrow\s+edema\b", r"\bbone\s+contusion\b", r"\bbone\s+bruise\b"]
        }
    },
    # Cortical / Avulsion / Depressed Fracture
    {
        "canonical_concept": "CORTICAL_FRACTURE",
        "target_candidates": ["Fracture"],
        "clinical_type": "FRACTURE",
        "polarity": "positive",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\bknöchernem\s+ausriss\b",
                r"\bfraktur\b",
                r"\babrissfraktur\b",
                r"\bflake\b",
            ],
            "ES": [
                r"\blínea\s+de\s+fractura\b",
                r"\bfractura\s+cortical\b",
                r"\bavulsión\s+ósea\b",
            ],
            "TR": [
                r"\bfraktür\s+hattı\b",
                r"\bnondeplase\s+fraktür\b",
                r"\bsubartiküler\s+fraktür\b",
                r"\bavülsiyon\s+fraktürü\b",
                r"\bçökme\s+kırığı\b",
            ],
            "HR/SR": [
                r"\bfraktura\b",
                r"\bavulzijska\s+fraktura\b",
                r"\bprijelom\b",
                r"\bimpresijska\s+fraktura\b",
            ],
            "GR": [
                r"\bκάταγμα\b",
                r"\bρωγμώδες\s+κάταγμα\b",
                r"\bαποσπαστικό\s+κάταγμα\b",
            ],
            "BG": [
                r"\bостеохондрална\s+фрактура\b",
                r"\bфрактурна\s+линия\b",
                r"\bфрактура\b",
            ],
            "NL": [
                r"\bverplaatste\s+fractuur\b",
                r"\bavulsiefractuur\b",
                r"\bcorticale\s+breuk\b",
            ],
            "EN": [
                r"\bfracture\s+line\b",
                r"\bcortical\s+disruption\b",
                r"\bdepressed\s+fracture\b",
                r"\bavulsion\s+fracture\b",
            ]
        }
    },

    # --------------------------------------------------------------------------
    # 6. CYST (Baker's Cyst)
    # --------------------------------------------------------------------------
    {
        "canonical_concept": "BAKERS_CYST",
        "target_candidates": ["Baker's"],
        "clinical_type": "CYST",
        "polarity": "positive",
        "anatomical_hint": "POPLITEAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\bbaker[\s-]*(?:zyste|cyst)\b", r"\bpoplitealzyste\b"],
            "ES": [r"\bquiste\s+de\s+baker\b", r"(?<!pequeño\s)(?<!pequeno\s)\bquiste\s+poplíteo\b", r"(?<!pequeño\s)(?<!pequeno\s)\bquiste\s+popliteo\b"],
            "TR": [r"\bbaker\s+kisti\b", r"\bpopliteal\s+kist\b"],
            "HR/SR": [r"\bbakerova\s+cista\b", r"\bbaker\s+cista\b", r"\bpoplitealna\s+cista\b", r"\bbakerovoj\s+cisti\b"],
            "GR": [r"\bκύστη\s+baker\b", r"\bιγνυακή\s+κύστη\b"],
            "BG": [r"\bбейкерова\s+киста\b", r"\bкиста\s+на\s+бейкър\b", r"\bпоплитеална\s+киста\b"],
            "NL": [r"\bbaker[\s-]*cyste\b", r"\bbakers\s+cyste\b", r"\bpopliteale\s+cyste\b"],
            "EN": [r"\bbaker'?s\s+cyst\b", r"\bpopliteal\s+cyst\b"]
        }
    }
]


# Pre-compile all patterns for performance
COMPILED_LEXICON = []
for entry in MULTILINGUAL_LEXICON:
    compiled_patterns = {}
    for lang, pat_list in entry["patterns"].items():
        compiled_patterns[lang] = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in pat_list]
    compiled_entry = dict(entry)
    compiled_entry["compiled_patterns"] = compiled_patterns
    COMPILED_LEXICON.append(compiled_entry)


def extract_multilingual_concepts(report_text: str) -> List[ConceptRecord]:
    """
    Extract all matching multilingual clinical concepts from report text with clause-level scoping.
    """
    norm_rep = normalize_multilingual_text(report_text)
    clauses = split_into_scoped_clauses(norm_rep)
    records = []

    for clause in clauses:
        clause_is_neg = is_clause_truly_negative(clause)
        c_low = clause.lower()

        for entry in COMPILED_LEXICON:
            canonical = entry["canonical_concept"]
            targets = list(entry["target_candidates"])
            clin_type = entry["clinical_type"]
            pol = entry["polarity"]
            sev_hint = entry.get("severity_hint")
            anat_hint = entry.get("anatomical_hint")
            conf = entry.get("confidence", 0.90)

            # Compartment-aware scoping for meniscus targets
            scoped_targets = []
            for t in targets:
                if t == "Medial Meniscus":
                    if any(w in c_low for w in ["lateral", "externo", "aussen", "außen", "lateralni", "латерал", "έξω"]) and not any(w in c_low for w in ["medial", "interno", "innen", "medijalni", "медиал", "έσω"]):
                        continue
                if t == "Lateral Meniscus":
                    if any(w in c_low for w in ["medial", "interno", "innen", "medijalni", "медиал", "έσω"]) and not any(w in c_low for w in ["lateral", "externo", "aussen", "außen", "lateralni", "латерал", "έξω"]):
                        continue
                scoped_targets.append(t)
            if not scoped_targets:
                continue

            for lang, pattern_list in entry["compiled_patterns"].items():
                for pat in pattern_list:
                    if pat.search(clause):
                        actual_pol = pol
                        if clause_is_neg and pol == "positive":
                            actual_pol = "negative"

                        rec = ConceptRecord(
                            surface_forms=[clause.strip()],
                            canonical_concept=canonical if actual_pol == "positive" else f"{canonical}_NEGATIVE",
                            target_candidates=scoped_targets,
                            clinical_type=clin_type,
                            language=lang,
                            polarity=actual_pol,
                            severity_hint=sev_hint,
                            anatomical_hint=anat_hint,
                            confidence=conf,
                            evidence_text=clause.strip()
                        )
                        records.append(rec)

    return records



def get_multilingual_evidence_for_target(
    target: str,
    full_report: str,
    baseline_evidence: str = ""
) -> Optional[ConceptRecord]:
    """
    Retrieve the most relevant multilingual concept for a specific target.
    """
    if not full_report:
        return None

    all_concepts = extract_multilingual_concepts(full_report)
    target_concepts = [c for c in all_concepts if target in c.target_candidates]

    if not target_concepts:
        return None

    # Priority sorting:
    # 1. Positive definite pathology
    # 2. Negative explicit confirmation
    # 3. Confidence score
    def sort_key(c: ConceptRecord) -> Tuple[int, float]:
        score = 0
        if c.polarity == "positive":
            score += 10
        elif c.polarity == "negative":
            score += 5
        if c.clinical_type in ["TEAR", "FRACTURE", "CYST", "CARTILAGE_WEAR"]:
            score += 2
        return (score, c.confidence)

    target_concepts.sort(key=sort_key, reverse=True)
    return target_concepts[0]

