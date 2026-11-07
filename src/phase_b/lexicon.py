"""
Phase B Step 3 Remediation: Multilingual Clinical Concept Expansion & Normalization Engine (Step 3B).

Provides structured mappings for 8 language groups (EN, DE, ES, TR, HR/SR, GR, BG, NL)
normalizing free-text medical phrasing into canonical clinical concepts.
Preserves exact character spans [start_char, end_char] without text mutation.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import re

from phase_b.config import TARGETS
from phase_b.parser import (
    SpanOffset,
    ScopedClause,
    split_text_into_scoped_clauses,
    analyze_fracture_displacement_relations
)


@dataclass
class ConceptRecord:
    surface_forms: List[str]
    canonical_concept: str
    target_candidates: List[str]
    clinical_type: str
    language: str
    polarity: str  # "positive", "negative", "uncertain", "neutral"
    severity_hint: Optional[str] = None
    anatomical_hint: Optional[str] = None
    confidence: float = 0.90
    evidence_text: str = ""
    evidence_span: Optional[SpanOffset] = None
    rule_id: str = "RULE_LEXICON_MATCH"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.evidence_span:
            d["evidence_span"] = self.evidence_span.to_dict()
        return d


def normalize_multilingual_text(text: str) -> str:
    """Normalize text across unicode encodings (e.g. Greek micro sign, Turkish i, ligatures)."""
    if not text:
        return ""
    text = text.replace("\u00b5", "\u03bc")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text


NEGATIVE_MODIFIERS = [
    # English
    r"\bno\b", r"\bnone\b", r"\bwithout\b", r"\bnegative\b", r"\bintact\b", r"\bunremarkable\b",
    r"\bfree\s+of\b", r"\bnot\s+seen\b", r"\bno\s+evidence\s+of\b", r"\bno\s+signs?\s+of\b",
    # German
    r"\bkein(?:e|en|er|es)?\b", r"\bohne\b", r"\bintakt(?:e|en|er|es)?\b", r"\bunauffällig(?:e|en|er|es)?\b",
    r"\bregelrecht(?:e|en|er|es)?\b", r"\bohne\s+fissur\b", r"\bkeine\s+fissur\b", r"\bkeine\s+infraktion\b",
    r"\bdurchgehend\s+intakt\b",
    # Spanish
    r"\bsin\b", r"\bno\s+se\s+observa\b", r"\bno\s+hay\b", r"\bintacto\b", r"\bintacta\b",
    r"\bnormal(?:es)?\b", r"\bconservado\b", r"\bconservada\b", r"\bíntegro\b", r"\bíntegra\b",
    r"\bsin\s+alteraciones\b", r"\bsin\s+signos\s+de\b", r"\bsin\s+evidencia\s+de\b",
    # Turkish
    r"\byok\b", r"\bsaptanmadı\b", r"\bsaptanmadi\b", r"\bizlenmedi\b", r"\bizlenmemiştir\b",
    r"\bizlenmemistir\b", r"\bsaptanmamıştır\b", r"\bdoğaldır\b", r"\bdogaldir\b", r"\bdoğal\b",
    r"\bdogal\b", r"\bintakt\b", r"\bnormaldir\b", r"\bnormal\b", r"\bbütünlüğü\s+korunmuştur\b",
    # Croatian / Serbian
    r"\bbez\b", r"\bnema\b", r"\bnije\s+nađen\b", r"\bnije\s+viđeno\b", r"\bintaktan\b",
    r"\bintakta\b", r"\buredan\b", r"\buredna\b", r"\bočuvan\b", r"\bočuvana\b",
    r"\bbez\s+znakova\b", r"\bbez\s+rupture\b",
    # Greek
    r"\bχωρίς\b", r"\bδεν\s+παρατηρείται\b", r"\bδεν\s+διαπιστώνεται\b", r"\bδεν\s+αναδεικνύεται\b",
    r"\bάθικτος\b", r"\bάθικτη\b", r"\bάθικτο\b", r"\bφυσιολογικός\b", r"\bφυσιολογική\b",
    # Bulgarian
    r"\bбез\b", r"\bняма\b", r"\bинтактна\b", r"\bинтактен\b", r"\bинтактно\b",
    r"\bбез\s+данни\s+за\b", r"\bняма\s+данни\s+за\b", r"\bзапазен\b", r"\bзапазена\b",
    # Dutch
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b", r"\bintact\b", r"\bnormaal\b", r"\bonopvallend\b",
    r"\bzonder\s+afwijkingen\b", r"\bgeen\s+aanwijzingen\s+voor\b"
]

COMPILED_NEG_MODIFIERS = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in NEGATIVE_MODIFIERS]


def is_clause_truly_negative_non_destructive(clause_text: str, base_offset: int = 0) -> bool:
    """Non-destructively evaluate if a clause is negative, checking relational non-displacement semantics."""
    rel = analyze_fracture_displacement_relations(clause_text, base_offset)
    if rel["is_affirmative_non_displaced"]:
        return False
    if rel["is_coordinated_negation"]:
        return True

    for pat in COMPILED_NEG_MODIFIERS:
        if pat.search(clause_text):
            return True
    return False


# ==============================================================================
# MASTER STRUCTURED MULTILINGUAL CLINICAL LEXICON
# ==============================================================================
MULTILINGUAL_LEXICON: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # 1. LIGAMENTS (ACL, MCL)
    # --------------------------------------------------------------------------
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
                r"\bvorderes?\s+kreuzband[\s-]*(?:ruptur|riss|läsion|teilruptur|anriss)\b",
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
            ],
            "ES": [
                r"\besguince\s+(?:de\s+grado\s+1\s+|leve\s+)?del\s+lca\b",
            ],
            "DE": [
                r"\bgrad\s+1\s+zerrung\s+des\s+vorderen\s+kreuzbandes\b",
            ]
        }
    },
    {
        "canonical_concept": "ACL_INTACT",
        "target_candidates": ["ACL"],
        "clinical_type": "NORMAL",
        "polarity": "negative",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\bunauffälliger\s+vkb\b",
                r"\bvkb\s+intakt\b",
                r"\bvorderes?\s+kreuzband.*?(?:intakt|regelrecht|unauffällig)\b",
                r"\bvordere\s+kreuzband\s+zeigt\s+sich\s+durchgehend\s+intakt\b",
                r"\bintaktes\s+vorderes\s+kreuzband\b",
                r"\bkeine\s+vkb[\s-]*(?:ruptur|läsion)\b",
            ],
            "ES": [
                r"\blca\s+(?:está\s+)?(?:conservado|íntegro|intacto|normal)\b",
                r"\bligamento\s+cruzado\s+anterior\s+(?:conservado|íntegro|intacto|normal)\b",
                r"\bsin\s+(?:evidencia\s+de\s+)?rotura\s+del\s+(?:lca|ligamento\s+cruzado\s+anterior)\b",
            ],
            "TR": [
                r"\bön\s+çapraz\s+bağ\s+(?:doğaldır|doğal|intakt|normaldir)\b",
                r"\böçb\s+(?:doğaldır|intakt|normal)\b",
                r"\bön\s+çapraz\s+bağda\s+yırtık\s+(?:izlenmedi|saptanmadı|izlenmemiştir)\b",
            ],
            "HR/SR": [
                r"\bprednji\s+križni\s+ligament\s+(?:uredan|očuvan|intaktan)\b",
                r"\bpkl\s+(?:uredan|očuvan|intaktan)\b",
                r"\bbez\s+rupture\s+prednjeg\s+križnog\s+ligamenta\b",
            ],
            "GR": [
                r"\bδεν\s+παρατηρείται\s+ρήξη\s+πρόσθιου\s+χιαστού\b",
                r"\bπρόσθιος\s+χιαστός\s+σύνδεσμος\s+άθικτος\b",
                r"\bχωρίς\s+ρήξη\s+πρόσθιου\s+χιαστού\b",
            ],
            "BG": [
                r"\bпредна\s+кръстна\s+връзка\s+е\s+интактна\b",
                r"\bбез\s+данни\s+за\s+руптура\s+на\s+пкв\b",
            ],
            "NL": [
                r"\bvoorste\s+kruisband\s+is\s+intact\b",
                r"\bgeen\s+scheur\s+van\s+de\s+voorste\s+kruisband\b",
            ],
            "EN": [
                r"\bacl\s+is\s+intact\b",
                r"\banterior\s+cruciate\s+ligament\s+is\s+intact\b",
                r"\bno\s+acl\s+tear\b",
            ]
        }
    },
    {
        "canonical_concept": "MCL_TEAR",
        "target_candidates": ["MCL"],
        "clinical_type": "TEAR",
        "polarity": "positive",
        "severity_hint": "moderate",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\binnenband(?:ruptur|riss|läsion|zerrung)\b",
                r"\bmcl[\s-]*(?:ruptur|riss|läsion)\b",
                r"\bmediales\s+kollateralband.*?(?:ruptur|riss)\b",
            ],
            "ES": [
                r"\brotura\s+(?:parcial\s+|completa\s+)?del\s+(?:lcm|ligamento\s+colateral\s+medial|ligamento\s+colateral\s+interno)\b",
                r"\blesión\s+(?:de\s+grado\s+[23]\s+)?del\s+lcm\b",
            ],
            "TR": [
                r"\biç\s+yan\s+bağ.*?(?:yırtık|rüptür|grade\s+[23]\s+sprain)\b",
                r"\bmcl.*?(?:yırtık|rüptür|sprain)\b",
            ],
            "HR/SR": [
                r"\bruptura\s+(?:medijalnog\s+kolateralnog\s+ligamenta|mkl|mcl)\b",
                r"\blesio\s+ligamenti\s+collateralis\s+medialis\b",
            ],
            "GR": [
                r"\bέσω\s+πλάγιος\s+σύνδεσμος.*?ρήξ\b",
            ],
            "BG": [
                r"\bмедиален\s+колатерален\s+лигамент.*?(?:руптур|разкъсван)\b",
            ],
            "NL": [
                r"\b(?:partiële\s+|volledige\s+)?scheur\s+van\s+de\s+mediale\s+band\b",
                r"\bmediale\s+collaterale\s+band.*?(?:scheur|letsel)\b",
            ],
            "EN": [
                r"\b(?:mcl|medial\s+collateral\s+ligament)\s+(?:tear|sprain\s+grade\s+[23])\b",
                r"\btear\s+of\s+(?:the\s+)?(?:mcl|medial\s+collateral\s+ligament)\b",
            ]
        }
    },
    {
        "canonical_concept": "MCL_INTACT",
        "target_candidates": ["MCL"],
        "clinical_type": "NORMAL",
        "polarity": "negative",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\bunauffälliges\s+innenband\b",
                r"\binnenband\s+intakt\b",
                r"\bkeine\s+innenband(?:ruptur|läsion)\b",
            ],
            "ES": [
                r"\bligamento\s+colateral\s+(?:medial|interno)\s+(?:conservado|íntegro|intacto|normal)\b",
                r"\blcm\s+conservado\b",
            ],
            "TR": [
                r"\biç\s+yan\s+bağ\s+(?:doğaldır|doğal|intakt|normal)\b",
            ],
            "HR/SR": [
                r"\bmedijalni\s+kolateralni\s+ligament\s+(?:uredan|očuvan|intaktan)\b",
            ],
            "GR": [
                r"\bέσω\s+πλάγιος\s+σύνδεσμος\s+άθικτος\b",
            ],
            "BG": [
                r"\bмедиален\s+колатерален\s+лигамент\s+е\s+интактен\b",
            ],
            "NL": [
                r"\bmediale\s+collaterale\s+band\s+is\s+intact\b",
            ],
            "EN": [
                r"\bmcl\s+is\s+intact\b",
                r"\bmedial\s+collateral\s+ligament\s+is\s+intact\b",
            ]
        }
    },

    # --------------------------------------------------------------------------
    # 2. MENISCUS (Medial Meniscus, Lateral Meniscus)
    # --------------------------------------------------------------------------
    {
        "canonical_concept": "MEDIAL_MENISCUS_TEAR",
        "target_candidates": ["Medial Meniscus"],
        "clinical_type": "TEAR",
        "polarity": "positive",
        "anatomical_hint": "MEDIAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\binnenmeniskus[\s-]*(?:riss|ruptur|läsion|teilruptur|hinterhornriss)\b",
                r"\bdegenerativer\s+innenmeniskusriss\b",
                r"\briss\s+des\s+innenmeniskushinterhorns\b",
                r"\bkomplexe\s+innenmeniskusläsion\b",
            ],
            "ES": [
                r"\brotura\s+(?:compleja|oblicua|horizontal|radial|en\s+asa\s+de\s+cubo|longitudinal|[a-z]+\s+)*?(?:del\s+cuerno\s+(?:anterior|posterior)\s+del\s+|del\s+cuerpo\s+del\s+|de\s+la\s+raíz\s+del\s+|del\s+)?(?:menisco\s+interno|menisco\s+medial)\b",
                r"\b(?:menisco\s+interno|menisco\s+medial)\s+(?:con\s+)?(?:rotura|desgarro)\b",
                r"\bdesgarro\s+del\s+(?:menisco\s+interno|menisco\s+medial)\b",
            ],

            "TR": [
                r"\bmedial\s+menisküs.*?(?:yırtık|horizontal\s+yırtık|grade\s+3\s+yırtık|eklem\s+yüzüne\s+uzanan)\b",
                r"\bmedial\s+menisküs\s+posterior\s+hornda\s+eklem\s+yüzüne\s+uzanan\b",
            ],
            "HR/SR": [
                r"\bruptura\s+(?:medijalnog\s+meniska|medijalnog\s+meniskusa)\b",
                r"\blesio\s+menisci\s+medialis\b",
            ],
            "GR": [
                r"\bρήξη\s+(?:έσω\s+μηνίσκου|του\s+έσω\s+μηνίσκου)\b",
            ],
            "BG": [
                r"\bразкъсване\s+на\s+медиалния\s+менискус\b",
                r"\bруптура\s+на\s+медиален\s+менискус\b",
            ],
            "NL": [
                r"\b(?:complexe\s+|horizontale\s+)?scheur\s+van\s+de\s+mediale\s+meniscus\b",
            ],
            "EN": [
                r"\b(?:tear|rupture)\s+of\s+(?:the\s+)?medial\s+meniscus\b",
                r"\bmedial\s+menisc(?:us|al)\s+tear\b",
            ]
        }
    },
    {
        "canonical_concept": "MEDIAL_MENISCUS_INTACT",
        "target_candidates": ["Medial Meniscus"],
        "clinical_type": "NORMAL",
        "polarity": "negative",
        "anatomical_hint": "MEDIAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\binnenmeniskus\s+(?:intakt|unauffällig|regelrecht)\b",
                r"\bkeine\s+innenmeniskusläsion\b",
                r"\bkeine\s+meniskusläsion\b",
            ],
            "ES": [
                r"\bmenisco\s+(?:interno|medial)\s+(?:conservado|íntegro|intacto|normal|sin\s+alteraciones)\b",
                r"\bsin\s+rotura\s+del\s+menisco\s+(?:interno|medial)\b",
            ],
            "TR": [
                r"\bmedial\s+menisküs\s+(?:doğaldır|doğal|intakt|normaldir)\b",
                r"\bmedial\s+menisküste\s+yırtık\s+(?:izlenmedi|saptanmadı|izlenmemiştir)\b",
            ],
            "HR/SR": [
                r"\bmedijalni\s+menisk\s+(?:uredan|očuvan|intaktan|bez\s+znakova\s+rupture)\b",
                r"\bdegeneracija\s+medijalnog\s+meniska\s+bez\s+znakova\s+rupture\b",
            ],
            "GR": [
                r"\bέσω\s+μηνίσκος\s+άθικτος\b",
                r"\bχωρίς\s+ρήξη\s+έσω\s+μηνίσκου\b",
            ],
            "BG": [
                r"\bмедиален\s+менискус\s+е\s+интактен\b",
                r"\bбез\s+данни\s+за\s+лезия\s+на\s+медиалния\s+менискус\b",
            ],
            "NL": [
                r"\bmediale\s+meniscus\s+is\s+intact\b",
            ],
            "EN": [
                r"\bmedial\s+meniscus\s+is\s+intact\b",
                r"\bno\s+medial\s+meniscal\s+tear\b",
            ]
        }
    },
    {
        "canonical_concept": "LATERAL_MENISCUS_TEAR",
        "target_candidates": ["Lateral Meniscus"],
        "clinical_type": "TEAR",
        "polarity": "positive",
        "anatomical_hint": "LATERAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\baussenmeniskus[\s-]*(?:riss|ruptur|läsion)\b",
                r"\baußenmeniskus[\s-]*(?:riss|ruptur|läsion)\b",
            ],
            "ES": [
                r"\brotura\s+(?:compleja|oblicua|horizontal|radial|en\s+asa\s+de\s+cubo|longitudinal|[a-z]+\s+)*?(?:del\s+cuerno\s+(?:anterior|posterior)\s+del\s+|del\s+cuerpo\s+del\s+|de\s+la\s+raíz\s+del\s+|del\s+)?(?:menisco\s+externo|menisco\s+lateral)\b",
                r"\b(?:menisco\s+externo|menisco\s+lateral)\s+(?:con\s+)?(?:rotura|desgarro)\b",
            ],

            "TR": [
                r"\blateral\s+menisküs.*?(?:yırtık|kova\s+sapı\s+yırtığı|grade\s+3\s+yırtık)\b",
                r"\blateral\s+menisküste\s+kova\s+sapı\s+yırtığı\b",
            ],
            "HR/SR": [
                r"\bruptura\s+(?:lateralnog\s+meniska|lateralnog\s+meniskusa)\b",
                r"\blesio\s+menisci\s+lateralis\b",
            ],
            "GR": [
                r"\bρήξη\s+(?:έξω\s+μηνίσκου|του\s+έξω\s+μηνίσκου)\b",
            ],
            "BG": [
                r"\bразкъсване\s+на\s+латералния\s+менискус\b",
                r"\bруптура\s+на\s+латерален\s+менискус\b",
            ],
            "NL": [
                r"\b(?:complexe\s+|horizontale\s+)?scheur\s+van\s+de\s+laterale\s+meniscus\b",
            ],
            "EN": [
                r"\b(?:tear|rupture)\s+of\s+(?:the\s+)?lateral\s+meniscus\b",
                r"\blateral\s+menisc(?:us|al)\s+tear\b",
            ]
        }
    },
    {
        "canonical_concept": "LATERAL_MENISCUS_INTACT",
        "target_candidates": ["Lateral Meniscus"],
        "clinical_type": "NORMAL",
        "polarity": "negative",
        "anatomical_hint": "LATERAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [
                r"\b(?:aussenmeniskus|außenmeniskus)\s+(?:intakt|unauffällig|regelrecht)\b",
                r"\bkeine\s+(?:aussenmeniskus|außenmeniskus)läsion\b",
            ],
            "ES": [
                r"\bmenisco\s+(?:externo|lateral)\s+(?:conservado|íntegro|intacto|normal|sin\s+alteraciones)\b",
                r"\bsin\s+rotura\s+del\s+menisco\s+(?:externo|lateral)\b",
            ],
            "TR": [
                r"\blateral\s+menisküs\s+(?:doğaldır|doğal|intakt|normaldir)\b",
                r"\blateral\s+menisküste\s+yırtık\s+(?:izlenmedi|saptanmadı)\b",
            ],
            "HR/SR": [
                r"\blateralni\s+menisk\s+(?:uredan|očuvan|intaktan|bez\s+znakova\s+rupture)\b",
            ],
            "GR": [
                r"\bέξω\s+μηνίσκος\s+άθικτος\b",
                r"\bχωρίς\s+ρήξη\s+έξω\s+μηνίσκου\b",
            ],
            "BG": [
                r"\bлатерален\s+менискус\s+е\s+интактен\b",
            ],
            "NL": [
                r"\blaterale\s+meniscus\s+is\s+intact\b",
            ],
            "EN": [
                r"\blateral\s+meniscus\s+is\s+intact\b",
                r"\bno\s+lateral\s+meniscal\s+tear\b",
            ]
        }
    },
    {
        "canonical_concept": "MENISCUS_DEGENERATION",
        "target_candidates": ["Medial Meniscus", "Lateral Meniscus"],
        "clinical_type": "DEGENERATION",
        "polarity": "neutral",
        "confidence": 0.85,
        "patterns": {
            "TR": [
                r"\bgrade\s+[12iI]{1,2}\s+sinyal\s+artışı\b",
                r"\bgrade\s+[12iI]{1,2}\s+dejenerasyon\b",
            ],
            "ES": [
                r"\bmeniscopatía\s+grado\s+[12iI]{1,2}\b",
                r"\bdegeneración\s+mucinoide\b",
            ],
            "DE": [
                r"\bgrad\s+[12]\s+meniskopathie\b",
                r"\bmukoide\s+degeneration\b",
            ],
            "EN": [
                r"\bgrade\s+[12]\s+meniscal\s+signal\b",
                r"\bintrasubstance\s+signal\b",
            ]
        }
    },

    # --------------------------------------------------------------------------
    # 3. OSTEOARTHRITIS (Medial OA, Lateral OA, PF OA)
    # --------------------------------------------------------------------------
    {
        "canonical_concept": "MEDIAL_TF_OA",
        "target_candidates": ["Medial OA"],
        "clinical_type": "OA",
        "polarity": "positive",
        "anatomical_hint": "MEDIAL",
        "confidence": 0.90,
        "patterns": {
            "DE": [r"\bmediale\s+gonarthrose\b"],
            "ES": [r"\bgonartrosis\s+medial\b", r"\bartrosis\s+femorotibial\s+medial\b"],
            "TR": [r"\bmedial\s+kompartmanda\s+osteoartrit\b"],
            "HR/SR": [r"\bmedijalna\s+gonartroza\b"],
            "EN": [r"\bmedial\s+compartment\s+osteoarthritis\b"]
        }
    },
    {
        "canonical_concept": "LATERAL_TF_OA",
        "target_candidates": ["Lateral OA"],
        "clinical_type": "OA",
        "polarity": "positive",
        "anatomical_hint": "LATERAL",
        "confidence": 0.90,
        "patterns": {
            "DE": [r"\blaterale\s+gonarthrose\b"],
            "ES": [r"\bgonartrosis\s+lateral\b", r"\bartrosis\s+femorotibial\s+lateral\b"],
            "TR": [r"\blateral\s+kompartmanda\s+osteoartrit\b"],
            "HR/SR": [r"\blateralna\s+gonartroza\b"],
            "EN": [r"\blateral\s+compartment\s+osteoarthritis\b"]
        }
    },
    {
        "canonical_concept": "PF_OA",
        "target_candidates": ["PF OA"],
        "clinical_type": "OA",
        "polarity": "positive",
        "anatomical_hint": "PATELLOFEMORAL",
        "confidence": 0.90,
        "patterns": {
            "DE": [r"\bretropatellare\s+arthrose\b", r"\bpatellofemorale\s+arthrose\b", r"\bpatellofemoralarthrose\b", r"\bchondropathia\s+patellae\s+grad\s+[34]\b"],
            "ES": [r"\bartrosis\s+patelofemoral\b", r"\bartrosis\s+femoropatelar\b", r"\bcondromalacia\s+rotuliana\s+grado\s+[34]\b", r"\bpérdida\s+completa\s+del\s+cartílago\s+rotuliano\b"],
            "TR": [r"\bpatellofemoral\s+osteoartrit\b", r"\bpatellofemoral\s+artroz\b", r"\bpatellar\s+kartilajda\s+grade\s+[34]\s+kondromalazi\b"],
            "HR/SR": [r"\bfemoropatelarna\s+artroza\b", r"\bhondromalacija\s+patele\s+(?:gradus\s+[34]|iii°|iv°)\b"],
            "EN": [r"\bpatellofemoral\s+osteoarthritis\b", r"\bgrade\s+[34]\s+patellar\s+chondromalacia\b"]
        }
    },
    {
        "canonical_concept": "PF_MILD_CHONDROMALACIA",
        "target_candidates": ["PF OA"],
        "clinical_type": "MILD_CHONDROMALACIA",
        "polarity": "neutral",
        "anatomical_hint": "PATELLOFEMORAL",
        "confidence": 0.90,
        "patterns": {
            "DE": [r"\bgeringe\s+chondropathia\s+patellae\s+grad\s+1\b", r"\bchondropathia\s+patellae\s+grad\s+[12]\b"],
            "ES": [r"\bcondromalacia\s+rotuliana\s+grado\s+[12]\b"],
            "TR": [r"\bpatellada\s+grade\s+[12]\s+kondromalazi\b"],
            "HR/SR": [r"\bhondromalacija\s+patele\s+(?:gradus\s+[12]|i°|ii°)\b"],
            "EN": [r"\bmild\s+patellofemoral\s+chondromalacia\b", r"\bgrade\s+[12]\s+patellar\s+chondromalacia\b"]
        }
    },

    # --------------------------------------------------------------------------
    # 4. EFFUSION & SYNOVITIS
    # --------------------------------------------------------------------------
    {
        "canonical_concept": "SIGNIFICANT_EFFUSION",
        "target_candidates": ["Effusion"],
        "clinical_type": "EFFUSION",
        "polarity": "positive",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\b(?:deutlicher|massiver|ausgeprägter|mäßiger)?\s*gelenkerguss\b", r"\berguss\s+im\s+kniegelenk\b"],
            "ES": [r"\bderrame\s+articular\b", r"\babundante\s+líquido\s+intraarticular\b"],
            "TR": [r"\beklem\s+içi\s+(?:belirgin\s+)?sıvı\s+artışı\b", r"\beklem\s+sıvısında\s+artış\b"],
            "HR/SR": [r"\bizljev\s+u\s+koljenu\b", r"\bveći\s+intraartikularni\s+izljev\b"],
            "GR": [r"\bσυλλογή\s+υγρού\s+στο\s+γόνατο\b"],
            "BG": [r"\bставен\s+излив\b"],
            "NL": [r"\bhydrops\s+van\s+het\s+kniegewricht\b"],
            "EN": [r"\b(?:moderate|large|significant|trace)?\s*joint\s+effusion\b"]
        }
    },
    {
        "canonical_concept": "EFFUSION_NEGATIVE",
        "target_candidates": ["Effusion"],
        "clinical_type": "NORMAL",
        "polarity": "negative",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\bkein\s+gelenkerguss\b", r"\bohne\s+gelenkerguss\b"],
            "ES": [r"\bsin\s+derrame\s+articular\b", r"\bno\s+se\s+observa\s+derrame\b"],
            "TR": [r"\beklem\s+sıvısı\s+doğaldır\b", r"\bsıvı\s+artışı\s+izlenmedi\b"],
            "HR/SR": [r"\bnema\s+izljeva\s+u\s+koljenu\b", r"\bbez\s+intraartikularnog\s+izljeva\b"],
            "GR": [r"\bχωρίς\s+συλλογή\s+υγρού\b", r"\bδεν\s+παρατηρείται\s+συλλογή\s+υγρού\b"],
            "BG": [r"\bняма\s+ставен\s+излив\b", r"\bбез\s+данни\s+за\s+ставен\s+излив\b"],
            "NL": [r"\bgeen\s+hydrops\b"],
            "EN": [r"\bno\s+joint\s+effusion\b", r"\bwithout\s+effusion\b"]
        }
    },
    {
        "canonical_concept": "SYNOVIAL_PROLIFERATION",
        "target_candidates": ["Synovitis"],
        "clinical_type": "SYNOVITIS",
        "polarity": "positive",
        "confidence": 0.92,
        "patterns": {
            "DE": [r"\bsynovialitis\b", r"\bsynoviale\s+hypertrophie\b"],
            "ES": [r"\bsinovitis\b", r"\bhipertrofia\s+sinovial\b"],
            "TR": [r"\bsinovit\b", r"\bsinovyal\s+hipertrofi\b"],
            "HR/SR": [r"\bsinovitis\b", r"\bsinovijalna\s+hipertrofija\b"],
            "GR": [r"\bυμενίτιδα\b"],
            "BG": [r"\bсиновит\b"],
            "NL": [r"\bsynovitis\b"],
            "EN": [r"\bsynovitis\b", r"\bsynovial\s+hypertrophy\b"]
        }
    },

    # --------------------------------------------------------------------------
    # 5. BAKER'S CYST
    # --------------------------------------------------------------------------
    {
        "canonical_concept": "BAKERS_CYST",
        "target_candidates": ["Baker's"],
        "clinical_type": "CYST",
        "polarity": "positive",
        "anatomical_hint": "POPLITEAL",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\bbaker[\s-]*(?:zyste|cyste)\b", r"\bpoplitealzyste\b"],
            "ES": [r"\bquiste\s+de\s+baker\b", r"(?<!pequeño\s)(?<!pequeno\s)\bquiste\s+poplíteo\b", r"(?<!pequeño\s)(?<!pequeno\s)\bquiste\s+popliteo\b"],
            "TR": [r"\bbaker\s+kisti\b", r"\bpopliteal\s+kist\b"],
            "HR/SR": [r"\bbakerova\s+cista\b", r"\bpoplitealna\s+cista\b", r"\bcista\s+u\s+poplitealnoj\s+jami\b"],
            "GR": [r"\bκύστη\s+baker\b", r"\bιγνυακή\s+κύστη\b"],
            "BG": [r"\bбейкърова\s+киста\b", r"\bпоплитеална\s+киста\b"],
            "NL": [r"\bbaker-cyste\b", r"\bpopliteale\s+cyste\b"],
            "EN": [r"\bbaker'?s\s+cyst\b", r"\bpopliteal\s+cyst\b"]
        }
    },

    # --------------------------------------------------------------------------
    # 6. CONTUSION (Bone Bruise / Subchondral Edema)
    # --------------------------------------------------------------------------
    {
        "canonical_concept": "BONE_CONTUSION",
        "target_candidates": ["Contusion"],
        "clinical_type": "CONTUSION",
        "polarity": "positive",
        "confidence": 0.92,
        "patterns": {
            "DE": [r"\bknochenödem\b", r"\bknochenkontusion\b", r"\bknochenmarködem\b", r"\bspongiosaeinblutung\b", r"\bknochenmarksödem\b"],
            "ES": [r"\bedema\s+óseo\b", r"\bcontusión\s+ósea\b", r"\bedema\s+trabecular\b"],

            "TR": [r"\bkemik\s+iliği\s+ödemi\b", r"\bkemik\s+kontüzyonu\b"],
            "HR/SR": [r"\bkoštani\s+edem\b", r"\bkontuzija\s+kosti\b"],
            "GR": [r"\bοστικό\s+οίδημα\b"],
            "BG": [r"\bкостен\s+оток\b", r"\bкостна\s+контузия\b"],
            "NL": [r"\bbotoedeem\b", r"\bbotcontusie\b", r"\bbeenmergoedeem\b"],
            "EN": [r"\bbone\s+contusion\b", r"\bbone\s+bruise\b", r"\bbone\s+marrow\s+edema\b"]
        }
    },

    # --------------------------------------------------------------------------
    # 7. FRACTURE
    # --------------------------------------------------------------------------
    {
        "canonical_concept": "CORTICAL_FRACTURE",
        "target_candidates": ["Fracture"],
        "clinical_type": "FRACTURE",
        "polarity": "positive",
        "severity_hint": "complete",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\bfraktur\b", r"\bkortikalisunterbrechung\b", r"\bnicht\s+dislozierte\s+fraktur\b", r"\bfissur\b"],
            "ES": [r"\bfractura\b", r"\binterrupción\s+cortical\b", r"\bfractura\s+sin\s+desplazamiento\b"],
            "TR": [r"\bkırık\b", r"\bnondeplase\s+kırık\b"],
            "HR/SR": [r"\bprijelom\b", r"\bfraktura\b"],
            "GR": [r"\bκάταγμα\b", r"\bδιακοπή\s+της\s+συνέχειας\s+του\s+φλοιώδους\b"],
            "BG": [r"\bфрактура\b", r"\bфрактура\s+без\s+дислокация\b"],
            "NL": [r"\bfractuur\b", r"\bniet-verplaatste\s+fractuur\b"],
            "EN": [r"\b(?:cortical\s+|subchondral\s+|avulsion\s+|comminuted\s+)?fracture\b", r"\bnon-displaced\s+fracture\b"]
        }
    },
    {
        "canonical_concept": "FRACTURE_NEGATIVE",
        "target_candidates": ["Fracture"],
        "clinical_type": "NORMAL",
        "polarity": "negative",
        "confidence": 0.95,
        "patterns": {
            "DE": [r"\bkeine\s+fraktur\b", r"\bohne\s+fissur\b", r"\bohne\s+fissur\s+oder\s+dislokation\b", r"\bkeine\s+fraktur\s+oder\s+dislokation\b", r"\bkeine\s+infraktion\b"],
            "ES": [r"\bsin\s+(?:evidencia\s+de\s+)?fractura\b", r"\bsin\s+evidencia\s+de\s+fractura\s+ni\s+desplazamiento\b"],
            "TR": [r"\bkırık\s+(?:saptanmadı|izlenmedi|saptanmamıştır)\b", r"\bkırık\s+yoktur\b"],
            "HR/SR": [r"\bnema\s+prijeloma\b", r"\bbez\s+znakova\s+prijeloma\b"],
            "GR": [r"\bχωρίς\s+κάταγμα\b", r"\bδεν\s+παρατηρείται\s+κάταγμα\b"],
            "BG": [r"\bняма\s+данни\s+за\s+фрактура\b", r"\bбез\s+фрактура\b"],
            "NL": [r"\bgeen\s+fractuur\b"],
            "EN": [r"\bno\s+fracture\b", r"\bwithout\s+fracture\b"]
        }
    }
]


# Compile concepts into indexed patterns
COMPILED_CONCEPT_REGISTRY: List[Dict[str, Any]] = []

for entry in MULTILINGUAL_LEXICON:
    canon = entry["canonical_concept"]
    targets = entry["target_candidates"]
    ctype = entry["clinical_type"]
    pol = entry["polarity"]
    sev = entry.get("severity_hint")
    anat = entry.get("anatomical_hint")
    conf = entry.get("confidence", 0.90)

    compiled_langs: Dict[str, List[re.Pattern]] = {}
    for lang, pat_strs in entry.get("patterns", {}).items():
        compiled_langs[lang] = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in pat_strs]

    COMPILED_CONCEPT_REGISTRY.append({
        "canonical_concept": canon,
        "target_candidates": targets,
        "clinical_type": ctype,
        "polarity": pol,
        "severity_hint": sev,
        "anatomical_hint": anat,
        "confidence": conf,
        "compiled_patterns": compiled_langs
    })


def extract_multilingual_concepts_non_destructive(report_text: str) -> List[ConceptRecord]:
    """
    Extract all matching multilingual clinical concepts from report text with exact spans.
    Does NOT use string mutation.
    """
    norm_rep = normalize_multilingual_text(report_text)
    clauses = split_text_into_scoped_clauses(norm_rep)
    records: List[ConceptRecord] = []

    for cl in clauses:
        clause_is_neg = is_clause_truly_negative_non_destructive(cl.text, cl.span.start_char)
        c_low = cl.text.lower()

        for entry in COMPILED_CONCEPT_REGISTRY:
            canonical = entry["canonical_concept"]
            targets = list(entry["target_candidates"])
            clin_type = entry["clinical_type"]
            pol = entry["polarity"]
            sev_hint = entry.get("severity_hint")
            anat_hint = entry.get("anatomical_hint")
            conf = entry.get("confidence", 0.90)

            # Compartment scoping for meniscus targets
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

            for lang, pat_list in entry["compiled_patterns"].items():
                for pat in pat_list:
                    match = pat.search(cl.text)
                    if match:
                        s, e = match.span()
                        abs_s = cl.span.start_char + s
                        abs_e = cl.span.start_char + e
                        span = SpanOffset(abs_s, abs_e, cl.text[s:e])

                        actual_pol = pol
                        if clause_is_neg and pol == "positive":
                            actual_pol = "negative"

                        rec = ConceptRecord(
                            surface_forms=[cl.text.strip()],
                            canonical_concept=canonical if actual_pol == "positive" else f"{canonical}_NEGATIVE",
                            target_candidates=scoped_targets,
                            clinical_type=clin_type,
                            language=lang,
                            polarity=actual_pol,
                            severity_hint=sev_hint,
                            anatomical_hint=anat_hint,
                            confidence=conf,
                            evidence_text=cl.text.strip(),
                            evidence_span=span,
                            rule_id=f"RULE_MULTILINGUAL_{canonical}"
                        )
                        records.append(rec)

    return records


def get_remediated_multilingual_evidence(
    target: str,
    full_report: str,
    baseline_evidence: str = "",
    precomputed_concepts: Optional[List[ConceptRecord]] = None
) -> Optional[ConceptRecord]:
    """Retrieve the most relevant multilingual concept for a specific target."""
    if precomputed_concepts is not None:
        all_concepts = precomputed_concepts
    else:
        if not full_report:
            return None
        all_concepts = extract_multilingual_concepts_non_destructive(full_report)

    target_concepts = [c for c in all_concepts if target in c.target_candidates]

    if not target_concepts:
        return None

    def sort_key(c: ConceptRecord) -> Tuple[int, float]:
        score = 0
        if c.polarity == "positive":
            score += 10
        elif c.polarity == "negative":
            score += 5
        if c.clinical_type in ["TEAR", "FRACTURE", "CYST", "OA", "CONTUSION", "SYNOVITIS"]:
            score += 2
        return (score, c.confidence)

    target_concepts.sort(key=sort_key, reverse=True)
    return target_concepts[0]



# Alias for test compatibility
match_concepts_in_clause = extract_multilingual_concepts_non_destructive

