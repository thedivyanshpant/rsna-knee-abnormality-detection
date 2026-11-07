"""
Phase B Step 3 Remediation: Non-Destructive Scoped Parser (Step 3A).

Preserves exact character spans [start_char, end_char] without string mutation.
Replaces text-stripping regex modifications with relational modifier graphs
and scoped clause segmentation.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import re


@dataclass(frozen=True)
class SpanOffset:
    """Represents an exact, non-mutated character slice within the original text."""
    start_char: int
    end_char: int
    matched_text: str

    def to_tuple(self) -> Tuple[int, int]:
        return (self.start_char, self.end_char)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "matched_text": self.matched_text
        }


@dataclass
class ScopedClause:
    """A semantic clause maintaining exact global offset bounds within the parent document."""
    text: str
    span: SpanOffset
    clause_index: int
    is_negated: bool = False
    negation_spans: List[SpanOffset] = field(default_factory=list)
    modifiers: List[Dict[str, Any]] = field(default_factory=list)


# Punctuation and contrastive conjunction boundaries for clause segmentation
CLAUSE_DELIMITER_REGEX = re.compile(
    r"(?:;|,|(?<=[\w\)])\.\s+|\b(?:but|however|although|while|whereas|aber|jedoch|pero|sin\s+embargo|več|ali|doch|хотя|но|obaçe)\b|\n+)",
    re.IGNORECASE | re.UNICODE
)


# Explicit phrase patterns where non-displacement qualifies a positive fracture
AFFIRMATIVE_NON_DISPLACED_PATTERNS = [
    re.compile(r"\b(?:fractura|fracture|fraktur|фрактура|kırık|prijelom)\s+(?:sin|without|ohne|без|niet)\s+(?:evidencia\s+de\s+|signos\s+de\s+|данни\s+за\s+|major\s+|significant\s+|wesentliche\s+|значима\s+)?(?:desplazamiento|dislocation|displacement|dislokation|разместване|verplaatsing)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:nicht\s+disloziert(?:e|en|er|es)?|non[\s-]displaced|nondeplase|niet[\s-]verplaatst(?:e)?)\s+(?:fraktur|fracture|fractura|kırık|prijelom|фрактура)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:fractura|fracture|fraktur|фрактура|kırık|prijelom)\b[^\.;\n]{1,80}\b(?:без\s+дислокация|sin\s+desplazamiento|ohne\s+dislokation|without\s+displacement|nondeplase)\b", re.IGNORECASE | re.UNICODE),
]

# Patterns where negation governs both fracture and displacement
COORDINATED_FRACTURE_NEGATION_PATTERNS = [
    re.compile(r"\b(?:sin\s+evidencia\s+de|sin\s+signos\s+de|sin)\s+fractura\s+(?:ni|o|u|y)\s+(?:de\s+)?desplazamiento\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:keine|ohne)\s+(?:fraktur|fissur|infraktion)\s+(?:oder|noch|und)\s+(?:dislokation|verschiebung)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:no|without)\s+(?:evidence\s+of\s+)?(?:fracture|fissure)\s+(?:or|nor|and)\s+(?:displacement|dislocation)\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:без\s+данни\s+за|няма)\s+фрактура\s+(?:или|нито)\s+дислокация\b", re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(?:geen)\s+(?:fractuur|fissuur)\s+(?:of|noch)\s+verplaatsing\b", re.IGNORECASE | re.UNICODE),
]


def split_text_into_scoped_clauses(full_text: str) -> List[ScopedClause]:
    """
    Split text into scoped clauses while preserving exact global character offsets.
    Does not modify or delete any characters from the source text.
    """
    if not full_text or not full_text.strip():
        span = SpanOffset(0, len(full_text), full_text)
        return [ScopedClause(text=full_text, span=span, clause_index=0)]

    clauses: List[ScopedClause] = []
    current_pos = 0
    clause_idx = 0

    for match in CLAUSE_DELIMITER_REGEX.finditer(full_text):
        match_start, match_end = match.span()
        clause_str = full_text[current_pos:match_start]
        stripped = clause_str.strip()
        if stripped:
            rel_start = clause_str.find(stripped)
            abs_start = current_pos + rel_start
            abs_end = abs_start + len(stripped)
            span = SpanOffset(abs_start, abs_end, full_text[abs_start:abs_end])
            clauses.append(ScopedClause(text=stripped, span=span, clause_index=clause_idx))
            clause_idx += 1
        current_pos = match_end

    # Tail clause
    if current_pos < len(full_text):
        tail_str = full_text[current_pos:]
        stripped = tail_str.strip()
        if stripped:
            rel_start = tail_str.find(stripped)
            abs_start = current_pos + rel_start
            abs_end = abs_start + len(stripped)
            span = SpanOffset(abs_start, abs_end, full_text[abs_start:abs_end])
            clauses.append(ScopedClause(text=stripped, span=span, clause_index=clause_idx))

    if not clauses:
        span = SpanOffset(0, len(full_text), full_text)
        return [ScopedClause(text=full_text.strip(), span=span, clause_index=0)]

    return clauses


def extract_span_matches(pattern: re.Pattern, text: str, base_offset: int = 0) -> List[SpanOffset]:
    """Find all regex matches in text and return exact absolute character spans."""
    results = []
    for m in pattern.finditer(text):
        s, e = m.span()
        results.append(SpanOffset(
            start_char=base_offset + s,
            end_char=base_offset + e,
            matched_text=text[s:e]
        ))
    return results


def analyze_fracture_displacement_relations(clause_text: str, base_offset: int = 0) -> Dict[str, Any]:
    """
    Analyzes relational semantics of fracture and displacement phrasing without string mutation.
    """
    for pat in COORDINATED_FRACTURE_NEGATION_PATTERNS:
        match = pat.search(clause_text)
        if match:
            s, e = match.span()
            return {
                "is_coordinated_negation": True,
                "is_affirmative_non_displaced": False,
                "fracture_polarity": "negative",
                "matched_span": SpanOffset(base_offset + s, base_offset + e, clause_text[s:e]),
                "rule_id": "REL_FRACTURE_COORDINATED_NEGATION"
            }

    for pat in AFFIRMATIVE_NON_DISPLACED_PATTERNS:
        match = pat.search(clause_text)
        if match:
            s, e = match.span()
            return {
                "is_coordinated_negation": False,
                "is_affirmative_non_displaced": True,
                "fracture_polarity": "positive",
                "modifier": "NON_DISPLACED",
                "matched_span": SpanOffset(base_offset + s, base_offset + e, clause_text[s:e]),
                "rule_id": "REL_FRACTURE_AFFIRMATIVE_NON_DISPLACED"
            }

    return {
        "is_coordinated_negation": False,
        "is_affirmative_non_displaced": False,
        "fracture_polarity": "neutral",
        "matched_span": None,
        "rule_id": "REL_FRACTURE_DEFAULT"
    }
