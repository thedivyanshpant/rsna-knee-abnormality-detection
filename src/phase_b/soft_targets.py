"""
Phase B Step 3 Remediation: Decoupled Soft Targets & Confidence Calibration Engine (Step 3C).

Decouples label probability Y in [0.0, 1.0] from extraction confidence C in [0.0, 1.0],
and derives vision model sample weights:
W = 0.25 + 0.75 * C (with W = 0.0 for abstained contradictions).
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass(frozen=True)
class SoftTargetResult:
    predicted_label: int
    label_probability: float  # Y in [0.0, 1.0]
    confidence_score: float   # C in [0.0, 1.0]
    sample_weight: float      # W in [0.0, 1.0]
    abstain_flag: int         # 0 = include in training, 1 = abstain (W=0.0)
    calibration_reason: str


def compute_soft_targets(
    binary_label: int,
    rule_applied: str,
    base_confidence: float = 0.90,
    has_contradiction: bool = False,
    is_mild_gated: bool = False,
    is_explicit_concept: bool = False
) -> SoftTargetResult:
    """
    Computes calibrated continuous probability Y, extraction confidence C,
    and training loss sample weight W.
    """
    # 1. Contradiction / Abstention Handling
    if has_contradiction:
        return SoftTargetResult(
            predicted_label=binary_label,
            label_probability=0.50,
            confidence_score=0.20,
            sample_weight=0.0,
            abstain_flag=1,
            calibration_reason="Contradictory evidence detected in report; record abstained from training loss."
        )

    # 2. Confidence score C calibration
    if is_explicit_concept:
        conf = max(base_confidence, 0.95)
    elif is_mild_gated:
        conf = 0.90
    elif "BASELINE" in rule_applied:
        conf = min(base_confidence, 0.75)
    else:
        conf = base_confidence

    conf = max(0.0, min(1.0, conf))

    # 3. Probability score Y calibration
    if binary_label == 1:
        if is_explicit_concept:
            prob = 0.95
        elif is_mild_gated:
            prob = 0.80
        else:
            prob = max(0.70, conf)
    else:  # binary_label == 0
        if is_explicit_concept:
            prob = 0.05
        elif is_mild_gated:
            prob = 0.20
        else:
            prob = min(0.30, 1.0 - conf)

    prob = max(0.0, min(1.0, prob))

    # 4. Training Sample Weight W = 0.25 + 0.75 * C
    weight = round(0.25 + 0.75 * conf, 4)
    weight = max(0.25, min(1.0, weight))

    return SoftTargetResult(
        predicted_label=binary_label,
        label_probability=round(prob, 4),
        confidence_score=round(conf, 4),
        sample_weight=weight,
        abstain_flag=0,
        calibration_reason=f"Calibrated via {rule_applied} (C={conf:.2f}, Y={prob:.2f}, W={weight:.2f})"
    )
