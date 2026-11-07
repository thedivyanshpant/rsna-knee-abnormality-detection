"""
Configuration constants and target definitions for Phase B.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from typing import List

TARGETS: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

OA_TARGETS: List[str] = ["Medial OA", "Lateral OA", "PF OA"]
MENISCUS_TARGETS: List[str] = ["Medial Meniscus", "Lateral Meniscus"]
LIGAMENT_TARGETS: List[str] = ["ACL", "MCL"]

# Phase B Step 1 Rules
RULE_ASSERTION_NEGATIVE = "RULE_ASSERTION_SCOPED_NEGATIVE"
RULE_ANATOMY_PF_ISOLATION = "RULE_ANATOMY_PF_ISOLATION"
RULE_ANATOMY_MISMATCH_SUPPRESSION = "RULE_ANATOMY_MISMATCH_SUPPRESSION"

# Phase B Step 2 Rules
RULE_OA_SEVERITY_GATE = "RULE_OA_SEVERITY_GATE"
RULE_EFFUSION_SEVERITY_GATE = "RULE_EFFUSION_SEVERITY_GATE"
RULE_SYNOVITIS_PROLIFERATION_GATE = "RULE_SYNOVITIS_PROLIFERATION_GATE"
RULE_MENISCUS_TEAR_GATE = "RULE_MENISCUS_TEAR_GATE"
RULE_BONE_FRACTURE_GATE = "RULE_BONE_FRACTURE_GATE"
RULE_BAKER_ANATOMICAL_GATE = "RULE_BAKER_ANATOMICAL_GATE"
RULE_LIGAMENT_SPRAIN_GATE = "RULE_LIGAMENT_SPRAIN_GATE"
