"""
Unit Tests for Phase B Step 3 Remediation Modules (Step 3A, 3B, 3C, 3D).
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


import unittest
from phase_b.parser import (
    SpanOffset,
    split_text_into_scoped_clauses,
    analyze_fracture_displacement_relations
)
from phase_b.lexicon import match_concepts_in_clause
from phase_b.severity_gates import apply_remediated_severity_gates
from phase_b.soft_targets import compute_soft_targets
from phase_b.pipeline import process_remediated_study_target, run_remediated_pipeline_batch


class TestRemediationParser(unittest.TestCase):
    def test_non_destructive_clause_splitting(self):
        text = "Keine Meniskusläsion, unauffälliger VKB. Geringe Chondropathia patellae Grad 1."
        clauses = split_text_into_scoped_clauses(text)
        self.assertGreaterEqual(len(clauses), 2)
        # Verify spans match exact substrings
        for cl in clauses:
            self.assertEqual(text[cl.span.start_char:cl.span.end_char], cl.text)

    def test_fracture_relational_parsing(self):
        # Case 1: Coordinated negation
        res1 = analyze_fracture_displacement_relations("Sin evidencia de fractura ni desplazamiento óseo.")
        self.assertTrue(res1["is_coordinated_negation"])
        self.assertEqual(res1["fracture_polarity"], "negative")

        # Case 2: Affirmative non-displaced fracture
        res2 = analyze_fracture_displacement_relations("Fractura sin desplazamiento en el platillo tibial.")
        self.assertTrue(res2["is_affirmative_non_displaced"])
        self.assertEqual(res2["fracture_polarity"], "positive")
        self.assertEqual(res2["modifier"], "NON_DISPLACED")

        # Case 3: German coordinated negation with Fissur
        res3 = analyze_fracture_displacement_relations("Ohne Fissur oder Dislokation der knöchernen Strukturen.")
        self.assertTrue(res3["is_coordinated_negation"])
        self.assertEqual(res3["fracture_polarity"], "negative")


class TestRemediationLexicon(unittest.TestCase):
    def test_greek_negative_assertion(self):
        sent = "Δεν παρατηρείται ρήξη πρόσθιου χιαστού."
        matches = match_concepts_in_clause(sent)
        acl_neg = [m for m in matches if "ACL" in m.canonical_concept and m.polarity == "negative"]
        self.assertGreaterEqual(len(acl_neg), 1)
        self.assertEqual(acl_neg[0].polarity, "negative")

    def test_turkish_normal_assertion(self):
        sent = "Medial menisküs doğaldır."
        matches = match_concepts_in_clause(sent)
        mm_neg = [m for m in matches if "MEDIAL_MENISCUS" in m.canonical_concept and m.polarity == "negative"]
        self.assertGreaterEqual(len(mm_neg), 1)
        self.assertEqual(mm_neg[0].polarity, "negative")

    def test_german_intact_acl(self):
        sent = "Das vordere Kreuzband zeigt sich durchgehend intakt."
        matches = match_concepts_in_clause(sent)
        acl_neg = [m for m in matches if "ACL" in m.canonical_concept and m.polarity == "negative"]
        self.assertGreaterEqual(len(acl_neg), 1)


class TestRemediationSeverityGates(unittest.TestCase):
    def test_german_grade1_chondromalacia_gated(self):
        evidence = "Geringe Chondropathia patellae Grad 1 ohne Zeichen einer Arthrose."
        pred, conf, rule, _ = apply_remediated_severity_gates(
            target="PF OA",
            incoming_pred=1,
            incoming_conf=0.8,
            evidence_text=evidence,
            full_report=evidence
        )
        self.assertEqual(pred, 0)
        self.assertIn("MILD_CHONDRO_EXCLUSION", rule)

    def test_turkish_grade1_meniscus_signal_gated(self):
        evidence = "Medial menisküs posterior hornda grade 1 sinyal artışı izlenmektedir."
        pred, conf, rule, _ = apply_remediated_severity_gates(
            target="Medial Meniscus",
            incoming_pred=1,
            incoming_conf=0.8,
            evidence_text=evidence,
            full_report=evidence
        )
        self.assertEqual(pred, 0)
        self.assertIn("MILD_SIGNAL_EXCLUSION", rule)

    def test_subchondral_impaction_gating(self):
        evidence = "Impactiefractuur op centraal dragende deel met botoedeem."
        # Fracture should be gated to 0
        p_fx, _, r_fx, _ = apply_remediated_severity_gates("Fracture", 1, 0.8, evidence, evidence)
        self.assertEqual(p_fx, 0)
        self.assertIn("GATE_FRACTURE_SUBCHONDRAL_CONTUSION_EXCLUSION", r_fx)
        # Contusion pass through when incoming is 0
        p_cont, _, r_cont, _ = apply_remediated_severity_gates("Contusion", 0, 0.8, evidence, evidence)
        self.assertEqual(p_cont, 0)



class TestRemediationSoftTargets(unittest.TestCase):
    def test_sample_weight_and_probability(self):
        res = compute_soft_targets(
            binary_label=1,
            rule_applied="RULE_REMEDIATED_ACL_TEAR",
            base_confidence=0.95,
            is_explicit_concept=True
        )
        self.assertEqual(res.predicted_label, 1)
        self.assertAlmostEqual(res.label_probability, 0.95, places=2)
        self.assertAlmostEqual(res.confidence_score, 0.95, places=2)
        self.assertAlmostEqual(res.sample_weight, round(0.25 + 0.75 * 0.95, 4), places=2)
        self.assertEqual(res.abstain_flag, 0)

    def test_abstention_on_contradiction(self):
        res = compute_soft_targets(
            binary_label=1,
            rule_applied="CONTRADICTION",
            has_contradiction=True
        )
        self.assertEqual(res.abstain_flag, 1)
        self.assertEqual(res.sample_weight, 0.0)


class TestRemediationPipelineSchema(unittest.TestCase):
    def test_20_field_provenance_schema(self):
        res = process_remediated_study_target(
            study_uid="TEST_STUDY_001",
            target="ACL",
            baseline_pred=1,
            baseline_conf=0.8,
            evidence="Δεν παρατηρείται ρήξη πρόσθιου χιαστού.",
            full_report="Δεν παρατηρείται ρήξη πρόσθιου χιαστού.",
            gold_label=0
        )
        required_fields = [
            "StudyInstanceUID", "target", "gold_label", "baseline_predicted_label",
            "baseline_confidence", "step1_predicted_label", "step1_rule_applied",
            "step2_predicted_label", "step2_rule_applied", "step3_predicted_label",
            "step3_rule_applied", "final_predicted_label", "label_probability",
            "confidence_score", "sample_weight", "abstain_flag",
            "matched_canonical_concept", "detected_language", "evidence_char_span",
            "evidence_text"
        ]
        self.assertEqual(len(res), 20)
        for f in required_fields:
            self.assertIn(f, res)
        self.assertEqual(res["final_predicted_label"], 0)


if __name__ == "__main__":
    unittest.main()
