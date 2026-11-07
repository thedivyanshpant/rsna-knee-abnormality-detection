"""
Unit and Regression Test Suite for Phase B Step 1: Assertion and Anatomical Scoping.
"""

import unittest
from phase_b.assertion import check_scoped_negative_assertion, split_into_scoped_clauses
from phase_b.anatomy import extract_anatomical_compartments, evaluate_compartment_scoping
from phase_b.pipeline import process_study_target


class TestPhaseBStep1(unittest.TestCase):

    # ==========================================
    # PART 1: ASSERTION / POLARITY TESTS
    # ==========================================

    def test_case_01_acl_intact(self):
        """1. 'ACL is intact.' → ACL = 0"""
        evidence = "ACL is intact."
        res = process_study_target("test_01", "ACL", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res["phase_b_predicted_label"], 0)
        self.assertEqual(res["rule_applied"], "RULE_ASSERTION_SCOPED_NEGATIVE")

    def test_case_02_mcl_intact(self):
        """2. 'MCL is intact.' → MCL = 0"""
        evidence = "The MCL is intact."
        res = process_study_target("test_02", "MCL", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res["phase_b_predicted_label"], 0)
        self.assertEqual(res["rule_applied"], "RULE_ASSERTION_SCOPED_NEGATIVE")

    def test_case_03_no_meniscal_tear_acl_torn(self):
        """3. 'No meniscal tear; ACL is torn.' → Meniscus = 0, ACL = 1"""
        evidence = "No meniscal tear; ACL is torn."
        res_med = process_study_target("test_03", "Medial Meniscus", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        res_lat = process_study_target("test_03", "Lateral Meniscus", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        res_acl = process_study_target("test_03", "ACL", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        
        self.assertEqual(res_med["phase_b_predicted_label"], 0, "Medial meniscus must be negative")
        self.assertEqual(res_lat["phase_b_predicted_label"], 0, "Lateral meniscus must be negative")
        self.assertEqual(res_acl["phase_b_predicted_label"], 1, "ACL must remain positive")

    def test_case_04_acl_intact_mcl_torn(self):
        """4. 'ACL intact, but MCL torn.' → ACL = 0, MCL = 1"""
        evidence = "ACL intact, but MCL torn."
        res_acl = process_study_target("test_04", "ACL", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        res_mcl = process_study_target("test_04", "MCL", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        
        self.assertEqual(res_acl["phase_b_predicted_label"], 0, "ACL must be negative")
        self.assertEqual(res_mcl["phase_b_predicted_label"], 1, "MCL must remain positive")

    def test_case_05_medial_meniscus_normal(self):
        """5. 'Medial meniscus normal.' → Medial Meniscus = 0"""
        evidence = "Medial meniscus normal."
        res = process_study_target("test_05", "Medial Meniscus", baseline_pred=1, baseline_conf=0.85, evidence=evidence)
        self.assertEqual(res["phase_b_predicted_label"], 0)

    def test_case_06_intact_menisci(self):
        """6. 'Intact menisci.' → Medial Meniscus = 0, Lateral Meniscus = 0"""
        evidence = "Intact menisci."
        res_med = process_study_target("test_06", "Medial Meniscus", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        res_lat = process_study_target("test_06", "Lateral Meniscus", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        self.assertEqual(res_med["phase_b_predicted_label"], 0)
        self.assertEqual(res_lat["phase_b_predicted_label"], 0)

    def test_case_07_spanish_sin_signos_desgarro(self):
        """7. 'Cambios de señal en el menisco interno sin signos de desgarro.' → Medial Meniscus = 0"""
        evidence = "Cambios de señal en el menisco interno sin signos de desgarro."
        res = process_study_target("test_07", "Medial Meniscus", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        self.assertEqual(res["phase_b_predicted_label"], 0)

    def test_case_multilingual_german_intakt(self):
        """Multilingual German: 'VKB, HKB, MCL und LCL intakt und physiologisch signalgebend.' → ACL=0, MCL=0"""
        evidence = "VKB, HKB, MCL und LCL intakt und physiologisch signalgebend."
        res_acl = process_study_target("test_de", "ACL", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        res_mcl = process_study_target("test_de", "MCL", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res_acl["phase_b_predicted_label"], 0)
        self.assertEqual(res_mcl["phase_b_predicted_label"], 0)

    def test_case_multilingual_croatian_bez_znakova(self):
        """Multilingual Croatian: 'Lateralni menisk bez znakova degeneracije ili rupture.' → Lateral Meniscus = 0"""
        evidence = "Lateralni menisk bez znakova degeneracije ili rupture."
        res = process_study_target("test_hr", "Lateral Meniscus", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        self.assertEqual(res["phase_b_predicted_label"], 0)

    # ==========================================
    # PART 2: ANATOMICAL COMPARTMENT SCOPING TESTS
    # ==========================================

    def test_case_08_medial_patellar_facet(self):
        """8. 'medial patellar facet chondrosis' → PF OA (Medial OA = 0)"""
        evidence = "medial patellar facet chondrosis"
        res_med = process_study_target("test_08", "Medial OA", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        res_pf = process_study_target("test_08", "PF OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res_med["phase_b_predicted_label"], 0, "Medial OA must be suppressed (PF isolation)")
        self.assertEqual(res_med["rule_applied"], "RULE_ANATOMY_PF_ISOLATION")
        self.assertEqual(res_pf["phase_b_predicted_label"], 1, "PF OA must be positive")

    def test_case_09_lateral_patellar_facet(self):
        """9. 'lateral patellar facet chondrosis' → PF OA (Lateral OA = 0)"""
        evidence = "lateral patellar facet chondrosis"
        res_lat = process_study_target("test_09", "Lateral OA", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        res_pf = process_study_target("test_09", "PF OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res_lat["phase_b_predicted_label"], 0, "Lateral OA must be suppressed")
        self.assertEqual(res_pf["phase_b_predicted_label"], 1, "PF OA must be positive")

    def test_case_10_medial_trochlear_cartilage_loss(self):
        """10. 'medial trochlear cartilage loss' → PF OA (Medial OA = 0)"""
        evidence = "medial trochlear cartilage loss"
        res_med = process_study_target("test_10", "Medial OA", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        res_pf = process_study_target("test_10", "PF OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res_med["phase_b_predicted_label"], 0)
        self.assertEqual(res_pf["phase_b_predicted_label"], 1)

    def test_case_11_medial_femoral_condyle(self):
        """11. 'medial femoral condyle cartilage loss' → Medial OA"""
        evidence = "medial femoral condyle cartilage loss"
        res_med = process_study_target("test_11", "Medial OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        res_pf = process_study_target("test_11", "PF OA", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        self.assertEqual(res_med["phase_b_predicted_label"], 1, "Medial OA must be positive")
        self.assertEqual(res_pf["phase_b_predicted_label"], 0, "PF OA must be suppressed if evidence is purely tibiofemoral")

    def test_case_12_lateral_femoral_condyle(self):
        """12. 'lateral femoral condyle cartilage loss' → Lateral OA"""
        evidence = "lateral femoral condyle cartilage loss"
        res_lat = process_study_target("test_12", "Lateral OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        res_pf = process_study_target("test_12", "PF OA", baseline_pred=1, baseline_conf=0.8, evidence=evidence)
        self.assertEqual(res_lat["phase_b_predicted_label"], 1)
        self.assertEqual(res_pf["phase_b_predicted_label"], 0)

    def test_case_13_medial_tibial_plateau(self):
        """13. 'medial tibial plateau cartilage loss' → Medial OA"""
        evidence = "medial tibial plateau cartilage loss"
        res_med = process_study_target("test_13", "Medial OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res_med["phase_b_predicted_label"], 1)

    def test_case_14_lateral_tibial_plateau(self):
        """14. 'lateral tibial plateau cartilage loss' → Lateral OA"""
        evidence = "lateral tibial plateau cartilage loss"
        res_lat = process_study_target("test_14", "Lateral OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence)
        self.assertEqual(res_lat["phase_b_predicted_label"], 1)

    def test_case_15_dual_compartment_clause(self):
        """15. 'Cartilage loss at the medial patellar facet and medial femoral condyle.' → PF OA + Medial OA"""
        evidence_pf = "Cartilage loss at the medial patellar facet"
        evidence_med = "medial femoral condyle cartilage loss"
        
        res_pf = process_study_target("test_15", "PF OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence_pf)
        res_med = process_study_target("test_15", "Medial OA", baseline_pred=1, baseline_conf=0.9, evidence=evidence_med)
        res_lat = process_study_target("test_15", "Lateral OA", baseline_pred=1, baseline_conf=0.8, evidence=evidence_pf)
        
        self.assertEqual(res_pf["phase_b_predicted_label"], 1, "PF OA must be positive")
        self.assertEqual(res_med["phase_b_predicted_label"], 1, "Medial OA must be positive")
        self.assertEqual(res_lat["phase_b_predicted_label"], 0, "Lateral OA must be negative")


if __name__ == "__main__":
    unittest.main()
