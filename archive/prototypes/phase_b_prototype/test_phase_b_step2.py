"""
Unit and Regression Test Suite for Phase B Step 2: Clinical Severity & Conflation Gating.
"""

import unittest
from phase_b.pipeline import process_study_target
from phase_b.severity import apply_clinical_severity_gates


class TestPhaseBStep2(unittest.TestCase):

    # ==========================================
    # GATE 1: OA / CHONDROSIS SEVERITY TESTS
    # ==========================================

    def test_oa_mild_chondromalacia_suppression(self):
        """Mild/low-grade chondrosis should be gated to OA = 0."""
        evid = "Scattered foci of mild chondrosis throughout the patellofemoral compartment."
        res = process_study_target("test_oa_01", "PF OA", baseline_pred=1, baseline_conf=0.9, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_OA_SEVERITY_GATE")

    def test_oa_severe_osteoarthritis_preserved(self):
        """Severe full-thickness loss or explicit osteoarthritis must remain OA = 1."""
        evid = "Advanced osteoarthritis with severe joint space narrowing and large marginal osteophytes."
        res = process_study_target("test_oa_02", "Medial OA", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 1)

    def test_oa_full_thickness_denudation_preserved(self):
        """Full thickness cartilage defect with bone denudation must remain OA = 1."""
        evid = "Full-thickness cartilage defect of medial femoral condyle with subchondral bone denudation."
        res = process_study_target("test_oa_03", "Medial OA", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 1)

    # ==========================================
    # GATE 2: EFFUSION & SYNOVITIS TESTS
    # ==========================================

    def test_effusion_physiological_suppression(self):
        """Physiological fluid or extraction syntax artifact should be gated to Effusion = 0."""
        evid = "Minimal amount of right knee effusion is 1."
        res = process_study_target("test_eff_01", "Effusion", baseline_pred=1, baseline_conf=0.8, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_EFFUSION_SEVERITY_GATE")

    def test_effusion_moderate_large_preserved(self):
        """Moderate or large joint effusion must remain Effusion = 1."""
        evid = "Moderate joint effusion with marked distension of suprapatellar bursa."
        res = process_study_target("test_eff_02", "Effusion", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 1)

    def test_synovitis_with_thickening_preserved(self):
        """Synovial membrane thickening and proliferative synovitis must remain Synovitis = 1."""
        evid = "Joint effusion with synovial membrane thickening and proliferative synovitis."
        res_eff = process_study_target("test_syn_01", "Effusion", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        res_syn = process_study_target("test_syn_01", "Synovitis", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res_eff["step2_predicted_label"], 1)
        self.assertEqual(res_syn["step2_predicted_label"], 1)

    def test_synovitis_pure_enthesopathy_suppressed(self):
        """Isolated enthesopathy without synovial proliferation must be gated to Synovitis = 0."""
        evid = "Entezopatske promjene hvatišta tetive m. quadriceps."
        res = process_study_target("test_syn_02", "Synovitis", baseline_pred=1, baseline_conf=0.85, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_SYNOVITIS_PROLIFERATION_GATE")

    # ==========================================
    # GATE 3: MENISCUS TEAR vs DEGENERATION TESTS
    # ==========================================

    def test_meniscus_mucoid_degeneration_suppressed(self):
        """Mucoid degeneration without tear line must be gated to Meniscus = 0."""
        evid = "Medial meniscus mucoid degeneration."
        res = process_study_target("test_men_01", "Medial Meniscus", baseline_pred=1, baseline_conf=0.85, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_MENISCUS_TEAR_GATE")

    def test_meniscus_grade_2_degeneration_suppressed(self):
        """Grade II degeneration without tear line must be gated to Meniscus = 0."""
        evid = "Lateral menisküste grade II dejenerasyon saptanmıştır."
        res = process_study_target("test_men_02", "Lateral Meniscus", baseline_pred=1, baseline_conf=0.85, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_MENISCUS_TEAR_GATE")

    def test_meniscus_radial_horizontal_tear_preserved(self):
        """True radial, horizontal, or bucket-handle tears must remain Meniscus = 1."""
        evid_rad = "Radial tear of the posterior horn of the medial meniscus extending to the inferior articular surface."
        evid_bh = "Kova sapı yırtığı ile uyumlu görünüm mevcuttur."
        res_rad = process_study_target("test_men_03", "Medial Meniscus", baseline_pred=1, baseline_conf=0.95, evidence=evid_rad)
        res_bh = process_study_target("test_men_04", "Lateral Meniscus", baseline_pred=1, baseline_conf=0.95, evidence=evid_bh)
        self.assertEqual(res_rad["step2_predicted_label"], 1)
        self.assertEqual(res_bh["step2_predicted_label"], 1)

    # ==========================================
    # GATE 4: CONTUSION vs FRACTURE TESTS
    # ==========================================

    def test_fracture_subcortical_microfracture_suppressed(self):
        """Subcortical micro-impaction / contusion without cortical breach must be gated to Fracture = 0."""
        evid = "Edema óseo periférico posterior de platillo tibial lateral con pequeñas fracturas subcorticales regionales."
        res = process_study_target("test_fx_01", "Fracture", baseline_pred=1, baseline_conf=0.85, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_BONE_FRACTURE_GATE")

    def test_fracture_subchondral_insufficiency_preserved(self):
        """Subchondral insufficiency fracture (SIFK) or macroscopic fracture must remain Fracture = 1."""
        evid = "Subchondral insufficiency fracture at the medial margin of the medial tibial plateau."
        res = process_study_target("test_fx_02", "Fracture", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 1)

    def test_fracture_cortical_stepoff_preserved(self):
        """Fracture with cortical disruption or step-off must remain Fracture = 1."""
        evid = "Split fracture of lateral tibial plateau with 3mm cortical step-off."
        res = process_study_target("test_fx_03", "Fracture", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 1)

    # ==========================================
    # GATE 5: BAKER'S CYST ANATOMICAL GATE TESTS
    # ==========================================

    def test_bakers_suprapatellar_fluid_suppressed(self):
        """Suprapatellar bursal fluid without popliteal cyst must be gated to Baker's = 0."""
        evid = "Suprapatellar bursada ,diz eklemi içerisinde yaygın sıvı artışı izlendi."
        res = process_study_target("test_bk_01", "Baker's", baseline_pred=1, baseline_conf=0.85, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_BAKER_ANATOMICAL_GATE")

    def test_bakers_true_popliteal_cyst_preserved(self):
        """True popliteal Baker's cyst must remain Baker's = 1."""
        evid = "There is popliteal cyst measuring 21 x 17 x 35 mm in the semimembranosus-gastrocnemius bursa."
        res = process_study_target("test_bk_02", "Baker's", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 1)

    # ==========================================
    # GATE 6: LIGAMENT SPRAIN vs TEAR TESTS
    # ==========================================

    def test_acl_grade_1_sprain_suppressed(self):
        """Interstitial signal / Grade 1 injury of ACL must be gated to ACL = 0."""
        evid = "Interstitial hyperintense signal of ACL, in favor of grade 1 injury."
        res = process_study_target("test_acl_01", "ACL", baseline_pred=1, baseline_conf=0.85, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 0)
        self.assertEqual(res["step2_rule_applied"], "RULE_LIGAMENT_SPRAIN_GATE")

    def test_acl_complete_tear_preserved(self):
        """Complete ACL rupture / disruption must remain ACL = 1."""
        evid = "Complete tear of the anterior cruciate ligament with discontinuity of fibers."
        res = process_study_target("test_acl_02", "ACL", baseline_pred=1, baseline_conf=0.95, evidence=evid)
        self.assertEqual(res["step2_predicted_label"], 1)


if __name__ == "__main__":
    unittest.main()
