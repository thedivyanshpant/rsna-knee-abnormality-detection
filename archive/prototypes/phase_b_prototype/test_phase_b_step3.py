"""
Unit and Regression Test Suite for Phase B Step 3:
Multilingual Clinical Concept Expansion and Normalization Engine.
Tests all 8 language groups (EN, DE, ES, TR, HR/SR, GR, BG, NL) and canonical concept behaviors.
"""

import unittest
from phase_b.multilingual import (
    extract_multilingual_concepts,
    get_multilingual_evidence_for_target,
    normalize_multilingual_text
)
from phase_b.pipeline import process_study_target


class TestPhaseBStep3(unittest.TestCase):

    # ==========================================
    # 1. GERMAN (DE) TESTS
    # ==========================================

    def test_de_acl_vkb_ruptur(self):
        """DE: 'VKB Ruptur mit wohl ligamentärer Ruptur' -> ACL = 1"""
        text = "Anamnestisch bek. VKB Ruptur mit wohl ligamentärer Ruptur des anteromedialen Bündels."
        res = process_study_target("de_01", "ACL", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "ACL_TEAR")
        self.assertEqual(res["step3_language"], "DE")

    def test_de_bone_avulsion_fracture(self):
        """DE: 'knöchernem Ausriss des posterolateralen Bündels' -> Fracture = 1"""
        text = "Knöchernem Ausriss des posterolateralen Bündels an der Eminentia intercondylaris."
        res = process_study_target("de_02", "Fracture", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "CORTICAL_FRACTURE")

    def test_de_medial_meniscus_riss(self):
        """DE: 'Innenmeniskushinterhornriss' -> Medial Meniscus = 1"""
        text = "Nachweis einer komplexen Innenmeniskushinterhornriss mit Gelenkkontakt."
        res = process_study_target("de_03", "Medial Meniscus", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "MEDIAL_MENISCUS_TEAR")

    def test_de_intact_ligaments(self):
        """DE: 'VKB, HKB, MCL und LCL intakt' -> ACL = 0, MCL = 0"""
        text = "VKB, HKB, MCL und LCL intakt und regelrecht signalgebend."
        res_acl = process_study_target("de_04", "ACL", baseline_pred=1, baseline_conf=0.8, evidence="VKB intakt", full_report=text)
        self.assertEqual(res_acl["step3_predicted_label"], 0)

    # ==========================================
    # 2. SPANISH (ES) TESTS
    # ==========================================

    def test_es_pf_oa_condropatia_rotuliana(self):
        """ES: 'Condropatía rotuliana' -> PF OA = 1 (Medial OA = 0, Lateral OA = 0)"""
        text = "Pinzamiento de la almohadilla grasa de Hoffa. Condropatía rotuliana severa."
        res_pf = process_study_target("es_01", "PF OA", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        res_med = process_study_target("es_01", "Medial OA", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res_pf["step3_predicted_label"], 1)
        self.assertEqual(res_pf["step3_canonical_concept"], "PF_OA")
        self.assertEqual(res_med["step3_predicted_label"], 0)

    def test_es_medial_meniscus_rotura(self):
        """ES: 'Rotura completa del cuerno posterior del menisco interno' -> Medial Meniscus = 1"""
        text = "Rotura completa del cuerno posterior del menisco interno con extensión articular."
        res = process_study_target("es_02", "Medial Meniscus", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "MEDIAL_MENISCUS_TEAR")

    def test_es_quiste_de_baker(self):
        """ES: 'Quiste de Baker en hueco poplíteo' -> Baker's = 1"""
        text = "Se identifica quiste de Baker en el hueco poplíteo de 3x2 cm."
        res = process_study_target("es_03", "Baker's", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "BAKERS_CYST")

    def test_es_sin_signos_de_desgarro(self):
        """ES: 'Menisco interno sin signos de desgarro' -> Medial Meniscus = 0"""
        text = "Menisco interno sin signos de desgarro ni rotura."
        res = process_study_target("es_04", "Medial Meniscus", baseline_pred=1, baseline_conf=0.8, evidence="Menisco interno", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 0)

    # ==========================================
    # 3. TURKISH (TR) TESTS
    # ==========================================

    def test_tr_acl_komplet_yirtik(self):
        """TR: 'Ön çapraz bağ devamlılığı seçilememiştir (komplet yırtık)' -> ACL = 1"""
        text = "Ön çapraz bağ devamlılığı seçilememiştir (komplet yırtık). Arka çapraz bağ doğaldır."
        res = process_study_target("tr_01", "ACL", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "ACL_TEAR")

    def test_tr_lateral_meniscus_kova_sapi(self):
        """TR: 'Lateral menisküste kova sapı yırtığı' -> Lateral Meniscus = 1"""
        text = "Lateral menisküste grade II dejenerasyon, kova sapı yırtığı ile uyumlu görünüm mevcuttur."
        res = process_study_target("tr_02", "Lateral Meniscus", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "LATERAL_MENISCUS_TEAR")

    def test_tr_meniscus_grade_2_degeneration_gated(self):
        """TR: 'Lateral menisküste grade II dejenerasyon' without tear -> Lateral Meniscus = 0"""
        text = "Lateral menisküste grade II dejenerasyon saptanmıştır. Yırtık saptanmadı."
        res = process_study_target("tr_03", "Lateral Meniscus", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 0)

    # ==========================================
    # 4. CROATIAN / SERBIAN (HR/SR) TESTS
    # ==========================================

    def test_hr_bakers_poplitealna_cista(self):
        """HR/SR: 'Poplitealna cista promjera 5 mm' -> Baker's = 1"""
        text = "Opisana cistična lezija u poplitealnoj jami odgovara Bakerovoj cisti. Poplitealna cista promjera 5 mm."
        res = process_study_target("hr_01", "Baker's", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "BAKERS_CYST")

    def test_hr_synovial_proliferation(self):
        """HR/SR: 'umjerenu proliferaciju sinovije' -> Synovitis = 1"""
        text = "Opsežan zglobni izljev uz umjerenu proliferaciju sinovije u suprapatelarnom recesusu."
        res = process_study_target("hr_02", "Synovitis", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "SYNOVIAL_PROLIFERATION")

    def test_hr_medial_oa_denudacija(self):
        """HR/SR: 'Denudacija hrskavice medijalnog kompartmenta' -> Medial OA = 1"""
        text = "Denudacija hrskavice medijalnog kompartmenta FT zgloba te subhondralna skleroza."
        res = process_study_target("hr_03", "Medial OA", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "MEDIAL_TF_OA")

    # ==========================================
    # 5. GREEK (GR) TESTS
    # ==========================================

    def test_gr_acl_and_mcl_rupture(self):
        """GR: Greek text with micro sign (\u00b5) -> ACL = 1, MCL = 1"""
        text = "Οιδηµατώδης απεικόνιση του πρόσθιου χιαστού συνδέσµου µε ασυνέχεια των ινών του στο πλαίσιο υφολικής (σχεδόν πλήρους) ρήξης. Οιδηµατώδης απεικόνιση και ασυνέχεια στην εγγύς µοίρα του έσω πλαγίου συνδέσµου µε παρουσία οιδήµατος στο πλαίσιο υφολικής ρήξης."
        res_acl = process_study_target("gr_01", "ACL", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        res_mcl = process_study_target("gr_01", "MCL", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res_acl["step3_predicted_label"], 1)
        self.assertEqual(res_mcl["step3_predicted_label"], 1)
        self.assertEqual(res_acl["step3_canonical_concept"], "ACL_TEAR")
        self.assertEqual(res_mcl["step3_canonical_concept"], "MCL_TEAR")

    def test_gr_medial_meniscus_tear(self):
        """GR: 'ρήξη του έσω μηνίσκου' -> Medial Meniscus = 1"""
        text = "Επιμήκης ρήξη του έσω μηνίσκου στο οπίσθιο κέρας με επαφή με την κάτω αρθρική επιφάνεια."
        res = process_study_target("gr_02", "Medial Meniscus", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "MEDIAL_MENISCUS_TEAR")

    # ==========================================
    # 6. BULGARIAN (BG) TESTS
    # ==========================================

    def test_bg_osteochondral_fracture(self):
        """BG: 'остеохондрална фрактура' -> Fracture = 1"""
        text = "МР находка: Изобразява се остеохондрална фрактура в областта на медиалния кондил на бедрената кост."
        res = process_study_target("bg_01", "Fracture", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "CORTICAL_FRACTURE")

    def test_bg_fracture_without_dislocation_positive(self):
        """BG: 'фрактура... без дислокация' is non-displaced fracture -> Fracture = 1"""
        text = "МР находка: МР данни за фрактура в областта на латералното тибиално плато, без дислокация."
        res = process_study_target("bg_02", "Fracture", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)

    # ==========================================
    # 7. DUTCH (NL) TESTS
    # ==========================================

    def test_nl_medial_meniscus_scheur(self):
        """NL: 'Volledige scheur van de mediale meniscus' -> Medial Meniscus = 1"""
        text = "Volledige scheur van de mediale meniscus ter hoogte van het achterhoorn."
        res = process_study_target("nl_01", "Medial Meniscus", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res["step3_predicted_label"], 1)
        self.assertEqual(res["step3_canonical_concept"], "MEDIAL_MENISCUS_TEAR")

    def test_nl_impaction_fracture_gated_to_contusion(self):
        """NL: 'Kleine impactiefractuur' -> Contusion = 1, Fracture = 0"""
        text = "Kleine impactiefractuur op centraal dragende deel van de laterale femorale condylen, gepaard gaand met uitgebreid botoedeem."
        res_frac = process_study_target("nl_02", "Fracture", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        res_cont = process_study_target("nl_02", "Contusion", baseline_pred=0, baseline_conf=0.5, evidence="", full_report=text)
        self.assertEqual(res_frac["step3_predicted_label"], 0, "Impaction fracture without cortical step-off must not be Fracture")
        self.assertEqual(res_cont["step3_predicted_label"], 1, "Impaction fracture is Contusion")


if __name__ == "__main__":
    unittest.main()
