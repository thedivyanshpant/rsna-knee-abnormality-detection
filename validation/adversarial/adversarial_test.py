"""
Adversarial Verification Suite for Phase B Step 3 Remediation.
Evaluates the remediated pipeline across all 25 multilingual perturbation cases.
"""

import sys, os
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


import os
import csv
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False
from phase_b.pipeline import process_remediated_study_target

OUTPUT_DIR = os.environ.get("PHASE_B_OUTPUT_DIR", os.path.join(_ROOT_DIR, "validation", "adversarial"))

ADVERSARIAL_CASES = [
    # 1. Spanish Fracture Negation + Alignment
    {
        "id": "ADV_01",
        "lang": "ES",
        "sentence": "Sin evidencia de fractura ni desplazamiento óseo.",
        "eval_target": "Fracture",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Spanish: 'Sin evidencia de fractura ni desplazamiento óseo.' -> Fracture = 0"
    },
    # 2. German Meniscus + ACL Negation
    {
        "id": "ADV_02_MM",
        "lang": "DE",
        "sentence": "Keine Meniskusläsion, unauffälliger VKB.",
        "eval_target": "Medial Meniscus",
        "expected": 0,
        "baseline_pred": 1,
        "description": "German: 'Keine Meniskusläsion, unauffälliger VKB.' -> Medial Meniscus = 0"
    },
    {
        "id": "ADV_02_LM",
        "lang": "DE",
        "sentence": "Keine Meniskusläsion, unauffälliger VKB.",
        "eval_target": "Lateral Meniscus",
        "expected": 0,
        "baseline_pred": 1,
        "description": "German: 'Keine Meniskusläsion, unauffälliger VKB.' -> Lateral Meniscus = 0"
    },
    {
        "id": "ADV_02_ACL",
        "lang": "DE",
        "sentence": "Keine Meniskusläsion, unauffälliger VKB.",
        "eval_target": "ACL",
        "expected": 0,
        "baseline_pred": 1,
        "description": "German: 'Keine Meniskusläsion, unauffälliger VKB.' -> ACL = 0"
    },
    # 3. German Grade 1 Chondromalacia without OA
    {
        "id": "ADV_03",
        "lang": "DE",
        "sentence": "Geringe Chondropathia patellae Grad 1 ohne Zeichen einer Arthrose.",
        "eval_target": "PF OA",
        "expected": 0,
        "baseline_pred": 1,
        "description": "German: 'Geringe Chondropathia patellae Grad 1 ohne Zeichen einer Arthrose.' -> PF OA = 0"
    },
    # 4. Greek ACL Negation
    {
        "id": "ADV_04",
        "lang": "GR",
        "sentence": "Δεν παρατηρείται ρήξη πρόσθιου χιαστού.",
        "eval_target": "ACL",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Greek: 'Δεν παρατηρείται ρήξη πρόσθιου χιαστού.' -> ACL = 0"
    },
    # 5. Fracture + "without displacement" (Bulgarian true fracture)
    {
        "id": "ADV_05",
        "lang": "BG",
        "sentence": "МР данни за фрактура в областта на латералното тибиално плато, без дислокация.",
        "eval_target": "Fracture",
        "expected": 1,
        "baseline_pred": 0,
        "description": "Bulgarian: Fracture with non-displacement -> Fracture = 1"
    },
    # 6. Fracture + "without displacement" (German true fracture)
    {
        "id": "ADV_06",
        "lang": "DE",
        "sentence": "Nicht dislozierte Fraktur des medialen Tibiaplateaus.",
        "eval_target": "Fracture",
        "expected": 1,
        "baseline_pred": 0,
        "description": "German: 'Nicht dislozierte Fraktur' -> Fracture = 1"
    },
    # 7. Fracture negation + dislocation (German no fracture)
    {
        "id": "ADV_07",
        "lang": "DE",
        "sentence": "Keine Fraktur oder Dislokation nachweisbar.",
        "eval_target": "Fracture",
        "expected": 0,
        "baseline_pred": 1,
        "description": "German: 'Keine Fraktur oder Dislokation' -> Fracture = 0"
    },
    # 8. Fracture negation (German 'Ohne Fissur oder Dislokation')
    {
        "id": "ADV_08",
        "lang": "DE",
        "sentence": "Ohne Fissur oder Dislokation der knöchernen Strukturen.",
        "eval_target": "Fracture",
        "expected": 0,
        "baseline_pred": 1,
        "description": "German: 'Ohne Fissur oder Dislokation' -> Fracture = 0"
    },
    # 9. Meniscal degeneration without tear (Croatian)
    {
        "id": "ADV_09",
        "lang": "HR/SR",
        "sentence": "Degeneracija medijalnog meniska bez znakova rupture.",
        "eval_target": "Medial Meniscus",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Croatian: 'Degeneracija medijalnog meniska bez znakova rupture' -> Medial Meniscus = 0"
    },
    # 10. Degenerative meniscal tear (German true tear)
    {
        "id": "ADV_10",
        "lang": "DE",
        "sentence": "Degenerativer Innenmeniskusriss im Hinterhorn.",
        "eval_target": "Medial Meniscus",
        "expected": 1,
        "baseline_pred": 0,
        "description": "German: 'Degenerativer Innenmeniskusriss' -> Medial Meniscus = 1"
    },
    # 11. Grade I meniscal signal (Turkish non-tear)
    {
        "id": "ADV_11",
        "lang": "TR",
        "sentence": "Medial menisküs posterior hornda grade 1 sinyal artışı izlenmektedir.",
        "eval_target": "Medial Meniscus",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Turkish: Grade 1 meniscal signal -> Medial Meniscus = 0"
    },
    # 12. Grade II meniscal signal (Spanish non-tear)
    {
        "id": "ADV_12",
        "lang": "ES",
        "sentence": "Meniscopatía grado II en cuerno posterior de menisco interno.",
        "eval_target": "Medial Meniscus",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Spanish: Grade 2 meniscal signal -> Medial Meniscus = 0"
    },
    # 13. Grade III surface-reaching tear (Turkish true tear)
    {
        "id": "ADV_13",
        "lang": "TR",
        "sentence": "Medial menisküs posterior hornda eklem yüzüne uzanan grade 3 yırtık.",
        "eval_target": "Medial Meniscus",
        "expected": 1,
        "baseline_pred": 0,
        "description": "Turkish: Grade 3 surface-reaching tear -> Medial Meniscus = 1"
    },
    # 14. Patellar chondromalacia without OA (Spanish mild)
    {
        "id": "ADV_14",
        "lang": "ES",
        "sentence": "Condromalacia rotuliana grado 1 sin signos de artrosis patelofemoral.",
        "eval_target": "PF OA",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Spanish: Grade 1 chondromalacia without OA -> PF OA = 0"
    },
    # 15. Patellar cartilage loss with explicit OA (Spanish severe)
    {
        "id": "ADV_15",
        "lang": "ES",
        "sentence": "Artrosis patelofemoral severa con pérdida completa del cartílago rotuliano.",
        "eval_target": "PF OA",
        "expected": 1,
        "baseline_pred": 0,
        "description": "Spanish: Severe PF OA with complete cartilage loss -> PF OA = 1"
    },
    # 16. Intact ACL (German normal)
    {
        "id": "ADV_16",
        "lang": "DE",
        "sentence": "Das vordere Kreuzband zeigt sich durchgehend intakt und regelrecht.",
        "eval_target": "ACL",
        "expected": 0,
        "baseline_pred": 1,
        "description": "German: Intact ACL -> ACL = 0"
    },
    # 17. ACL tear (Dutch true rupture)
    {
        "id": "ADV_17",
        "lang": "NL",
        "sentence": "Volledige scheur van de voorste kruisband met oedeem.",
        "eval_target": "ACL",
        "expected": 1,
        "baseline_pred": 0,
        "description": "Dutch: Full ACL tear -> ACL = 1"
    },
    # 18. Multilingual Effusion Negation (Greek no effusion)
    {
        "id": "ADV_18",
        "lang": "GR",
        "sentence": "Χωρίς συλλογή υγρού στο γόνατο.",
        "eval_target": "Effusion",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Greek: 'Χωρίς συλλογή υγρού' -> Effusion = 0"
    },
    # 19. Multilingual Anatomical Mismatch (Spanish Lateral tear evaluated on Medial)
    {
        "id": "ADV_19",
        "lang": "ES",
        "sentence": "Rotura compleja del menisco externo. Menisco interno conservado.",
        "eval_target": "Medial Meniscus",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Spanish: Lateral tear only -> Medial Meniscus = 0"
    },
    # 20. Dutch Impaction Contusion vs Fracture
    {
        "id": "ADV_20_Cont",
        "lang": "NL",
        "sentence": "Impactiefractuur op centraal dragende deel van de laterale femorale condylen, gepaard gaand met uitgebreid botoedeem.",
        "eval_target": "Contusion",
        "expected": 1,
        "baseline_pred": 0,
        "description": "Dutch: 'Impactiefractuur' with botoedeem -> Contusion = 1"
    },
    {
        "id": "ADV_20_Frac",
        "lang": "NL",
        "sentence": "Impactiefractuur op centraal dragende deel van de laterale femorale condylen, gepaard gaand met uitgebreid botoedeem.",
        "eval_target": "Fracture",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Dutch: Subchondral 'Impactiefractuur' gated from cortical Fracture -> Fracture = 0"
    },
    # 21. Turkish Bucket-handle Lateral Tear (Anatomical Scoping)
    {
        "id": "ADV_21",
        "lang": "TR",
        "sentence": "Lateral menisküste kova sapı yırtığı izlendi. Medial menisküs doğaldır.",
        "eval_target": "Medial Meniscus",
        "expected": 0,
        "baseline_pred": 1,
        "description": "Turkish: Lateral bucket-handle tear only -> Medial Meniscus = 0"
    },
    # 22. Croatian Baker's Cyst
    {
        "id": "ADV_22",
        "lang": "HR/SR",
        "sentence": "U poplitealnoj jami vidi se manja Bakerova cista promera 12 mm.",
        "eval_target": "Baker's",
        "expected": 1,
        "baseline_pred": 0,
        "description": "Croatian: Bakerova cista -> Baker's = 1"
    }
]

def run_adversarial_verification():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print("Running Remediation Adversarial Perturbation Verification...")
    for item in ADVERSARIAL_CASES:
        cid = item["id"]
        lang = item["lang"]
        sent = item["sentence"]
        target = item["eval_target"]
        expected = item["expected"]
        base_pred = item["baseline_pred"]

        res = process_remediated_study_target(
            study_uid=cid,
            target=target,
            baseline_pred=base_pred,
            baseline_conf=0.8,
            evidence=sent,
            full_report=sent,
            gold_label=expected
        )

        final_pred = res["final_predicted_label"]
        is_pass = (final_pred == expected)

        results.append({
            "case_id": cid,
            "language": lang,
            "input_sentence": sent,
            "target": target,
            "expected_label": expected,
            "baseline_pred": base_pred,
            "remediated_final_pred": final_pred,
            "label_probability": res["label_probability"],
            "confidence_score": res["confidence_score"],
            "sample_weight": res["sample_weight"],
            "abstain_flag": res["abstain_flag"],
            "step3_rule_applied": res["step3_rule_applied"],
            "matched_concept": res["matched_canonical_concept"],
            "evidence_span": res["evidence_char_span"],
            "status": "PASS" if is_pass else "FAIL",
            "description": item["description"]
        })

    csv_path = f"{OUTPUT_DIR}/phase_b_step3_remediated_adversarial_audit.csv"
    if HAS_PANDAS:
        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} ({len(df_res)} test cases)")
        n_pass = int((df_res["status"] == "PASS").sum())
        n_total = len(df_res)
    else:
        if results:
            keys = list(results[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
        print(f"Saved {csv_path} ({len(results)} test cases)")
        n_pass = sum(1 for r in results if r["status"] == "PASS")
        n_total = len(results)
        df_res = results

    pass_pct = (n_pass / n_total) * 100 if n_total > 0 else 0
    print(f"\n===================================================================================================")
    print(f"REMEDIATION ADVERSARIAL VERIFICATION SUMMARY: {n_pass}/{n_total} PASSED ({pass_pct:.2f}%)")
    print(f"===================================================================================================")

    if n_pass < n_total:
        print("\nFailed Cases:")
        if HAS_PANDAS:
            print(df_res[df_res["status"] == "FAIL"][["case_id", "target", "expected_label", "remediated_final_pred", "input_sentence", "description"]].to_string(index=False))
        else:
            for r in [x for x in results if x["status"] == "FAIL"]:
                print(f"  {r['case_id']}: {r['target']} expected {r['expected_label']} got {r['remediated_final_pred']} | {r['input_sentence']}")

    return df_res

if __name__ == "__main__":
    run_adversarial_verification()
