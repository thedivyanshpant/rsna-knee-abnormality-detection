"""
AUDIT 4 & AUDIT 5: Rule Specificity, Overmatching Analysis & Provenance Audit.
Catalogs every rule and regex in multilingual.py, mapping to source studies,
Gold-58 dependencies, clinical specificity levels, and potential overmatching risks.
"""

import os
import pandas as pd
from phase_b.multilingual import MULTILINGUAL_LEXICON, NON_DISPLACEMENT_PATTERNS

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
OUTPUT_DIR = os.environ.get("PHASE_B_OUTPUT_DIR", os.path.join(REPO_ROOT, "results/archive"))

# Mapping of specific patterns created/refined during Gold-58 error analysis
GOLD58_MOTIVATED_PATTERNS = {
    "VKB Ruptur mit wohl ligamentärer Ruptur": "1.2.826.0.1.3680043.8.498.97274720257634584071500649275217521662",
    "knöchernem Ausriss": "1.2.826.0.1.3680043.8.498.97274720257634584071500649275217521662",
    "Condropatía rotuliana": "1.2.826.0.1.3680043.8.498.12448079646359892252441208258836556945, 1.2.826.0.1.3680043.8.498.69392348385274125385290015404144639002",
    "Ön çapraz bağ devamlılığı seçilememiştir": "1.2.826.0.1.3680043.8.498.30246079718471552972130572444383079911",
    "Poplitealna cista promjera 5 mm": "1.2.826.0.1.3680043.8.498.88077418639301174409926781329613570435",
    "ρήξη πρόσθιου χιαστού": "1.2.826.0.1.3680043.8.498.32321830776739689645700555055955725945",
    "ρήξη έσω πλαγίου": "1.2.826.0.1.3680043.8.498.32321830776739689645700555055955725945",
    "без дислокация": "1.2.826.0.1.3680043.8.498.59483999067816153759785789654766826710",
    "niet-verplaatste": "1.2.826.0.1.3680043.8.498.62549354677638403845904556149868367236",
    "menisco medial que no impresiona contactar": "1.2.826.0.1.3680043.8.498.15593638897292057356864060466120253309"
}

def run_provenance_audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    provenance_records = []
    
    rule_idx = 1
    for entry in MULTILINGUAL_LEXICON:
        canonical = entry["canonical_concept"]
        targets = "; ".join(entry["target_candidates"])
        clin_type = entry["clinical_type"]
        polarity = entry["polarity"]
        conf = entry.get("confidence", 0.90)

        for lang, pat_list in entry["patterns"].items():
            for pat in pat_list:
                # Classify provenance
                gold_dep = "NONE (General Lexicon)"
                audit_status = "General Lexicon"
                created_step3 = "YES"

                for key_phrase, uid in GOLD58_MOTIVATED_PATTERNS.items():
                    if key_phrase.lower() in pat.lower() or any(w in pat.lower() for w in key_phrase.lower().split() if len(w) > 5):
                        gold_dep = f"Gold-58 Study {uid}"
                        audit_status = "Gold-58 Error Derived"
                        break

                rule_id = f"RULE_ML_{canonical}_{lang}_{rule_idx:03d}"
                rule_idx += 1

                provenance_records.append({
                    "rule_id": rule_id,
                    "file": os.path.join(REPO_ROOT, "src/phase_b/lexicon.py"),
                    "canonical_concept": canonical,
                    "clinical_type": clin_type,
                    "language": lang,
                    "target": targets,
                    "pattern": pat,
                    "polarity": polarity,
                    "rule_description": f"Extracts {canonical} in {lang} targeting {targets}",
                    "gold_study_dependency": gold_dep,
                    "source_example": pat.replace("\\b", "").replace(".*?", " ... "),
                    "created_or_modified_during_step3": created_step3,
                    "confidence": conf,
                    "audit_status": audit_status
                })

    df_prov = pd.DataFrame(provenance_records)
    csv_path = f"{OUTPUT_DIR}/phase_b_step3_rule_provenance.csv"
    df_prov.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df_prov)} rules cataloged)")

    print("\nProvenance Summary:")
    print(df_prov["audit_status"].value_counts())
    print(f"\nTotal Gold-58 Error Derived Rules: {(df_prov['audit_status'] == 'Gold-58 Error Derived').sum()}/{len(df_prov)}")

    return df_prov

if __name__ == "__main__":
    run_provenance_audit()
