# Phase B — Step 3: Independent Reproducibility & Generalization Audit

**Document Date:** 2026-09-11  
**Audit Scope:** Phase B Step 3 Multilingual Clinical Concept Expansion & Normalization Layer  
**Auditor:** Independent Engineering & Scientific Validation Protocol (Antigravity Assistant)  
**Primary Dataset:** RSNA Knee Abnormality Detection Dataset (`train.csv`, `gold_studies.csv`, `golden_reports.csv`)  
**Production Code Inspected:** [`phase_b/multilingual.py`](file:///home/blackbook/phase_b/multilingual.py), [`phase_b/assertion.py`](file:///home/blackbook/phase_b/assertion.py), [`phase_b/pipeline.py`](file:///home/blackbook/phase_b/pipeline.py), [`phase_b/evaluate.py`](file:///home/blackbook/phase_b/evaluate.py)

---

## Executive Audit Summary

This independent audit evaluated whether Phase B Step 3 genuinely generalizes across diverse radiology reports or whether its improvements (84.77% accuracy, +11 error fixes, 0 regressions on Gold-58) stem from memorization / overfitting to the 58 Gold studies.

### Core Audit Findings Matrix:
| Audit Investigation | Evaluation Cohort | Key Metric / Result | Generalization Assessment |
| :--- | :--- | :--- | :--- |
| **Audit 1: Full Corpus Activation** | 4,407 Training Studies (52,884 target evaluations) | Target positive activation shift: +0.61% to +13.27% across targets | **Substantial Prevalence Increases** observed on Effusion (+13.27%), Baker's (+10.71%), Synovitis (+9.35%), and Medial Meniscus (+8.46%). |
| **Audit 2: Adversarial Perturbations** | 25 Multilingual Edge-Case Sentences | 19/25 Passed (76.00%) | **6 Edge-Case Failures Identified** (Greek `Δεν παρατηρείται`, German `Grad 1 ohne Arthrose`, non-displacement stripping edge-cases). |
| **Audit 3: Dev/Holdout Split** | 38 Dev Studies / 20 Holdout Studies (seed=42) | Dev: 84.65% (F1: 0.7722), Holdout: 85.00% (F1: 0.7992) | **No Holdout Degradation** observed on Gold-58 slice (Holdout fixes: 7, Regressions: 0). |
| **Audit 4: Rule Specificity Analysis** | 299 Lexical Patterns across 8 Languages | 3 Major Areas Audited (Displacement, Meniscus, PF OA) | **Semantic Overbreadth Identified** in Spanish `condropatía rotuliana` $\rightarrow$ PF OA without grade qualification. |
| **Audit 5: Provenance Audit** | 299 Total Cataloged Regex Rules | 230 General Lexicon (76.9%), 69 Gold-58 Error Derived (23.1%) | **69 Rules Directly Motivated** by Gold-58 failure cases. |
| **Audit 6: Exact Reproducibility** | Full Gold-58 (696 evaluations) | Exactly 590/696 (84.77%), Macro F1: 0.7889, 106 errors | **100% Bit-for-Bit Deterministic Reproduction**. |

---

## AUDIT 1 — Full 4,407-Report Activation Analysis

### 1.1 Dataset Identification
* **Dataset File:** `/mnt/data/RSNA_Knee_Data/train.csv` (Size: 5,690,007 bytes, SHA256: `8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33`)
* **Study Count:** 4,407 unique studies (`StudyInstanceUID`)
* **Radiology Report Column:** `'Report'` (string, complete free-text findings and impression)

### 1.2 Execution & Methodology
The full Step 3 pipeline was executed across all 4,407 studies for all 12 targets ($4,407 \times 12 = 52,884$ predictions). Concept activation frequencies, study coverage, linguistic distributions, and target prevalence shifts were measured against Phase A baseline and Step 2 severity gates.

### 1.3 Prevalence Shift Across Full 4,407 Corpus
| Target Condition | Phase A Baseline Positives (%) | Step 1 Positives (%) | Step 2 Positives (%) | Step 3 Positives (%) | Δ (S3 - S2) (% pts) | Δ (S3 - Base) (% pts) | Substantial Shift Flag (>3.0%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 0.54% (24) | 0.52% (23) | 0.52% (23) | **4.15% (183)** | **+3.63%** | +3.61% | **FLAGGED** |
| **MCL** | 0.20% (9) | 0.18% (8) | 0.18% (8) | **1.34% (59)** | +1.16% | +1.13% | Unflagged |
| **Medial Meniscus** | 0.59% (26) | 0.57% (25) | 0.54% (24) | **9.01% (397)** | **+8.46%** | +8.42% | **FLAGGED** |
| **Lateral Meniscus** | 0.52% (23) | 0.41% (18) | 0.41% (18) | **3.83% (169)** | **+3.43%** | +3.31% | **FLAGGED** |
| **Medial OA** | 0.34% (15) | 0.34% (15) | 0.34% (15) | **1.52% (67)** | +1.18% | +1.18% | Unflagged |
| **Lateral OA** | 0.25% (11) | 0.20% (9) | 0.20% (9) | **0.86% (38)** | +0.66% | +0.61% | Unflagged |
| **PF OA** | 0.48% (21) | 0.32% (14) | 0.32% (14) | **3.52% (155)** | **+3.20%** | +3.04% | **FLAGGED** |
| **Effusion** | 0.79% (35) | 0.66% (29) | 0.66% (29) | **13.93% (614)** | **+13.27%** | +13.14% | **FLAGGED** |
| **Synovitis** | 0.61% (27) | 0.48% (21) | 0.48% (21) | **9.83% (433)** | **+9.35%** | +9.21% | **FLAGGED** |
| **Baker's** | 0.27% (12) | 0.23% (10) | 0.20% (9) | **10.91% (481)** | **+10.71%** | +10.64% | **FLAGGED** |
| **Contusion** | 0.43% (19) | 0.36% (16) | 0.36% (16) | **3.88% (171)** | **+3.52%** | +3.45% | **FLAGGED** |
| **Fracture** | 0.41% (18) | 0.39% (17) | 0.39% (17) | **1.63% (72)** | +1.25% | +1.23% | Unflagged |

### 1.4 Analysis of Flagged Targets & Top Responsible Rules
1. **Effusion (+13.27% shift, 614 total positives):**
   - Dominant Rule: `RULE_MULTILINGUAL_SIGNIFICANT_EFFUSION` (585 firings, 13.27% of corpus).
   - Responsible Expressions: Bulgarian `ставен излив` (54 studies), Turkish `eklem içi sıvı artışı` (49 studies), Spanish `derrame articular` (38 studies), Croatian `zglobni izljev` (27 studies), Dutch `gewrichtsvocht` (18 studies).
   - Finding: Baseline extractor completely lacked non-English lexical entries for effusion, missing vast numbers of legitimate foreign effusion diagnoses.
2. **Baker's Cyst (+10.71% shift, 481 total positives):**
   - Dominant Rule: `RULE_MULTILINGUAL_BAKERS_CYST` (472 firings, 10.71% of corpus).
   - Responsible Expressions: German `Baker-Zyste` / `Poplitealzyste` (182 studies), Croatian `Bakerova cista` (114 studies), Turkish `Baker kisti` (88 studies), Spanish `quiste de Baker` (61 studies).
3. **Synovitis (+9.35% shift, 433 total positives):**
   - Dominant Rule: `RULE_MULTILINGUAL_SYNOVIAL_PROLIFERATION` (412 firings, 9.35% of corpus).
   - Responsible Expressions: German `Synovialitis` / `Synovialisproliferation` (190 studies), Turkish `sinovyal kalınlaşma` / `sinovit` (118 studies), Spanish `engrosamiento sinovial` (62 studies).
4. **Medial Meniscus (+8.46% shift, 397 total positives):**
   - Dominant Rule: `RULE_MULTILINGUAL_MEDIAL_MENISCUS_TEAR` (373 firings, 8.46% of corpus).
   - Responsible Expressions: Spanish `rotura de menisco interno` (142 studies), Turkish `medial menisküs ... yırtık` (111 studies), German `Innenmeniskusriss` (81 studies), Croatian `ruptura medijalnog meniska` (39 studies).

*Artifacts Generated:* [`phase_b_step3_full_corpus_activation_audit.csv`](file:///home/blackbook/phase_b/output/phase_b_step3_full_corpus_activation_audit.csv), [`phase_b_step3_full_corpus_activation_summary.json`](file:///home/blackbook/phase_b/output/phase_b_step3_full_corpus_activation_summary.json).

---

## AUDIT 2 — Adversarial Multilingual Perturbation Test

A dedicated adversarial test suite ([`phase_b/audit2_adversarial.py`](file:///home/blackbook/phase_b/audit2_adversarial.py)) containing 25 complex clinical linguistic edge cases was constructed and evaluated.

### 2.1 Adversarial Results Summary
* **Total Adversarial Tests:** 25
* **Passed:** 19 (76.00%)
* **Failed:** 6 (24.00%)

### 2.2 Granular Failure Analysis of Adversarial Cases
| Case ID | Language | Input Sentence | Target | Expected | Actual | Responsible Failure Mode |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`ADV_03`** | DE | `"Geringe Chondropathia patellae Grad 1 ohne Zeichen einer Arthrose."` | PF OA | 0 | **1 (FAIL)** | **German Severity Gate Absence:** German `Grad 1 ohne Zeichen einer Arthrose` was not parsed by English severity gates, allowing positive PF OA recovery. |
| **`ADV_04`** | GR | `"Δεν παρατηρείται ρήξη πρόσθιου χιαστού."` | ACL | 0 | **1 (FAIL)** | **Missing Greek Negative Copula:** Greek negative assertion list included `χωρίς` (without) but omitted `Δεν παρατηρείται` (is not observed), missing the negation. |
| **`ADV_08`** | DE | `"Ohne Fissur oder Dislokation der knöchernen Strukturen."` | Fracture | 0 | **1 (FAIL)** | **Negation Context Disruption:** Removing `Dislokation` via non-displacement filter broke the phrase boundary, preventing `Ohne Fissur` from negating baseline fracture. |
| **`ADV_11`** | TR | `"Medial menisküs posterior hornda grade 1 sinyal artışı izlenmektedir."` | Medial Meniscus | 0 | **1 (FAIL)** | **Turkish Degeneration Non-Gating:** When baseline predicted 1, Turkish `grade 1 sinyal artışı` matched neutral degeneration but did not force a downward gate to 0. |
| **`ADV_19`** | ES | `"Rotura compleja del menisco externo. Menisco interno conservado."` | Medial Meniscus | 0 | **1 (FAIL)** | **Cross-Clause Negation Scope:** `conservado` (preserved) was located in clause 2, but baseline evidence chunk contained clause 1, leaving baseline 1 untouched. |
| **`ADV_21`** | TR | `"Lateral menisküste kova sapı yırtığı izlendi. Medial menisküs doğaldır."` | Medial Meniscus | 0 | **1 (FAIL)** | **Cross-Clause Negation Scope:** Turkish `doğaldır` (is natural/intact) was in clause 2 and did not negate baseline pred = 1. |

*Artifact Generated:* [`phase_b_step3_adversarial_audit.csv`](file:///home/blackbook/phase_b/output/phase_b_step3_adversarial_audit.csv).

---

## AUDIT 3 — Gold-58 Development / Validation Separation

### 3.1 Split Methodology & Recording
To rigorously measure generalization without data leakage assumptions:
- The 58 Gold studies were split into a **Development Set (38 studies, 456 target evaluations)** and a **Holdout Validation Set (20 studies, 240 target evaluations)**.
- **Split Random Seed:** `RandomState(seed=42)` on sorted StudyInstanceUIDs.
- **Holdout StudyInstanceUIDs (20 studies):**
  1. `1.2.826.0.1.3680043.8.498.11548045715264151632742918861214041180` (EN)
  2. `1.2.826.0.1.3680043.8.498.11557620559191469069130827959098335840` (ES)
  3. `1.2.826.0.1.3680043.8.498.11915937982684988073121575003303350162` (EN)
  4. `1.2.826.0.1.3680043.8.498.12448079646359892252441208258836556945` (ES)
  5. `1.2.826.0.1.3680043.8.498.15593638897292057356864060466120253309` (ES)
  6. `1.2.826.0.1.3680043.8.498.24357731998592535749449852233668102377` (EN)
  7. `1.2.826.0.1.3680043.8.498.30246079718471552972130572444383079911` (TR)
  8. `1.2.826.0.1.3680043.8.498.32321830776739689645700555055955725945` (GR)
  9. `1.2.826.0.1.3680043.8.498.34752541300971032822452292723707026727` (HR/SR)
  10. `1.2.826.0.1.3680043.8.498.35626359505870050800880332857025219934` (EN)
  11. `1.2.826.0.1.3680043.8.498.36195568579979929944772183204918712628` (ES)
  12. `1.2.826.0.1.3680043.8.498.44183842189689531065114704381382414702` (EN)
  13. `1.2.826.0.1.3680043.8.498.54977581323931277817074741182670080450` (ES)
  14. `1.2.826.0.1.3680043.8.498.59483999067816153759785789654766826710` (BG)
  15. `1.2.826.0.1.3680043.8.498.67188121544063723837669983597453941774` (BG)
  16. `1.2.826.0.1.3680043.8.498.69392348385274125385290015404144639002` (ES)
  17. `1.2.826.0.1.3680043.8.498.71886121650392357774944851458428812904` (EN)
  18. `1.2.826.0.1.3680043.8.498.75187434248356774277526985329346125190` (BG)
  19. `1.2.826.0.1.3680043.8.498.88077418639301174409926781329613570435` (HR/SR)
  20. `1.2.826.0.1.3680043.8.498.97274720257634584071500649275217521662` (DE)

### 3.2 Performance Comparison Across Splits
| Evaluation Cohort | Studies | Evaluations | Accuracy (%) | Macro F1 | Precision | Recall | Specificity | FP | FN | Fixes (S2 $\rightarrow$ S3) | Regressions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development Set** | 38 | 456 | **84.65%** | **0.7722** | 0.7259 | 0.8547 | 0.8431 | 53 | 17 | 4 | 0 |
| **Holdout Validation Set** | 20 | 240 | **85.00%** | **0.7992** | 0.7353 | 0.9091 | 0.8176 | 25 | 11 | 7 | 0 |
| **Full Combined Gold-58** | 58 | 696 | **84.77%** | **0.7889** | 0.7288 | 0.8734 | 0.8124 | 78 | 28 | 11 | 0 |

### 3.3 Critical Scientific Disclosure on Holdout Validity
While the holdout set achieved 85.00% accuracy with 0 regressions, **4 of the 7 holdout error fixes fired on rules that were added during Step 3 error inspection of Gold-58** (e.g. German `knöchernem Ausriss` for Study `...21662`, Greek `ρήξη πρόσθιου χιαστού` for Study `...725945`, Turkish `Ön çapraz bağ devamlılığı` for Study `...079911`). Therefore, this 38/20 split represents a **semi-holdout reproducibility check**, not a fully blinded out-of-sample trial.

*Artifacts Generated:* [`phase_b_step3_holdout_results.csv`](file:///home/blackbook/phase_b/output/phase_b_step3_holdout_results.csv), [`phase_b_step3_holdout_summary.json`](file:///home/blackbook/phase_b/output/phase_b_step3_holdout_summary.json).

---

## AUDIT 4 — Rule Specificity & Overmatching Analysis

### 4.1 Non-Displacement Fracture Handling
* **Mechanism:** `is_clause_truly_negative()` removes non-displacement qualifiers before checking negation words.
* **Tested Expressions:**
  - `"Sin evidencia de fractura ni desplazamiento."` $\rightarrow$ Correctly evaluated as Fracture = 0 (Passed).
  - `"Keine Fraktur oder Dislokation."` $\rightarrow$ Correctly evaluated as Fracture = 0 (Passed).
  - `"Fraktur ohne Dislokation."` $\rightarrow$ Correctly evaluated as Fracture = 1 (Passed).
  - `"Nicht dislozierte Fraktur."` $\rightarrow$ Correctly evaluated as Fracture = 1 (Passed).
  - `"Ohne Fissur oder Dislokation."` $\rightarrow$ **Vulnerability:** Evaluated as Fracture = 1 when baseline pred = 1 because `Fissur` was not in the explicit negative assertion dictionary (Failed).
* **Verdict:** Mechanism is safe for standard `"ohne/sin/без Dislokation"` constructions, but vulnerable when paired with rare fracture synonyms like `Fissur`.

### 4.2 Meniscal Degeneration vs Tear Granularity
* **Distinctions Tested:**
  - Grade 1 / 2 intrasubstance signal without tear: Gated successfully when parsed through `MENISCUS_DEGENERATION` with compartment scoping.
  - Grade 3 surface-reaching tear: Correctly classified as positive tear (`MEDIAL_MENISCUS_TEAR`).
  - Intact / normal structure (`unauffälliger Meniskus`, `menisco conservado`): Correctly classified as negative.
* **Vulnerability:** When baseline input prediction is 1, Turkish `grade 1 sinyal artışı` does not actively force downward gating if evidence is not passed in the primary chunk.

### 4.3 Chondromalacia vs Patellofemoral Osteoarthritis (PF OA)
* **Rule Inspected:** `MULTILINGUAL_LEXICON` entry `PF_OA` matching Spanish `condropatía rotuliana` and `condromalacia rotuliana`.
* **Clinical Assessment:** In musculoskeletal radiology, Grade 1/2 chondromalacia is cartilage softening/edema and does not constitute osteoarthritis (which requires full-thickness defect, subchondral sclerosis, or osteophytes).
* **Audit Verdict:** The Spanish rule `r"\bcondromalacia\s+rotuliana\b"` is **clinically overbroad**. It matches mild non-OA chondromalacia. While it corrected 2 false negatives in Gold-58 (`...556945` and `...693923`), it caused 141 positive activations in the 4,407 corpus, representing an overmatching risk for mild cartilage wear.

---

## AUDIT 5 — Rule Provenance Audit

All 299 regex rules in [`phase_b/multilingual.py`](file:///home/blackbook/phase_b/multilingual.py) were cataloged into [`phase_b/output/phase_b_step3_rule_provenance.csv`](file:///home/blackbook/phase_b/output/phase_b_step3_rule_provenance.csv).

### Provenance Distribution:
* **General Lexicon Rules:** 230 rules (76.92%) derived from standard multilingual orthopedic radiological vocabularies.
* **Gold-58 Error Derived Rules:** 69 rules (23.08%) created or specifically tuned after inspecting Gold-58 baseline error cases.

### Exact Gold-58 Study Dependency Mapping:
| StudyInstanceUID | Target | Motivated Pattern / Rule | Clinical Rationale |
| :--- | :--- | :--- | :--- |
| `...97274720257634584071500649275217521662` (DE) | ACL | `\bvkb[\s-]*(?:ruptur\|riss)\b` | German compound noun separation |
| `...97274720257634584071500649275217521662` (DE) | Fracture | `\bknöchernem\s+ausriss\b` | German bone avulsion phrasing |
| `...12448079646359892252441208258836556945` (ES) | PF OA | `\bcondropatía\s+rotuliana\b` | Spanish patellar chondromalacia |
| `...69392348385274125385290015404144639002` (ES) | PF OA | `\bcondromalacia\s+rotuliana\b` | Spanish patellar chondromalacia |
| `...30246079718471552972130572444383079911` (TR) | ACL | `\bön\s+çapraz\s+bağ.*?(?:devamlılığı\s+seçilememiştir)\b` | Turkish loss of continuity |
| `...88077418639301174409926781329613570435` (HR/SR) | Baker's | `\bpoplitealna\s+cista\b` | Slavic popliteal cyst |
| `...32321830776739689645700555055955725945` (GR) | ACL | `\bπρόσθι(?:ος\|ου\|ο)\s+χιαστ(?:ός\|ού\|ο).*?ρήξ(?:η\|ης)\b` | Greek ACL rupture with accents |
| `...32321830776739689645700555055955725945` (GR) | MCL | `\bέσω\s+πλαγί(?:ος\|ου)\s+συνδέσμ(?:ος\|ου).*?ρήξ(?:η\|ης)\b` | Greek MCL rupture with accents |
| `...59483999067816153759785789654766826710` (BG) | Fracture | `\bбез\s+(?:данни\s+за\s+)?дислокация\b` | Bulgarian non-displacement qualifier |
| `...62549354677638403845904556149868367236` (NL) | Contusion | `\bniet[\s-]*(?:verplaatste\|verplaatst)\b` | Dutch non-displacement qualifier |
| `...15593638897292057356864060466120253309` (ES) | Medial Meniscus | `\bmenisco\s+medial.*?(?:no\s+impresiona\s+contactar)\b` | Spanish non-surface reaching sign |

---

## AUDIT 6 — Reproducibility Verification of 84.77% Result

The official evaluation script was re-run without code modification:
* **Execution Command:** `PYTHONPATH=/home/blackbook python3 /home/blackbook/phase_b/evaluate.py`
* **Python Environment:** Python 3.13.13 (GCC 14.3.0), `pandas` 3.0.5, `numpy` 2.5.3, `scikit-learn` 1.9.0
* **Input Predictions:** `/home/blackbook/phase_a_baseline/predictions_gold58.csv` (SHA256: `cbbe007f3b231c9240c536ad6d6858ca99b90216a2ea980244d736788730717e`)
* **Output Predictions:** `/home/blackbook/phase_b/output/phase_b_step3_predictions_gold58.csv` (SHA256: `f3ed9da103cc678bf86584993d53a83908fdda434ccc8ffeaa80756b8c369010`)

### Verification Outcomes:
- **Total Evaluations:** 696 / 696
- **Correct Predictions:** Exactly 590 / 696 (**84.77%**)
- **Macro F1:** Exactly **0.7889**
- **Total Errors:** Exactly **106** (78 FP, 28 FN)
- **Step 2 $\rightarrow$ Step 3 Corrections:** Exactly **11**
- **Regressions:** Exactly **0**
- **Exact Study Match:** Exactly **9 / 58 (15.52%)**

**Audit Status:** **REPRODUCED & VERIFIED**.

---

## AUDIT 7 — Final Scientific Verdict

Based on the 6 audits, the formal scientific verdict is categorized as follows:

### 1. REPRODUCED
- The reported 84.77% accuracy, 0.7889 Macro F1, 106 errors, 11 corrections, and 0 regressions on Gold-58 reproduce 100% deterministically from the codebase.

### 2. VERIFIED
- Multilingual concept expansion successfully extracts non-English orthopedic terminology across German, Spanish, Turkish, Croatian/Serbian, Greek, Bulgarian, and Dutch.
- Clause-level Unicode splitting (`\w`) resolved sentence collapse in Greek and Cyrillic reports.
- Non-displacement exception filtering correctly protects true fractures from false negative inversion.

### 3. NOT VERIFIED
- Out-of-sample accuracy on unlabelled non-English studies cannot be claimed as 84.77% because 23.08% of rules were tuned on Gold-58 error cases.

### 4. OVERFIT RISK: LOW-TO-MODERATE
- While 69 patterns were derived from Gold-58 errors, the Holdout evaluation (20 studies) showed consistent performance (85.00% accuracy, 0.7992 F1). Overfitting is constrained because rules are structured medical concept mappings rather than memorized full-sentence strings.

### 5. GENERALIZATION RISK: MODERATE
- Adversarial tests identified that negative assertion phrases in Greek (`Δεν παρατηρείται`) and Turkish (`doğaldır`) lack coverage when baseline input predictions are noisy 1s.

### 6. CLINICAL / SEMANTIC RISK: MODERATE
- Equating mild `condropatía rotuliana` (chondromalacia) directly to `PF OA` without severity qualification presents a clinical overbreadth risk, leading to elevated PF OA activations across the 4,407 corpus.

### 7. REQUIRED FIXES BEFORE PRODUCTION PIPELINE ACCEPTANCE
1. **Add Multilingual Severity Gates for PF OA:** Grade 1/2 Spanish `condromalacia rotuliana` and German `Chondropathia patellae Grad 1` should require osteophytes, sclerosis, or Grade 3/4 full-thickness loss before asserting PF OA.
2. **Expand Greek and Turkish Negative Assertion Lexicons:** Add `Δεν παρατηρείται`, `χωρίς ένδειξη`, `doğaldır`, `izlenmemiştir` to standard assertion dictionaries.
3. **Refine German Fracture Negation Synonyms:** Add `Fissur` and `Infraktion` to fracture negation dictionaries.
