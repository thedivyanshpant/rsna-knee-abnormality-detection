# Phase D3-A: Clinical Label Extraction & Sparsity Root-Cause Audit Report

## 1. Executive Summary

During Phase D1.2, evaluation against the locked Gold-58 cohort revealed an extreme prevalence mismatch between the 4,349-study development corpus and the locked clinical Gold cohort for Osteoarthritis targets:
- **Lateral OA**: 3 positives / 4,349 in development (0.07%) vs. 11 / 58 in Gold (18.97%).
- **PF OA**: 1 positive / 4,349 in development (0.02%) vs. 21 / 58 in Gold (36.21%).
- **Medial OA**: 15 positives / 4,349 in development (0.34%) vs. 19 / 58 in Gold (32.76%).

This comprehensive root-cause audit investigates the exact natural language processing, regex extraction, severity thresholds, and anatomical scoping implemented in Phase B Step 3 (`src/phase_b/lexicon.py`, `src/phase_b/pipeline.py`, `src/phase_b/severity_gates.py`) across all 4,349 development clinical reports.

---

## 2. Root Causes of Phase B Step 3 Sparsity

The audit identified three primary structural failure modes in the Phase B Step 3 extraction pipeline:

### Failure Mode 1: Romance Language Syntax Incompatibility (Noun-Adjective Word Order)
In `src/phase_b/pipeline.py` (lines 63–65), `TARGET_KEYWORD_MAP` defined compartment keywords as:
```python
"Medial OA": [r"\bmedial\s+(?:compartment\s+)?(?:oa|osteoarthritis|arthrose|artrosis)\b", r"\bgonartrosi\b"]
"Lateral OA": [r"\blateral\s+(?:compartment\s+)?(?:oa|osteoarthritis|arthrose|artrosis)\b"]
"PF OA": [r"\bpatellofemoral\s+(?:oa|osteoarthritis|arthrose|artrosis)\b", r"\bpf\s+oa\b", r"\bretropatellare\s+arthrose\b"]
```
- **The Defect**: English puts the anatomical modifier before the noun ("lateral osteoarthritis", "medial arthrosis"). In Spanish (which represents over 90% of non-English reports in this corpus), adjectives follow the noun ("artrosis femorotibial lateral", "artrosis femorotibial medial", "artrosis femoropatelar").
- **Clinical Impact**: The regex `r"\blateral\s+(?:compartment\s+)?(?:oa|osteoarthritis|arthrose|artrosis)\b"` had **zero matches** on Spanish reports with "artrosis femorotibial lateral". The regexes failed simply because of inverted Romance language syntax.

### Failure Mode 2: Rigid Medical Vocabulary & Missing Latin American Terminology
In `src/phase_b/lexicon.py` (lines 510–572):
- The lexicon searched exclusively for `condromalacia rotuliana` and `gonartrosis`.
- **The Defect**: In Latin American (specifically Chilean) musculoskeletal MRI reporting, radiologists standardly describe articular cartilage degradation using the term **"condropatía"** rather than "condromalacia" or "artrosis":
  - `condromalacia / condropatía`: Mentioned in **326 reports (7.50% of the corpus)**.
  - `femorotibial`: Mentioned in **322 reports (7.40% of the corpus)**.
  - `rotuliana / patelar`: Mentioned in **273 reports (6.28% of the corpus)**.
  - `femoropatelar / patelofemoral`: Mentioned in **223 reports (5.13% of the corpus)**.
  - `pinzamiento`: Mentioned in **99 reports (2.28% of the corpus)**.
- **Clinical Impact**: Because "condropatía" was completely omitted from `phase_b/lexicon.py` and `phase_b/pipeline.py`, hundreds of clear, high-grade cartilage loss descriptions were completely discarded.

### Failure Mode 3: Omission of Compartment Sub-Anatomy
In `src/phase_b/anatomy.py`:
- The compartment mapper expected literal "medial" or "lateral" compartment tokens, but completely lacked:
  - `platillo tibial lateral` / `platillo tibial medial` (lateral/medial tibial plateau)
  - `cóndilo femoral lateral` / `cóndilo femoral medial` (lateral/medial femoral condyle)
  - `rótula` / `patela` / `tróclea femoral` (patellar and trochlear groove components of the PF joint)
  - `artrosis tricompartimental` / `gonartrosis tricompartimental` (global three-compartment joint failure)
- **Clinical Impact**: Radiologists frequently report findings at specific sub-structures (e.g., *"Condropatía lesiones grado 3 del platillo tibial lateral"*). Because "platillo tibial lateral" was not mapped to Lateral OA, these cases were lost.

---

## 3. Quantification of Affected Reports in Development Corpus (4,349 Studies)

| Clinical Phenomenon / Concept in Free Text | Reports in Dev Corpus | % of Dev Corpus | Discarded in Phase B Step 3? |
| :--- | :---: | :---: | :---: |
| **Condropatía / Condromalacia** | 326 | 7.50% | **YES (~85% discarded)** |
| **Femorotibial Cartilage Alteration** | 322 | 7.40% | **YES (~80% discarded)** |
| **Patellar / Trochlear Cartilage Alteration** | 273 | 6.28% | **YES (~95% discarded)** |
| **Joint Space Pinzamiento (Narrowing)** | 99 | 2.28% | **YES (100% discarded)** |
| **Explicit Artrosis / Osteoarthritis** | 46 | 1.06% | Partial (English caught, Spanish syntax missed) |
| **Tricompartmental Osteoarthritis** | 37 | 0.85% | **YES (100% discarded)** |

---

## 4. Remediation Architecture & Governance Rules for Phase D3

To ensure rigorous scientific calibration without label corruption:
1. **Decoupled Soft Targets**:
   - **Explicit Severe / Advanced OA** (Outerbridge Grade 3–4, full-thickness loss, subchondral marrow edema, bone-on-bone): Assigned calibrated probability $Y = 0.95$, confidence $= 0.95$, sample weight $= 1.00$.
   - **Moderate Chondral Wear** (Grade 2, unclassified condropatía, superficial-to-moderate ulceration): Assigned calibrated probability $Y = 0.80$, confidence $= 0.85$, sample weight $= 0.85$.
   - **Explicit Negative** (normal cartilage, preserved joint space, intact surfaces): Assigned calibrated probability $Y = 0.05$, confidence $= 0.95$, sample weight $= 1.00$.
   - **Unmentioned / Neutral**: Preserves Phase B baseline $Y = 0.20$, confidence $= 0.80$, sample weight $= 0.85$.
2. **Preservation of Non-OA Targets**:
   - All 9 non-OA targets (ACL, MCL, Medial Meniscus, Lateral Meniscus, Effusion, Synovitis, Baker's, Contusion, Fracture) remain **100% byte-for-byte identical** to Phase B Step 3.
3. **Strict Isolation & Governance**:
   - The frozen production corpus `/scratch/deepanthpcu/RSNA_Code/labels/production/phase_b_step3_remediated_training_corpus.csv` is NEVER modified.
   - All candidate labels are versioned independently in `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/labels/`.
   - Gold-58 studies are strictly quarantined with 0 leakage.
