# Phase C3.12 Stage A: Out-of-Fold (OOF) Audit Report

## 1. OOF Integrity Summary

- **Total Non-Gold Development Studies**: `4,349`
- **Unique Studies Covered**: `4,349` (`100.0%`)
- **Duplicate Records**: `0`
- **Missing Development Studies**: `0`
- **Gold-58 Leakage**: **`0` (PASS)**
- **Provenance Alignment**: `100.0%` (Every study predicted strictly by its held-out fold checkpoint)
- **Consolidated OOF Path**: `phase_c/c3/results/oof_predictions_5folds.csv`

## 2. Global Study-Level OOF Metrics vs Fold Means

- **Global Study-Level OOF Macro ROC-AUC**: **`0.6190`** (concatenated 4,349 predictions)
- **Historical C3.11 Fold-Level Mean Macro ROC-AUC**: `0.6333 ± 0.0326`

### Per-Target Global OOF Performance

| Target Pathology | Global OOF ROC-AUC | Spearman rho | Pearson r | Positives (Y>=0.5) | Prevalence (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ACL** | `0.6200` | `0.3901` | `0.2271` | `137 / 4,349` | `3.15%` |
| **MCL** | `0.5098` | `0.3908` | `0.2008` | `109 / 4,349` | `2.51%` |
| **Medial Meniscus** | `0.6772` | `0.2992` | `0.2373` | `206 / 4,349` | `4.74%` |
| **Lateral Meniscus** | `0.5964` | `0.3966` | `0.2438` | `99 / 4,349` | `2.28%` |
| **Medial OA** | `0.5571` | `0.3555` | `0.2966` | `15 / 4,349` | `0.34%` |
| **Lateral OA** | `0.6419` | `0.3978` | `0.3822` | `3 / 4,349` | `0.07%` |
| **PF OA** | `0.4449` | `0.3839` | `0.3726` | `1 / 4,349` | `0.02%` |
| **Effusion** | `0.7624` | `0.2179` | `0.3341` | `489 / 4,349` | `11.24%` |
| **Synovitis** | `0.7411` | `0.4473` | `0.3578` | `413 / 4,349` | `9.50%` |
| **Baker's** | `0.6714` | `0.3263` | `0.2487` | `385 / 4,349` | `8.85%` |
| **Contusion** | `0.6352` | `0.3559` | `0.2293` | `387 / 4,349` | `8.90%` |
| **Fracture** | `0.5705` | `0.4173` | `0.2679` | `101 / 4,349` | `2.32%` |
