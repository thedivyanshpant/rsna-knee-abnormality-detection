# Phase D1.1: Full 5-Fold Production Cross-Validation & Global OOF Evaluation

- **Execution Date**: `2026-09-12T22:00:56.199006+00:00`
- **Total Development Studies**: **4,349** (100% complete disjoint CV coverage)
- **Integrity Audit**: **PASS (0 anomalies)**
- **Gold-58 Leakage**: **0** (Strictly excluded)

## 1. Headline Benchmark: D1.1 vs Phase C Control Baseline

| Evaluation Scope | Phase C Global OOF | Phase D D1.1 Global OOF | Absolute Delta ($\Delta$) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Global Macro ROC-AUC (Unweighted)** | 0.6198 | **0.7510** | **+0.1312** | **+21.17%** |
| **Global Macro ROC-AUC (Sample-Weighted)** | 0.6198 | **0.7573** | **+0.1375** | **+22.18%** |
| **Global Macro PR-AUC** | -- | **0.2212** | -- | -- |

## 2. Fold-Level Performance Progression

| Fold | Validation Studies | Phase C Baseline | D1.1 Unweighted AUC | D1.1 Weighted AUC | Delta ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 872 | 0.6146 | **0.7512** | **0.7566** | **+0.1366** |
| **Fold 1** | 871 | 0.6400 | **0.7277** | **0.7323** | **+0.0877** |
| **Fold 2** | 870 | 0.6603 | **0.8067** | **0.8115** | **+0.1464** |
| **Fold 3** | 869 | 0.6647 | **0.8161** | **0.8303** | **+0.1514** |
| **Fold 4** | 867 | 0.6122 | **0.8252** | **0.8336** | **+0.2130** |

## 3. Target-by-Target Breakdown (All 12 Pathologies)

| Pathology Target | Positives / Total | Prevalence | Phase C OOF AUC | D1.1 OOF AUC (Unw) | D1.1 PR-AUC | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 137 / 4,349 | 3.1% | 0.6096 | **0.7674** | 0.1250 | **+0.1578** |
| **MCL** | 109 / 4,349 | 2.5% | 0.5126 | **0.6148** | 0.0480 | **+0.1022** |
| **Medial Meniscus** | 206 / 4,349 | 4.7% | 0.6839 | **0.8198** | 0.2314 | **+0.1359** |
| **Lateral Meniscus** | 99 / 4,349 | 2.3% | 0.5861 | **0.7234** | 0.0634 | **+0.1373** |
| **Medial OA** | 15 / 4,349 | 0.3% | 0.5571 | **0.8617** | 0.0689 | **+0.3046** |
| **Lateral OA** | 3 / 4,349 | 0.1% | 0.6419 | **0.9025** | 0.0065 | **+0.2606** |
| **PF OA** | 1 / 4,349 | 0.0% | 0.4449 | **0.2282** | 0.0003 | -0.2167 |
| **Effusion** | 489 / 4,349 | 11.2% | 0.7666 | **0.8962** | 0.6996 | **+0.1296** |
| **Synovitis** | 413 / 4,349 | 9.5% | 0.7398 | **0.8754** | 0.5871 | **+0.1356** |
| **Baker's** | 385 / 4,349 | 8.8% | 0.6777 | **0.8408** | 0.4356 | **+0.1631** |
| **Contusion** | 387 / 4,349 | 8.9% | 0.6365 | **0.7715** | 0.3023 | **+0.1350** |
| **Fracture** | 101 / 4,349 | 2.3% | 0.5811 | **0.7108** | 0.0870 | **+0.1297** |

## 4. Integrity and Governance Audit

- Total rows in global OOF table: **4,349** (Target: 4,349)
- Unique StudyInstanceUIDs: **4,349**
- Duplicate studies: **0**
- Missing development studies: **0**
- Gold-58 studies: **0** (Strictly excluded)
- NaN / Inf anomalies: **0 / 0**
- Fold provenance breakdown: `{0: 872, 1: 871, 2: 870, 3: 869, 4: 867}`
