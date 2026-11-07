# PHASE D8 — LOCKED GOLD-58 BENCHMARK REPORT (D8-A, D8-B, D8-C, D8-D)

**Date**: 2026-09-15 17:47:45 UTC  
**Cohort**: 58 Permanently Isolated Clinical Studies (`csv/gold_studies.csv`)  
**Evaluation Pass**: ONE-TIME AUTHORIZED LOCKED BENCHMARK (Strictly No Retraining/Tuning Afterward)  
**Governance**: Single-Pass Evaluation; Permanent Quarantine Immediately Resealed.  

---

## 1. Executive Gold-58 Benchmark Matrix

| Model Identifier | Architecture | Volumetric Sampling | Gold Macro ROC-AUC | Gold PR-AUC | $\Delta$ vs D7-C Record (0.6474) | $\Delta$ vs D6 Champion (0.6331) | $\Delta$ vs D1.2-B2 (0.6174) | $\Delta$ vs Phase C (0.5986) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase C Baseline** | 3D ResNet-18 Control | Central Band | `0.5986` | `0.4526` | -0.0488 | -0.0345 | -0.0188 | — |
| **Phase D1.2-B2** | ViT-S/14 Standalone | Central Band | `0.6174` | `0.4938` | -0.0300 | -0.0157 | — | +0.0188 |
| **Phase D5-B2** | ViT-S/14 (D3 Repr) | Central Band | `0.6307` | `0.5285` | -0.0167 | -0.0024 | +0.0133 | +0.0321 |
| **Phase D6-B2 Champion**| ViT-S/14 (6 Slots) | Central Band | `0.6331` | `0.5125` | -0.0143 | — | +0.0157 | +0.0345 |
| **Phase D7-C Record** | ViT-B/14 (87.2M) | Multi-Window | **`0.6474`** | **`0.5290`** | — | **+0.0143** | **+0.0300** | **+0.0488** |
| **D8-A** | ViT-S/14 + PathologyCrossAttn | Central Band (12 grp) | **`0.6311`** | `0.5133` | **`-0.0163`** | **`-0.0020`** | `+0.0137` | `+0.0325` |
| **D8-B** | ViT-S/14 + HighBudget (24 grp) | High-Budget (24 grp) | **`0.6312`** | `0.4952` | **`-0.0162`** | **`-0.0019`** | `+0.0138` | `+0.0326` |
| **D8-C** | Hybrid Conv+Attn (CoAtNet) | Central Band (12 grp) | **`0.6487`** | `0.5120` | **`+0.0013`** | **`+0.0156`** | `+0.0313` | `+0.0501` |
| **D8-D** | Full Combined D8 Configuration | High-Budget (24 grp) | **`0.6322`** | `0.5214` | **`-0.0152`** | **`-0.0009`** | `+0.0148` | `+0.0336` |

---

## 2. Per-Pathology Clinical Breakdown on Gold-58 (Rank-Averaged AUC)

| Pathology Target | Pos / Neg | D6 Champion | D7-C Record | D8-A (CrossAttn) | D8-B (HighBudget) | D8-C (Hybrid) | D8-D (Combined) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 24 / 34 | `0.6538` | `0.6577` | `0.6238` | `0.6415` | `0.6483` | `0.5876` |
| **MCL** | 9 / 49 | `0.7211` | `0.6868` | `0.5941` | `0.6961` | `0.6825` | `0.5986` |
| **Medial Meniscus** | 26 / 32 | `0.6178` | `0.6148` | `0.6587` | `0.6412` | `0.6268` | `0.6647` |
| **Lateral Meniscus** | 23 / 35 | `0.6118` | `0.5898` | `0.6286` | `0.5863` | `0.6043` | `0.6298` |
| **Medial OA** | 15 / 43 | `0.5341` | `0.5250` | `0.5566` | `0.5140` | `0.6085` | `0.5829` |
| **Lateral OA** | 11 / 47 | `0.5522` | `0.6422` | `0.6412` | `0.6141` | `0.6412` | `0.5803` |
| **PF OA** | 21 / 37 | `0.5817` | `0.6486` | `0.5946` | `0.5997` | `0.6075` | `0.6692` |
| **Effusion** | 35 / 23 | `0.6205` | `0.6646` | `0.6484` | `0.6621` | `0.6689` | `0.6683` |
| **Synovitis** | 27 / 31 | `0.6039` | `0.6074` | `0.6828` | `0.5830` | `0.6404` | `0.6589` |
| **Baker's** | 12 / 46 | `0.7346` | `0.7726` | `0.7264` | `0.7835` | `0.7264` | `0.7835` |
| **Contusion** | 19 / 39 | `0.6242` | `0.6205` | `0.5175` | `0.5574` | `0.5958` | `0.5074` |
| **Fracture** | 18 / 40 | `0.7410` | `0.7385` | `0.7007` | `0.6951` | `0.7340` | `0.6556` |

---

## 3. Individual Fold Checkpoint Performance on Gold-58 (Macro ROC-AUC)

| Model Identifier | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 5-Fold Rank Ensemble |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **D8A** | `0.6138` | `0.6218` | `0.6154` | `0.6346` | `0.6117` | **`0.6311`** |
| **D8B** | `0.6349` | `0.6167` | `0.6194` | `0.6180` | `0.6252` | **`0.6312`** |
| **D8C** | `0.6457` | `0.6454` | `0.6287` | `0.6343` | `0.6437` | **`0.6487`** |
| **D8D** | `0.6181` | `0.6338` | `0.6165` | `0.6055` | `0.6278` | **`0.6322`** |

---

## 4. Governance Sign-Off: Gold-58 Permanently Resealed

Following this one-time authorized benchmark:
1. **Zero Retraining or Tuning**: The Gold-58 results are frozen permanently. No hyperparameters, weights, or architectures will be tuned on these results.
2. **Quarantine Resealed**: Access to `csv/gold_studies.csv` and `phase_c/c1/processed_gold` is permanently locked.
3. **Artifact Integrity**: Checkpoint predictions and evaluation summaries have been hashed and recorded in `phase_d/d8/manifests/d8_gold58_sha256_manifest.txt`.
