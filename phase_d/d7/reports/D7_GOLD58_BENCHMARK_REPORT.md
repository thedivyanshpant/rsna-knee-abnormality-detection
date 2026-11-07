# PHASE D7 — LOCKED GOLD-58 BENCHMARK REPORT (D7-A, D7-B, D7-C, D7-D)

**Date**: 2026-09-15 07:35:08 UTC  
**Cohort**: 58 Permanently Isolated Clinical Studies (`csv/gold_studies.csv`)  
**Evaluation Pass**: ONE-TIME AUTHORIZED LOCKED BENCHMARK  

---

## 1. Executive Benchmark Matrix

| Model Identifier | Architecture | Volumetric Sampling | Gold Macro ROC-AUC | Gold PR-AUC | $\Delta$ vs D6 Gold (0.6331) | $\Delta$ vs D1.2-B2 (0.6174) | $\Delta$ vs Phase C (0.5986) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase C Baseline** | 3D ResNet-18 Control | Central Band | `0.5986` | `0.4526` | -0.0345 | -0.0188 | — |
| **Phase D1.2-B2** | ViT-S/14 Standalone | Central Band | `0.6174` | `0.4938` | -0.0157 | — | +0.0188 |
| **Phase D5-B2** | ViT-S/14 (D3 Repr) | Central Band | `0.6307` | `0.5285` | -0.0024 | +0.0133 | +0.0321 |
| **Phase D6-B2 Champion**| ViT-S/14 (6 Slots) | Central Band | **`0.6331`** | **`0.5125`** | — | **+0.0157** | **+0.0345** |
| **D7-A** | ViT-B/14 (87.0M) | Central Band | **`0.6362`** | `0.5249` | **`+0.0031`** | `+0.0188` | `+0.0376` |
| **D7-B** | ViT-S/14 (22.4M) | Multi-Window | **`0.6428`** | `0.5201` | **`+0.0097`** | `+0.0254` | `+0.0442` |
| **D7-C** | ViT-B/14 (87.2M) | Multi-Window | **`0.6474`** | `0.5290` | **`+0.0143`** | `+0.0300` | `+0.0488` |
| **D7-D** | ViT-B/14 + DepthAttn | Multi-Window | **`0.6208`** | `0.5138` | **`-0.0123`** | `+0.0034` | `+0.0222` |

---

## 2. Per-Pathology Clinical Breakdown on Gold-58 (Rank-Averaged AUC)

| Pathology Target | Pos / Neg | D6 Champion | D7-A | D7-B | D7-C | D7-D |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 24 / 34 | `0.6538` | `0.6311` | `0.6550` | `0.6471` | `0.6299` |
| **MCL** | 9 / 49 | `0.7211` | `0.7041` | `0.7302` | `0.7370` | `0.6905` |
| **Medial Meniscus** | 26 / 32 | `0.6178` | `0.6334` | `0.6298` | `0.6430` | `0.6190` |
| **Lateral Meniscus** | 23 / 35 | `0.6118` | `0.6031` | `0.5863` | `0.5863` | `0.5466` |
| **Medial OA** | 15 / 43 | `0.5341` | `0.5791` | `0.5202` | `0.5907` | `0.5357` |
| **Lateral OA** | 11 / 47 | `0.5522` | `0.6170` | `0.6422` | `0.6306` | `0.6199` |
| **PF OA** | 21 / 37 | `0.5817` | `0.6268` | `0.6030` | `0.6486` | `0.6107` |
| **Effusion** | 35 / 23 | `0.6205` | `0.6466` | `0.6199` | `0.6646` | `0.6385` |
| **Synovitis** | 27 / 31 | `0.6039` | `0.5597` | `0.6440` | `0.5962` | `0.5902` |
| **Baker's** | 12 / 46 | `0.7346` | `0.7563` | `0.7726` | `0.7509` | `0.6957` |
| **Contusion** | 19 / 39 | `0.6242` | `0.5405` | `0.5769` | `0.5445` | `0.5391` |
| **Fracture** | 18 / 40 | `0.7410` | `0.7361` | `0.7333` | `0.7292` | `0.7333` |

---

## 3. Governance Sign-Off: Gold-58 Access Sealed
Following this benchmark evaluation, access to Gold-58 is **PERMANENTLY RESEALED**.
No Gold data will be accessed, loaded, or evaluated during Phase D8.
