# PHASE D7 — BACKBONE CAPACITY & MULTI-WINDOW DEPTH COVERAGE FACTORIAL REPORT

**Generated:** 2026-09-15 00:37:35 UTC  
**Status:** COMPLETED AND AUDITED  
**Cohort:** 4,349 Development Studies (5-Fold Stratified Cross-Validation)  
**Supervision:** Remediated Phase D3 Candidate Label Corpus (`phase_d3_candidate_label_corpus.csv`)  
**Gold-58 Status:** PERMANENTLY LOCKED & UNTOUCHED (Zero Exposure)  

---

## 1. EXECUTIVE SUMMARY & FACTORIAL RESULTS MATRIX

Phase D7 rigorously evaluated two independent architectural axes and their factorial interaction:
1. **Axis 1 (Backbone Capacity)**: Scaling pretrained foundation model from DINOv2 ViT-S/14 (~22.4M params) to ViT-B/14 (~86.6M params).
2. **Axis 2 (Multi-Window Depth Coverage)**: Expanding anatomical slice depth coverage from $62.5\%$ central band to $96.9\%$ volumetric coverage ($[0.08, 0.92]$ slice depth) via 3 overlapping anatomical depth windows.
3. **Interaction (Combined)**: Joint deployment of ViT-B/14 and Multi-Window Depth Coverage.

### Factorial Performance Matrix

| Experiment | Backbone | Depth Coverage | Slices Covered | Params (Trainable / Total) | Remediated Macro ROC-AUC | Remediated Weighted AUC | Remediated PR-AUC | 5-Fold CV Mean ± Std | Legacy Phase B AUC | $\Delta$ vs D6 Champion |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **D6 Baseline** | ViT-S/14 | Single Window (0.2, 0.8) | 62.5% (Slices 6–25) | 10.8M / 22.2M (48.6%) | **0.8185** | 0.8258 | 0.3045 | 0.8254 ± 0.0114 | 0.8248 | — |
| **D7-A** | ViT-B/14 | Single Window (0.2, 0.8) | 62.5% (Slices 6–25) | 42.9M / 87.0M (49.4%) | **0.8137** | 0.8212 | 0.2979 | 0.8232 ± 0.0134 | 0.8174 | **-0.0048** (-0.59%) |
| **D7-B** | ViT-S/14 | Multi-Window (3 windows) | 96.9% (Slices 1–31) | 11.0M / 22.4M (49.0%) | **0.8100** | 0.8159 | 0.2864 | 0.8198 ± 0.0132 | 0.8153 | **-0.0085** (-1.04%) |
| **D7-C** | ViT-B/14 | Multi-Window (3 windows) | 96.9% (Slices 1–31) | 43.1M / 87.2M (49.5%) | **0.8079** | 0.8129 | 0.2821 | 0.8177 ± 0.0115 | 0.8089 | **-0.0106** (-1.30%) |

**NEW DEVELOPMENT CHAMPION:** **D6 Champion** (**0.8185** Macro ROC-AUC, **0.3045** PR-AUC).

---

## 2. FORMAL ANSWERS TO THE THREE SCIENTIFIC HYPOTHESES

### Hypothesis 1: Backbone Capacity Impact (ViT-B vs ViT-S)
**ViT-B/14 Alone (D7-A)** achieved **0.8137** Macro ROC-AUC (-0.0048 vs D6 ViT-S 0.8185, -0.59% relative). PR-AUC changed by -0.0066 (0.2979 vs 0.3045).

### Hypothesis 2: Multi-Window Depth Coverage Impact
**Multi-Window Depth Coverage Alone (D7-B)** achieved **0.8100** Macro ROC-AUC (-0.0085 vs D6 0.8185, -1.04% relative). PR-AUC changed by -0.0181 (0.2864 vs 0.3045).

### Hypothesis 3: Complementarity of Capacity and Multi-Window Depth
**Combined ViT-B + Multi-Window (D7-C)** achieved **0.8079** Macro ROC-AUC (-0.0106 vs D6 0.8185, -1.30% relative). Comparison with individual components: D7-A (0.8137) and D7-B (0.8100).

---

## 3. PER-PATHOLOGY FACTORIAL BREAKDOWN (ALL 12 TARGETS)

| Pathology Target | Pos / Neg | D6 AUC | D7-A AUC | $\Delta$ D7-A | D7-B AUC | $\Delta$ D7-B | D7-C AUC | $\Delta$ D7-C |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 137 / 4212 | 0.7646 | 0.7654 | +0.0008 | 0.7642 | -0.0004 | 0.7514 | -0.0132 |
| **MCL** | 109 / 4240 | 0.6216 | 0.6248 | +0.0032 | 0.6380 | +0.0164 | 0.6312 | +0.0096 |
| **Medial Meniscus** | 206 / 4143 | 0.8292 | 0.8259 | -0.0033 | 0.8262 | -0.0030 | 0.8217 | -0.0075 |
| **Lateral Meniscus** | 99 / 4250 | 0.7780 | 0.7970 | +0.0190 | 0.7433 | -0.0347 | 0.7558 | -0.0222 |
| **Medial OA** | 123 / 4226 | 0.8621 | 0.8470 | -0.0151 | 0.8723 | +0.0102 | 0.8505 | -0.0116 |
| **Lateral OA** | 78 / 4271 | 0.8490 | 0.8978 | +0.0488 | 0.8407 | -0.0083 | 0.8782 | +0.0292 |
| **PF OA** | 292 / 4057 | 0.9455 | 0.9466 | +0.0011 | 0.9478 | +0.0023 | 0.9478 | +0.0023 |
| **Effusion** | 489 / 3860 | 0.9102 | 0.8976 | -0.0126 | 0.9030 | -0.0072 | 0.8982 | -0.0120 |
| **Synovitis** | 413 / 3936 | 0.8827 | 0.8653 | -0.0174 | 0.8635 | -0.0192 | 0.8669 | -0.0158 |
| **Baker's** | 385 / 3964 | 0.8567 | 0.8228 | -0.0339 | 0.8153 | -0.0414 | 0.8094 | -0.0473 |
| **Contusion** | 387 / 3962 | 0.7818 | 0.7731 | -0.0087 | 0.7690 | -0.0128 | 0.7758 | -0.0060 |
| **Fracture** | 101 / 4248 | 0.7403 | 0.7015 | -0.0388 | 0.7369 | -0.0034 | 0.7079 | -0.0324 |

---

## 4. RIGOROUS GOVERNANCE & AUDIT VERIFICATION

All 3 experiments satisfy 100% of the project's strict reproducibility and governance rules:
1. **Zero Gold-58 Leakage**: Exactly 0 Gold studies loaded or present in training or validation splits.
2. **Full Cohort Coverage**: Exactly 4,349 studies out-of-fold evaluated across all 5 folds.
3. **Zero Missing or Invalid Values**: Exactly 0 NaNs and 0 Infs across all prediction files.
4. **Hardware & Resource Efficiency**:
   - D7-A Peak VRAM: 8.36 GB / 80 GB per GPU.
   - D7-B Peak VRAM: 3.42 GB / 80 GB per GPU.
   - D7-C Peak VRAM: 8.41 GB / 80 GB per GPU.
   - Zero out-of-memory errors across all 15 trained fold models.
