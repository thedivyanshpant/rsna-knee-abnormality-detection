# PHASE R/B — AUTHORITATIVE DATASET & OOF PREDICTION INVENTORY

**Date**: September 16, 2026  
**Auditor**: Autonomous Agentic AI (Google DeepMind Antigravity)  
**Scope**: Complete inventory of frozen datasets, representations, and master OOF prediction assets on HPC.

---

## 1. Authoritative Development Cohort & Volumetric Assets

```text
Development Studies:       4,349 Non-Gold Studies (cv_splits.json, 5 Folds)
Total Series Recorded:     24,033 Series in csv/train_series.csv across 4,349 studies (Mean: 5.53 series/study)
Total Processed Volumes:   24,033 .npz 3D Tensors in phase_c/c1/processed/ (73.66 GB apparent / 144.1 GB Lustre)
Tensor Dimensions:         [32, 256, 256] at 3.0 x 0.5 x 0.5 mm spacing (float16 dtype)
Gold-58 Clinical Studies:  58 Permanently Isolated Studies (csv/gold_studies.csv, 336 volumes in processed_gold/)
Gold-58 Overlap in Dev:    EXACTLY ZERO (100% Verified Quarantine)
```

---

## 2. Representation Evolution: 5-Slot vs. 6-Slot Architecture

| Slot Index | Canonical Slot Name | Target Plane & Contrast | Phase C / D1..D5 Coverage | Phase D6..D8 Coverage | D6 Enhancement Mechanism |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Slot 0** | `SAG_FLUID_FS` | Sagittal Fluid-Sensitive Fat-Suppressed | ~98.4% | ~98.4% | Rule-based metadata classification |
| **Slot 1** | `COR_FLUID_FS` | Coronal Fluid-Sensitive Fat-Suppressed | ~98.2% | ~98.2% | Rule-based metadata classification |
| **Slot 2** | `AX_FLUID_FS` | Axial Fluid-Sensitive Fat-Suppressed | ~96.5% | ~96.5% | Rule-based metadata classification |
| **Slot 3** | `SAG_FLUID_NOFS`| Sagittal Fluid-Sensitive Non-Fat-Suppressed | **0.0% (0 studies)** | **19.6% (852 studies)** | **Deterministic intensity profile & fat-attenuation resolver** |
| **Slot 4** | `COR_T1` | Coronal T1 Non-Fat-Suppressed | ~97.1% | ~97.1% | Rule-based metadata classification |
| **Slot 5** | `SAG_T1` | Sagittal T1 Non-Fat-Suppressed | ~97.8% | ~97.8% | Rule-based metadata classification |

---

## 3. Master OOF Prediction Assets Catalog (15 Authoritative Models)

All 15 master out-of-fold prediction CSVs are verified on HPC. Each file contains predictions for exactly **4,349 studies** across all 12 pathology targets.

| Model / Experiment | Phase | Relative Path on HPC | Size | Prediction Type | Supervision Corpus | Architecture & Parameter Regime |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **D1.1 ViT-S Baseline** | D1.1 | `phase_d/d1/results/oof/d1_1_global_oof_predictions.csv` | 1.36 MB | Raw Probability | Phase B Production | DINOv2 ViT-S/14 (22.2M), 5 Slots |
| **D1.2-B2 Production** | D1.2 | `phase_d/d1/predictions/d1_2_b_hflip/d1_2_b_hflip_global_oof.csv` | 3.01 MB | Flip TTA + Rank Norm | Phase B Production | DINOv2 ViT-S/14 (22.2M), 5 Slots |
| **D2 ResNet-50** | D2 | `phase_d/d2/predictions/oof/d2_global_oof_predictions.csv` | 1.36 MB | Raw Probability | Phase B Production | RadImageNet ResNet-50 (23.5M), 5 Slots |
| **D3 Remediated ViT** | D3 | `phase_d/d3/predictions/d3_global_oof_predictions.csv` | 1.36 MB | Raw Probability | Phase D3 Remediated | DINOv2 ViT-S/14 (22.2M), 5 Slots |
| **D4 Multi-Node CNN** | D4 | `phase_d/d4/predictions/oof/d4_cnn_oof_predictions.csv` | 1.36 MB | Raw Probability | Phase D3 Remediated | RadImageNet ResNet-50 (23.5M), 5 Slots |
| **D5-B2 Calibrated** | D5 | `phase_d/d5/predictions/d5_b2_oof_predictions.csv` | 0.83 MB | Flip TTA + Rank Norm | Phase D3 Remediated | DINOv2 ViT-S/14 (22.2M), 5 Slots |
| **D6 Standalone ViT** | **D6** | `phase_d/d6/oof/d6_global_oof_predictions.csv` | 3.14 MB | Raw Probability | Phase D3 Remediated | **DINOv2 ViT-S/14 (22.2M), 6 Slots [DEV CHAMPION: 0.8185]** |
| **D7-A ViT-B Alone** | D7 | `phase_d/d7/oof/d7a_global_oof_predictions.csv` | 3.14 MB | Raw Probability | Phase D3 Remediated | DINOv2 ViT-B/14 (87.0M), 6 Slots |
| **D7-B Multi-Window** | D7 | `phase_d/d7/oof/d7b_global_oof_predictions.csv` | 3.14 MB | Raw Probability | Phase D3 Remediated | DINOv2 ViT-S/14 (22.4M), Multi-Window Depth |
| **D7-C ViT-B + Depth**| D7 | `phase_d/d7/oof/d7c_global_oof_predictions.csv` | 3.14 MB | Raw Probability | Phase D3 Remediated | DINOv2 ViT-B/14 (87.2M), Multi-Window Depth |
| **D7-D Full ViT-B** | D7 | `phase_d/d7/oof/d7d_global_oof_predictions.csv` | 3.14 MB | Raw Probability | Phase D3 Remediated | DINOv2 ViT-B/14 + DepthPosAttn (87.4M) |
| **D8-A Cross-Attn** | D8 | `phase_d/d8/oof/d8a_global_oof_predictions.csv` | 2.80 MB | Raw Probability | Phase D3 Remediated | ViT-S/14 + PathologyCrossAttn (22.6M) |
| **D8-B High-Budget** | D8 | `phase_d/d8/oof/d8b_global_oof_predictions.csv` | 3.14 MB | Raw Probability | Phase D3 Remediated | ViT-S/14 Dense Core 24 Groups (22.5M) |
| **D8-C CoAtNet Hybrid**| **D8** | `phase_d/d8/oof/d8c_global_oof_predictions.csv` | 3.14 MB | Raw Probability | Phase D3 Remediated | **CoAtNet Hybrid Conv+Attn (22.4M) [GOLD RECORD: 0.6487]** |
| **D8-D Full Combined** | D8 | `phase_d/d8/oof/d8d_global_oof_predictions.csv` | 2.80 MB | Raw Probability | Phase D3 Remediated | Hybrid + 24 Groups + CrossAttn (22.8M) |

---

## 4. Pathology Targets Reference (12 Clinical Targets)

All prediction CSVs and ground truth corpora consistently track the exact 12 RSNA knee abnormality targets:
1. `ACL` (Anterior Cruciate Ligament Tear)
2. `MCL` (Medial Collateral Ligament Tear)
3. `Medial Meniscus` (Medial Meniscus Tear)
4. `Lateral Meniscus` (Lateral Meniscus Tear)
5. `Medial OA` (Medial Compartment Osteoarthritis)
6. `Lateral OA` (Lateral Compartment Osteoarthritis)
7. `PF OA` (Patellofemoral Osteoarthritis)
8. `Effusion` (Joint Effusion)
9. `Synovitis` (Synovitis)
10. `Baker's Cyst` (Popliteal / Baker's Cyst)
11. `Bone Contusion` (Bone Marrow Edema / Contusion)
12. `Fracture` (Bone Fracture)
