# Phase D5: D3 ViT Inference-Time Calibration & Test-Time Augmentation (TTA) — Comprehensive Development & Verification Report

---

## 1. Executive Summary

Phase D5 tested whether the inference-time post-processing and test-time augmentation (TTA) strategies established during Phase D1.2 (specifically Horizontal-Flip TTA and Within-Fold Percentile Rank Normalization, Candidate `D1.2-B2`) transfer meaningfully to the substantially stronger, experimentally remediated Phase D3 Vision Transformer (DINOv2 ViT-S/14 + Anatomical SlotHead).

Phase D5 was executed under strict **inference-only governance**:
- **Zero model retraining or fine-tuning**: Exact frozen 5-fold Phase D3 checkpoints were evaluated.
- **Zero label modification**: Phase D3 Candidate Remediated Labels (`phase_d3_candidate_label_corpus.csv`) and Phase B Production Corpus (`phase_b_step3_remediated_training_corpus.csv`) remained 100% immutable.
- **Zero CV split changes**: Authoritative 4,349-study 5-fold split (`phase_c/c3/configs/cv_splits.json`) was strictly preserved.
- **Gold-58 permanently locked**: Zero Gold-58 studies were loaded, inspected, evaluated, or calibrated against (`gold_leakage = 0`).
- **SLURM Cluster Safety**: Zero `scancel` calls executed.

### Core Headline Findings:
1. **D3 Baseline Exact Reproduction (D5-A)**:
   - On Remediated D3 Ground Truth: Macro ROC-AUC = **`0.8135`** (Weighted = **`0.8204`**, PR-AUC = **`0.3080`**).
   - On Legacy Phase B Ground Truth: Macro ROC-AUC = **`0.8184`** (Weighted = **`0.8253`**, PR-AUC = **`0.2319`**).
   - Numerical difference vs saved D3 validation predictions: **`max = 0.000000`**, **`mean = 0.000000`** (100% bit-identical reproduction across all 4,349 studies).
2. **Horizontal-Flip TTA Alone (D5-B)**:
   - $P_{\text{TTA}} = 0.5 \cdot P_{\text{std}} + 0.5 \cdot P_{\text{flip}}$.
   - Remediated GT: Macro ROC-AUC = **`0.8143`** (+0.0008 / +0.10% gain, PR-AUC = `0.3086` / +0.20%).
   - Legacy GT: Macro ROC-AUC = **`0.8171`** (-0.0013 / -0.16% drop, PR-AUC = `0.2293` / -1.12%).
3. **Within-Fold Rank Normalization Alone (D5-C)**:
   - $R_{\text{std}} = \text{rank}(P_{\text{std}}) / N_k$ per fold $k$.
   - Remediated GT: Macro ROC-AUC = **`0.8147`** (+0.0012 / +0.15% gain, PR-AUC = `0.3141` / +1.98%).
   - Legacy GT: Macro ROC-AUC = **`0.8188`** (+0.0004 / +0.05% gain, PR-AUC = `0.2334` / +0.65%).
4. **Primary D1.2-B2 Protocol Replication (D5-D)**:
   - $P_{\text{avg}} = 0.5 \cdot P_{\text{std}} + 0.5 \cdot P_{\text{flip}}$, followed by within-fold percentile rank normalization $R_{\text{avg}} = \text{rank}(P_{\text{avg}}) / N_k$.
   - Remediated GT: Macro ROC-AUC = **`0.8154`** (**`+0.0019`** / **`+0.23% relative gain`** vs D3 baseline `0.8135`), Weighted ROC-AUC = **`0.8226`** (+0.0022), Macro PR-AUC = **`0.3149`** (**`+0.0069`** / **`+2.24% relative gain`** vs D3 baseline `0.3080`).
   - Legacy GT: Macro ROC-AUC = **`0.8162`** (**`-0.0022`** / **`-0.27% relative drop`** vs D3 baseline `0.8184`), Weighted ROC-AUC = **`0.8234`** (-0.0019), Macro PR-AUC = **`0.2318`** (-0.0001).
5. **Scientific Verdict**:
   - **CASE B — MARGINAL TRANSFER**.
   - Unlike in Phase D1.2 (where D1.2-B2 delivered a substantial +0.0082 / +1.09% ROC-AUC gain from 0.7510 to 0.7592 on noisy supervision), on the clean, remediated D3 ViT representation the gain is small (+0.0019 / +0.23% on Remediated GT) and accompanied by a slight negative trade-off on the legacy label standard (-0.0022 / -0.27%).
   - Therefore, standard D3 ViT inference remains the primary, parsimonious production model, while D5-B2 represents an optional calibrated inference configuration if higher PR-AUC on remediated labels (+2.24%) is prioritized over legacy cross-standard consistency.

---

## 2. Scientific Question

During Phase D1.2, inference-time post-processing yielded notable performance improvements on the baseline DINOv2 ViT model trained under noisy Phase B supervision:
$$\text{D1.1 Baseline (0.7510)} \xrightarrow{\text{H-Flip TTA + RankNorm}} \text{D1.2-B2 (0.7592, +0.0082 / +1.09\%)}$$

In Phase D3, NLP label remediation recovered hundreds of previously omitted osteoarthritis cases, lifting standalone ViT performance to:
$$\text{Macro ROC-AUC} = 0.8135 \quad (\text{PR-AUC} = 0.3080)$$

The scientific question for Phase D5 is:
> **Do the inference-time post-processing techniques (horizontal-flip TTA and within-fold rank normalization) transfer to the substantially stronger, well-calibrated Phase D3 ViT checkpoints, or does the remediated ViT representation already incorporate spatial and cross-fold invariance?**

---

## 3. Frozen Phase D3 Baseline Reference

Authoritative Phase D3 Out-of-Fold (OOF) benchmark performance across the 4,349 development studies:

| Evaluation Ground Truth Standard | Macro ROC-AUC (Unweighted) | Macro ROC-AUC (Sample-Weighted) | Macro PR-AUC |
| :--- | :---: | :---: | :---: |
| **Phase D3 Candidate Remediated Labels** | **`0.8135`** | **`0.8204`** | **`0.3080`** |
| **Phase B Legacy Production Labels** | **`0.8184`** | **`0.8253`** | **`0.2319`** |

Per-fold baseline breakdown (Remediated GT):
- Fold 0: 0.8280 (Weighted: 0.8332)
- Fold 1: 0.7914 (Weighted: 0.7960)
- Fold 2: 0.8071 (Weighted: 0.8138)
- Fold 3: 0.8241 (Weighted: 0.8378)
- Fold 4: 0.8296 (Weighted: 0.8365)

---

## 4. Input Artifact Provenance & Verification

Every input file used in Phase D5 was verified on disk with size, timestamp, and SHA256 checksum:

| Artifact Description | File Path | Size (Bytes) | SHA256 Checksum |
| :--- | :--- | :---: | :--- |
| **D3 Checkpoint Fold 0** | `phase_d/d3/checkpoints/fold_0/best_model.pt` | 176,111,451 | `52eaf45a0934c6d86a32a34f0fa8ef1fac2a8713363b484c7c735bb1b186d1cc` |
| **D3 Checkpoint Fold 1** | `phase_d/d3/checkpoints/fold_1/best_model.pt` | 176,111,451 | `f600d29f8b0cbfb9d0c32ba9e3f736b0289e15725a3eaf45d447d19a2211ef5b` |
| **D3 Checkpoint Fold 2** | `phase_d/d3/checkpoints/fold_2/best_model.pt` | 176,111,451 | `cf1cdea350850aea17a878220e7a5cf12dd271b6deff9e8437dec84669f0914c` |
| **D3 Checkpoint Fold 3** | `phase_d/d3/checkpoints/fold_3/best_model.pt` | 176,111,451 | `b48ce28a0442723565795739eed3b0fe51e9d79b925c95eb51ecbc9feebc6cff` |
| **D3 Checkpoint Fold 4** | `phase_d/d3/checkpoints/fold_4/best_model.pt` | 176,111,451 | `9d29143c6d6f43bf6f4745044ec4997bc696bb95a218492e9ac209f2a068ce3b` |
| **D3 Remediated Labels** | `phase_d/d3/labels/phase_d3_candidate_label_corpus.csv` | 23,191,014 | `a90cd5c0e44da3256ecad118a0e69ad52aa38a1193b3cb35a084657323958ac1` |
| **D3 Candidate Manifest** | `phase_d/d3/labels/phase_d3_candidate_manifest.json` | 1,486 | `05134744f93c2f9332285a4d3fbc8a18833efba5894b37c2c06830da1ae02041` |
| **Legacy Phase B Labels** | `labels/production/phase_b_step3_remediated_training_corpus.csv` | 23,787,696 | `bd064fc882ee9c344fc46adedaef2e592071ab032a632bcba9e2fc699a741c30` |
| **Frozen 5-Fold CV Splits** | `phase_c/c3/configs/cv_splits.json` | 1,653,606 | `648ab88d9bb2484ed49dcfae826cfb51efdc7b8f56bdf218803ee46f1d451db8` |
| **Slot Mapping Manifest** | `phase_d/d1/manifests/d1_slot_mapping_manifest.csv` | 1,671,697 | `3e314c84200cf3b0b7023ca17f9c4e76c353aef00f69fcfe4ac1aad1a56ea883` |
| **Model Architecture Code** | `phase_d/d1/scripts/model_d1.py` | 9,727 | `d12985d6f6ed92cd3a26baf34813deba174ad7e72db25cbe21f7ff18d9f05742` |
| **Dataset Loader Code** | `phase_d/d1/scripts/dataset_d1.py` | 10,616 | `c0159486e1cefe4ec06d519426847b2da0c5ac5564d2e842777d07c449721b7a` |
| **Evaluation Utilities** | `phase_d/d1/scripts/d1_2_eval_utils.py` | 8,345 | `8ce5c881141e324f9664b5eb65d0b52c532aa29d3e5dce1e64b122ddbc55a300` |

---

## 5. Experimental Protocol & Mathematical Definitions

All experiments operated on the frozen validation cohort of each fold without data shuffling, retraining, or leakage.

### Stream 1: Standard Prediction ($P_{\text{std}}$)
For each input volume tensor $\mathbf{X} \in \mathbb{R}^{6 \times 12 \times 3 \times 224 \times 224}$ and slot mask $\mathbf{m} \in \{0, 1\}^6$:
$$\mathbf{z}_{\text{std}} = \text{Model}(\mathbf{X}, \mathbf{m}), \quad P_{\text{std}} = \sigma(\mathbf{z}_{\text{std}})$$

### Stream 2: Horizontal-Flip TTA ($P_{\text{flip}}$)
Spatial reflection along the width axis (dimension -1):
$$\mathbf{X}_{\text{flip}} = \text{torch.flip}(\mathbf{X}, \text{dims}=[-1])$$
$$\mathbf{z}_{\text{flip}} = \text{Model}(\mathbf{X}_{\text{flip}}, \mathbf{m}), \quad P_{\text{flip}} = \sigma(\mathbf{z}_{\text{flip}})$$

### Experiment D5-A: Standard Baseline
$$P_{\text{D5-A}} = P_{\text{std}}$$

### Experiment D5-B: Arithmetic Probability Mean TTA
$$P_{\text{D5-B}} = 0.5 \cdot P_{\text{std}} + 0.5 \cdot P_{\text{flip}}$$
No rank transform is applied before or after averaging.

### Experiment D5-C: Standard Prediction + Within-Fold Rank Normalization
For each fold $k \in \{0, 1, 2, 3, 4\}$ with $N_k$ studies, and for each pathology target $c \in \{0, \dots, 11\}$:
$$R_{\text{std}}(i, c) = \frac{\text{rank}(P_{\text{std}}[:, c])_i}{N_k}, \quad i \in \text{Fold}_k$$
Folds are never pooled or mixed during rank normalization.

### Experiment D5-D: Primary D1.2-B2 Replication
Combines probability-space horizontal-flip averaging with within-fold percentile rank normalization:
1. Compute probability average: $P_{\text{avg}} = 0.5 \cdot P_{\text{std}} + 0.5 \cdot P_{\text{flip}}$
2. Compute within-fold percentile rank:
$$R_{\text{avg}}(i, c) = \frac{\text{rank}(P_{\text{avg}}[:, c])_i}{N_k}, \quad i \in \text{Fold}_k$$

### Experiment D5-D_alt: Within-Fold Equal Rank Average
Computes within-fold ranks first, then averages ranks:
$$R_{\text{alt}}(i, c) = 0.5 \cdot \frac{\text{rank}(P_{\text{std}}[:, c])_i}{N_k} + 0.5 \cdot \frac{\text{rank}(P_{\text{flip}}[:, c])_i}{N_k}$$

### Diagnostic Stream: Horizontal Flip Alone
$$P_{\text{diag}} = P_{\text{flip}}$$

---

## 6. Global Evaluation Matrix

The table below presents the authoritative evaluation results across all tested inference configurations evaluated against both ground-truth standards:

| Experiment ID | Configuration Description | Remediated ROC-AUC | Remediated Weighted AUC | Remediated PR-AUC | Legacy ROC-AUC | Legacy Weighted AUC | Legacy PR-AUC | Δ vs D3 (Remediated) | Δ vs D3 (Legacy) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D5-A_STANDARD** | Standard D3 ViT (Baseline) | **`0.8135`** | **`0.8204`** | **`0.3080`** | **`0.8184`** | **`0.8253`** | **`0.2319`** | `+0.0000` | `+0.0000` |
| **D5-B_FLIP_TTA** | 50/50 Prob Mean ($0.5 \cdot P_{\text{std}} + 0.5 \cdot P_{\text{flip}}$) | **`0.8143`** | **`0.8213`** | **`0.3086`** | **`0.8171`** | **`0.8241`** | **`0.2293`** | `+0.0008` | `-0.0013` |
| **D5-C_RANKNORM** | Within-Fold RankNorm($P_{\text{std}}$) | **`0.8147`** | **`0.8219`** | **`0.3141`** | **`0.8188`** | **`0.8261`** | **`0.2334`** | `+0.0012` | `+0.0004` |
| **D5-D_B2_PRIMARY** | RankNorm($0.5 \cdot P_{\text{std}} + 0.5 \cdot P_{\text{flip}}$) | **`0.8154`** | **`0.8226`** | **`0.3149`** | **`0.8162`** | **`0.8234`** | **`0.2318`** | **`+0.0019`** | **`-0.0022`** |
| **D5-D_B2_ALT** | Rank Avg ($0.5 \cdot R_{\text{std}} + 0.5 \cdot R_{\text{flip}}$) | **`0.8149`** | **`0.8221`** | **`0.3149`** | **`0.8160`** | **`0.8233`** | **`0.2313`** | `+0.0014` | `-0.0024` |
| **D5-DIAG_FLIP** | Horizontal Flip Alone (Diagnostic) | **`0.8102`** | **`0.8171`** | **`0.3021`** | **`0.8108`** | **`0.8178`** | **`0.2229`** | `-0.0033` | `-0.0076` |

---

## 7. Per-Target Detailed Breakdown (Primary D5-B2 vs D3 Baseline)

Evaluated against Phase D3 Remediated Ground Truth across all 4,349 studies:

| Pathology Target | Positive Cohort | D3 Baseline ROC-AUC | D5-B2 Primary ROC-AUC | Δ ROC-AUC | D3 Baseline PR-AUC | D5-B2 Primary PR-AUC | Δ PR-AUC | Target Direction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 137 | 0.7842 | 0.7807 | -0.0035 | 0.1440 | 0.1469 | +0.0029 | Neutral / Minor Drop |
| **MCL** | 109 | 0.6177 | 0.6295 | **+0.0118** | 0.0506 | 0.0531 | +0.0025 | **Improved** |
| **Medial Meniscus** | 206 | 0.8260 | 0.8293 | +0.0033 | 0.2407 | 0.2421 | +0.0014 | **Improved** |
| **Lateral Meniscus** | 99 | 0.7471 | 0.7542 | **+0.0071** | 0.1001 | 0.0895 | -0.0106 | Mixed (AUC+, PR-) |
| **Medial OA** | 123 | 0.8445 | 0.8435 | -0.0010 | 0.2545 | 0.2763 | **+0.0218** | Mixed (AUC~, PR+) |
| **Lateral OA** | 78 | 0.8330 | 0.8213 | -0.0117 | 0.1772 | 0.2036 | **+0.0264** | Mixed (AUC-, PR+) |
| **PF OA** | 292 | 0.9536 | 0.9540 | +0.0004 | 0.5914 | 0.6009 | **+0.0095** | **Improved** |
| **Effusion** | 489 | 0.9034 | 0.9037 | +0.0003 | 0.7219 | 0.7172 | -0.0047 | Stable |
| **Synovitis** | 413 | 0.8791 | 0.8823 | +0.0032 | 0.5846 | 0.5858 | +0.0012 | **Improved** |
| **Baker's Cyst** | 385 | 0.8394 | 0.8445 | **+0.0051** | 0.4064 | 0.4288 | **+0.0224** | **Improved** |
| **Contusion** | 387 | 0.7853 | 0.7853 | +0.0000 | 0.3135 | 0.3047 | -0.0088 | Stable |
| **Fracture** | 101 | 0.7487 | 0.7564 | **+0.0077** | 0.1113 | 0.1294 | **+0.0181** | **Improved** |
| **Macro Average** | — | **0.8135** | **0.8154** | **+0.0019** | **0.3080** | **0.3149** | **+0.0069** | **Net Positive** |

### Per-Target Summary:
- **ROC-AUC Direction**: 7 improved, 2 stable ($\pm 0.0004$), 3 decreased (ACL -0.0035, Medial OA -0.0010, Lateral OA -0.0117).
- **PR-AUC Direction**: 9 improved, 3 decreased (Lateral Meniscus -0.0106, Effusion -0.0047, Contusion -0.0088).
- **Largest AUC Gain**: MCL (+0.0118), Fracture (+0.0077), Lateral Meniscus (+0.0071), Baker's (+0.0051).
- **Largest PR-AUC Gain**: Lateral OA (+0.0264), Baker's (+0.0224), Medial OA (+0.0218), Fracture (+0.0181).

---

## 8. Per-Fold Performance Breakdown (Remediated Ground Truth)

| Fold Index | Val Study Count | D3 Baseline AUC | D5-B (Flip TTA) | D5-C (RankNorm) | D5-D (Primary D5-B2) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 872 | 0.8280 | 0.8291 | 0.8286 | **0.8296** (+0.0016) |
| **Fold 1** | 871 | 0.7914 | 0.7917 | 0.7891 | **0.7892** (-0.0022) |
| **Fold 2** | 870 | 0.8071 | 0.8074 | 0.8051 | **0.8052** (-0.0019) |
| **Fold 3** | 869 | 0.8241 | 0.8252 | 0.8255 | **0.8263** (+0.0022) |
| **Fold 4** | 867 | 0.8296 | 0.8306 | 0.8327 | **0.8334** (+0.0038) |
| **Global OOF**| **4,349** | **0.8135** | **0.8143** | **0.8147** | **0.8154** (+0.0019) |

---

## 9. Comprehensive Governance & Integrity Audit

The governance audit verified strict adherence to all operational constraints:
```text
[x] Exactly 4,349 unique OOF studies evaluated
[x] Zero missing studies
[x] Zero duplicate studies
[x] Zero Gold-58 studies loaded, evaluated, or touched (gold_leakage = 0)
[x] Zero NaN values across all prediction matrices
[x] Zero Inf values across all prediction matrices
[x] All 12 canonical pathology targets present
[x] Deterministic StudyInstanceUID ordering matching Phase D template
[x] Frozen 5-fold CV splits preserved without modification
[x] D3 checkpoints byte-for-byte unmodified
[x] D3 candidate label corpus byte-for-byte unmodified
[x] Phase B production corpus byte-for-byte unmodified
[x] Zero model retraining or fine-tuning executed
[x] Zero scancel commands executed
```

---

## 10. Computational Execution Profile

- **Cluster Hostname**: `rpgpu007` (Head node of SLURM Job `653724`)
- **Hardware**: 2 × NVIDIA A100-PCIE-80GB GPUs
- **Execution Topology**: 2 concurrent worker processes via Python multiprocessing
  - Worker 0: GPU 0 (Folds 0, 1)
  - Worker 1: GPU 1 (Folds 2, 3, 4)
- **Batch Size**: 4 studies per GPU with 4 DataLoader workers per GPU
- **Precision**: BFloat16 forward inference with explicit float32 NumPy probability casting
- **Inference Wallclock Time**: **298.6 seconds (~4.98 minutes)** across all 5 folds (8,698 forward passes total: 4,349 standard + 4,349 flipped)
- **Total Execution & Evaluation Time**: **301.8 seconds (5.03 minutes)**

---

## 11. Scientific Interpretation & Cross-Phase Analysis

### 11.1 Comparison with Phase D1.2 Behavior
In Phase D1.2, the base ViT model had been trained on noisy Phase B weak labels. The baseline had substantial inter-fold score calibration drift and imperfect directional representations:
- D1.1 Baseline: `0.7510`
- D1.2-B1 (Flip TTA Prob Mean): `0.7521` (+0.0011)
- D1.2-B2 (Rank-Normalized TTA): `0.7592` (**+0.0082** / **+1.09% gain**)

In Phase D5, the model had been trained on clean, remediated D3 labels:
- D3 Baseline: `0.8135`
- D5-B (Flip TTA Prob Mean): `0.8143` (+0.0008)
- D5-C (RankNorm Alone): `0.8147` (+0.0012)
- D5-D (D5-B2 Primary): `0.8154` (**+0.0019** / **+0.23% gain**)

### 11.2 Why Did the TTA / Rank Calibration Gain Contract from +1.09% to +0.23%?
1. **Remediated Representations Are Already Well-Calibrated**:
   Phase D3 supervision provided dense, clinically coherent positive annotations (e.g. 292 patellofemoral OA cases vs 1 legacy case). The resulting ViT features learned tight, well-separated cluster margins, reducing inter-fold calibration drift. Percentile ranking within folds therefore provides only a marginal adjustment (+0.0012) rather than the massive recalibration observed in D1.2 (+0.0071).
2. **Anatomical Symmetry vs Asymmetry**:
   Horizontal flipping in knee MRI mirrors medial and lateral compartments (e.g. lateral meniscus flips to medial position in pixel coordinates). While self-attention across 2.5D slice groups and inductive slot priors can tolerate spatial mirroring for diffuse fluid abnormalities (effusion, synovitis, Baker's cyst, fracture), it introduces subtle positional tension for compartment-specific wear (Medial OA dropped -0.0010, Lateral OA dropped -0.0117). However, PR-AUC improved significantly on these targets (+0.0218 on Medial OA, +0.0264 on Lateral OA), indicating that TTA acts as a regularizer on extreme confidence predictions.
3. **Cross-Standard Trade-Off**:
   On Remediated Ground Truth, D5-B2 achieves a net positive gain on both ROC-AUC (+0.0019) and PR-AUC (+0.0069 / +2.24%). However, when evaluated against the legacy Phase B production standard, D5-B2 drops slightly from `0.8184` to `0.8162` (-0.0022). Standard D3 inference exhibits superior cross-standard stability.

---

## 12. Final Decision Logic & Protocol Decision

Following Section 24 of the governing Phase D5 specification:
- **CASE A (Clear Transfer)**: Rejected. The ROC-AUC gain is +0.0019 (+0.23%), which is below the threshold for an unambiguous, transformative leap, and exhibits a -0.0022 drop on legacy labels.
- **CASE B (Marginal Transfer)**: **CONFIRMED AND ADOPTED**.
  - Small positive improvement on the primary development standard (+0.0019 ROC-AUC, +0.0069 PR-AUC).
  - Accompanied by minor cross-standard tradeoff on legacy labels (-0.0022).
  - Test-time inference compute overhead: 2× forward passes required for horizontal flip.

### Official Phase D5 Decision:
1. **Primary Recommendation**: **Retain Standard D3 ViT Inference (`D5-A`) as the authoritative, parsimonious baseline model.** Standard inference achieves `0.8135` Remediated AUC / `0.8184` Legacy AUC with minimal computational overhead and zero cross-standard degradation.
2. **Secondary Calibrated Candidate**: **Freeze `D5-B2` (`d5_b2_oof_predictions.csv`, SHA256: `30c774a28be316bb291e7964d81da83bd61904adb219d42166a1687a1d3dd41e`)** as an officially audited post-processing configuration for downstream applications where maximizing Precision-Recall AUC on remediated pathology definitions (+2.24% PR-AUC gain) is desired.

---

## 13. Limitations

1. **Test-Time Augmentation Scope**: Multi-window slice variation (evaluating anterior/posterior slice offset bands) was not included in D5-B2, as D1.2 previously established that slice-window shifts degraded performance relative to simple horizontal flipping.
2. **Deterministic Flips**: Only bilateral horizontal flipping was tested; vertical flipping is biologically invalid for weight-bearing knee MRI.
3. **Absence of External Calibration Data**: All calibration and rank transformations were derived strictly out-of-fold across the 4,349-study development cohort.

---

## 14. Gold-58 Permanent Lock Declaration

> **Gold-58 Status**: Gold-58 remains permanently locked. Zero Gold-58 images, predictions, labels, metrics, or calibration signals were accessed, loaded, evaluated, or tuned during Phase D5. The locked historical Gold-58 benchmark remains fixed at **`0.6174`**.

---

## 15. Hard-Stop Declaration

**PHASE D5 COMPLETE — HARD STOP. AWAITING EXPLICIT AUTHORIZATION FOR ANY SUBSEQUENT PHASE.**
