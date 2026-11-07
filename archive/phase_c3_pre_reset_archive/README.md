# Phase C3: Downstream Multi-Series 3D Vision Modeling

This directory implements the production Phase C3 3D vision modeling pipeline for the RSNA Knee Abnormality Detection challenge on HPC (`rpgpu018`).

---

## 1. Directory Structure

```text
phase_c/c3/
├── README.md                 # This documentation
├── scripts/
│   ├── loss.py               # Exact confidence-weighted soft BCE loss formulation
│   ├── model.py              # MultiSeriesKneeModel (shared 3D ResNet encoder, masked slot pooling/attention)
│   ├── dataset.py            # Study-level MultiSeriesKneeDataset with 5 canonical sequence slots
│   └── split.py              # Pure-NumPy 5-fold stratified cross-validation splitter (seed 42)
├── configs/
│   └── cv_splits.json        # 5-fold disjoint CV splits across all 4,349 non-Gold development studies
├── tests/
│   └── test_c3_integrity.py  # Automated scientific data integrity gate
├── logs/                     # Training run logs
├── checkpoints/              # Model weights (.pt)
├── results/                  # Validation predictions, metrics, and ROC-AUC/F1 tables
└── manifests/                # Provenance and slot allocation logs
```

---

## 2. Canonical Sequence Slots & Model Architecture

### Five Canonical Sequence Slots:
1. **Slot 0**: `sagittal_fluid_sensitive` (Sagittal, Fluid-Sensitive / FS)
2. **Slot 1**: `coronal_fluid_sensitive` (Coronal, Fluid-Sensitive / FS)
3. **Slot 2**: `axial_fluid_sensitive` (Axial, Fluid-Sensitive / FS)
4. **Slot 3**: `sagittal_non_fs_t1` (Sagittal, Non-FS / T1)
5. **Slot 4**: `coronal_non_fs_t1` (Coronal, Non-FS / T1)

### Architecture Pipeline:
```text
Study Input: 5 sequence slots [B, 5, 1, 32, 256, 256] + slot_mask [B, 5]
      ↓
Shared 3D ResNet Encoder (SmallResNet3DEncoder: 2,052,188 parameters)
      ↓
5 Slot Feature Vectors [B, 5, 128] + Learned Slot Type Bias [1, 5, 128]
      ↓
Masked Slot Aggregation (Masked Pooling: 2,052,828 params | Masked Attention: 2,119,260 params)
      ↓
Study-Level Representation Vector [B, 128]
      ↓
Linear Prediction Head [128 → 12]
      ↓
Raw Unnormalized Logits [B, 12]
```

---

## 3. Label Integration & Loss Objective

* **Label Source**: [`labels/production/phase_b_step3_remediated_training_corpus.csv`](../../labels/production/phase_b_step3_remediated_training_corpus.csv) (52,884 rows = 4,407 studies $\times$ 12 targets).
* **Continuous Soft Target**: $Y \in [0.0, 1.0]$ (`label_probability`).
* **Sample Weight**: $W \ge 0.0$ (`sample_weight`, with $W=0.0$ for `abstain_flag == 1`).
* **Exact Loss**:
  $$\mathcal{L} = \frac{\sum_{m=1}^{12} W_m \cdot \text{BCEWithLogits}(z_m, Y_m)}{\sum_{m=1}^{12} W_m}$$

---

## 4. Stratified 5-Fold Cross-Validation

Generated deterministically via pure NumPy stratification based on positive pathology count per study (Seed `42`):
* **Fold 0**: Train = 3,477, Val = 872
* **Fold 1**: Train = 3,478, Val = 871
* **Fold 2**: Train = 3,479, Val = 870
* **Fold 3**: Train = 3,480, Val = 869
* **Fold 4**: Train = 3,482, Val = 867
* **Total Non-Gold Studies Validated**: Exactly 4,349 studies (0 overlap across validation folds).
* **Gold-58 Holdout Leakage**: Exactly 0.

---

## 5. Verification & Scientific Integrity Gate

Run the automated test suite:
```bash
python3 phase_c/c3/tests/test_c3_integrity.py
```
Asserts:
1. Gold-58 complete exclusion (0 leakage across all folds and directories).
2. Non-Gold development cohort coverage: exactly 4,349 unique studies.
3. Production label completeness: 52,188 targets, $Y \in [0, 1]$, $W \ge 0$.
4. Processed Pipeline C volumes: shape `(32, 256, 256)`, `float16`, finite, normalized in $[0, 1]$.
5. Model forward and backward gradient passes with weighted soft BCE.
6. 5-fold CV splits on-disk disjointness.
7. Series metadata presence check for deterministic slot classification.
- **Status**: **ALL 7 CHECKS PASSED (GREEN)**.

---

## 6. Execution Tracker & Milestone Gates

| Milestone | Gate Description | Status | Key Findings / Verification Artifacts |
| :--- | :--- | :---: | :--- |
| **C3.1–6** | Scientific Data Integrity Gate | **PASS** | 7/7 checks passed (`phase_c/c3/tests/test_c3_integrity.py`) |
| **C3.7** | Canonical Series Slot Mapping & Audit | **PASS** | 24,033 volumes joined, 0 anomalies, 4,118 studies (94.7%) with $\ge 4$ slots (`manifests/c3_canonical_slot_mapping_audit.json`) |
| **C3.8** | 8-Study Overfit Engineering Sanity | **PASS** | Loss dropped from 0.713 to 0.430 (gap to Shannon bound = 0.022, MAE = 0.0516) (`results/c3_stage8_overfit_audit.md`) |
| **C3.9** | Single-A100 Sanity Run | **PASS** | Peak VRAM 691 MB, 5.6 studies/s, checkpoint saving & resume verified (`results/c3_stage9_single_a100_audit.md`) |
| **C3.10** | 2-GPU DDP Sanity Test | **PASS** | 2 × A100 PCIe (NCCL), 11.0 studies/s, bitwise cross-rank parameter sync verified (`results/c3_stage10_ddp_audit.md`) |
| **C3.11** | Full 5-Fold Cross-Validation | **READY** | Pre-configured for launch via `torchrun --nproc_per_node=2` |
| **C3.12** | Locked Final Gold-58 Evaluation | **PLANNED** | Strictly isolated from training and threshold tuning |

