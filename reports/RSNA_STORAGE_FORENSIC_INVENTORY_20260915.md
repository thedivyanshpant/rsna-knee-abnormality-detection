# STORAGE FORENSIC INVENTORY REPORT — RSNA KNEE ABNORMALITY DETECTION

**Date**: September 15, 2026  
**Auditor**: Autonomous Agentic AI (Google DeepMind Antigravity)  
**Host Filesystem**: Lustre Distributed Storage (`/scratch`, `/home`)  
**Scope**: Read-Only Forensic Audit of `/scratch/deepanthpcu` and `/scratch/deepanthpcu/RSNA_Code`  
**Execution Directives Observed**: Strictly 0 deletions, 0 moves, 0 compressions, 0 SLURM job cancellations (`653724` intact), Gold-58 quarantined.

---

## 1. Executive Storage Summary & Quota Status

```text
/scratch Quota:   175.7 GB used / 190.0 GB soft / 200.0 GB hard limit (87.85% utilized)
/scratch Free:    14.3 GB to soft quota | 24.3 GB to hard limit
Total Files:      29,900 files across user account

RSNA_Code Usage:  95.0 GB apparent size / 175.7 GB physical Lustre block allocation
  ├── phase_c:    74.98 GB apparent / 145.0 GB Lustre block allocation (82.5% of quota)
  ├── phase_d:    19.07 GB apparent /  31.1 GB Lustre block allocation (17.7% of quota)
  └── other:       0.95 GB apparent /   0.6 GB Lustre block allocation ( 0.3% of quota)

/home Quota:      32.05 GB used / 48.0 GB soft / 50.0 GB hard limit (64.1% utilized)
  ├── marked_train_series.tar.gz: 3.23 GB (Raw Gold-58 DICOM Archive)
  └── ml_dft_preprocessed.tar.gz: 11.56 GB (DFT domain archive)
```

> [!IMPORTANT]
> **Key Forensic Discovery — Apparent vs. Block Allocation on Lustre**:
> The primary storage consumer in `/scratch` is `phase_c/c1/processed`. While its apparent file size is **73.66 GB** (79.09 GB decimal), Lustre's 1 MB stripe allocation and physical block overhead across 24,033 individual `.npz` files results in **144.1 GB of physical quota consumption**.
> Addressing this single dataset will instantly release **144.1 GB** (82.0% of user quota), dropping total `/scratch` usage down to **31.6 GB**.

---

## 2. Priority Identification: The ~72–74 GB Training Dataset

* **Status**: **FOUND & TRACED**
* **Identity**: Phase C1 Master Multi-Series 3D Volumetric Tensor Corpus
* **Exact Path**: `/scratch/deepanthpcu/RSNA_Code/phase_c/c1/processed`
* **Apparent Size**: **`73.66 GiB`** (`79.09 GB decimal`, `79,091,732,842 bytes`)
* **Lustre Block Allocation**: **`144.1 GiB`** (`150,994,944 KB`)
* **Study Count**: **4,349 studies** (Exact 1:1 match with frozen development cohort in `cv_splits.json`)
* **Series / File Count**: **24,033 files** (all `.npz` format, float16 tensors, shape `[32, 256, 256]`, voxel spacing `3.0 x 0.5 x 0.5 mm`)
* **Gold-58 Contamination Check**: **0 Gold studies present** (100% verified quarantine; Gold-58 studies are segregated in `phase_c/c1/processed_gold`).
* **Reconstructibility**: **100% Reconstructible** from the workstation's raw DICOM corpus (~523 GB) via [`phase_c/c1/scripts/run_c1_preprocessing.py`](file:///scratch/deepanthpcu/RSNA_Code/phase_c/c1/scripts/run_c1_preprocessing.py).
* **Necessity for Current Inference**: **NO**. Phase D development (D1..D8) and the locked Gold-58 benchmark are complete. Checkpoints and predictions are frozen.
* **Necessity for Future R/B Audit**: **NO**. Representation analysis can operate on cached model embeddings, existing prediction matrices, or can be run on the workstation.
* **Workstation Status**: **SAFE TO LEAVE ON WORKSTATION**. The workstation already houses the raw DICOMs and can retain the processed `.npz` copy.

---

## 3. Project Root Breakdown: Level 1 & Level 2

### Level 1: Top-Level Directory Inventory (`/scratch/deepanthpcu/RSNA_Code/`)

| Directory / File | Type | Apparent Size | Lustre Block Size | File Count | Regular Files | Symlinks | Project Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`phase_c`** | DIR | `74.98 GB` | **`145.0 GB`** | 24,489 | 24,489 | 0 | Phase C 3D preprocessing & ResNet-18 modeling |
| **`phase_d`** | DIR | `19.07 GB` | **`31.1 GB`** | 492 | 492 | 0 | Phase D deep learning evolution suite (D1..D8) |
| **`archive`** | DIR | `0.08 GB` | `0.15 GB` | 84 | 84 | 0 | Archived prototypes, LLM logs, and pre-reset C3 |
| **`labels`** | DIR | `0.03 GB` | `0.05 GB` | 7 | 7 | 0 | Authoritative NLP label sets (Phase A, B, Gold) |
| **`csv`** | DIR | `0.01 GB` | `0.02 GB` | 9 | 9 | 0 | Metadata tables (`train_series.csv`, `gold_studies.csv`) |
| **`results`** | DIR | `0.001 GB` | `0.003 GB` | 23 | 23 | 0 | Acceptance audit logs and phase transition records |
| **`src`** | DIR | `0.0003 GB` | `0.0005 GB` | 33 | 33 | 0 | Core Python pipeline source code |
| **`docs`** | DIR | `0.0002 GB` | `0.0003 GB` | 10 | 10 | 0 | Technical documentation markdown specs |
| **`docs_backup_...`** | DIR | `0.0001 GB` | `0.0001 GB` | 8 | 8 | 0 | Ad-hoc documentation backup |
| **`validation`** | DIR | `0.0001 GB` | `0.0002 GB` | 9 | 9 | 0 | Data validation scripts and checksum runners |
| **`manifests`** | DIR | `0.0001 GB` | `0.0001 GB` | 3 | 3 | 0 | Global project SHA256 manifests |
| **`README.md`** | FILE | `48 KB` | `48 KB` | 1 | 1 | 0 | Project overview and reproduction guidelines |

---

### Level 2: Recursive Breakdown of Major Consumers (> 1.0 GB)

#### A. `phase_c/` Breakdown (Total: 145.0 GB Lustre / 74.98 GB Apparent)
* **`phase_c/c1/processed`**: **144.1 GB Lustre** (73.66 GB apparent, 24,033 `.npz` files across 4,349 studies) — Primary training corpus.
* **`phase_c/c1/processed_gold`**: **1.1 GB Lustre** (1.02 GB apparent, 336 `.npz` files across 58 Gold studies) — **PROTECTED GOLD DATA**.
* **`phase_c/c3/`**: **390 MB Lustre** (196 MB apparent, 8 ResNet-18 checkpoints [188.6 MB], 22 CSVs, 13 JSONs).
* **`phase_c/c2/`**: **190 MB Lustre** (95 MB apparent, 4 checkpoints [94.3 MB]).
* **`phase_c/c0/`**: **19 MB Lustre** (9.2 MB apparent, 15 files including 17 MB visual PNG samples).

#### B. `phase_d/` Breakdown (Total: 31.1 GB Lustre / 19.07 GB Apparent)
* **`phase_d/d7`**: **19.0 GB Lustre** (10.5 GB apparent, 117 files, 20 ViT checkpoints)
  * `d7a` (ViT-B capacity alone): **6.5 GB Lustre** (3.3 GB apparent, 5 checkpoints @ 660 MB each)
  * `d7c` (ViT-B multi-window): **5.2 GB Lustre** (3.3 GB apparent, 5 checkpoints @ 662 MB each)
  * `d7d` (ViT-B combination): **5.9 GB Lustre** (3.3 GB apparent, 5 checkpoints @ 665 MB each)
  * `d7b` (ViT-S multi-window): **1.0 GB Lustre** (849 MB apparent, 5 checkpoints @ 170 MB each)
* **`phase_d/d8`**: **3.6 GB Lustre** (3.4 GB apparent, 115 files, 20 checkpoints)
  * `d8a` (Cross-attention): **0.84 GB** (5 checkpoints @ 172 MB)
  * `d8b` (High-budget): **0.83 GB** (5 checkpoints @ 171 MB)
  * `d8c` (Hybrid CoAtNet Champion): **0.83 GB** (5 checkpoints @ 171 MB) — **PROJECT RECORD MODEL**
  * `d8d` (Full combination): **0.85 GB** (5 checkpoints @ 174 MB)
* **`phase_d/d4`**: **2.1 GB Lustre** (1.4 GB apparent, 5 ResNet-50 checkpoints @ 267 MB each)
* **`phase_d/d2`**: **1.9 GB Lustre** (1.4 GB apparent, 5 ResNet-50 checkpoints @ 266 MB each)
* **`phase_d/d1`**: **1.8 GB Lustre** (875 MB apparent, 5 ViT-S checkpoints @ 170 MB each)
* **`phase_d/d3`**: **1.7 GB Lustre** (865 MB apparent, 5 ViT-S checkpoints @ 170 MB each)
* **`phase_d/d6`**: **1.0 GB Lustre** (852 MB apparent, 5 ViT-S checkpoints @ 170 MB each) — **DEV CHAMPION MODEL**
* **`phase_d/d5`**: **8.5 MB Lustre** (5.9 MB apparent, inference-only calibration, 0 checkpoints)

---

## 4. Top 20 Largest Storage Consumers (Sorted Descending)

| Rank | Path | Type | Apparent Size | Lustre Block Size | Description / Project Role |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `/scratch/.../phase_c/c1/processed` | DIR | **`73.66 GB`** | **`144.1 GB`** | 4,349 development studies (24,033 `.npz` volumes) |
| 2 | `/scratch/.../phase_d/d7/d7d` | DIR | `3.24 GB` | **`5.9 GB`** | 5 fold checkpoints of ViT-B D7-D combination |
| 3 | `/scratch/.../phase_d/d7/d7a` | DIR | `3.22 GB` | **`6.5 GB`** | 5 fold checkpoints of ViT-B D7-A capacity |
| 4 | `/scratch/.../phase_d/d7/d7c` | DIR | `3.23 GB` | **`5.2 GB`** | 5 fold checkpoints of ViT-B D7-C multi-window |
| 5 | `/scratch/.../phase_d/d4/checkpoints` | DIR | `1.30 GB` | **`2.1 GB`** | 5 fold checkpoints of ResNet-50 D4 |
| 6 | `/scratch/.../phase_d/d2/checkpoints` | DIR | `1.30 GB` | **`1.9 GB`** | 5 fold checkpoints of ResNet-50 D2 |
| 7 | `/scratch/.../phase_c/c1/processed_gold` | DIR | **`1.02 GB`** | **`1.1 GB`** | **PROTECTED GOLD-58 3D VOLUMES (336 .npz)** |
| 8 | `/scratch/.../phase_d/d7/d7b` | DIR | `0.83 GB` | **`1.0 GB`** | 5 fold checkpoints of ViT-S D7-B multi-window |
| 9 | `/scratch/.../phase_d/d6/checkpoints` | DIR | `0.82 GB` | **`1.0 GB`** | **DEVELOPMENT CHAMPION ViT-S CHECKPOINTS** |
| 10 | `/scratch/.../phase_d/d8/d8c` | DIR | `0.83 GB` | **`0.9 GB`** | **ALL-TIME GOLD RECORD CoAtNet CHECKPOINTS** |
| 11 | `/scratch/.../phase_d/d8/d8d` | DIR | `0.85 GB` | **`0.9 GB`** | 5 fold checkpoints of ViT-S D8-D combination |
| 12 | `/scratch/.../phase_d/d8/d8a` | DIR | `0.84 GB` | **`0.9 GB`** | 5 fold checkpoints of ViT-S D8-A cross-attn |
| 13 | `/scratch/.../phase_d/d8/d8b` | DIR | `0.83 GB` | **`0.9 GB`** | 5 fold checkpoints of ViT-S D8-B high-budget |
| 14 | `/scratch/.../phase_d/d1/checkpoints` | DIR | `0.82 GB` | **`1.8 GB`** | 5 fold checkpoints of ViT-S D1 baseline |
| 15 | `/scratch/.../phase_d/d3/checkpoints` | DIR | `0.82 GB` | **`1.7 GB`** | 5 fold checkpoints of ViT-S D3 remediated |
| 16 | `/scratch/.../phase_c/c3/checkpoints` | DIR | `0.18 GB` | **`0.38 GB`** | 8 checkpoints of SmallResNet3D |
| 17 | `/scratch/.../archive/phase_c3_pre_reset...` | DIR | `0.07 GB` | **`0.15 GB`** | Archived pre-reset Phase C3 artifacts |
| 18 | `/scratch/.../phase_c/c2/checkpoints` | DIR | `0.09 GB` | **`0.19 GB`** | 4 checkpoints of Phase C2 experiments |
| 19 | `/scratch/.../labels/gold58` | DIR | `1.26 MB` | `2.5 MB` | Gold-58 NLP label predictions |
| 20 | `/scratch/.../csv/Copy-of-train_series.csv`| FILE | `3.30 MB` | `3.3 MB` | Exact duplicate of canonical `train_series.csv` |

---

## 5. Gold-58 Absolute Protection Registry

Every Gold-58 asset is audited, cataloged, and permanently designated:  
**`PROTECTED — DO NOT TOUCH`**.

| Asset Path | Size | Nature | Role & Contents | Protection Status |
| :--- | :---: | :---: | :--- | :---: |
| `/home/deepanthpcu/marked_train_series.tar.gz` | `3.23 GB` | Raw DICOM | Full DICOM series for all 58 Gold studies | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_c/c1/processed_gold/` | `1.02 GB` | Processed | 336 `.npz` 3D tensors for 58 Gold studies | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../csv/gold_studies.csv` | `14.2 KB` | Authoritative | Ground truth clinical labels for 58 studies | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../csv/golden_reports.csv` | `83.5 KB` | Authoritative | Original Spanish radiology reports | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_c/c1/manifests/c1_gold58_manifest.csv` | `55.8 KB` | Manifest | Series processing records for Gold-58 | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_c/c1/output/c1_gold58_execution_summary.json` | `503 B` | Metadata | Execution audit for Gold-58 volume prep | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_c/c3/results/gold58/` | `0.12 MB` | Predictions | Phase C3 ResNet-18 Gold predictions & report | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_d/d1/predictions/gold58/` | `0.12 MB` | Predictions | Phase D1.2 ViT Gold predictions & report | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_d/d5/predictions/gold58/` | `0.21 MB` | Predictions | Phase D5-B ViT Gold predictions & report | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_d/d6/predictions/gold58/` | `0.21 MB` | Predictions | Phase D6-B ViT Gold predictions & report | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_d/d7/predictions/gold58/` | `0.08 MB` | Predictions | Phase D7 Gold benchmark predictions & report | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../phase_d/d8/predictions/gold58/` | `0.08 MB` | Predictions | Phase D8 Gold benchmark predictions (Record) | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../labels/gold58/` | `1.26 MB` | Labels | Phase A/B NLP predictions on Gold-58 | **PROTECTED — DO NOT TOUCH** |
| `/scratch/.../archive/llm_experiments/gold*.jsonl` | `0.90 MB` | NLP Audits | DeepSeek/Qwen/BioLLM test evaluations | **PROTECTED — DO NOT TOUCH** |

---

## 6. Frozen Authoritative Assets Verification

All core project ground truth files were hashed and verified bit-for-bit:

1. **Canonical `train_series.csv`**:
   * Path: `/scratch/deepanthpcu/RSNA_Code/csv/train_series.csv`
   * Size: `3,462,820 bytes` (3.30 MB)
   * Expected SHA256: `573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3`
   * Actual SHA256:   `573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3` — **VERIFIED MATCH**
2. **Frozen Cross-Validation Splits**:
   * Path: `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/configs/cv_splits.json`
   * Size: `1,677,955 bytes` (1.60 MB)
   * Expected SHA256: `648ab88d9bb2484ed49dcfae826cfb51efdc7b8f56bdf218803ee46f1d451db8`
   * Actual SHA256:   `648ab88d9bb2484ed49dcfae826cfb51efdc7b8f56bdf218803ee46f1d451db8` — **VERIFIED MATCH**
3. **Phase D3 Remediated Supervision Corpus**:
   * Path: `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/labels/phase_d3_candidate_label_corpus.csv`
   * Size: `8,767,882 bytes` (8.36 MB)
   * SHA256: `a90cd5c0e44da3256ecad118a0e69ad52aa38a1193b3cb35a084657323958ac1` — **RECORDED**
4. **Phase B Production Remediated Corpus**:
   * Path: `/scratch/deepanthpcu/RSNA_Code/labels/production/phase_b_step3_remediated_training_corpus.csv`
   * Size: `8,266,419 bytes` (7.88 MB)
   * SHA256: `bd064fc882ee9c344fc46adedaef2e592071ab032a632bcba9e2fc699a741c30` — **RECORDED**

---

## 7. Duplicate File Forensic Audit

1. **Exact Metadata Duplicate**:
   * File A: `/scratch/deepanthpcu/RSNA_Code/csv/Copy-of-train_series.csv` (3.30 MB)
   * File B: `/scratch/deepanthpcu/RSNA_Code/csv/train_series.csv` (3.30 MB)
   * SHA256: `573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3`
   * Rationale: Redundant manual copy. Safe to clean up.
2. **Intra-Study Repeated Series Volumes**:
   * 36 `.npz` files in `phase_c/c1/processed/` share identical hashes with another series file within the same study directory (accounting for 117.6 MB). These represent identical DICOM series acquired twice during the same patient exam.
3. **Documentation Backup Duplication**:
   * `docs_backup_20260914_1825/` (144 KB) is a duplicate snapshot of older technical reports in `docs/`.

---

## 8. Asset Classification Matrix (A–E)

* **Class A (Irreplaceable / Must Remain on HPC)**:
  * Gold-58 datasets, ground truths, and evaluation reports (`csv/gold_studies.csv`, `processed_gold`, all Gold prediction CSVs).
  * Frozen CV splits (`phase_c/c3/configs/cv_splits.json`).
  * Authoritative series metadata (`csv/train_series.csv`).
  * Supervision corpora (`phase_d/d3/...`, `phase_b_step3_remediated...`).
  * All study-level OOF predictions across D1..D8 (`phase_d/*/oof/*.csv`).
  * All-Time Gold Champion Checkpoints: `phase_d/d8/d8c/checkpoints/` (CoAtNet Hybrid).
  * Development Champion Checkpoints: `phase_d/d6/checkpoints/` (ViT-S 6-Slot).
* **Class B (Important but Archivable to Workstation or /home)**:
  * 70 non-champion model checkpoints across D1, D2, D3, D4, D7A/B/C/D, D8A/B/D, C2, C3 (**17.6 GB apparent / 29.5 GB Lustre blocks**).
* **Class C (Reconstructible from Workstation Raw DICOM)**:
  * Master training processed volumes: `phase_c/c1/processed` (**73.66 GB apparent / 144.1 GB Lustre blocks**).
* **Class D (Temporary / Intermediate / Inspection)**:
  * `phase_c/c0/visual_samples/` (17 MB PNG visualizations).
  * `archive/intermediate/` (1.8 MB temporary files).
  * `docs_backup_20260914_1825/` (144 KB).
* **Class E (Duplicate)**:
  * `csv/Copy-of-train_series.csv` (3.30 MB).

---

## 9. Future Phase R/B (Representation & Bottleneck Audit) Requirements

| Analysis Dimension | Existing HPC Data Sufficient? | Raw DICOM Required? | Scientific Technical Rationale |
| :--- | :---: | :---: | :--- |
| **D3 / Gold Label Agreement** | **YES** | **NO** | Label CSVs (`phase_d3_candidate_label_corpus.csv`, `gold_studies.csv`) exist on HPC. |
| **OOF Error Analysis** | **YES** | **NO** | All study-level prediction matrices and ground-truth vectors exist on HPC. |
| **D6 / D7 / D8 Model Disagreement** | **YES** | **NO** | All 4,349 OOF probability CSVs are preserved in `phase_d/*/oof/`. |
| **Sequence Slot Ablation** | **YES** | **NO** | Zeroing or masking slot embeddings can run on cached tensors or feature vectors. |
| **Slice-Density Analysis** | **YES** | **NO** | 3D `.npz` tensors (`32 x 256 x 256`) can be downsampled along slice depth on HPC. |
| **Resolution Analysis** | **PARTIALLY** | **YES** | Downsampling below 256x256 is supported on HPC; higher native resolution (512x512) requires raw DICOM. |
| **DICOM Metadata Audit** | **YES** | **NO** | `csv/train_series.csv` contains all extracted acquisition tags for all 24,033 series. |
| **Discarded-Series Audit** | **YES** | **NO** | Comparing `train_series.csv` (24,033) vs `c1_study_manifest.csv` (23,548) identifies all 485 skipped series. |
| **Original Acquisition Audit** | **PARTIALLY** | **YES** | Standard tags are in CSV; proprietary vendor shadow tags require raw DICOM headers. |
| **Full Preprocessing Audit** | **PARTIALLY** | **YES** | Verifying pixel resampling fidelity against source requires raw DICOM comparison. |

**Summary**: 7 of 10 R/B audits can be performed 100% on existing HPC data without any raw DICOM. The 3 remaining audits require raw DICOM on the workstation.

---

## 10. Storage Priority & Recovery Matrix

| Priority | Path | Apparent Size | Lustre Block Size | Purpose & Role | Action Recommendation |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **RED** | `phase_c/c1/processed_gold/` | `1.02 GB` | `1.1 GB` | Gold-58 evaluation tensors (336 `.npz`) | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `csv/gold_studies.csv` | `14 KB` | `16 KB` | Gold-58 clinical ground truth labels | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `csv/train_series.csv` | `3.3 MB` | `3.3 MB` | Canonical series metadata table | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `phase_c/c3/configs/cv_splits.json`| `1.6 MB` | `1.6 MB` | Frozen 5-fold cross-validation splits | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `phase_d/d3/labels/...` | `8.4 MB` | `8.4 MB` | Phase D3 candidate supervision corpus | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `phase_d/d8/d8c/checkpoints/` | `830 MB` | `900 MB` | **All-Time Gold Champion Checkpoints** | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `phase_d/d6/checkpoints/` | `820 MB` | `1.0 GB` | **Development Champion Checkpoints** | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `phase_d/*/oof/*.csv` | `100 MB` | `100 MB` | Authoritative OOF prediction CSVs | **MUST REMAIN ON HPC (PROTECTED)** |
| **RED** | `phase_d/*/predictions/gold58/` | `10 MB` | `10 MB` | Authoritative Gold-58 benchmark CSVs | **MUST REMAIN ON HPC (PROTECTED)** |
| **YELLOW**| `phase_c/c1/processed/` | **`73.66 GB`** | **`144.1 GB`** | Development training volumes (4,349 studies) | **Transfer/Verify on Workstation -> Safe Cleanup** |
| **YELLOW**| `phase_d/d7/` (d7a, d7b, d7c, d7d) | `10.5 GB` | `18.6 GB` | Non-champion Phase D7 ViT checkpoints (20 .pt) | **Archive to Workstation / User Approval** |
| **YELLOW**| `phase_d/d8/` (d8a, d8b, d8d) | `2.5 GB` | `2.7 GB` | Non-champion Phase D8 checkpoints (15 .pt) | **Archive to Workstation / User Approval** |
| **YELLOW**| `phase_d/d4/checkpoints/` | `1.3 GB` | `2.1 GB` | Phase D4 ResNet-50 checkpoints (5 .pt) | **Archive to Workstation / User Approval** |
| **YELLOW**| `phase_d/d2/checkpoints/` | `1.3 GB` | `1.9 GB` | Phase D2 ResNet-50 checkpoints (5 .pt) | **Archive to Workstation / User Approval** |
| **YELLOW**| `phase_d/d1/checkpoints/` | `0.82 GB` | `1.8 GB` | Phase D1 ViT-S checkpoints (5 .pt) | **Archive to Workstation / User Approval** |
| **YELLOW**| `phase_d/d3/checkpoints/` | `0.82 GB` | `1.7 GB` | Phase D3 ViT-S checkpoints (5 .pt) | **Archive to Workstation / User Approval** |
| **YELLOW**| `phase_c/c3/checkpoints/` | `0.18 GB` | `0.38 GB` | SmallResNet-3D checkpoints (8 .pt) | **Archive to Workstation / User Approval** |
| **GREEN** | `csv/Copy-of-train_series.csv` | `3.3 MB` | `3.3 MB` | Exact duplicate of train_series.csv | **Safe Cleanup after approval** |
| **GREEN** | `phase_c/c0/visual_samples/` | `8.4 MB` | `17 MB` | Temporary C0 inspection PNGs | **Safe Cleanup after approval** |
| **GREEN** | `docs_backup_20260914_1825/` | `140 KB` | `144 KB` | Outdated documentation snapshot | **Safe Cleanup after approval** |
| **GREEN** | `archive/intermediate/` | `1.8 MB` | `1.8 MB` | Temporary NLP intermediate files | **Safe Cleanup after approval** |

---

## 11. Exact Space Recovery Calculations

```text
1. Candidate 1 (Processed Training Corpus):
   Path: /scratch/deepanthpcu/RSNA_Code/phase_c/c1/processed
   Apparent Size:            73.66 GB
   Physical Quota Recovery: 144.10 GB
   Status: Safe after verifying workstation has copy

2. Candidate 2 (Archivable Non-Champion Checkpoints):
   Paths: D1, D2, D3, D4, D7A/B/C/D, D8A/B/D, C2, C3 checkpoints
   Apparent Size:            17.60 GB
   Physical Quota Recovery:  29.50 GB
   Status: Requires user approval to archive to workstation

3. Candidate 3 (Duplicates & Temporary Artifacts):
   Paths: Copy-of-train_series.csv, c0/visual_samples, docs_backup, intermediate
   Apparent Size:             0.02 GB
   Physical Quota Recovery:   0.03 GB
   Status: Safe cleanup after approval
-------------------------------------------------------------------------
MAXIMUM POTENTIAL STORAGE RECOVERY: 173.63 GB (Dropping usage to ~2.1 GB)
RECOMMENDED IMMEDIATE TARGET (Candidate 1 alone): 144.10 GB (Dropping usage to 31.6 GB)
```

---

## 12. Recommended Workstation Transfer Strategy (Proposals Only — Unexecuted)

1. **Step 1 — Workstation Verification**:
   * Verify whether `/scratch/.../phase_c/c1/processed` already exists on the workstation or can be regenerated from raw DICOMs using `phase_c/c1/scripts/run_c1_preprocessing.py`.
   * If not already on the workstation, transfer the 24,033 `.npz` files (73.66 GB) via parallel rsync/tar transfer to the workstation.
2. **Step 2 — Checkpoint Archival (Optional)**:
   * Bundle the 70 non-champion checkpoints (`tar -czf non_champion_checkpoints.tar.gz`) and transfer to workstation storage.
3. **Step 3 — Controlled HPC Purge (Only upon User Command)**:
   * Once MD5/SHA256 manifests confirm the workstation copy is 100% intact, purge `phase_c/c1/processed` on HPC.
   * Total HPC space freed: **`144.1 GB`** (leaving `/scratch` at **31.6 GB / 200 GB**, a healthy 15.8% utilization).
4. **Step 4 — Retain on HPC**:
   * Keep all Gold-58 assets, D8-C champion checkpoints, D6 champion checkpoints, all OOF prediction CSVs, code, manifests, and documentation permanently on HPC.
