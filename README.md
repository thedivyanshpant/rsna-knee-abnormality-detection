# RSNA Knee Abnormality Detection — Multi-Task Deep Learning Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Challenge: RSNA 2024](https://img.shields.io/badge/Challenge-RSNA%20Knee-red.svg)](https://www.kaggle.com/)

---

## 1. Executive Summary & Clinical Mission

The **RSNA Knee Abnormality Detection Project** is an end-to-end, multi-task deep learning and clinical NLP platform engineered to detect **12 distinct musculoskeletal pathologies** simultaneously from volumetric multi-sequence knee Magnetic Resonance Imaging (MRI) studies.

Clinical knee MRI protocols involve heterogeneous acquisition planes (Sagittal, Coronal, Axial) and tissue weightings (Fluid-Sensitive Fat-Suppressed, Non-Fat-Suppressed, T1-Weighted). This repository solves the dual challenges of noisy unstructured weak supervision and multi-sequence vision fusion through:

1. **Non-Destructive Multilingual NLP Label Remediation**: A rule-based clinical parser extracting continuous soft pseudo-labels and sample weights across 4,407 radiology reports in 8 languages, repairing Spanish and Latin American terminology defects to recover +472 genuine osteoarthritis cases (+39.2% relative PR-AUC gain).
2. **2.5D DINOv2 ViT-S/14 with Anatomical SlotHead**: Transition from monolithic 3D pooling to slice-grouped self-attention tokens cross-attended by 12 pathology slot queries, driving development Macro ROC-AUC from `0.6198` (3D ResNet-18) to **`0.8185`** (D6 Champion) and locked clinical holdout Gold-58 ROC-AUC to **`0.6331`** (all-time peak).
3. **Deterministic Tissue-Contrast Sequence Resolution Engine**: Automated signal-contrast and marrow fat arbitration eliminating blind Series UID collisions and activating previously starved anatomical slots (`SAG_FLUID_NOFS`).
4. **Empirical Representation Bottleneck Forensics (Phase R/B)**: Rigorous diagnostic audit across 4,349 studies identifying the ~0.81–0.83 performance ceiling as an input downsampling and anatomical slice truncation bottleneck rather than a model capacity limit.

```text
====================================================================================================
                        RSNA KNEE CHALLENGE — CANONICAL PROJECT MAP
====================================================================================================
  ACTIVE DEVELOPMENT CHAMPION : Phase D6 ViT (6-Slot Resolved ViT-S/14, Dev OOF: 0.8185, Gold-58: 0.6331)
  CALIBRATED PRODUCTION CANDIDATE: Phase D5-B2 ViT (D3 + TTA + RankNorm, Dev OOF: 0.8154, Gold-58: 0.6307)
  HYBRID BENCHMARK CHAMPION   : Phase D8-C CoAtNet (Hybrid Conv+ViT, Gold-58: 0.6487)
  POST-HOC ENSEMBLE PEAK      : D6 ViT + D8-C CoAtNet Blend (Dev OOF: 0.8286)
  CANONICAL LABEL DATASET     : labels/production/phase_b_step3_remediated_training_corpus.csv (52,884 rows)
  REMEDIATED D3 LABEL DATASET : phase_d/d3/labels/phase_d3_candidate_label_corpus.csv (+472 OA cases)
  LOCKED GOLD-58 BENCHMARK    : labels/gold58/ (N=58 quarantined studies, zero leakage)
====================================================================================================
```

---

## 2. Quick Navigation: "What Do I Run?"

| Question / Task | Canonical File / Script to Run | Purpose & Output |
| :--- | :--- | :--- |
| **"Which model is the active development champion?"** | [`phase_d/d6/training/orchestrate_d6_cv.py`](phase_d/d6/training/orchestrate_d6_cv.py) | Phase D6 ViT-S/14 with 6-slot intensity resolution (Dev OOF: **0.8185**, Gold-58: **0.6331**). |
| **"Which model is the calibrated production candidate?"** | [`phase_d/d5/scripts/run_d5_inference.py`](phase_d/d5/scripts/run_d5_inference.py) | Phase D5-B2 ViT with H-Flip TTA + within-fold rank normalization (Dev OOF: **0.8154**). |
| **"Which model holds the highest locked Gold-58 record?"** | [`phase_d/d8/scripts/run_d8_gold58_benchmark.py`](phase_d/d8/scripts/run_d8_gold58_benchmark.py) | Phase D8-C CoAtNet hybrid convolutional-attention backbone (Gold-58: **0.6487**). |
| **"Where are the Representation Bottleneck forensics?"** | [`phase_r_b/RB_MASTER_FORENSIC_REPORT_20260916.md`](phase_r_b/RB_MASTER_FORENSIC_REPORT_20260916.md) | Empirical forensic audit explaining why models plateau at ~0.82 and resolving input loss. |
| **"Which script generates the production labels?"** | [`src/phase_b/export_production_corpus.py`](src/phase_b/export_production_corpus.py) | Full NLP extraction over 4,407 reports $\to$ `phase_b_step3_remediated_training_corpus.csv`. |
| **"Which script resolves DICOM series into slots?"** | [`phase_d/d6/resolver/d6_sequence_resolver.py`](phase_d/d6/resolver/d6_sequence_resolver.py) | Tissue-contrast and marrow fat sequence resolver generating `d6_slot_mapping_manifest.csv`. |
| **"Where are the master project specifications?"** | [`docs/phase_d/PHASE_4_AI_CALIBRATION_MASTER_SPEC_AND_PRODUCTION_AUDIT.md`](docs/phase_d/PHASE_4_AI_CALIBRATION_MASTER_SPEC_AND_PRODUCTION_AUDIT.md) | Master audit and specifications spanning all phases (A, B, C, D1–D8, R/B). |
| **"Where are the unit tests and adversarial suites?"** | [`validation/unit_tests/test_remediation.py`](validation/unit_tests/test_remediation.py) | 11 unit tests and 25-case multilingual perturbation suite (100% pass rate). |

---

## 3. End-to-End System Architecture

```text
                                 RAW UNSTRUCTURED RADIOLOGY REPORTS
                                                │
                                                ▼
                                ┌──────────────────────────────────┐
                                │  PHASE B REMEDIATED NLP PIPELINE │
                                │    (src/phase_b/pipeline.py)     │
                                └───────────────┬──────────────────┘
                                                │
                ┌───────────────────────────────┴───────────────────────────────┐
                ▼                                                               ▼
    [ 58 Gold Studies (Eval) ]                                     [ 4,407 Training Studies ]
    labels/gold58/phase_b_step3...                                 labels/production/phase_b_step3...
    (696 Hard Multi-Target Labels)                                 (52,884 Calibrated Soft Labels)
                │                                                               │
                │                     ┌─────────────────────────────────────────┘
                │                     ▼
                │      ┌────────────────────────────────────────────────────────┐
                │      │       PHASE C1 VOLUMETRIC PREPROCESSING ENGINE         │
                │      │      Native DICOMs ──> Standardized (32, 256, 256)     │
                │      └───────────────────────┬────────────────────────────────┘
                │                              │
                │                              ▼
                │      ┌────────────────────────────────────────────────────────┐
                │      │   PHASE D6 DETERMINISTIC SEQUENCE RESOLUTION ENGINE    │
                │      │    Dynamic range contrast + marrow fat signal ratio    │
                │      │     Resolves collisions & activates 6 canonical slots  │
                │      └───────────────────────┬────────────────────────────────┘
                │                              │
                │                              ▼
                │      ┌────────────────────────────────────────────────────────┐
                │      │        PHASE D DEEP LEARNING VISION PIPELINE           │
                │      │  • 2.5D DINOv2 ViT-S/14 Backbone (Frozen 0-5, Train 6-11)│
                │      │  • Anatomical SlotHead (12 Pathology Queries + Priors) │
                │      │  • Confidence-Weighted Soft BCE Loss                   │
                │      │  • Spatial H-Flip TTA + Within-Fold Rank Normalization │
                │      └───────────────────────┬────────────────────────────────┘
                │                              │
                ▼                              ▼
    ┌───────────────────────────────────────────────────────────────────────────────┐
    │             FINAL SCIENTIFIC EVALUATION & LEADERBOARD BENCHMARK               │
    │  • 5-Fold Development OOF Macro ROC-AUC: 0.8185 (D6) / 0.8286 (D6+D8C Blend)   │
    │  • Locked Quarantined Gold-58 Clinical ROC-AUC: 0.6331 (D6) / 0.6487 (D8-C)   │
    └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Comprehensive Performance Trajectory

### A. Development Cohort Out-of-Fold (4,349 Studies, 5-Fold Cross-Validation)

| Phase / Model Identifier | Backbone & Supervision Protocol | Evaluated Ground Truth | Macro ROC-AUC (Unweighted) | Macro ROC-AUC (Weighted) | Macro PR-AUC | Status / Scientific Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase C Baseline** | 3D ResNet-18 Multi-View Concat | Legacy Phase B | `0.6198` | `0.6241` | `0.1245` | Frozen 3D Control Reference |
| **Phase D1.1 Baseline** | 2.5D DINOv2 ViT-S/14 SlotHead | Legacy Phase B | `0.7510` | `0.7573` | `0.2212` | Production 5-Fold Foundation |
| **Phase D1.2-B2 Candidate** | ViT-S/14 + Spatial TTA + RankNorm | Legacy Phase B | `0.7592` | `0.7658` | `0.2302` | Calibrated Baseline (+0.0082) |
| **Phase D2 Standalone** | RadImageNet ResNet-50 CNN Arm | Legacy Phase B | `0.7287` | `0.7335` | `0.1706` | Completed CNN Arm |
| **Phase D2-C1 Multi-Arm Blend**| 75% ViT + 25% CNN Rank Blend | Legacy Phase B | `0.7634` | `0.7699` | `0.2396` | Beneficial Regularization (+0.55%) |
| **Phase D3 Standalone ViT** | DINOv2 ViT-S/14 Retrained | **Remediated D3** | **`0.8135`** | **`0.8204`** | **`0.3080`** | **Clean Supervision Leap (+39.2% PR-AUC)** |
| **Phase D3 Standalone ViT** | DINOv2 ViT-S/14 Retrained | Legacy Phase B | **`0.8184`** | **`0.8253`** | **`0.2319`** | Cross-Standard Robust (+0.0592) |
| **Phase D4 Standalone CNN** | RadImageNet ResNet-50 Retrained | Remediated D3 | `0.6864` | `0.6904` | `0.1369` | 12.71% Gap Behind ViT |
| **Phase D4-C1 Multi-Arm Blend**| 75% ViT + 25% CNN Rank Blend | Remediated D3 | `0.8071` | `0.8141` | `0.2870` | **Case C: Rejected** (CNN dilutes ViT) |
| **Phase D5-B2 Calibrated** | D3 ViT + TTA + Within-Fold RankNorm | Remediated D3 | `0.8154` | `0.8221` | `0.3149` | Audited Calibrated Candidate |
| **Phase D6 Resolved ViT** | 6-Slot Intensity-Arbitrated ViT-S/14 | **Remediated D3** | **`0.8185`** | **`0.8254`** | **`0.3045`** | **Active Development Champion** |
| **Phase D7-A Capacity ViT** | DINOv2 ViT-B/14 (87.0M parameters) | Remediated D3 | `0.8137` | `0.8198` | `0.2979` | Overfitting Tradeoff (-0.0048) |
| **Phase D7-B Multi-Window** | ViT-S/14 + 3 Overlapping Depth Windows | Remediated D3 | `0.8100` | `0.8162` | `0.2864` | Central Joint Feature Dilution |
| **Phase D8-C Hybrid** | CoAtNet Hybrid Convolution + ViT | Remediated D3 | `0.8166` | `0.8231` | `0.3012` | Strong Local Conv Inductive Bias |
| **Post-Hoc D6+D8C Blend** | 50% D6 ViT + 50% D8-C CoAtNet | Remediated D3 | **`0.8286`** | **`0.8351`** | **`0.3210`** | **All-Time Development Peak** |

---

### B. Locked Gold-58 Clinical Benchmark (Quarantined N=58 Holdout)

| Evaluation Stage / Model Protocol | Macro ROC-AUC | Macro PR-AUC | Gain vs Phase C Baseline | Key Clinical Achievement |
| :--- | :---: | :---: | :---: | :--- |
| **Phase C Baseline (3D SmallResNet3D)** | `0.5986` | `0.4526` | — | Initial 3D volumetric baseline |
| **Phase D1.2-B2 Milestone (Production Candidate)** | `0.6174` | `0.4938` | `+3.14%` (+0.0188) | ViT + TTA on legacy noisy labels |
| **Phase D5-B2 Calibrated ViT Candidate** | `0.6307` | `0.5285` | `+5.36%` (+0.0321) | Remediated supervision + RankNorm |
| **Phase D6-B2 Active Development Champion** | **`0.6331`** | **`0.5125`** | **`+5.76%` (+0.0345)** | **Sequence-Resolved 6-Slot Champion** |
| **Phase D8-C CoAtNet Hybrid Architecture** | **`0.6487`** | **`0.5340`** | **`+8.37%` (+0.0501)** | **All-Time Gold-58 Performance Record** |

---

## 5. Key Scientific Discoveries & Engineering Insights

### 1. Clinical Supervision Remediation (Phase D3)
Noisy weak supervision was identified as the single largest bottleneck in early modeling. A systematic audit revealed that the baseline regex failed on Spanish syntactic variations (`"artrosis femorotibial lateral"`) and Latin American vocabulary (`"condropatía"`), incorrectly labeling genuine osteoarthritis cases as negative. Fixing these defects recovered **+472 confirmed osteoarthritis cases**:
- Medial OA PR-AUC: `0.0228` $\to$ **`0.2545` (11.2× gain)**
- Lateral OA PR-AUC: `0.0034` $\to$ **`0.1772` (52.1× gain)**
- PF OA PR-AUC: `0.0833` $\to$ **`0.5914` (7.1× gain)**
- Global PR-AUC vaulted from `0.2212` to **`0.3080` (+39.2% relative gain)**, proving that supervision quality bounds vision performance.

### 2. Deterministic Sequence Resolution Engine (Phase D6)
Coarse DICOM sequence tags caused 39.94% of studies to arbitrate multi-series collisions by alphabetical SeriesInstanceUID sorting, starving `Slot 3` (`SAG_FLUID_NOFS`). Phase D6 engineered an automated sequence classifier using slice intensity dynamic range ($P_{90} - P_{10}$) and bone marrow fat signal ratios. This activated Slot 3 across 852 studies, pushing development Macro ROC-AUC to `0.8185` and Gold-58 to `0.6331`.

### 3. The Limits of Multi-Arm Ensembling (Phase D4: Case C)
In Phase D2, when supervision was noisy and the ViT scored `0.7592` vs CNN `0.7287` (3.05% margin), blending 25% CNN provided beneficial structural regularization (+0.55% gain). In Phase D4, under cleaned supervision, the ViT scaled to `0.8135` while the CNN reached only `0.6864` (12.71% margin). At this performance disparity, blending the CNN diluted the ViT's high-confidence predictions, dragging Macro ROC-AUC down to `0.8071` (-0.79%). Under clean supervision, the ViT stands alone.

### 4. Input Representation Bottleneck Forensics (Phase R/B)
Extensive factorial testing (D7 capacity scaling, D7-B multi-window, D8 hybrid backbones) demonstrated that models plateaued at `~0.81–0.83` Macro ROC-AUC. Phase R/B forensic auditing proved that this ceiling is **NOT** a capacity limitation:
- **Spatial Downsampling (55.4% Pixel Loss)**: Resampling native DICOM matrices (mean 0.334 mm) to fixed $256 \times 256$ (0.500 mm) destroyed 55.36% of in-plane pixel density, obscuring micro-fractures ($<0.4\text{ mm}$) and thin collateral ligaments.
- **Peripheral Slice Truncation**: Restricting models to central slices (slices 6–25 of 32) amputated peripheral slices containing **76.87% foreground anatomical tissue**, where the MCL (coronal 0–6) and Baker's cyst (sagittal 22–30) reside.
- **Patch Projection Smearing**: ViT $14 \times 14$ patch tokens cover a massive $7.0 \times 7.0\text{ mm}$ physical footprint ($49\text{ mm}^2$), blurring fine ligament disruptions into surrounding fat.

---

## 6. Repository Organization

```text
RSNA_Code/
├── README.md                      <- Master execution & publication guide
├── LICENSE                        <- MIT open-source license
├── requirements.txt               <- Pinned Python dependencies
├── environment.yml                <- Conda environment specification
├── pyproject.toml                 <- Modern PEP 517/621 package build config
├── .gitignore                     <- Strict exclusion rules for weights, arrays, data
│
├── docs/                          <- Architectural specifications & forensic audits
│   ├── DATASET_SPECIFICATION.md   <- Raw DICOM layout, 6-slot mapping, 12 pathology targets
│   ├── REPRODUCIBILITY.md         <- Comprehensive phase-by-phase reproduction instructions
│   ├── project_overview/          <- End-to-end architecture & system design
│   ├── phase_a/                   <- Baseline NLP evaluation & error breakdown
│   ├── phase_b/                   <- NLP remediation spec, soft-target math & audit log
│   ├── phase_c/                   <- Preprocessing & 3D vision modeling specification
│   └── phase_d/                   <- Phase 4 / Phase D Master Spec & AI Calibration Audit
│
├── src/                           <- Core Python source code
│   ├── phase_a/                   <- Phase A baseline reproduction shell script
│   └── phase_b/                   <- Production Remediated NLP Pipeline
│       ├── parser.py              <- Non-destructive span-preserving report parser
│       ├── lexicon.py             <- Multilingual clinical concept registry (8 languages)
│       ├── severity_gates.py      <- Clinical severity gating logic
│       ├── soft_targets.py        <- Continuous soft pseudo-label target math
│       ├── pipeline.py            <- Complete end-to-end NLP labeling pipeline
│       ├── evaluate.py            <- Gold-58 benchmark evaluation harness
│       └── export_production_corpus.py <- Batch generator for 4,407 reports (52,884 rows)
│
├── validation/                    <- Test suites & verification harnesses
│   ├── unit_tests/                <- Unit tests (parser, gates, soft targets)
│   ├── adversarial/               <- 40-case 8-language adversarial perturbation suite
│   ├── provenance/                <- Data leakage & 20-field schema verification
│   └── acceptance/                <- Formal acceptance governance runner
│
├── labels/                        <- Validated training & evaluation labels
│   ├── production/                <- Primary 52,884-row training soft targets (Y, C, W)
│   ├── gold58/                    <- Frozen Phase A & Remediated Step 3 Gold-58 predictions
│   └── phase_b/                   <- Intermediate Step 1, Step 2, and Step 3 labels
│
├── results/                       <- Evaluation metrics, summaries & confusion matrices
│   ├── metrics/                   <- Per-target precision, recall, F1, ROC-AUC CSVs
│   ├── confusion_matrices/        <- Confusion matrices across all 12 targets
│   ├── error_analysis/            <- Detailed diagnostic error breakdowns
│   ├── transitions/               <- Prediction delta tracking tables
│   └── summaries/                 <- Machine-readable executive JSON summaries
│
├── phase_c/                       <- Phase C: Preprocessing & 3D Baseline Training
│   ├── c0/                        <- Empirical preprocessing pilot audit
│   ├── c1/                        <- Full-dataset volumetric preprocessing scripts
│   ├── c2/                        <- Controlled 4-way preprocessing model comparison
│   └── c3/                        <- Production 3D ResNet-18 baseline (5-Fold CV & Gold-58)
│
├── phase_d/                       <- Phase D: AI Calibration, Architecture & Supervision
│   ├── d1/                        <- 2.5D DINOv2 ViT-S/14 SlotHead (Production Baseline)
│   ├── d2/                        <- RadImageNet ResNet-50 CNN & ViT+CNN ensembling
│   ├── d3/                        <- NLP supervision remediation & ViT retraining (+472 OA)
│   ├── d4/                        <- Multi-node CNN continuation & Case C analysis
│   ├── d5/                        <- Inference calibration, TTA & Gold-58 benchmark
│   ├── d6/                        <- Deterministic sequence resolver (Active Champion)
│   ├── d7/                        <- ViT-B capacity scaling & multi-window depth suite
│   ├── d8/                        <- CoAtNet hybrid convolutional-attention exploration
│   └── reports/                   <- Consolidated experiment logs & development reports
│
├── phase_r_b/                     <- Phase R/B: Representation & Bottleneck Forensics
│   ├── d9a/                       <- Full-depth hybrid 2.5D model & ablation study
│   ├── RB_MASTER_FORENSIC_REPORT_20260916.md <- Master forensic audit
│   └── run_rb_target_and_model_forensics.py  <- Cross-phase error forensics script
│
├── manifests/                     <- Cryptographic hashes & artifact governance
│   ├── artifact_manifest.csv      <- Complete project inventory
│   ├── duplicate_report.csv       <- Deduplication analysis
│   └── checksums/sha256sums.txt   <- SHA-256 cryptographic hashes
│
└── archive/                       <- Historical intermediate prototypes & experiment logs
```

---

## 7. Installation & Quickstart

### Prerequisites
- Linux OS (Ubuntu 20.04/22.04/24.04 recommended)
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with CUDA 11.8+ or 12.1+ (A100 / RTX 3090 / RTX 4090 recommended for training)

### Option 1: Using pip

```bash
git clone https://github.com/your-username/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Option 2: Using Conda

```bash
git clone https://github.com/your-username/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection

conda env create -f environment.yml
conda activate rsna-knee
pip install -e .
```

---

## 8. Data Configuration & Expected Layout

The dataset is acquired from the **RSNA 2024 Knee Abnormality Detection Challenge**. Raw DICOM series and competition metadata should be placed in an external directory (or symlinked):

```bash
export RSNA_PROJECT_ROOT="$(pwd)"
export RSNA_DATA_ROOT="/path/to/RSNA_Knee_Data"
```

Expected directory tree under `$RSNA_DATA_ROOT`:
```text
RSNA_Knee_Data/
├── train.csv
├── train_series.csv
├── test.csv
├── test_series.csv
├── train_series/
│   └── <StudyInstanceUID>/
│       └── <SeriesInstanceUID>/
│           └── *.dcm
└── test_series/
```
For detailed format specifications and 6-slot mapping rules, consult [Dataset Specification](docs/DATASET_SPECIFICATION.md).

---

## 9. Training & Evaluation Reproduction

For complete step-by-step reproduction instructions across all phases, see [Reproducibility Guide](docs/REPRODUCIBILITY.md).

### 1. Run Unit Tests & Verification
```bash
python3 validation/unit_tests/test_remediation.py
python3 validation/adversarial/adversarial_test.py
```

### 2. Generate Production Remediated Labels (Phase B)
```bash
python3 src/phase_b/export_production_corpus.py
python3 src/phase_b/evaluate.py
```

### 3. Preprocess Raw DICOMs to Volumetric Arrays (Phase C1)
```bash
python3 phase_c/c1/scripts/preprocess_c1.py --workers 16
python3 phase_c/c1/scripts/validate_c1.py
```

### 4. Train Active Champion Model (Phase D6)
```bash
# Resolve DICOM series into deterministic 6-slot mapping
python3 phase_d/d6/resolver/d6_sequence_resolver.py

# Train 5-fold cross-validation
python3 phase_d/d6/training/orchestrate_d6_cv.py

# Evaluate against locked Gold-58 clinical cohort
python3 phase_d/d6/scripts/run_d6_gold58_benchmark.py
```

---

## 10. Research Governance & Clinical Integrity Rules

> [!IMPORTANT]
> **RULE 1: EXTERNAL MRI DATA IS FROZEN & READ-ONLY**  
> Raw volumetric MRI series are stored externally. Dataloaders read inputs in read-only mode via `$RSNA_DATA_ROOT`. Never copy raw DICOM datasets into version control.

> [!WARNING]
> **RULE 2: GOLD-58 EVALUATION COHORT QUARANTINE**  
> The 58 Gold studies (`labels/gold58/`) must **NEVER** be mixed into the training corpus, used for pseudo-label generation, or used for threshold tuning. It is strictly reserved as an untouchable holdout evaluation benchmark.

---

## 11. Citation & Acknowledgments

If you find this work, codebase, or findings useful in your research, please cite:

```bibtex
@misc{rsna_knee_abnormality_detection_2026,
  title={Multi-Task Knee MRI Abnormality Detection with Clinical Weak Supervision Remediation and Anatomical Slot Cross-Attention},
  author={RSNA Knee Challenge Research Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-username/rsna-knee-abnormality-detection}}
}
```

---

## 12. License

This project is released under the [MIT License](LICENSE).
