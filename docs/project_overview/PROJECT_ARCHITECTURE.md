# RSNA Knee Abnormality Detection — Project Architecture & Governance

## 1. System Overview & End-to-End Workflow

The RSNA Knee Abnormality Detection system is an end-to-end multi-task deep learning pipeline designed to detect 12 distinct musculoskeletal pathologies from volumetric knee MRI series.

```text
                                  UNSTRUCTURED RADIOLOGY REPORTS
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
               │                                                               ▼
               │                                              ┌──────────────────────────────────┐
               │                                              │      PHASE C 3D VISION BASELINE  │
               │                                              │    (3D ResNet-18 Multi-Series)   │
               │                                              │     Dev OOF: 0.6198 Macro AUC    │
               │                                              └────────────────┬─────────────────┘
               │                                                               │
               │                                                               ▼
               │                                              ┌──────────────────────────────────┐
               │                                              │   PHASE D AI VISION ARCHITECTURE │
               │                                              │ 2.5D DINOv2 ViT-S/14 + SlotHead  │
               │                                              │  D1.1: 0.7510 | D1.2-B2: 0.7592  │
               │                                              │  D2 ViT+CNN Ensemble: 0.7634     │
               │                                              │  D3 Remediated ViT: 0.8135       │
               │                                              │  D4 Multi-Node Case C: 0.8071    │
               │                                              └────────────────┬─────────────────┘
               │                                                               │
               ▼                                                               ▼
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │             FINAL SCIENTIFIC EVALUATION & LEADERBOARD SUBMISSION              │
   │           Frozen Benchmark: Gold-58 Macro ROC-AUC = 0.6174 (D1.2-B2)          │
   └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```text
RSNA_Code/
├── README.md                      <- Master "What Do I Run?" execution & reproduction guide
├── docs/                          <- Architectural specifications, phase reports & audits
│   ├── project_overview/          <- Architecture, file maps, data pointers
│   ├── phase_a/                   <- Baseline evaluation & error analysis docs
│   ├── phase_b/                   <- Remediation specs, audit logs, validation reports
│   ├── phase_c/                   <- Phase C vision model training protocol
│   └── phase_d/                   <- Phase 4 / Phase D Master Spec, AI Calibration & Audits
├── src/                           <- Python source code & executables
│   ├── phase_a/                   <- Frozen baseline reproduction script
│   └── phase_b/                   <- Production remediated NLP pipeline & modules
├── validation/                    <- Test suites & verification runners
│   ├── unit_tests/                <- Unit tests for parsing, gating, soft targets
│   ├── adversarial/               <- 40-case 8-language adversarial perturbation suite
│   ├── provenance/                <- Data leakage & 20-field schema audit
│   └── acceptance/                <- Comprehensive acceptance audit suite & verdict
├── labels/                        <- Cleaned & validated label datasets
│   ├── production/                <- Primary 52,884-row training pseudo-labels (Y, C, W)
│   ├── gold58/                    <- Frozen Phase A & Remediated Step 3 Gold-58 predictions
│   └── phase_b/                   <- Step 1, Step 2, and Step 3 intermediate labels
├── results/                       <- Evaluation outputs, metrics & error analysis
│   ├── metrics/                   <- Per-target precision, recall, F1, accuracy CSVs
│   ├── confusion_matrices/        <- Target confusion matrices
│   ├── error_analysis/            <- Detailed diagnostic error breakdowns
│   ├── transitions/               <- Prediction delta & transition tables
│   └── summaries/                 <- JSON executive summaries
├── phase_c/                       <- Phase C Preprocessing & 3D Baseline Training (C0..C3)
├── phase_d/                       <- Phase D AI Calibration, Modeling & Supervision (D1..D4)
│   ├── d1/                        <- 2.5D DINOv2 ViT-S/14 SlotHead (Frozen Production Candidate)
│   ├── d2/                        <- RadImageNet ResNet-50 CNN & ViT-CNN Ensembling
│   ├── d3/                        <- NLP Weak Supervision Remediation & ViT Retraining
│   ├── d4/                        <- 3-Node 6-GPU Multi-Arm Continuation on Remediated Supervision
│   └── reports/                   <- Consolidated development reports & experiment logs
├── manifests/                     <- Cryptographic hashes & artifact governance
│   ├── artifact_manifest.csv      <- Complete project inventory
│   ├── duplicate_report.csv       <- Deduplication & hash collision report
│   └── checksums/                 <- sha256sums.txt
└── archive/                       <- Historical prototypes & intermediate files
    ├── intermediate/              <- Intermediate audit results & holdout evaluations
    ├── prototypes/                <- Phase B early prototype code
    └── llm_experiments/           <- Historical local LLM prompt extraction logs
```

---

## 3. External MRI Data Corpus Pointer

The volumetric MRI image dataset (~512 GB) is stored externally on the filesystem and **MUST NOT** be modified, moved, or duplicated into `RSNA_Code`:
- **External Data Corpus Path**: Configured via `$RSNA_DATA_ROOT` (e.g. `/path/to/RSNA_Knee_Data`)
  - `train_series/`: 4,351 volumetric MRI series subdirectories
  - `test_series/`: Test evaluation MRI series
  - `marked_train_series/`: Bounding-box and slice-marked series
  - `train.csv` / `train_series.csv`: Metadata and series indices
  - `golden_reports.csv` / `gold_studies.csv`: Raw report text and gold study UIDs
