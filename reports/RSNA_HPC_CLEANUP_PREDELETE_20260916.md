# RSNA HPC CLEANUP — PRE-DELETION AUDIT MANIFEST

**Audit Date**: September 16, 2026  
**Auditor**: Autonomous Agentic AI (Google DeepMind Antigravity)  
**Host Filesystem**: Lustre Distributed Storage (`/scratch`)  
**Target Root**: `/scratch/deepanthpcu/RSNA_Code/`  
**Governance Assertions**:
* Strictly 0 deletions have taken place yet.
* SLURM Job `653724` is untouched.
* Training data (`phase_c/c1/processed/`) is 100% PROTECTED.
* Gold-58 assets are 100% PROTECTED.
* Model checkpoints (`*.pt`) are 100% PROTECTED.
* OOF predictions, evaluation records, canonical metadata, and authoritative labels are 100% PROTECTED.

---

## 1. Executive Pre-Deletion Space Recovery Summary

* **Total Proposed File Deletions**: 78 files
* **Total Proposed Empty Directory Deletions**: 13 empty directories
* **Total Proposed Directory Removals**: 3 parent obsolete directories (`visual_samples/`, `docs_backup_20260914_1825/`, `archive/intermediate/`) + 21 `__pycache__` directories
* **TOTAL PROPOSED DELETION — APPARENT SIZE**: **`14.22 MB`** (`14907193 bytes`)
* **TOTAL PROPOSED DELETION — LUSTRE QUOTA SIZE**: **`23.14 MB`** (`24260608 bytes` allocated physical blocks)

---

## 2. Safety Gate Checklist & Verification Invariants

| Protected Class | Status | Confirmation Method |
| :--- | :---: | :--- |
| **Training Data (`phase_c/c1/processed/`)** | **100% PROTECTED (YES)** | Verified zero overlap with any proposed deletion path. |
| **Gold-58 Clinical Assets & Predictions** | **100% PROTECTED (YES)** | Strictly excluded any path containing `gold` or `marked`. |
| **All Model Checkpoints (`*.pt`, `*.ckpt`)**| **100% PROTECTED (YES)** | Verified zero checkpoint files in deletion list. |
| **All OOF & Evaluation Predictions** | **100% PROTECTED (YES)** | All master OOF and val prediction CSVs retained. |
| **Canonical Metadata (`train_series.csv`)** | **100% PROTECTED (YES)** | Canonical file retained; only `Copy-of-` duplicate targeted. |
| **Authoritative Labels & Splits** | **100% PROTECTED (YES)** | `cv_splits.json`, `phase_d3_candidate_...` retained. |
| **Project Source Code & Core Documentation**| **100% PROTECTED (YES)** | All active `.py` and `docs/` documentation retained. |

---

## 3. Comprehensive Itemized Deletion Manifest

| Index | Proposed Deletion Path | Apparent Size | Lustre Block Size | Obsolescence Rationale | Duplication Status |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `/scratch/deepanthpcu/RSNA_Code/csv/Copy-of-train_series.csv` | 3377.8 KB | 3380.0 KB | Exact duplicate of canonical train_series.csv (SHA256: 573c1d80...) | Yes, canonical copy at csv/train_series.csv |
| 2 | `/scratch/deepanthpcu/RSNA_Code/csv/.~lock.train.csv#` | 0.1 KB | 4.0 KB | Stale temporary editor/LibreOffice lock file | No (transient lock file) |
| 3 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c0/visual_samples/visual_preservation_study_1_4850458401.png` | 3645.4 KB | 7296.0 KB | Temporary C0 slice inspection diagnostic PNG images | Yes, initial pipeline validation artifacts |
| 4 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c0/visual_samples/visual_preservation_study_2_7250465738.png` | 1628.1 KB | 3264.0 KB | Temporary C0 slice inspection diagnostic PNG images | Yes, initial pipeline validation artifacts |
| 5 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c0/visual_samples/visual_preservation_study_3_5446885715.png` | 3263.3 KB | 6528.0 KB | Temporary C0 slice inspection diagnostic PNG images | Yes, initial pipeline validation artifacts |
| 6 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/D3_D_LABEL_QUALITY_AUDIT.md` | 9.6 KB | 12.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 7 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/D4_DEVELOPMENT_REPORT.md` | 14.5 KB | 16.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 8 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/D2_D3_EXPERIMENT_LOG.md` | 11.8 KB | 12.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 9 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/D3_PRETRAINING_RULE_AUDIT.md` | 10.9 KB | 12.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 10 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/D1_CHANGELOG.md` | 15.0 KB | 16.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 11 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/PHASE_4_AI_CALIBRATION_MASTER_SPEC_AND_PRODUCTION_AUDIT.md` | 46.4 KB | 48.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 12 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/D4_EXPERIMENT_LOG.md` | 7.5 KB | 8.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 13 | `/scratch/deepanthpcu/RSNA_Code/docs_backup_20260914_1825/D2_D3_DEVELOPMENT_REPORT.md` | 13.5 KB | 16.0 KB | Ad-hoc redundant documentation backup snapshot | Yes, current authoritative docs are in docs/ and phase_d/ |
| 14 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_remediated_adversarial_audit_10case.csv` | 6.3 KB | 8.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 15 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_full_corpus_activation_summary.json` | 15.4 KB | 16.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 16 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_rule_provenance.csv` | 93.9 KB | 192.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 17 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_confusion_matrices.csv` | 0.2 KB | 4.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 18 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_metrics.csv` | 1.5 KB | 4.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 19 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_summary.json` | 4.5 KB | 8.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 20 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_adversarial_audit.csv` | 8.7 KB | 12.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 21 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_holdout_summary.json` | 10.2 KB | 12.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 22 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_holdout_results.csv` | 1183.5 KB | 1184.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 23 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_error_cases.csv` | 185.9 KB | 188.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 24 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_transitions.csv` | 159.9 KB | 160.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 25 | `/scratch/deepanthpcu/RSNA_Code/archive/intermediate/phase_b_step3_prototype_full_corpus_activation_audit.csv` | 47.4 KB | 48.0 KB | Obsolete Phase B rule extraction prototype files superseded by production | Yes, production assets live in labels/production/ |
| 26 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/scripts/__pycache__/model.cpython-311.pyc` | 12.5 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 27 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/scripts/__pycache__/dataset.cpython-311.pyc` | 15.3 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 28 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/scripts/__pycache__/loss.cpython-311.pyc` | 2.7 KB | 8.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 29 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c0/scripts/__pycache__/preprocess.cpython-313.pyc` | 14.3 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 30 | `/scratch/deepanthpcu/RSNA_Code/phase_c/c2/scripts/__pycache__/config.cpython-311.pyc` | 2.7 KB | 8.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 31 | `/scratch/deepanthpcu/RSNA_Code/archive/phase_c3_pre_reset_archive/scripts/__pycache__/model.cpython-311.pyc` | 12.5 KB | 16.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 32 | `/scratch/deepanthpcu/RSNA_Code/archive/phase_c3_pre_reset_archive/scripts/__pycache__/loss.cpython-311.pyc` | 2.7 KB | 4.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 33 | `/scratch/deepanthpcu/RSNA_Code/archive/phase_c3_pre_reset_archive/scripts/__pycache__/dataset.cpython-311.pyc` | 15.3 KB | 16.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 34 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/lexicon.cpython-313.pyc` | 30.1 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 35 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/parser.cpython-313.pyc` | 8.7 KB | 12.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 36 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/severity_gates.cpython-311.pyc` | 12.7 KB | 16.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 37 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/anatomy.cpython-313.pyc` | 7.7 KB | 16.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 38 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/soft_targets.cpython-311.pyc` | 3.5 KB | 4.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 39 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/assertion.cpython-311.pyc` | 10.8 KB | 12.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 40 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/config.cpython-311.pyc` | 1.6 KB | 4.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 41 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/anatomy.cpython-311.pyc` | 8.3 KB | 12.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 42 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/soft_targets.cpython-313.pyc` | 3.2 KB | 4.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 43 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/lexicon.cpython-311.pyc` | 32.0 KB | 36.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 44 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/pipeline.cpython-311.pyc` | 17.0 KB | 40.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 45 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/assertion.cpython-313.pyc` | 10.0 KB | 24.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 46 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/pipeline.cpython-313.pyc` | 15.0 KB | 16.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 47 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/parser.cpython-311.pyc` | 9.4 KB | 12.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 48 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/__init__.cpython-311.pyc` | 0.6 KB | 4.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 49 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/severity_gates.cpython-313.pyc` | 11.1 KB | 12.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 50 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/severity.cpython-313.pyc` | 12.6 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 51 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/severity.cpython-311.pyc` | 15.1 KB | 16.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 52 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/__init__.cpython-313.pyc` | 0.6 KB | 4.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 53 | `/scratch/deepanthpcu/RSNA_Code/src/phase_b/__pycache__/config.cpython-313.pyc` | 1.5 KB | 4.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 54 | `/scratch/deepanthpcu/RSNA_Code/validation/unit_tests/__pycache__/test_remediation.cpython-313.pyc` | 9.6 KB | 12.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 55 | `/scratch/deepanthpcu/RSNA_Code/validation/adversarial/__pycache__/adversarial_test.cpython-313.pyc` | 7.9 KB | 16.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 56 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d4/scripts/__pycache__/train_ddp_d4.cpython-311.pyc` | 26.9 KB | 28.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 57 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d4/scripts/__pycache__/model_d4_cnn.cpython-311.pyc` | 9.9 KB | 12.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 58 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d4/scripts/__pycache__/orchestrate_d4_cv.cpython-311.pyc` | 26.0 KB | 56.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 59 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d8/__pycache__/dataset_d8.cpython-311.pyc` | 14.3 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 60 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d8/training/__pycache__/train_ddp_d8.cpython-311.pyc` | 33.1 KB | 36.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 61 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d8/training/__pycache__/orchestrate_d8.cpython-311.pyc` | 40.4 KB | 44.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 62 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d8/models/__pycache__/model_d8.cpython-311.pyc` | 35.6 KB | 36.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 63 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d6/training/__pycache__/train_ddp_d6.cpython-311.pyc` | 29.8 KB | 64.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 64 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d6/training/__pycache__/orchestrate_d6_cv.cpython-311.pyc` | 24.2 KB | 28.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 65 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d2/scripts/__pycache__/model_d2_cnn.cpython-311.pyc` | 10.9 KB | 24.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 66 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/scripts/__pycache__/dataset_d1.cpython-311.pyc` | 13.2 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 67 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/scripts/__pycache__/train_ddp_d1.cpython-311.pyc` | 23.9 KB | 48.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 68 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/scripts/__pycache__/d1_2_eval_utils.cpython-311.pyc` | 13.0 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 69 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/scripts/__pycache__/model_d1.cpython-311.pyc` | 12.6 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 70 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/scripts/__pycache__/slot_mapping_d1.cpython-311.pyc` | 13.4 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 71 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/scripts/__pycache__/loss_d1.cpython-311.pyc` | 2.6 KB | 8.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 72 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/scripts/__pycache__/d1_2_blend_experiments.cpython-311.pyc` | 8.6 KB | 24.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 73 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/__pycache__/dataset_d7.cpython-311.pyc` | 13.4 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 74 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/training/__pycache__/train_ddp_d7.cpython-311.pyc` | 30.8 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 75 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/training/__pycache__/orchestrate_d7d.cpython-311.pyc` | 23.9 KB | 24.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 76 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/training/__pycache__/orchestrate_d7_cv.cpython-311.pyc` | 42.2 KB | 44.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 77 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/models/__pycache__/model_d7.cpython-311.pyc` | 30.1 KB | 32.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |
| 78 | `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/scripts/__pycache__/orchestrate_d3_cv.cpython-311.pyc` | 11.1 KB | 24.0 KB | Python bytecode cache file (.pyc) automatically recompiled upon execution | Reconstructible automatically from source .py |

---

## 4. Itemized Empty Directory Removal List

The following 13 directories are empty artifacts from earlier experimental setups and will be cleanly removed via `rmdir`:
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/cache`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/preprocessing`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/features`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d4/configs`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d6/inference`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d6/configs`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d6/models`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d2/reports`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/configs`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/checkpoints`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/configs`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/audits`
* `/scratch/deepanthpcu/RSNA_Code/phase_d/d7/predictions/val`

---
*Safety gate completed. Awaiting authorization / programmatic confirmation before execution.*
