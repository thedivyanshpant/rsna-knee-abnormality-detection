# RSNA Knee Abnormality Detection — Reproducibility & Execution Guide

This document outlines the step-by-step instructions to reproduce each scientific phase of the project, from unstructured text label remediation through vision model training, calibration, and benchmark evaluation.

---

## 1. Environment Setup

### Option A: Conda Environment (Recommended)

```bash
# Create and activate environment from environment.yml
conda env create -f environment.yml
conda activate rsna-knee

# Install local package in editable development mode
pip install -e .
```

### Option B: Python Virtual Environment with pip

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

---

## 2. Environment Variables Configuration

Before executing pipeline scripts, configure paths to match your local storage:

```bash
# Mandatory: Root of this repository
export RSNA_PROJECT_ROOT="$(pwd)"

# Path to directory containing train.csv, train_series.csv, and raw DICOMs
export RSNA_DATA_ROOT="/path/to/RSNA_Knee_Data"

# Optional: Path to preprocessed .npz storage (defaults to phase_c/c1/processed)
export RSNA_PROCESSED_DIR="${RSNA_PROJECT_ROOT}/phase_c/c1/processed"

# Optional: Execution binaries for distributed training (defaults to current python/torchrun)
export PYTHON_BIN="$(which python3)"
export TORCHRUN_BIN="$(which torchrun)"
```

---

## 3. Phase-by-Phase Reproduction Workflow

### Phase A: Baseline NLP Extractor Reproduction
Reproduces the competitor baseline evaluation on the locked Gold-58 cohort:
```bash
bash src/phase_a/reproduce_phase_a.sh
```
Outputs:
- Predictions: `results/phase_a_baseline/predictions_gold58.csv`
- Metrics: `results/phase_a_baseline/metrics_per_target.csv`
- Confusion matrices: `results/phase_a_baseline/confusion_matrices.csv`

---

### Phase B: Production Remediated Label Generation
Executes the non-destructive multilingual rule-based NLP extraction pipeline across all 4,407 training reports, applying clinical severity gates and continuous soft-target math:

```bash
# Generate production training corpus with continuous soft targets (52,884 rows)
python3 src/phase_b/export_production_corpus.py

# Evaluate remediated NLP extraction against Gold-58
python3 src/phase_b/evaluate.py
```

Outputs:
- `labels/production/phase_b_step3_remediated_training_corpus.csv`
- `labels/gold58/phase_b_step3_remediated_predictions_gold58.csv`

---

### Phase C: Preprocessing & 3D Baseline Training

#### 1. Full Dataset Volumetric Preprocessing (Phase C1)
Converts raw DICOM series into standardized $(32, 256, 256)$ float16 volumes:
```bash
python3 phase_c/c1/scripts/preprocess_c1.py \
    --data_dir "${RSNA_DATA_ROOT}" \
    --out_dir "${RSNA_PROCESSED_DIR}" \
    --workers 16

# Validate processed dataset integrity (4,349 studies, ~24,033 series)
python3 phase_c/c1/scripts/validate_c1.py
```

#### 2. Phase C3: 3D ResNet-18 5-Fold Cross-Validation
Trains the baseline 3D ResNet-18 multi-series model across 5 stratified folds:
```bash
python3 phase_c/c3/scripts/train_cv_5folds.py --fold 0
```
Or distributed across 2 GPUs via PyTorch DDP:
```bash
torchrun --nproc_per_node=2 phase_c/c3/scripts/train_ddp_sanity.py
```

---

### Phase D: Deep Learning Vision Architecture & AI Calibration

#### Phase D1: 2.5D DINOv2 ViT-S/14 with Anatomical SlotHead
1. **Generate Anatomical Slot Mappings**:
   ```bash
   python3 phase_d/d1/scripts/slot_mapping_d1.py
   ```
2. **Train 5-Fold Cross-Validation**:
   ```bash
   python3 phase_d/d1/scripts/orchestrate_d1_cv.py
   ```
3. **Run Horizontal-Flip Test-Time Augmentation (TTA) & Rank Normalization**:
   ```bash
   python3 phase_d/d1/scripts/d1_2_hflip_experiment.py
   python3 phase_d/d1/scripts/d1_2_combined_tta.py
   ```

#### Phase D3: Retrained ViT on Remediated Supervision (+472 OA Cases)
```bash
# Generate candidate remediated labels
python3 phase_d/d3/scripts/generate_d3_labels.py

# Train 5 folds on remediated supervision
python3 phase_d/d3/scripts/orchestrate_d3_cv.py
```

#### Phase D6: Sequence Resolution Engine (Active Development Champion)
Deploys deterministic tissue-contrast sequence arbitration to resolve blind UID collisions and activate Slot 3 (`SAG_FLUID_NOFS`):
```bash
# Resolve series slots
python3 phase_d/d6/resolver/d6_sequence_resolver.py

# Execute full 5-fold training
python3 phase_d/d6/training/orchestrate_d6_cv.py

# Benchmark on locked Gold-58 cohort
python3 phase_d/d6/scripts/run_d6_gold58_benchmark.py
```

#### Phase D8: Hybrid Architecture Exploration (CoAtNet D8-C)
```bash
python3 phase_d/d8/training/orchestrate_d8.py
python3 phase_d/d8/scripts/run_d8_gold58_benchmark.py
```

#### Phase R/B: Representation & Bottleneck Forensics
Executes comprehensive target-level bottleneck analysis and model disagreement diagnostics:
```bash
python3 phase_r_b/run_rb_target_and_model_forensics.py
```

---

## 4. Testing & Verification Suite

Run all verification tests to validate codebase integrity:

```bash
# Unit tests for NLP parsing, clinical severity gating, and soft targets
python3 validation/unit_tests/test_remediation.py

# 40-case 8-language adversarial perturbation test suite
python3 validation/adversarial/adversarial_test.py

# Formal acceptance & scientific integrity audit
python3 validation/acceptance/audit_acceptance.py

# Phase C3 pipeline integrity verification
python3 phase_c/c3/tests/test_c3_integrity.py
```
