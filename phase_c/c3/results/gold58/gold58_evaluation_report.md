# Phase C3.12 Stage B: Locked Gold-58 Evaluation Report

**Date / Timestamp:** 2026-09-12 17:25:13  
**Host:** `rpgpu018`  
**Environment:** Conda `ML_DFT_V2` | Python `3.11.16` | PyTorch `2.7.0+cu126`  
**Hardware:** NVIDIA A100 80GB PCIe  
**Status:** **LOCKED EVALUATION COMPLETE — 100% PASS**  

---

## 1. Executive Summary

The **Phase C3.12 Stage B Locked Gold-58 Evaluation** has been executed using the **five frozen C3.11 production checkpoints** on the held-out 58-study Gold cohort.
In strict adherence to protocol:
- **0 Gold-58 studies** were ever exposed to training, validation, or checkpoint selection.
- **No fine-tuning, threshold tuning, or fold weighting** was performed on Gold-58 data.
- The ensemble was formed using the **pre-specified normalized rank averaging** procedure across all 5 models.

### Key Global Metrics Comparison

| Evaluation Stage | Scope | Cohort Size | Macro ROC-AUC | Note |
| :--- | :--- | :---: | :---: | :--- |
| **C3.11 Production CV** | Development Mean Fold Val | 869.8 / fold | **0.6333 ± 0.0326** | Fold-level CV development baseline |
| **C3.12 Stage A OOF** | Global Development OOF | 4,349 studies | **0.6190** | Global study-level non-Gold development OOF |
| **C3.12 Stage B Gold-58** | **Locked Rank Ensemble** | **58 studies** | **0.5986** | **Official locked out-of-sample Gold evaluation** |
| *C3.12 Stage B Gold-58* | *Locked Score Ensemble* | *58 studies* | *0.5985* | *Reference simple score average* |

---

## 2. Locked Gold-58 Per-Target Performance (Rank-Averaged Ensemble)

| Target Name | True Positives | Prevalence | ROC-AUC | PR-AUC | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 24/58 | 41.4% | **0.5411** | 0.5256 | 0.4815 | 0.5172 |
| **MCL** | 9/58 | 15.5% | **0.5782** | 0.2264 | 0.2500 | 0.4828 |
| **Medial Meniscus** | 26/58 | 44.8% | **0.6647** | 0.6550 | 0.6316 | 0.6379 |
| **Lateral Meniscus** | 23/58 | 39.7% | **0.5925** | 0.4938 | 0.5185 | 0.5517 |
| **Medial OA** | 15/58 | 25.9% | **0.5752** | 0.2953 | 0.3830 | 0.5000 |
| **Lateral OA** | 11/58 | 19.0% | **0.6064** | 0.2557 | 0.4186 | 0.5690 |
| **PF OA** | 21/58 | 36.2% | **0.5347** | 0.3829 | 0.4314 | 0.5000 |
| **Effusion** | 35/58 | 60.3% | **0.7130** | 0.7517 | 0.6562 | 0.6207 |
| **Synovitis** | 27/58 | 46.6% | **0.6583** | 0.6316 | 0.5455 | 0.5690 |
| **Baker's** | 12/58 | 20.7% | **0.6105** | 0.2898 | 0.4091 | 0.5517 |
| **Contusion** | 19/58 | 32.8% | **0.4764** | 0.3804 | 0.3830 | 0.5000 |
| **Fracture** | 18/58 | 31.0% | **0.6319** | 0.5431 | 0.5000 | 0.5862 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MACRO AVERAGE** | **--** | **--** | **0.5986** | **0.4526** | **0.4674** | **0.5489** |

---

## 3. Per-Checkpoint Gold Performance

| Model Checkpoint | Macro ROC-AUC | Macro PR-AUC |
| :--- | :---: | :---: |
| **Fold 0 Checkpoint** | 0.5935 | 0.4476 |
| **Fold 1 Checkpoint** | 0.5977 | 0.4648 |
| **Fold 2 Checkpoint** | 0.5984 | 0.4489 |
| **Fold 3 Checkpoint** | 0.5827 | 0.4443 |
| **Fold 4 Checkpoint** | 0.5908 | 0.4399 |
| **Predetermined Rank Ensemble** | **0.5986** | **0.4526** |

---

## 4. Integrity and Leakage Audit

- **Gold Studies Evaluated:** `58` (Expected: 58)
- **Gold Studies Duplicated:** `0`
- **Gold Studies Missing:** `0`
- **Development Studies Accidentally Included:** `0`
- **Gold Labels Used in Training:** `False`
- **Gold Labels Used for Threshold/Model Selection:** `False`
- **Pre-specified Rank Averaging Applied:** `True`
- **NaN / Inf Predictions:** `0`
- **Audit Verdict:** **PASS - 100% UNCOMPROMISED LOCKED EVALUATION**

---

## 5. Artifact Inventory

| Artifact Description | File Path |
| :--- | :--- |
| Rank-Averaged Ensemble Predictions | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/results/gold58/gold58_rank_averaged_predictions.csv` |
| Score-Averaged Ensemble Predictions | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/results/gold58/gold58_score_averaged_predictions.csv` |
| Integrity Audit JSON | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/results/gold58/gold58_integrity_audit.json` |
| Comprehensive Evaluation Summary JSON | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/results/gold58/gold58_evaluation_summary.json` |
| Evaluation Markdown Report | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/results/gold58/gold58_evaluation_report.md` |
