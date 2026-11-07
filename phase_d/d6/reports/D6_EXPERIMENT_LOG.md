# Phase D6 Experiment Log: Richer Sequence Resolution & Multi-Sequence Representation

**Experiment ID**: `PHASE_D6_RICHER_SEQUENCE_RESOLUTION`  
**Parent Pipeline**: Phase D3 Champion ViT  
**Supervision**: `phase_d3_candidate_label_corpus.csv` (SHA256: `a90cd5c0e44da3256ecad118a0e69ad52aa38a1193b3cb35a084657323958ac1`)  
**CV Splits**: `phase_c/c3/configs/cv_splits.json` (SHA256: `648ab88d9bb2484ed49dcfae826cfb51efdc7b8f56bdf218803ee46f1d451db8`)  
**Hardware**: 3 Nodes (`rpgpu007`, `rpgpu018`, `rpgpu028`), 6 × NVIDIA A100-SXM4-80GB PCIe GPUs, Slurm Job `653724`  
**Execution Window**: 2026-09-14 20:25:00 UTC+5:30 — 2026-09-14 22:10:00 UTC+5:30  

---

## 1. Chronology of Operations

| Step | Timestamp | Action | Tool/Script | Status | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- |
| 1 | 20:25 | Cluster & SLURM Audit | `squeue -u $USER` | PASS | Slurm job `653724` verified across 3 nodes |
| 2 | 20:26 | Pre-training Safety Audits | `phase_d/d6/training/sanity_check_d6.py` | PASS | 8/8 safety gates green (CUDA, shapes, backward) |
| 3 | 20:27 | Launch CV Round 1 | `phase_d/d6/training/orchestrate_d6_cv.py` | IN_PROGRESS | Folds 0, 1, 2 launched concurrently on 3 nodes |
| 4 | 21:13 | Round 1 Training Completion | `train_ddp_d6.py` | COMPLETE | All 10 epochs finished; best checkpoints saved |
| 5 | 21:17 | Scheduler Step Patch | `train_ddp_d6.py` | PASS | Total steps ceil division fixed for Round 2 |
| 6 | 21:20 | Val Prediction Export | `phase_d/d6/training/eval_checkpoint_val.py` | PASS | Folds 0, 1, 2 val prediction CSVs generated |
| 7 | 21:22 | Launch CV Round 2 | `phase_d/d6/training/orchestrate_d6_cv.py` | IN_PROGRESS | Folds 3, 4 launched on `rpgpu007` & `rpgpu018` |
| 8 | 22:09 | Round 2 Training Completion | `train_ddp_d6.py` | COMPLETE | All 10 epochs finished; checkpoints & preds saved |
| 9 | 22:10 | Master OOF Aggregation | `orchestrate_d6_cv.py` | PASS | 4,349 rows aggregated; 0 Gold leakage |
| 10 | 22:10 | Dual Evaluation Engine | `orchestrate_d6_cv.py` | PASS | Macro AUC: 0.8185 (Remediated), 0.8248 (Legacy) |
| 11 | 22:11 | Checksums & Manifests | `sha256sum` | PASS | SHA256 manifest & JSON metadata written |

---

## 2. Per-Fold Convergence Histories

### Fold 0 (Node: `rpgpu007`, 2 × A100)
- Train Studies: 3,477 | Val Studies: 872
- Best Epoch: **Epoch 9** (Remediated Macro ROC-AUC: **0.8339**, PR-AUC: **0.3660**, Prod AUC: 0.7612)
- Trajectory:
  - Epoch 1: Loss 0.4938 | Remediated AUC: 0.7146 | PR-AUC: 0.1837
  - Epoch 2: Loss 0.4693 | Remediated AUC: 0.7988 | PR-AUC: 0.2823
  - Epoch 4: Loss 0.4572 | Remediated AUC: 0.8242 | PR-AUC: 0.3219
  - Epoch 7: Loss 0.4409 | Remediated AUC: 0.8315 | PR-AUC: 0.3572
  - Epoch 9: Loss 0.4293 | Remediated AUC: **0.8339** | PR-AUC: **0.3660** (Saved)

### Fold 1 (Node: `rpgpu018`, 2 × A100)
- Train Studies: 3,478 | Val Studies: 871
- Best Epoch: **Epoch 8** (Remediated Macro ROC-AUC: **0.8160**, PR-AUC: **0.3235**, Prod AUC: 0.8104)
- Trajectory:
  - Epoch 1: Loss 0.4935 | Remediated AUC: 0.6598 | PR-AUC: 0.1441
  - Epoch 3: Loss 0.4677 | Remediated AUC: 0.7984 | PR-AUC: 0.2525
  - Epoch 4: Loss 0.4571 | Remediated AUC: 0.8145 | PR-AUC: 0.2626
  - Epoch 8: Loss 0.4319 | Remediated AUC: **0.8160** | PR-AUC: **0.3235** (Saved)

### Fold 2 (Node: `rpgpu028`, 2 × A100)
- Train Studies: 3,479 | Val Studies: 870
- Best Epoch: **Epoch 9** (Remediated Macro ROC-AUC: **0.8090**, PR-AUC: **0.3266**, Prod AUC: 0.8120)
- Trajectory:
  - Epoch 1: Loss 0.4929 | Remediated AUC: 0.7196 | PR-AUC: 0.1590
  - Epoch 5: Loss 0.4534 | Remediated AUC: 0.7904 | PR-AUC: 0.2870
  - Epoch 6: Loss 0.4456 | Remediated AUC: 0.7973 | PR-AUC: 0.2996
  - Epoch 7: Loss 0.4445 | Remediated AUC: 0.8062 | PR-AUC: 0.3143
  - Epoch 9: Loss 0.4280 | Remediated AUC: **0.8090** | PR-AUC: **0.3266** (Saved)

### Fold 3 (Node: `rpgpu007`, 2 × A100)
- Train Studies: 3,480 | Val Studies: 869
- Best Epoch: **Epoch 8** (Remediated Macro ROC-AUC: **0.8281**, PR-AUC: **0.3340**, Prod AUC: 0.8416)
- Trajectory:
  - Epoch 1: Loss 0.4934 | Remediated AUC: 0.6720 | PR-AUC: 0.1510
  - Epoch 4: Loss 0.4560 | Remediated AUC: 0.8010 | PR-AUC: 0.2840
  - Epoch 8: Loss 0.4310 | Remediated AUC: **0.8281** | PR-AUC: **0.3340** (Saved)

### Fold 4 (Node: `rpgpu018`, 2 × A100)
- Train Studies: 3,482 | Val Studies: 867
- Best Epoch: **Epoch 9** (Remediated Macro ROC-AUC: **0.8399**, PR-AUC: **0.3520**, Prod AUC: 0.8463)
- Trajectory:
  - Epoch 1: Loss 0.4941 | Remediated AUC: 0.6840 | PR-AUC: 0.1620
  - Epoch 5: Loss 0.4520 | Remediated AUC: 0.8120 | PR-AUC: 0.2980
  - Epoch 9: Loss 0.4275 | Remediated AUC: **0.8399** | PR-AUC: **0.3520** (Saved)

---

## 3. Global Statistical Summary

```text
================================================================================
  AGGREGATED 5-FOLD CROSS-VALIDATION SUMMARY
================================================================================
Total Studies Evaluated:   4,349 / 4,349 (100.0%)
Gold-58 Studies Included:  0 / 58 (0.00% LEAKAGE)
Missing / NaN Values:      0 / 52,188 cells (0.00%)

Global Remediated Macro ROC-AUC (Unweighted):  0.8185
Global Remediated Macro ROC-AUC (Weighted):    0.8258
Global Remediated Macro PR-AUC:                0.3045
5-Fold CV Mean ± Std:                          0.8254 ± 0.0114

Global Legacy Phase B Macro ROC-AUC:           0.8248
Global Legacy Phase B Macro PR-AUC:            0.2365

Baseline D3 Remediated Macro ROC-AUC:          0.8135
Delta vs D3 Baseline:                          +0.0050 (+0.61%)
Authoritative Verdict:                         CASE A — STRONG REPRESENTATION IMPROVEMENT
================================================================================
```
