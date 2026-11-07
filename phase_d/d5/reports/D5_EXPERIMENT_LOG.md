# Phase D5: Complete Experiment & Execution Log

---

## Experiment 1: Checkpoint Loading & State-Dict Verification
- **Timestamp**: 2026-09-14 18:49:24 IST
- **Command**: `python3 -c "import torch; from phase_d.d1.scripts.model_d1 import PhaseD1Model; model = PhaseD1Model(); ckpt = torch.load('/scratch/deepanthpcu/RSNA_Code/phase_d/d3/checkpoints/fold_0/best_model.pt', map_location='cpu'); print('Load result:', model.load_state_dict(ckpt['model_state_dict']))"`
- **Hostname**: `rpgpu007`
- **GPU Configuration**: CPU verification pass
- **Input Checkpoint**: `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/checkpoints/fold_0/best_model.pt` (SHA256: `52eaf45a0934c6d86a32a34f0fa8ef1fac2a8713363b484c7c735bb1b186d1cc`)
- **Experiment ID**: `D5-EXP-01-CKPT-VERIFY`
- **Configuration**: Standard `PhaseD1Model` instantiation with frozen blocks 0–5, trainable blocks 6–11 + SlotHead.
- **Output**: `<All keys matched successfully>`
- **Metric**: N/A (structural integrity check)
- **Decision**: PASS. Confirmed D3 checkpoints match `PhaseD1Model` architecture exactly.

---

## Experiment 2: Multiprocessing Spawn Verification (Interactive Subprocess Test)
- **Timestamp**: 2026-09-14 18:52:24 IST
- **Command**: `python3 -c "import torch; from multiprocessing import get_context; ..."`
- **Hostname**: `rpgpu007`
- **GPU Configuration**: 2 × NVIDIA A100-PCIE-80GB (PCIe Bus 3B:00.0, 5E:00.0)
- **Input Checkpoint**: N/A (inter-process synchronization check)
- **Experiment ID**: `D5-EXP-02-MP-SPAWN-TEST`
- **Configuration**: Tested `spawn` start method in an interactive one-liner.
- **Output**: `AttributeError: Can't get attribute 'worker' on <module '__main__' (built-in)>`
- **Metric**: FAILED (interactive Python `-c` execution limitation for pickle serialization).
- **Decision**: Refactored worker execution into standalone script `run_d5_inference.py` using standard multiprocessing `Process` and `Queue` with module-level top-level functions.

---

## Experiment 3: Full 5-Fold Dual-Stream Inference Run (Standard + Horizontal Flip TTA)
- **Timestamp**: 2026-09-14 18:53:06 IST – 18:58:16 IST
- **Command**: `python3 /scratch/deepanthpcu/RSNA_Code/phase_d/d5/scripts/run_d5_inference.py 2>&1 | tee /scratch/deepanthpcu/RSNA_Code/phase_d/d5/reports/d5_execution.log`
- **Hostname**: `rpgpu007` (Head node of SLURM Job `653724`)
- **GPU Configuration**: 2 × NVIDIA A100-PCIE-80GB
  - Worker 0: GPU 0 (Folds 0, 1)
  - Worker 1: GPU 1 (Folds 2, 3, 4)
- **Input Checkpoints**:
  - Fold 0: `phase_d/d3/checkpoints/fold_0/best_model.pt`
  - Fold 1: `phase_d/d3/checkpoints/fold_1/best_model.pt`
  - Fold 2: `phase_d/d3/checkpoints/fold_2/best_model.pt`
  - Fold 3: `phase_d/d3/checkpoints/fold_3/best_model.pt`
  - Fold 4: `phase_d/d3/checkpoints/fold_4/best_model.pt`
- **Experiment ID**: `D5-EXP-03-FULL-5FOLD-INFERENCE`
- **Configuration**:
  - Microbatch: 4 studies per GPU
  - DataLoader workers: 4 per GPU
  - Slice sampling: 12 slice groups per slot, central depth band $(0.20, 0.80)$
  - Resolution: $224 \times 224$, 130-mm FOV
  - Precision: BFloat16 forward inference with Float32 NumPy probability casting
- **Fold Execution Runtime & Baseline Consistency**:
  - Fold 0: 872 studies in 108.3s | diff vs saved: max=0.000000, mean=0.000000
  - Fold 1: 871 studies in 92.3s  | diff vs saved: max=0.000000, mean=0.000000
  - Fold 2: 870 studies in 108.4s | diff vs saved: max=0.000000, mean=0.000000
  - Fold 3: 869 studies in 92.1s  | diff vs saved: max=0.000000, mean=0.000000
  - Fold 4: 867 studies in 91.8s  | diff vs saved: max=0.000000, mean=0.000000
  - Total inference runtime: 298.6s (4.98 minutes)
- **Output Files**:
  - `phase_d/d5/predictions/d5_standard_oof_predictions.csv` (SHA256: `82c4e90830c17117da98c6eace43cafed24038daeb4f6cfa0a3be793b4082881`)
  - `phase_d/d5/predictions/d5_flip_tta_oof_predictions.csv` (SHA256: `b965fc34be6056923f444fcdb63de04ea56c81d5511bfd1cb147dff49656aed7`)
  - `phase_d/d5/predictions/d5_ranknorm_oof_predictions.csv` (SHA256: `c9e4971e604f444f905dadbb1a07211b961928523938115f18a190424823d1a6`)
  - `phase_d/d5/predictions/d5_b2_oof_predictions.csv` (SHA256: `30c774a28be316bb291e7964d81da83bd61904adb219d42166a1687a1d3dd41e`)
  - `phase_d/d5/predictions/d5_b2_alt_rank_avg_oof_predictions.csv` (SHA256: `b36b02c16882b6f29de0fae0578375aa363e7470e3a56c09b71cbd2c1716f624`)
  - `phase_d/d5/predictions/d5_diagnostic_flip_alone_oof_predictions.csv` (SHA256: `8b31815971035b91459f0843fb1d2897338002ddeed49fc3ba1efe26ac7ad612`)
- **Evaluated Metrics**:
  - `D5-A_STANDARD`: Remediated AUC = 0.8135 | Remediated Wtd = 0.8204 | Remediated PR = 0.3080 | Legacy AUC = 0.8184 | Legacy PR = 0.2319
  - `D5-B_FLIP_TTA`: Remediated AUC = 0.8143 (+0.0008) | Remediated PR = 0.3086 | Legacy AUC = 0.8171 (-0.0013)
  - `D5-C_RANKNORM`: Remediated AUC = 0.8147 (+0.0012) | Remediated PR = 0.3141 | Legacy AUC = 0.8188 (+0.0004)
  - `D5-D_B2_PRIMARY`: Remediated AUC = 0.8154 (+0.0019) | Remediated PR = 0.3149 (+0.0069) | Legacy AUC = 0.8162 (-0.0022)
  - `D5-D_B2_ALT`: Remediated AUC = 0.8149 (+0.0014) | Remediated PR = 0.3149 | Legacy AUC = 0.8160 (-0.0024)
  - `D5-DIAG_FLIP`: Remediated AUC = 0.8102 (-0.0033) | Remediated PR = 0.3021 | Legacy AUC = 0.8108 (-0.0076)
- **Decision**: Primary candidate D5-B2 achieves marginal gain on Remediated labels (+0.0019 / +0.23% ROC-AUC, +2.24% PR-AUC) with slight trade-off on Legacy labels (-0.0022 / -0.27%). Classified as **CASE B — MARGINAL TRANSFER**. Standard D3 ViT retained as authoritative reference model; D5-B2 retained as audited secondary calibrated candidate.
