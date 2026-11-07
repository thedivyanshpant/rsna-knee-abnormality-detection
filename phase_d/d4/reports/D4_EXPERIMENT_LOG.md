# Phase D4: Master Experiment Registry & Provenance Log

**Document Type:** Engineering Provenance & Execution Log  
**Project:** RSNA Knee Abnormality Detection  
**Phase:** Phase D4 (Controlled Multi-Arm ViT–CNN Continuation on Remediated Supervision)  
**Host Cluster:** 3-Node HPC GPU Cluster (`rpgpu007`, `rpgpu018`, `rpgpu028`) | SLURM Job ID: `653724`  
**Governance Standard:** Gold-58 Permanently Locked (`0.6174`, 0 Access / 0 Leakage) | Phase B Corpus Frozen (`bd064fc8...`) | SLURM Safety (`scancel` Prohibited)  

---

## 1. Scientific Objective & Governance Constraints

### 1.1 Core Hypothesis & Purpose
Phase D4 tests whether the ViT–CNN representation complementarity established in Phase D2 (where a 75/25 ViT–CNN within-fold rank blend improved Macro ROC-AUC from `0.7592` to **`0.7634`**, $+0.55\%$ gain, $+4.09\%$ PR-AUC) persists after both arms are trained on the experimentally remediated Phase D3 weak supervision corpus.

**Central Question:** Does a domain-specific CNN (RadImageNet ResNet-50) trained on remediated osteoarthritis supervision add complementary diagnostic signal to a Vision Transformer (DINOv2 ViT-S/14) trained on the same remediated supervision?

### 1.2 Absolute Governance Constraints
1. **Gold-58 Permanently Locked:** Historical locked benchmark remains **`0.6174 Macro ROC-AUC`**. Strictly zero access, zero loading, zero tuning, zero calibration, zero leakage during Phase D4. Audit verified `gold_leakage: 0`.
2. **Production Corpus Frozen:** `labels/production/phase_b_step3_remediated_training_corpus.csv` remains strictly immutable.
3. **Supervision Corpus:** Exclusively `phase_d/d3/labels/phase_d3_candidate_label_corpus.csv`.
4. **Frozen CV Splits:** Exactly the 4,349 development studies partitioned in `phase_c/c3/configs/cv_splits.json`. Zero split regeneration.
5. **Artifact Immutability:** Phases C, D1, D2, and D3 checkpoints, predictions, and reports remain strictly untouched.
6. **No Arbitrary Searches:** Exactly two pre-specified controlled ensembles tested:
   - Primary: **D4-C1** (75% D3 ViT + 25% D4 CNN within-fold rank blend)
   - Secondary Control: **D4-B1** (50% D3 ViT + 50% D4 CNN within-fold rank blend)
7. **Cluster Safety:** Zero `scancel` or cluster job termination under any circumstances.
8. **Hard Stop:** Final report produced upon completion; no D5 or subsequent phase launched without explicit authorization.

---

## 2. Cluster Allocation & Multi-Node Topology

- **SLURM Job ID:** `653724`
- **Node Allocation (3 nodes):**
  - Node 0: `rpgpu007` (Master / Head, IP: `172.5.3.7`, Ranks 0, 1)
  - Node 1: `rpgpu018` (Worker 1, IP: `172.5.3.18`, Ranks 2, 3)
  - Node 2: `rpgpu028` (Worker 2, IP: `172.5.3.28`, Ranks 4, 5)
- **GPU Inventory:** 2 × NVIDIA A100-PCIE-80GB per node = **6 × NVIDIA A100-PCIE-80GB GPUs total**
- **Interconnect / Subsystem:** Multi-node NCCL DDP over high-speed `eth2` network interface (`NCCL_SOCKET_IFNAME=eth2`, `GLOO_SOCKET_IFNAME=eth2`)
- **Master Communication Address:** `rpgpu007:29530`
- **6-GPU Smoke Test:** Executed on 2026-09-14 14:47:44 IST; all-reduce sum $= 6.0$ across 6 processes verified.

---

## 3. Pre-Training Cryptographic Provenance Audit

| Artifact Name | Exact Absolute Path | SHA256 Hash |
| :--- | :--- | :--- |
| **D3 Candidate Corpus** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/labels/phase_d3_candidate_label_corpus.csv` | `a90cd5c0e44da3256ecad118a0e69ad52aa38a1193b3cb35a084657323958ac1` |
| **Frozen CV Splits** | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/configs/cv_splits.json` | `648ab88d9bb2484ed49dcfae826cfb51efdc7b8f56bdf218803ee46f1d451db8` |
| **Phase B Corpus (Frozen)** | `/scratch/deepanthpcu/RSNA_Code/labels/production/phase_b_step3_remediated_training_corpus.csv` | `bd064fc882ee9c344fc46adedaef2e592071ab032a632bcba9e2fc699a741c30` |
| **D3 ViT Global OOF** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/predictions/d3_global_oof_predictions.csv` | `348d2512bee28e1ca7b984d0f5eb82a21a55ad1c2658ae9ca6172591da20aad4` |
| **D2 CNN Global OOF** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d2/predictions/oof/d2_global_oof_predictions.csv` | `ba7174dbc22121a9bc1c9b65b511341127cc01b25e30a2b3a75a465c1b050a78` |
| **RadImageNet Weights** | `/home/deepanthpcu/.cache/huggingface/hub/models--Lab-Rasool--RadImageNet/.../ResNet50.pt` | `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734` |

---

## 4. Experiment D4-A: Multi-Node 5-Fold Training on Remediated Supervision

- **Timestamp:** 2026-09-14 14:50:00 IST – 2026-09-14 17:01:00 IST
- **Experiment ID:** `D4_A_RADIMAGENET_CNN_REMEDIATED_SUPERVISION`
- **Execution Command:**
  ```bash
  python3 -u /scratch/deepanthpcu/RSNA_Code/phase_d/d4/scripts/orchestrate_d4_cv.py \
      --data_dir /scratch/deepanthpcu/RSNA_Code/data/processed/c1 \
      --labels_csv /scratch/deepanthpcu/RSNA_Code/phase_d/d3/labels/phase_d3_candidate_label_corpus.csv \
      --legacy_labels_csv /scratch/deepanthpcu/RSNA_Code/labels/production/phase_b_step3_remediated_training_corpus.csv \
      --splits_json /scratch/deepanthpcu/RSNA_Code/phase_c/c3/configs/cv_splits.json \
      --radimagenet_ckpt /home/deepanthpcu/.cache/huggingface/hub/models--Lab-Rasool--RadImageNet/snapshots/14460ee4c1276f6925611a63aa9a54a05d39eae0/ResNet50.pt \
      --output_dir /scratch/deepanthpcu/RSNA_Code/phase_d/d4 \
      --epochs 10 \
      --batch_size 1 \
      --backbone_lr 1e-4 \
      --head_lr 1e-3 \
      --weight_decay 0.02
  ```
- **Launcher Implementation:** Multi-node launch via `torchrun` on `rpgpu007`:
  ```bash
  torchrun \
      --nnodes=3 \
      --nproc_per_node=2 \
      --rdzv_id=653724 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=rpgpu007:29530 \
      /scratch/deepanthpcu/RSNA_Code/phase_d/d4/scripts/train_ddp_d4.py ...
  ```
- **Environment Variables Set:**
  ```bash
  export NCCL_SOCKET_IFNAME=eth2
  export GLOO_SOCKET_IFNAME=eth2
  export NCCL_DEBUG=INFO
  export OMP_NUM_THREADS=4
  ```
- **Model Architecture & Hyperparameters:**
  - Architecture: `RadImageNetResNet50SlotHead` (Identical parameter layout to D2).
  - Frozen Parameters: 8,131,328 (`conv1`, `bn1`, `layer1`, `layer2`).
  - Trainable Parameters: 16,043,084 (`layer3`, `layer4`, `slot_proj`, `SlotHead`).
  - Total Parameters: 24,174,412.
  - Optimizer: AdamW (`backbone_lr = 1e-4`, `head_lr = 1e-3`, `weight_decay = 0.02`).
  - Scheduler: OneCycleLR (`pct_start = 0.15`, 10 epochs, cosine annealing).
  - Effective Batch Size: 1 study/GPU × 6 GPUs = 6 studies per step.
  - Precision: BFloat16 forward with explicit Float32 promotion for validation NumPy array conversions.
  - Loss: `weighted_soft_bce_loss`.
- **Engineering Bug Recovery:**
  - Encountered `TypeError: Got unsupported ScalarType BFloat16` during initial validation predictions in NumPy.
  - Implemented explicit `.float().cpu().numpy()` casting in `train_ddp_d4.py` and `orchestrate_d4_cv.py`.
  - Re-run completed cleanly without errors.

### Per-Fold Execution Profile (7,542 Seconds Total Runtime)
- **Fold 0 (872 studies):** Completed in 1513s. Best Val Macro AUC: `0.6909` (Remediated) / `0.6466` (Legacy).
- **Fold 1 (871 studies):** Completed in 1508s. Best Val Macro AUC: `0.6688` (Remediated) / `0.7076` (Legacy).
- **Fold 2 (870 studies):** Completed in 1506s. Best Val Macro AUC: `0.6763` (Remediated) / `0.6785` (Legacy).
- **Fold 3 (869 studies):** Completed in 1504s. Best Val Macro AUC: `0.6979` (Remediated) / `0.7057` (Legacy).
- **Fold 4 (867 studies):** Completed in 1513s. Best Val Macro AUC: `0.7149` (Remediated) / `0.7258` (Legacy).

### Output Artifacts & Checkpoint Hashes:
- Checkpoint Fold 0: `phase_d/d4/checkpoints/fold_0/best_model.pt` | SHA256: `546260b0a9b6b654fa6bb49be8245b43c2773ca8fb5f226f46b5d263ec97e832`
- Checkpoint Fold 1: `phase_d/d4/checkpoints/fold_1/best_model.pt` | SHA256: `eb646ea397a416600642ad87cce6c4ad2b95c0120ce7f23fef0eecaf101ecb6e`
- Checkpoint Fold 2: `phase_d/d4/checkpoints/fold_2/best_model.pt` | SHA256: `94d13febe6e69b9033ff592cb666110463e9db5838bf99df8fe95adaee22352f`
- Checkpoint Fold 3: `phase_d/d4/checkpoints/fold_3/best_model.pt` | SHA256: `de7477bb5dcccb2a5ba1107d54b620be384483dfbad2430e783d1eee00bd62ae`
- Checkpoint Fold 4: `phase_d/d4/checkpoints/fold_4/best_model.pt` | SHA256: `066436a24e14d21b9042d829b76e87348885547ca100d29ebd0f85050eb106bf`
- Standalone D4 OOF Predictions: `phase_d/d4/predictions/oof/d4_cnn_oof_predictions.csv` | SHA256: `61fc8738b8d0029217f98a75661c7180f01f7a176eaf1cc260bd7330d5283b81`
- Evaluation Summary JSON: `phase_d/d4/reports/d4_oof_evaluation_summary.json`

---

## 5. Experiment D4-B & D4-C: Controlled Multi-Arm Ensembling

- **Timestamp:** 2026-09-14 17:01:30 IST
- **Experiment IDs:** `D4_C1_WEIGHT_75_25` (Primary), `D4_B1_WEIGHT_50_50` (Secondary)
- **Phase:** D4 (Controlled Ensembling)
- **Inputs:**
  - Frozen D3 ViT OOF: `phase_d/d3/predictions/d3_global_oof_predictions.csv` (`348d2512...`)
  - Standalone D4 CNN OOF: `phase_d/d4/predictions/oof/d4_cnn_oof_predictions.csv` (`61fc8738...`)
- **Ensemble Evaluations:**
  1. **Primary Blend `D4-C1` (75% D3 ViT + 25% D4 CNN Within-Fold Rank Blend):**
     - Evaluation against Remediated D3 GT:
       - Macro ROC-AUC (Unweighted): **`0.8071`** (Weighted: `0.8141` | PR-AUC: `0.2870`)
       - Δ vs D3 Standalone ViT (`0.8135`): **`-0.0064`** (`-0.79%` relative drop)
       - PR-AUC Δ vs D3 Standalone ViT (`0.3080`): **`-0.0210`** (`-6.82%` relative drop)
     - Evaluation against Legacy Phase B GT:
       - Macro ROC-AUC (Unweighted): **`0.8169`** (Weighted: `0.8239` | PR-AUC: `0.2544`)
       - Δ vs D3 Standalone ViT (`0.8184`): **`-0.0015`** (`-0.18%` relative drop)
     - Output Artifact: `phase_d/d4/predictions/blends/d4_ensemble_predictions.csv` | SHA256: `4a7eed17754ac945dc04a824926f5c748c0cef876ea24390dbfd29e8a20e30c5`
     - Decision: **CASE C — REJECTED**.
  2. **Secondary Blend `D4-B1` (50% D3 ViT + 50% D4 CNN Within-Fold Rank Blend):**
     - Evaluation against Remediated D3 GT:
       - Macro ROC-AUC (Unweighted): **`0.7815`** (Weighted: `0.7879` | PR-AUC: `0.2490`)
       - Δ vs D3 Standalone ViT (`0.8135`): **`-0.0320`** (`-3.93%` relative drop)
     - Evaluation against Legacy Phase B GT:
       - Macro ROC-AUC (Unweighted): **`0.7978`** (Weighted: `0.8043` | PR-AUC: `0.2725`)
       - Δ vs D3 Standalone ViT (`0.8184`): **`-0.0206`** (`-2.52%` relative drop)
     - Decision: **CASE C — REJECTED**.

---

## 6. Formal Decision & Governance Sign-Off

- **Formal Scientific Verdict:** **CASE C — NO COMPLEMENTARITY**.
  - The domain-specific CNN does NOT add complementary diagnostic signal to the Vision Transformer under remediated supervision.
  - The 12.71% standalone performance gap (ViT `0.8135` vs CNN `0.6864`) causes the CNN arm to dilute and corrupt the ViT's superior representation.
  - Standalone Phase D3 ViT remains the definitive uncompromised champion.
- **Gold-58 Status:** Permanently locked. Strictly 0 Gold-58 studies accessed, loaded, or evaluated.
- **Phase B Production Corpus:** Untouched and immutable.
- **Execution Status:** **100% COMPLETE & PRODUCTION AUDITED**. Hard stop observed.
