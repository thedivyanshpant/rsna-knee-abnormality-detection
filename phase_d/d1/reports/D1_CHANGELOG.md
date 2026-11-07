# Phase D — AI Calibration: Master Change & Provenance Log

**Document Type:** Master Cryptographic & Architecture Change Log  
**Project:** RSNA Knee Abnormality Detection  
**Phases Covered:** D1.0, D1.1, D1.2, Stage B (Gold-58), D2, D3, D4  
**Cohort:** 4,349 Development Studies (Frozen 5-Fold Cross-Validation, `phase_c/c3/configs/cv_splits.json`)  
**Governance Standard:** Gold-58 Permanently Locked (`0.6174`, 0 Touches) | Phase B Corpus Frozen (`bd064fc8...`) | SLURM Safety (`scancel` Prohibited)  

---

## 1. Master Cryptographic Provenance Registry

### 1.1 Python Source Scripts (20 Core Engines)

| Script Path | Purpose / Phase | SHA256 Hash |
| :--- | :--- | :--- |
| `phase_d/d1/scripts/slot_mapping_d1.py` | 6-slot DICOM classifier & collision resolver | `94eb49f81746aa10bdb5cb592984826de92d1731d5698d39c771ad26833164b2` |
| `phase_d/d1/scripts/dataset_d1.py` | 2.5D triplet slice extractor & normalizer | `c0159486e1cefe4ec06d519426847b2da0c5ac5564d2e842777d07c449721b7a` |
| `phase_d/d1/scripts/model_d1.py` | DINOv2 ViT-S/14 + Anatomical SlotHead | `d12985d6f6ed92cd3a26baf34813deba174ad7e72db25cbe21f7ff18d9f05742` |
| `phase_d/d1/scripts/loss_d1.py` | Weighted soft-BCE loss module | `afac08a3208a0165a034b8a61f57ab45794dcf3a06909d4033046570d1843426` |
| `phase_d/d1/scripts/train_ddp_d1.py` | Multi-GPU NCCL DDP training engine (D1.1) | `0bcb927cacb0c7a48bbefadaf6097f6aba4ec1260f0164d78d4055f5b2929d31` |
| `phase_d/d1/scripts/orchestrate_d1_cv.py` | 5-fold master orchestrator (D1.1) | `9c353b370f5c20021af3ca049d5b181ae8d5186c38eaa74d6502309db0682a1b` |
| `phase_d/d1/scripts/aggregate_d1_oof.py` | Global 5-fold OOF prediction aggregator | `ebd5cd7caf8586b3d36d1558e1177adf493599b8eac435b10f0b7c4ba46fcd30` |
| `phase_d/d1/scripts/d1_2_inference_engine.py`| Multi-TTA inference execution engine | `22cdaf418b65efd5fdb5daeaf43b725b0657868bea5984dd56296713747afa32` |
| `phase_d/d1/scripts/d1_2_eval_utils.py` | Governance audit & PR-AUC evaluation toolkit | `8ce5c881141e324f9664b5eb65d0b52c532aa29d3e5dce1e64b122ddbc55a300` |
| `phase_d/d1/scripts/d1_2_blend_experiments.py`| Rank normalization & ensembling engine | `fc2265d88d2011eab1ddb22ecf68c48555b62ee63e970853aa571c3f04d2225c` |
| `phase_d/d1/scripts/d1_gold58_benchmark.py` | Standalone Gold-58 evaluation engine | `afa81bff6bb6c3f22df3d59fc6f6d8dffc1b88741a7048c30664ebd32100b492` |
| `phase_d/d2/scripts/model_d2_cnn.py` | RadImageNet ResNet-50 + SlotHead (D2) | `cbd2e34cc010f6e4201f4af3ce5af9c8c994f0380a7c28e4737f9effc64252d0` |
| `phase_d/d2/scripts/train_ddp_d2.py` | 2 × A100 NCCL DDP training engine (D2) | `3ade6d55e7df201fc1c2586bdf8f8e353e4b38d749b63a7a8dc0d77dfa0950f7` |
| `phase_d/d2/scripts/orchestrate_d2_cv.py` | 5-fold CV orchestrator & aggregator (D2) | `728b51063ae5fd6ce03f10a3d0228c067e8422e11592587affee5fae04c1111e` |
| `phase_d/d3/scripts/generate_d3_labels.py` | Candidate label remediation NLP engine (D3) | `c3fa81e162a6dbcc36ad3d9cdc6754f0c0ee0eeb24feba4fbc651cc448d8e9cb` |
| `phase_d/d3/scripts/train_ddp_d3.py` | Retraining engine on remediated labels (D3) | `df6a92414d86f3d4ee2bf5523ff600da23fd026a8e41ce032e5086a0b1cf761b` |
| `phase_d/d3/scripts/orchestrate_d3_cv.py` | 5-fold CV dual-standard evaluator (D3) | `44f5d541e23c268be4e2a470f8f262930f123c375cceec961ed974b0ec84ca62` |
| `phase_d/d4/scripts/model_d4_cnn.py` | RadImageNet ResNet-50 SlotHead (D4) | `c6d2169e6effe2a119e795fbf4ae3ed9dbf1a39b9d32c0968bce0da5db6e1183` |
| `phase_d/d4/scripts/train_ddp_d4.py` | 3-node 6-GPU multi-node DDP engine (D4) | `1e5be57fd6e82ab481e3d5e1485016b729f36f1251c94f61803f8bf01dc9942f` |
| `phase_d/d4/scripts/orchestrate_d4_cv.py` | Multi-node 5-fold orchestrator & blender (D4)| `75a897f6295943d011784759890d6aacbb1fc3826443245244cdca36ef77192d` |

### 1.2 Model Checkpoints (20 Trained Fold Checkpoints)

| Phase & Fold | Model Architecture | Supervision Source | Checkpoint Path | SHA256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| **D1.1 Fold 0** | DINOv2 ViT-S/14 SlotHead | Phase B Legacy | `phase_d/d1/checkpoints/fold_0/best_model.pt` | `7a8e0dc4f04da352ba6290e63ac7446315fd498b3557c5f35c561687f920e07f` |
| **D1.1 Fold 1** | DINOv2 ViT-S/14 SlotHead | Phase B Legacy | `phase_d/d1/checkpoints/fold_1/best_model.pt` | `7151140eab4000f22061868b41c5bbfe72eb6804c66dadf205e803a3e528cd09` |
| **D1.1 Fold 2** | DINOv2 ViT-S/14 SlotHead | Phase B Legacy | `phase_d/d1/checkpoints/fold_2/best_model.pt` | `28331fca01b232f4629660441c7e7db95a242eba4061d484726b9a0b924ff4f6` |
| **D1.1 Fold 3** | DINOv2 ViT-S/14 SlotHead | Phase B Legacy | `phase_d/d1/checkpoints/fold_3/best_model.pt` | `55220321d615c959c8735238d5dc14cf254495a1088a952a7fc96fec659da60d` |
| **D1.1 Fold 4** | DINOv2 ViT-S/14 SlotHead | Phase B Legacy | `phase_d/d1/checkpoints/fold_4/best_model.pt` | `7253b1069a4efea902d674d9e60b396780c1e010391a609179c1be726c34c382` |
| **D2 Fold 0** | RadImageNet ResNet-50 | Phase B Legacy | `phase_d/d2/checkpoints/fold_0/best_model.pt` | `aeec34cce68a4e8df86a3d324341d3183b84a1411d74f4ff5d69d9c9e1de1c21` |
| **D2 Fold 1** | RadImageNet ResNet-50 | Phase B Legacy | `phase_d/d2/checkpoints/fold_1/best_model.pt` | `84f3c8ffd0ed2f600798d03bfcd57ca5d82ba467dfb9ae9d6a19380c1834a812` |
| **D2 Fold 2** | RadImageNet ResNet-50 | Phase B Legacy | `phase_d/d2/checkpoints/fold_2/best_model.pt` | `5d0fd964d65abed7852d0a47595fd62b09fd524a116ce6da10c386212c8ca850` |
| **D2 Fold 3** | RadImageNet ResNet-50 | Phase B Legacy | `phase_d/d2/checkpoints/fold_3/best_model.pt` | `3fd22076346e7fbba39a2aae61b5063c12dbd4e1aee89754589c3267009a454a` |
| **D2 Fold 4** | RadImageNet ResNet-50 | Phase B Legacy | `phase_d/d2/checkpoints/fold_4/best_model.pt` | `f3a01e50e6c05bee7ced31adf79ae18e3c0e372df251e8612bb9190165e8c49f` |
| **D3 Fold 0** | DINOv2 ViT-S/14 SlotHead | Remediated D3 | `phase_d/d3/checkpoints/fold_0/best_model.pt` | `52eaf45a0934c6d86a32a34f0fa8ef1fac2a8713363b484c7c735bb1b186d1cc` |
| **D3 Fold 1** | DINOv2 ViT-S/14 SlotHead | Remediated D3 | `phase_d/d3/checkpoints/fold_1/best_model.pt` | `f600d29f8b0cbfb9d0c32ba9e3f736b0289e15725a3eaf45d447d19a2211ef5b` |
| **D3 Fold 2** | DINOv2 ViT-S/14 SlotHead | Remediated D3 | `phase_d/d3/checkpoints/fold_2/best_model.pt` | `cf1cdea350850aea17a878220e7a5cf12dd271b6deff9e8437dec84669f0914c` |
| **D3 Fold 3** | DINOv2 ViT-S/14 SlotHead | Remediated D3 | `phase_d/d3/checkpoints/fold_3/best_model.pt` | `b48ce28a0442723565795739eed3b0fe51e9d79b925c95eb51ecbc9feebc6cff` |
| **D3 Fold 4** | DINOv2 ViT-S/14 SlotHead | Remediated D3 | `phase_d/d3/checkpoints/fold_4/best_model.pt` | `9d29143c6d6f43bf6f4745044ec4997bc696bb95a218492e9ac209f2a068ce3b` |
| **D4 Fold 0** | RadImageNet ResNet-50 | Remediated D3 | `phase_d/d4/checkpoints/fold_0/best_model.pt` | `546260b0a9b6b654fa6bb49be8245b43c2773ca8fb5f226f46b5d263ec97e832` |
| **D4 Fold 1** | RadImageNet ResNet-50 | Remediated D3 | `phase_d/d4/checkpoints/fold_1/best_model.pt` | `eb646ea397a416600642ad87cce6c4ad2b95c0120ce7f23fef0eecaf101ecb6e` |
| **D4 Fold 2** | RadImageNet ResNet-50 | Remediated D3 | `phase_d/d4/checkpoints/fold_2/best_model.pt` | `94d13febe6e69b9033ff592cb666110463e9db5838bf99df8fe95adaee22352f` |
| **D4 Fold 3** | RadImageNet ResNet-50 | Remediated D3 | `phase_d/d4/checkpoints/fold_3/best_model.pt` | `de7477bb5dcccb2a5ba1107d54b620be384483dfbad2430e783d1eee00bd62ae` |
| **D4 Fold 4** | RadImageNet ResNet-50 | Remediated D3 | `phase_d/d4/checkpoints/fold_4/best_model.pt` | `066436a24e14d21b9042d829b76e87348885547ca100d29ebd0f85050eb106bf` |

### 1.3 Datasets, Manifests & Global Out-of-Fold Prediction Artifacts

| Description | Exact Absolute Path | SHA256 Hash |
| :--- | :--- | :--- |
| **Phase B Production Corpus (Frozen)** | `/scratch/deepanthpcu/RSNA_Code/labels/production/phase_b_step3_remediated_training_corpus.csv` | `bd064fc882ee9c344fc46adedaef2e592071ab032a632bcba9e2fc699a741c30` |
| **Phase D3 Candidate Corpus** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/labels/phase_d3_candidate_label_corpus.csv` | `a90cd5c0e44da3256ecad118a0e69ad52aa38a1193b3cb35a084657323958ac1` |
| **Phase D3 Candidate Manifest** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/labels/phase_d3_candidate_manifest.json` | `05134744f93c2f9332285a4d3fbc8a18833efba5894b37c2c06830da1ae02041` |
| **Phase D4 Manifest** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d4/manifests/d4_manifest.json` | `0eff478fbc114130e07c5872c8df201e415bd0c0abee7416c78a9044e3151c31` |
| **Frozen 5-Fold CV Splits** | `/scratch/deepanthpcu/RSNA_Code/phase_c/c3/configs/cv_splits.json` | `648ab88d9bb2484ed49dcfae826cfb51efdc7b8f56bdf218803ee46f1d451db8` |
| **D1.1 Global OOF Table** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d1/results/oof/d1_1_global_oof_predictions.csv` | `1315655092611e75a01b23d41d38cd270ac887baab6d5716dad6c678b0a541e8` |
| **D2 Standalone Global OOF Table** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d2/predictions/oof/d2_global_oof_predictions.csv` | `ba7174dbc22121a9bc1c9b65b511341127cc01b25e30a2b3a75a465c1b050a78` |
| **D3 Standalone Global OOF Table** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d3/predictions/d3_global_oof_predictions.csv` | `348d2512bee28e1ca7b984d0f5eb82a21a55ad1c2658ae9ca6172591da20aad4` |
| **D4 Standalone Global OOF Table** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d4/predictions/oof/d4_cnn_oof_predictions.csv` | `61fc8738b8d0029217f98a75661c7180f01f7a176eaf1cc260bd7330d5283b81` |
| **D4-C1 Ensemble Table** | `/scratch/deepanthpcu/RSNA_Code/phase_d/d4/predictions/blends/d4_ensemble_predictions.csv` | `4a7eed17754ac945dc04a824926f5c748c0cef876ea24390dbfd29e8a20e30c5` |

---

## 2. Configuration & Hyperparameter Evolution Across Phases

| Parameter / Dimension | Phase D1.1 (Production ViT) | Phase D2 (CNN Arm) | Phase D3 (Retrained ViT) | Phase D4 (Multi-Node CNN) | Rationale for Modification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Backbone Architecture** | DINOv2 ViT-S/14 | RadImageNet ResNet-50 | DINOv2 ViT-S/14 | RadImageNet ResNet-50 | Testing ViT vs CNN representations |
| **Pretrained Weights Source** | `dinov2_vits14` (Meta LVD-142M)| `Lab-Rasool/RadImageNet` | `dinov2_vits14` | `Lab-Rasool/RadImageNet` | Medical vs natural pretraining |
| **Backbone Parameters** | 22,056,204 | 24,174,412 | 22,056,204 | 24,174,412 | Standard model capacities |
| **Trainable Parameters** | 10,985,484 (Blocks 6–11 + Head)| 16,043,084 (Layers 3–4 + Head)| 10,985,484 | 16,043,084 | Deep layer adaptation |
| **Frozen Parameters** | 11,070,720 (Blocks 0–5) | 8,131,328 (Layers 1–2 + Conv1)| 11,070,720 | 8,131,328 | Low-level feature retention |
| **Backbone Learning Rate** | $8.0 \times 10^{-6}$ | $1.0 \times 10^{-4}$ | $8.0 \times 10^{-6}$ | $1.0 \times 10^{-4}$ | CNNs require higher LR than ViTs |
| **SlotHead Learning Rate** | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-3}$ | Stable head convergence |
| **Weight Decay** | $0.02$ | $0.02$ | $0.02$ | $0.02$ | Standard regularization |
| **Scheduler** | OneCycleLR (`pct_start=0.15`) | OneCycleLR (`pct_start=0.15`)| OneCycleLR (`pct_start=0.15`) | OneCycleLR (`pct_start=0.15`) | Cosine annealing schedule |
| **Supervision Corpus** | Phase B Step 3 Legacy | Phase B Step 3 Legacy | Candidate Remediated D3 | Candidate Remediated D3 | Remediation of OA false negatives |
| **Effective Batch Size** | 8 studies (2/GPU × 2 GPUs × 2) | 8 studies (2/GPU × 2 GPUs × 2)| 8 studies (2/GPU × 2 GPUs × 2)| 6 studies (1/GPU × 6 GPUs) | Multi-node scaling on 3 nodes |
| **Hardware Topology** | 2 × A100 (`rpgpu018`) | 2 × A100 (`rpgpu018`) | 2 × A100 (`rpgpu018`) | 6 × A100 (`rpgpu007, 018, 028`)| Multi-node cluster utilization |
| **Interconnect Interface** | Local PCIe NCCL | Local PCIe NCCL | Local PCIe NCCL | Multi-Node `eth2` Socket NCCL | High-speed cluster interconnect |
| **Evaluation Precision** | BFloat16 forward | BFloat16 forward | BFloat16 forward | BFloat16 with Float32 cast | Fix NumPy BFloat16 incompatibility |

---

## 3. Engineering Bug Recoveries & Fixes Log

### 3.1 D1.1 Scheduler Step Count Off-by-One Exception
- **Issue:** Under uneven dataset lengths in cross-validation folds, `len(loader) // 2` dropped the trailing odd batch. Calling `scheduler.step()` on the final batch threw:
  `ValueError: Tried to step 1241 times. The specified number of total steps is 1240.`
- **Correction:** Implemented ceiling division in `train_ddp_d1.py`:
  `total_steps = ((len(loader) + 1) // 2) * epochs`
- **Result:** Fully unattended 5-fold training executed across all folds with zero exceptions.

### 3.2 D4 BFloat16 Evaluation Cast Exception in NumPy
- **Issue:** Under multi-node DDP evaluation with PyTorch AMP, validation probability tensors remained in BFloat16. Calling `.cpu().numpy()` triggered:
  `TypeError: Got unsupported ScalarType BFloat16`
- **Correction:** Added explicit Float32 type promotion in `train_ddp_d4.py` and `orchestrate_d4_cv.py`:
  `preds = torch.sigmoid(logits).float().cpu().numpy()`
- **Result:** Clean OOF aggregation and dual-standard evaluation completed with 100% data audit pass.

### 3.3 D3 NLP Label Remediation for Spanish Syntax & Vocabulary
- **Issue:** Spanish reports with `"artrosis femorotibial lateral"` and `"condropatía"` were dropped by English-centric regex rules in Phase B, leaving OA prevalence at 0.02%–0.34%.
- **Correction:** Developed `generate_d3_labels.py` with Spanish post-positive adjective parsing, `"condropatía"` vocabulary expansion, and multi-compartment mapping.
- **Result:** Recovered 472 true clinical positives with 100% qualitative precision across 30 adjudicated cases, boosting ViT Macro ROC-AUC from `0.7510` to `0.8135` (+8.32%) and PR-AUC by +39.2%.
