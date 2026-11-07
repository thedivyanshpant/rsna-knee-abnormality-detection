# Phase C 3D Vision Model Training Specification

## 1. Supervised Training Loss Formulation
For each training MRI study $i$ with raw logit outputs $\hat{z}_{i,m}$ across 12 targets:

$$\mathcal{L}_i = -\frac{1}{M} \sum_{m=1}^{M} W_{i,m} \cdot \Big[ Y_{i,m} \log\big(\sigma(\hat{z}_{i,m})\big) + (1 - Y_{i,m}) \log\big(1 - \sigma(\hat{z}_{i,m})\big) \Big]$$

Where:
- $Y_{i,m} \in [0.02, 0.98]$ is the soft pseudo-label target from `labels/production/phase_b_step3_remediated_training_corpus.csv`.
- $W_{i,m} \in [0.0, 1.0]$ is the sample weight ($W_{i,m} = 0.0$ for abstained/contradictory records).
- $M = 12$ abnormality targets.

## 2. Multi-Stage Curriculum Protocol
1. **Stage 1 (High-Confidence Warmup, Epochs 1–5)**:
   - Filter training batch to samples with $W_{i,m} \ge 0.90$ (encompassing 99.74% of corpus).
   - Establish stable visual feature representations in 3D backbone.
2. **Stage 2 (Full Soft-Target Regularization, Epochs 6–30)**:
   - Train on all samples with $W_{i,m} > 0.0$.
   - Enable mixup/cutmix 3D volumetric augmentations.
3. **Stage 3 (Validation & Checkpointing)**:
   - Evaluate model checkpoints against untouched 58 Gold studies in `labels/gold58/phase_b_step3_remediated_predictions_gold58.csv`.
   - Primary metric: Macro F1 and Macro ROC-AUC.
