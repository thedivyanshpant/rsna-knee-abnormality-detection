#!/usr/bin/env python3
"""
Phase C3 Stage 8: 8-Study Overfit Engineering Sanity Test
RSNA Knee Abnormality Detection

Purpose:
Verifies that the complete dataset -> slot collation -> MultiSeriesKneeModel ->
weighted soft BCE loss -> AdamW optimizer pipeline can strongly learn and overfit a tiny cohort.

Requirements:
- 8 non-Gold studies only.
- 0 Gold-58 leakage.
- Exact Phase B weighted soft BCE loss.
- Canonical 5-slot representation.
- Fixed seed 42.
- BF16 AMP mixed precision.
"""

import os
import sys
import time
import json
import random
import datetime
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(REPO_ROOT, "phase_c", "c3", "scripts"))

from model import MultiSeriesKneeModel, count_parameters, TARGETS, NUM_SLOTS
from loss import weighted_soft_bce_loss
from dataset import MultiSeriesKneeDataset, multi_series_collate_fn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_overfit_test(epochs: int = 50, batch_size: int = 2, lr: float = 1e-3, device_id: int = 0):
    start_time = time.time()
    set_seed(42)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[STAGE 8] Running on device: {device} ({torch.cuda.get_device_name(device)})")

    # 1. Paths
    processed_dir = os.path.join(REPO_ROOT, "phase_c", "c1", "processed")
    meta_path = os.path.join(REPO_ROOT, "csv", "train_series.csv")
    labels_path = os.path.join(REPO_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")
    gold_path = os.path.join(REPO_ROOT, "csv", "gold_studies.csv")
    cv_path = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
    results_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "results")
    checkpoints_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "checkpoints", "overfit_8study")
    logs_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "logs")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 2. Select 8 non-Gold studies from Fold 0 train split
    with open(cv_path) as f:
        splits = json.load(f)
    f0_train = splits["folds"]["fold_0"]["train_studies"]
    selected_uids = f0_train[:8]

    # Verify Gold-58 isolation
    gold_df = pd.read_csv(gold_path)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    overlap = set(selected_uids) & gold_uids
    assert len(overlap) == 0, f"CRITICAL LEAKAGE: Gold-58 overlap in overfit cohort: {overlap}"

    print(f"[STAGE 8] Selected 8 non-Gold StudyInstanceUIDs:")
    for idx, uid in enumerate(selected_uids):
        print(f"  {idx+1}: {uid}")

    # 3. Load metadata & labels
    labels_df = pd.read_csv(labels_path)
    series_meta_df = pd.read_csv(meta_path)

    dataset = MultiSeriesKneeDataset(
        study_uids=selected_uids,
        processed_dir=processed_dir,
        labels_df=labels_df,
        series_meta_df=series_meta_df,
        gold_uids=gold_uids,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=multi_series_collate_fn,
    )

    # 4. Initialize Model
    model = MultiSeriesKneeModel(
        embed_dim=128,
        num_targets=len(TARGETS),
        num_slots=NUM_SLOTS,
        aggregation="masked_pool",
        dropout=0.0,  # 0.0 dropout for pure overfit capacity verification
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"[STAGE 8] MultiSeriesKneeModel instantiated. Total params: {total_params:,} (Trainable: {trainable_params:,})")

    # 5. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # 6. Training Loop
    history = []
    initial_loss = None
    best_loss = float("inf")

    print(f"\n{'Epoch':>6} | {'Loss':>10} | {'LR':>10} | {'Time (s)':>10}")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        epoch_loss = 0.0
        total_weight = 0.0

        for batch in dataloader:
            volumes = batch["volumes"].to(device, non_blocking=True)      # (B, 5, 1, 32, 256, 256)
            slot_mask = batch["slot_mask"].to(device, non_blocking=True)  # (B, 5)
            targets = batch["targets"].to(device, non_blocking=True)      # (B, 12)
            weights = batch["weights"].to(device, non_blocking=True)      # (B, 12)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(volumes, slot_mask)
                loss = weighted_soft_bce_loss(logits, targets, weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_w_sum = weights.sum().item()
            epoch_loss += loss.item() * max(batch_w_sum, 1.0)
            total_weight += max(batch_w_sum, 1.0)

        mean_loss = epoch_loss / max(total_weight, 1.0)
        ep_duration = time.time() - ep_start

        if initial_loss is None:
            initial_loss = mean_loss

        if mean_loss < best_loss:
            best_loss = mean_loss
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "selected_uids": selected_uids,
            }, os.path.join(checkpoints_dir, "best_model.pt"))

        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "duration_sec": ep_duration,
        })

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{epoch:6d} | {mean_loss:10.5f} | {lr:10.6f} | {ep_duration:10.2f}")

    # 7. Final Evaluation on Overfit Cohort
    model.eval()
    all_preds = []
    all_targets = []
    all_weights = []

    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=multi_series_collate_fn)
    with torch.no_grad():
        for batch in eval_loader:
            volumes = batch["volumes"].to(device)
            slot_mask = batch["slot_mask"].to(device)
            targets = batch["targets"].to(device)
            weights = batch["weights"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(volumes, slot_mask)
                probs = torch.sigmoid(logits)

            all_preds.append(probs.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())
            all_weights.append(weights.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)      # (8, 12)
    all_targets = np.concatenate(all_targets, axis=0)  # (8, 12)
    all_weights = np.concatenate(all_weights, axis=0)  # (8, 12)

    # Calculate theoretical minimum entropy bound for soft targets
    eps = 1e-12
    y_clamped = np.clip(all_targets, eps, 1 - eps)
    entropy = - (all_targets * np.log(y_clamped) + (1 - all_targets) * np.log(1 - y_clamped))
    w_sum = np.sum(all_weights)
    theoretical_min_loss = float(np.sum(entropy * all_weights) / max(w_sum, 1.0))

    # Calculate final mean absolute error and distance to theoretical minimum
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    final_loss = float(history[-1]["loss"])
    loss_drop_pct = float(((initial_loss - final_loss) / initial_loss) * 100.0)
    entropy_gap = float(abs(final_loss - theoretical_min_loss))

    # Strong overfit criteria for continuous soft BCE: MAE < 0.15 and gap to Shannon bound < 0.05
    overfit_passed = (mae < 0.15) and (entropy_gap < 0.05)

    wall_clock = time.time() - start_time

    # 8. Document Results
    summary = {
        "stage": "C3.8_8_study_overfit_test",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU",
        "cohort": "8 non-Gold studies from Fold 0 train split",
        "selected_study_uids": selected_uids,
        "gold_leakage": 0,
        "architecture": "MultiSeriesKneeModel (Shared 3D ResNet + Masked Pool + Linear)",
        "parameter_count": total_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "precision": "bfloat16 (AMP)",
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "best_loss": float(best_loss),
        "theoretical_min_loss": float(theoretical_min_loss),
        "entropy_gap": float(entropy_gap),
        "loss_reduction_pct": float(loss_drop_pct),
        "final_mae": float(mae),
        "wall_clock_sec": float(wall_clock),
        "decision": "PASS" if overfit_passed else "FAIL",
    }

    summary_json_path = os.path.join(results_dir, "c3_stage8_overfit_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    history_csv_path = os.path.join(logs_dir, "c3_stage8_overfit_history.csv")
    pd.DataFrame(history).to_csv(history_csv_path, index=False)

    md_path = os.path.join(results_dir, "c3_stage8_overfit_audit.md")
    with open(md_path, "w") as f:
        f.write("# Phase C3 Stage 8: 8-Study Overfit Sanity Audit\n\n")
        f.write(f"- **Execution Timestamp**: {summary['timestamp']}\n")
        f.write(f"- **Hardware**: {summary['device_name']} (Device `{summary['device']}`)\n")
        f.write(f"- **Total Trainable Parameters**: {total_params:,}\n")
        f.write(f"- **Initial Loss**: `{initial_loss:.5f}`\n")
        f.write(f"- **Final Loss**: `{final_loss:.5f}`\n")
        f.write(f"- **Best Loss**: `{best_loss:.5f}`\n")
        f.write(f"- **Theoretical Minimum Shannon Entropy Bound**: `{theoretical_min_loss:.5f}`\n")
        f.write(f"- **Gap to Theoretical Minimum**: `{entropy_gap:.5f}` (Tolerance: < 0.05)\n")
        f.write(f"- **Relative Loss Reduction**: **{loss_drop_pct:.2f}%**\n")
        f.write(f"- **Mean Absolute Error (Y vs Pred)**: `{mae:.4f}` (Tolerance: < 0.15)\n")
        f.write(f"- **Total Runtime**: `{wall_clock:.2f} s`\n")
        f.write(f"- **Gate Status**: **{summary['decision']}**\n\n")

        f.write("## 1. Selected Non-Gold Studies\n\n")
        for i, u in enumerate(selected_uids):
            f.write(f"{i+1}. `{u}`\n")

        f.write("\n## 2. Target Level Predictions vs Soft Targets (Sample Study 1)\n\n")
        f.write("| Target | Soft Ground Truth Y | Model Output Prob | Sample Weight W |\n| :--- | :---: | :---: | :---: |\n")
        for t_idx, t_name in enumerate(TARGETS):
            f.write(f"| {t_name} | {all_targets[0, t_idx]:.4f} | {all_preds[0, t_idx]:.4f} | {all_weights[0, t_idx]:.4f} |\n")

    print("\n" + "=" * 80)
    print(f"[STAGE 8] 8-Study Overfit Result: {summary['decision']}")
    print(f"  - Initial Loss: {initial_loss:.5f} -> Final Loss: {final_loss:.5f} (Reduction: {loss_drop_pct:.2f}%)")
    print(f"  - Best Checkpoint: {os.path.join(checkpoints_dir, 'best_model.pt')}")
    print(f"  - Summary JSON: {summary_json_path}")
    print(f"  - Audit Markdown: {md_path}")
    print(f"  - Wall clock: {wall_clock:.2f}s")
    print("=" * 80)

    if not overfit_passed:
        print("[STAGE 8 CRITICAL] Model failed to strongly overfit 8 studies. Halting pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    run_overfit_test()
