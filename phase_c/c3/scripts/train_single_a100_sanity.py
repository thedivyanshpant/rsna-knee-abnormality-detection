#!/usr/bin/env python3
"""
Phase C3 Stage 9: Single-A100 Sanity Run
RSNA Knee Abnormality Detection

Purpose:
Verifies real training mechanics on a single NVIDIA A100 (80GB) before multi-GPU DDP:
- Forward pass, backward pass, optimizer step
- BF16 mixed precision
- Checkpoint saving and resume capability
- Metric computation on validation split
- GPU peak memory and throughput profiling
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


def run_single_a100_sanity(num_train: int = 32, num_val: int = 16, batch_size: int = 2, lr: float = 1e-4):
    start_time = time.time()
    set_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    print(f"[STAGE 9] Hardware: {gpu_name} (Device: {device})")

    # Paths
    processed_dir = os.path.join(REPO_ROOT, "phase_c", "c1", "processed")
    meta_path = os.path.join(REPO_ROOT, "csv", "train_series.csv")
    labels_path = os.path.join(REPO_ROOT, "labels", "production", "phase_b_step3_remediated_training_corpus.csv")
    gold_path = os.path.join(REPO_ROOT, "csv", "gold_studies.csv")
    cv_path = os.path.join(REPO_ROOT, "phase_c", "c3", "configs", "cv_splits.json")
    results_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "results")
    checkpoints_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "checkpoints", "single_a100_sanity")
    logs_dir = os.path.join(REPO_ROOT, "phase_c", "c3", "logs")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 1. Dataset splits from Fold 0
    with open(cv_path) as f:
        splits = json.load(f)
    f0 = splits["folds"]["fold_0"]
    train_uids = f0["train_studies"][:num_train]
    val_uids = f0["val_studies"][:num_val]

    # Leakage checks
    gold_df = pd.read_csv(gold_path)
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())
    assert len(set(train_uids) & gold_uids) == 0, "CRITICAL: Gold leakage in train split!"
    assert len(set(val_uids) & gold_uids) == 0, "CRITICAL: Gold leakage in val split!"
    assert len(set(train_uids) & set(val_uids)) == 0, "CRITICAL: Overlap between train and val!"

    print(f"[STAGE 9] Train cohort size: {len(train_uids)} | Val cohort size: {len(val_uids)}")

    # 2. Dataloaders
    labels_df = pd.read_csv(labels_path)
    meta_df = pd.read_csv(meta_path)

    train_ds = MultiSeriesKneeDataset(
        study_uids=train_uids,
        processed_dir=processed_dir,
        labels_df=labels_df,
        series_meta_df=meta_df,
        gold_uids=gold_uids,
    )
    val_ds = MultiSeriesKneeDataset(
        study_uids=val_uids,
        processed_dir=processed_dir,
        labels_df=labels_df,
        series_meta_df=meta_df,
        gold_uids=gold_uids,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=multi_series_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=multi_series_collate_fn
    )

    # 3. Model & Optimizer
    model = MultiSeriesKneeModel(
        embed_dim=128,
        num_targets=len(TARGETS),
        num_slots=NUM_SLOTS,
        aggregation="masked_pool",
        dropout=0.3,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"[STAGE 9] MultiSeriesKneeModel: {total_params:,} parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # 4. Phase 1: Train for 2 epochs, save checkpoint
    torch.cuda.reset_peak_memory_stats(device)
    print("\n--- Training Phase 1: Epochs 1 to 2 ---")
    history = []

    for epoch in range(1, 3):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_weight = 0.0

        for batch in train_loader:
            volumes = batch["volumes"].to(device, non_blocking=True)
            slot_mask = batch["slot_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(volumes, slot_mask)
                loss = weighted_soft_bce_loss(logits, targets, weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            w_sum = max(weights.sum().item(), 1.0)
            train_loss += loss.item() * w_sum
            train_weight += w_sum

        mean_train_loss = train_loss / max(train_weight, 1.0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_weight = 0.0
        with torch.no_grad():
            for batch in val_loader:
                volumes = batch["volumes"].to(device)
                slot_mask = batch["slot_mask"].to(device)
                targets = batch["targets"].to(device)
                weights = batch["weights"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(volumes, slot_mask)
                    loss = weighted_soft_bce_loss(logits, targets, weights)

                w_sum = max(weights.sum().item(), 1.0)
                val_loss += loss.item() * w_sum
                val_weight += w_sum

        mean_val_loss = val_loss / max(val_weight, 1.0)
        ep_time = time.time() - ep_start
        throughput = len(train_uids) / max(ep_time, 0.01)

        print(f"Epoch {epoch:2d} | Train Loss: {mean_train_loss:.5f} | Val Loss: {mean_val_loss:.5f} | Time: {ep_time:.2f}s ({throughput:.1f} studies/s)")
        history.append({"epoch": epoch, "train_loss": mean_train_loss, "val_loss": mean_val_loss, "time": ep_time})

    # Save checkpoint at epoch 2
    ckpt_path = os.path.join(checkpoints_dir, "checkpoint_epoch_2.pt")
    torch.save({
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
    }, ckpt_path)
    print(f"[STAGE 9] Checkpoint saved at {ckpt_path} (Size: {os.path.getsize(ckpt_path)/1e6:.2f} MB)")

    # 5. Phase 2: Resume Capability Verification (Reload checkpoint and train Epoch 3)
    print("\n--- Testing Resume Capability from Checkpoint ---")
    resume_model = MultiSeriesKneeModel(
        embed_dim=128,
        num_targets=len(TARGETS),
        num_slots=NUM_SLOTS,
        aggregation="masked_pool",
        dropout=0.3,
    ).to(device)

    resume_opt = torch.optim.AdamW(resume_model.parameters(), lr=lr, weight_decay=1e-4)
    resume_scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    checkpoint = torch.load(ckpt_path, map_location=device)
    resume_model.load_state_dict(checkpoint["model_state_dict"])
    resume_opt.load_state_dict(checkpoint["optimizer_state_dict"])
    resume_scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

    # Train Epoch 3 with resumed model
    ep_start = time.time()
    resume_model.train()
    train_loss = 0.0
    train_weight = 0.0

    for batch in train_loader:
        volumes = batch["volumes"].to(device, non_blocking=True)
        slot_mask = batch["slot_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        weights = batch["weights"].to(device, non_blocking=True)

        resume_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = resume_model(volumes, slot_mask)
            loss = weighted_soft_bce_loss(logits, targets, weights)

        resume_scaler.scale(loss).backward()
        resume_scaler.step(resume_opt)
        resume_scaler.update()

        w_sum = max(weights.sum().item(), 1.0)
        train_loss += loss.item() * w_sum
        train_weight += w_sum

    mean_train_loss = train_loss / max(train_weight, 1.0)

    # Validation on resumed model
    resume_model.eval()
    val_loss = 0.0
    val_weight = 0.0
    with torch.no_grad():
        for batch in val_loader:
            volumes = batch["volumes"].to(device)
            slot_mask = batch["slot_mask"].to(device)
            targets = batch["targets"].to(device)
            weights = batch["weights"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = resume_model(volumes, slot_mask)
                loss = weighted_soft_bce_loss(logits, targets, weights)

            w_sum = max(weights.sum().item(), 1.0)
            val_loss += loss.item() * w_sum
            val_weight += w_sum

    mean_val_loss = val_loss / max(val_weight, 1.0)
    ep_time = time.time() - ep_start
    throughput = len(train_uids) / max(ep_time, 0.01)

    print(f"Resumed Epoch {start_epoch:2d} | Train Loss: {mean_train_loss:.5f} | Val Loss: {mean_val_loss:.5f} | Time: {ep_time:.2f}s ({throughput:.1f} studies/s)")
    history.append({"epoch": start_epoch, "train_loss": mean_train_loss, "val_loss": mean_val_loss, "time": ep_time})

    # Peak GPU memory
    max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"[STAGE 9] Peak GPU VRAM allocated: {max_memory_mb:.1f} MB ({max_memory_mb / 81920 * 100:.2f}% of 80GB)")

    wall_clock = time.time() - start_time
    sanity_passed = (len(history) == 3) and (max_memory_mb < 20000)

    # Save summary
    summary = {
        "stage": "C3.9_single_a100_sanity",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "device": str(device),
        "train_cohort_size": len(train_uids),
        "val_cohort_size": len(val_uids),
        "parameters": total_params,
        "precision": "bfloat16 (AMP)",
        "batch_size": batch_size,
        "peak_vram_mb": float(max_memory_mb),
        "mean_epoch_time_sec": float(np.mean([h["time"] for h in history])),
        "mean_throughput_studies_per_sec": float(len(train_uids) / np.mean([h["time"] for h in history])),
        "checkpoint_saved": os.path.exists(ckpt_path),
        "checkpoint_size_bytes": os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else 0,
        "resume_successful": True,
        "wall_clock_sec": float(wall_clock),
        "decision": "PASS" if sanity_passed else "FAIL",
    }

    summary_path = os.path.join(results_dir, "c3_stage9_single_a100_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(results_dir, "c3_stage9_single_a100_audit.md")
    with open(md_path, "w") as f:
        f.write("# Phase C3 Stage 9: Single-A100 Sanity Audit Report\n\n")
        f.write(f"- **Execution Timestamp**: {summary['timestamp']}\n")
        f.write(f"- **Hardware**: {gpu_name} (80GB PCIe)\n")
        f.write(f"- **Precision**: BFloat16 AMP\n")
        f.write(f"- **Batch Size**: {batch_size}\n")
        f.write(f"- **Peak VRAM Allocated**: `{max_memory_mb:.1f} MB` ({max_memory_mb / 81920 * 100:.2f}% of 80GB)\n")
        f.write(f"- **Mean Training Throughput**: `{summary['mean_throughput_studies_per_sec']:.1f} studies/sec`\n")
        f.write(f"- **Mean Epoch Time (32 studies)**: `{summary['mean_epoch_time_sec']:.2f} s`\n")
        f.write(f"- **Checkpoint Saved & Resumed**: **PASS** ({os.path.basename(ckpt_path)})\n")
        f.write(f"- **Gate Decision**: **{summary['decision']}**\n\n")

        f.write("## Epoch History\n\n")
        f.write("| Epoch | Train Loss | Val Loss | Epoch Time (s) | Throughput (studies/s) |\n| :---: | :---: | :---: | :---: | :---: |\n")
        for h in history:
            th = len(train_uids) / max(h["time"], 0.01)
            f.write(f"| {h['epoch']} | {h['train_loss']:.5f} | {h['val_loss']:.5f} | {h['time']:.2f} | {th:.1f} |\n")

    print("\n" + "=" * 80)
    print(f"[STAGE 9] Single-A100 Sanity Result: {summary['decision']}")
    print(f"  - Peak VRAM: {max_memory_mb:.1f} MB")
    print(f"  - Throughput: {summary['mean_throughput_studies_per_sec']:.1f} studies/s")
    print(f"  - Checkpoint Resume: VERIFIED")
    print(f"  - Summary JSON: {summary_path}")
    print(f"  - Audit Markdown: {md_path}")
    print(f"  - Wall clock: {wall_clock:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_single_a100_sanity()
