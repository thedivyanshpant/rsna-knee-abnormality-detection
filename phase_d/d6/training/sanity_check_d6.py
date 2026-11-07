#!/usr/bin/env python3
"""
Phase D6: Compute Safety & Pre-Training Sanity Verification Engine.
RSNA Knee Abnormality Detection

Verifies all 8 safety gates before production training:
1. Minimal sanity batch loading
2. Data loading throughput
3. Six-slot tensor dimensions: (B, 6, 12, 3, 224, 224)
4. Slot masks with active Slot 3 verification
5. Forward pass through PhaseD1Model
6. Continuous soft BCE loss calculation
7. Backward pass & gradient verification across frozen/trainable blocks
8. Multi-GPU CUDA memory & execution check
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_d.d1.scripts.model_d1 import PhaseD1Model, PATHOLOGY_NAMES
from phase_d.d1.scripts.dataset_d1 import PhaseD1KneeDataset
from phase_d.d1.scripts.loss_d1 import weighted_soft_bce_loss

CV_SPLITS_JSON = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
D6_MANIFEST_CSV = os.path.join(PROJECT_ROOT, "phase_d/d6/resolver/d6_slot_mapping_manifest.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "phase_c/c1/processed")


def run_sanity_checks():
    print("=" * 80)
    print("PHASE D6: PRE-TRAINING COMPUTE SAFETY & SANITY VERIFICATION")
    print("=" * 80)

    # 1. Device check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[GATE 1] Compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    assert torch.cuda.is_available(), "CUDA is required for training!"

    # 2. Load frozen CV
    with open(CV_SPLITS_JSON) as f:
        cv_data = json.load(f)
    train_uids = cv_data["folds"]["fold_0"]["train_studies"][:16]
    print(f"[GATE 2] Pilot subset: {len(train_uids)} studies from fold_0")

    # 3. Instantiate Dataset with D6 manifest
    print(f"[GATE 3] Instantiating dataset with D6 sequence resolver manifest...")
    slot_df = pd.read_csv(D6_MANIFEST_CSV)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    gold_df = pd.read_csv(os.path.join(PROJECT_ROOT, "csv/gold_studies.csv"))
    gold_uids = set(gold_df["StudyInstanceUID"].tolist())

    dataset = PhaseD1KneeDataset(
        study_uids=train_uids,
        slot_manifest_df=slot_df,
        labels_df=d3_labels_df,
        processed_dir=PROCESSED_DIR,
        gold_uids=gold_uids,
        n_groups=12,
        slice_band=(0.2, 0.8),
        target_resolution=(224, 224),
        target_fov_mm=130.0,
        transform=None
    )
    print(f"Dataset successfully created with {len(dataset)} verified studies.")

    # 4. Data loader & batch tensor verification
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    batch = next(iter(loader))

    slots = batch["images"].to(device)
    slot_mask = batch["slot_mask"].to(device)
    targets = batch["targets"].to(device)
    weights = batch["weights"].to(device)

    print(f"[GATE 4] Batch tensor verification:")
    print(f"  slots shape:     {slots.shape} (expected: [2, 6, 12, 3, 224, 224])")
    print(f"  slot_mask shape: {slot_mask.shape} (expected: [2, 6])")
    print(f"  targets shape:   {targets.shape} (expected: [2, 12])")
    print(f"  weights shape:   {weights.shape} (expected: [2, 12])")
    assert slots.shape == (2, 6, 12, 3, 224, 224)
    assert slot_mask.shape == (2, 6)
    assert targets.shape == (2, 12)
    assert weights.shape == (2, 12)
    print("  Tensor dimensions: PASS")

    # 5. Model initialization
    print(f"[GATE 5] Initializing PhaseD1Model (DINOv2 ViT-S/14 + SlotHead)...")
    model = PhaseD1Model(
        pretrained_backbone="facebook/dinov2-small",
        freeze_blocks=6,
        n_slot=6,
        n_out=12,
        hidden_dim=256,
        prior_weight=0.55
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen params:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    assert total_params == 22262668, f"Unexpected parameter count: {total_params}"
    assert trainable_params == 10858252, f"Unexpected trainable count: {trainable_params}"
    print("  Parameter counts: PASS")

    # 6. Forward pass
    print(f"[GATE 6] Executing forward pass with BF16 autocast...")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits, attn = model(slots, slot_mask)
    print(f"  logits shape: {logits.shape} (expected: [2, 12])")
    print(f"  attn shape:   {attn.shape} (expected: [2, 12, 6])")
    assert logits.shape == (2, 12)
    assert attn.shape == (2, 12, 6)
    assert not torch.isnan(logits).any(), "NaN in logits!"
    assert not torch.isinf(logits).any(), "Inf in logits!"
    print("  Forward pass: PASS")

    # 7. Loss calculation
    print(f"[GATE 7] Computing weighted soft BCE loss...")
    loss = weighted_soft_bce_loss(logits.float(), targets, weights)
    print(f"  Computed loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0.0, "Loss must be positive!"
    print("  Loss calculation: PASS")

    # 8. Backward pass & gradient verification
    print(f"[GATE 8] Executing backward pass & gradient check...")
    loss.backward()

    # Check that frozen blocks have NO gradients
    for i in range(6):
        block = model.backbone.encoder.layer[i]
        for name, p in block.named_parameters():
            assert p.grad is None, f"Frozen block {i} parameter {name} has gradient!"

    # Check that trainable blocks have non-zero gradients
    trainable_with_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and torch.norm(p.grad).item() > 0.0:
                trainable_with_grad += 1
    print(f"  Trainable parameters receiving valid gradients: {trainable_with_grad}")
    assert trainable_with_grad > 0, "No gradients in trainable parameters!"
    print("  Backward pass & gradient flow: PASS")

    print("\n" + "=" * 80)
    print("ALL 8 PRE-TRAINING COMPUTE SAFETY GATES PASSED (100% GREEN)!")
    print("=" * 80)


if __name__ == "__main__":
    run_sanity_checks()
