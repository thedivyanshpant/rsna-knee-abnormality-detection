#!/usr/bin/env python3
"""
Populate Master Project Documentation & Knowledge Base in Notion
RSNA Knee Abnormality Detection Project

Creates and updates standardized audit documentation pages in the 'RSNA Knee' Notion database.
"""

import os
import subprocess
import json
import time
from typing import Dict, Any

NOTION_BIN = os.environ.get("NOTION_BIN", "notion-mcp-server")
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "3d5ad6e2-2d00-80a0-ae68-000bb9f368e4")


def get_mcp_proc():
    if not NOTION_TOKEN:
        print("[NOTION] Warning: NOTION_TOKEN environment variable not set. Aborting Notion sync.")
        return None
    return subprocess.Popen(
        [NOTION_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin"),
            "NOTION_TOKEN": NOTION_TOKEN,
        }
    )


def call_notion_write(proc, operation: str, payload: Dict[str, Any], req_id: int):
    msg = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {
            "name": "notion_write",
            "arguments": {
                "operation": operation,
                "payload": payload,
            }
        }
    }
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()

    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        data = json.loads(line)
        if data.get("id") == req_id:
            return json.loads(data["result"]["content"][0]["text"])


def main():
    print("[NOTION] Starting Notion master documentation sync...")
    proc = get_mcp_proc()

    # Handshake
    proc.stdin.write(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "notion-sync", "version": "1.0"}}
    }) + "\n")
    proc.stdin.flush()
    proc.stdout.readline()

    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    proc.stdin.flush()

    pages_to_create = [
        {
            "title": "01 — Project Overview & Master Scope",
            "markdown": """# RSNA Knee Abnormality Detection: Master Project Overview

## 1. Project Directives
- **Competition / Goal**: Detect knee abnormalities from multimodal MRI studies.
- **Master Dataset Dimensions**:
  - Total studies: **4,407**
  - Locked Gold-58 evaluation cohort: **58** (strictly isolated from all training, CV, threshold tuning, and model selection)
  - Non-Gold development cohort: **4,349**
- **Target Records**:
  - Total: 4,407 × 12 = **52,884** records
  - Development cohort: 4,349 × 12 = **52,188** records

## 2. The 12 Clinical Prediction Targets
1. `ACL` (Anterior Cruciate Ligament)
2. `MCL` (Medial Collateral Ligament)
3. `Medial Meniscus`
4. `Lateral Meniscus`
5. `Medial OA` (Osteoarthritis)
6. `Lateral OA`
7. `PF OA` (Patellofemoral Osteoarthritis)
8. `Effusion`
9. `Synovitis`
10. `Baker's` (Baker's Cyst)
11. `Contusion` (Bone Contusion / Bruise)
12. `Fracture`
"""
        },
        {
            "title": "02 — Dataset & Data Governance Specification",
            "markdown": """# Data Governance & Dataset Registry

## 1. Core Data Policy
- **Zero Raw Data Copying**: Raw DICOM data remains strictly separated in source storage. No raw MRI duplicate (523+ GB) is created on HPC or inside `RSNA_Code`.
- **Gold-58 Strict Isolation**: Gold-58 studies are permanently quarantined. Zero leakage is verified before any training run.
- **Canonical Processed Pipeline C Path**: `phase_c/c1/processed/`
- **Canonical Series Metadata Path**: `csv/train_series.csv`
- **Canonical Label Corpus**: `labels/production/phase_b_step3_remediated_training_corpus.csv`

## 2. Dataset Cohort Summary
| Cohort | Study Count | Target Records | Status | Allowed for Training |
| :--- | :---: | :---: | :---: | :---: |
| **Development Cohort** | 4,349 | 52,188 | Processed (Pipeline C) | YES |
| **Gold-58 Evaluation** | 58 | 696 | Locked Final Eval | NO (Locked) |
| **Total Master Set** | 4,407 | 52,884 | Complete | - |
"""
        },
        {
            "title": "06 — Loss Function Specification: Weighted Soft BCE",
            "markdown": """# Mathematical & Implementation Specification: Weighted Soft BCE

## 1. Mathematical Formulation
$$\\mathcal{L} = \\frac{\\sum_{m=1}^{12} W_m \\cdot \\text{BCEWithLogits}(z_m, Y_m)}{\\sum_{m=1}^{12} W_m + \\epsilon}$$

Where:
- $z_m \\in \\mathbb{R}$: Raw unnormalized model logit for target $m$.
- $Y_m \\in [0.02, 0.98]$: Continuous soft label probability derived from Phase B weak supervision.
- $C_m = |2Y_m - 1|$: Confidence metric.
- $W_m = 0.25 + 0.75 \\cdot C_m$: Sample weight ($W_m = 0.0$ when `abstain_flag == 1`).

## 2. PyTorch Production Implementation
```python
def weighted_soft_bce_loss(logits, targets, weights):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weighted_bce = bce * weights
    loss = weighted_bce.sum() / weights.sum().clamp(min=1.0)
    return loss
```
"""
        },
        {
            "title": "08 — Phase C3: Canonical Series Slot Mapping & Audit",
            "markdown": """# Phase C3: Canonical Series Slot Mapping & Verification Audit

## 1. The 5 Canonical Sequence Slots
- **Slot 0**: `sagittal_fluid_sensitive` (Sagittal + Fluid_Sensitive==1)
- **Slot 1**: `coronal_fluid_sensitive` (Coronal + Fluid_Sensitive==1)
- **Slot 2**: `axial_fluid_sensitive` (Axial + Fluid_Sensitive==1)
- **Slot 3**: `sagittal_non_fs_t1` (Sagittal + Fluid_Sensitive==0)
- **Slot 4**: `coronal_non_fs_t1` (Coronal + Fluid_Sensitive==0)

## 2. Empirical Allocation Statistics (4,349 Non-Gold Studies)
- **Total Processed Volumes on Disk**: 24,033 `.npz` files.
- **Matched Series**: 24,033 (100.0% matched to metadata).
- **Unmatched Processed Series**: 0 (PASS).
- **Metadata Rows with No Processed NPZ**: exactly 2 (both verified as the two corrupt raw series in partial studies).
- **Studies with all 5 slots populated**: 3,038 (69.86%).
- **Studies with >= 4 slots populated**: 4,118 (94.69%).
- **Collision Resolution**: 2,669 collisions deterministically resolved using `min(SeriesInstanceUID)` and logged.

## 3. Known Partial Studies Handling
- `1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035`: Mask `[1, 1, 0, 1, 1]`, safely handled.
- `1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680`: Mask `[1, 0, 1, 1, 1]`, safely handled.
"""
        },
        {
            "title": "13 — Hardware & Software Environment Registry",
            "markdown": """# Hardware & Software Environment Registry

## 1. Hardware Specification
- **Node Hostname**: `rpgpu018`
- **GPUs**: 2 × NVIDIA A100 PCIe 80 GB
- **Driver Version**: `550.54.14`
- **CUDA Runtime**: `12.4` / `12.6`

## 2. Software & Python Environment
- **Conda Environment**: `ML_DFT_V2`
- **Python**: `3.11.10`
- **PyTorch**: `2.7.0+cu126`
- **torchvision**: `0.22.0`
- **NumPy**: `2.2.6`
- **Pandas**: `3.0.5`
- **SciPy**: `1.17.1`
- **H5py**: `3.16.0`
"""
        },
        {
            "title": "15 — Decisions & Frozen Specifications Log",
            "markdown": """# Master Decision Log & Frozen Specifications

| Decision ID | Date | Specification | Impact | Status |
| :--- | :---: | :--- | :--- | :---: |
| **DEC-01** | 2026-09-11 | Phase B Step 3 Remediated Labels Frozen | Canonical label source for all vision models | **FROZEN** |
| **DEC-02** | 2026-09-11 | Pipeline C Selected as Production MRI Representation | Best empirical engineering/predictive trade-off | **FROZEN** |
| **DEC-03** | 2026-09-12 | 5-Fold Stratified Cross-Validation Splits Frozen | 4,349 studies partitioned into 5 disjoint folds (seed 42) | **FROZEN** |
| **DEC-04** | 2026-09-12 | Gold-58 Complete Isolation | Never used for training, threshold tuning, or model selection | **LOCKED** |
| **DEC-05** | 2026-09-12 | Canonical 5-Slot Multi-Series Mapping | Deterministic slot assignment via metadata join | **FROZEN** |
| **DEC-06** | 2026-09-12 | Zero Raw MRI Duplication | No duplicate raw DICOM on HPC | **FROZEN** |
"""
        },
        {
            "title": "16 — Failures, Issues & Recovery Log",
            "markdown": """# Failure, Issue & Recovery Log

## 1. Issue: Missing Series Metadata on HPC
- **Date**: 2026-09-12
- **Severity**: Blocker (C3.2 halted execution)
- **Root Cause**: `train_series.csv` had not been transferred to the HPC directory.
- **Resolution**: `train_series.csv` was restored to `csv/train_series.csv`.
- **Verification**: Check 7 of `test_c3_integrity.py` passed with 100% green status.

## 2. Issue: Two Corrupt DICOM Series in Raw Dataset
- **Date**: 2026-09-12
- **Severity**: Low (2 partial studies out of 4,349)
- **Studies Affected**:
  - `...35583035` (corrupted series `...02928348`)
  - `...995680` (corrupted series `...115703`)
- **Resolution**: Missing slots are kept zero-padded with `slot_mask=0.0`. Valid slots are utilized without fabricating data.
"""
        },
        {
            "title": "22 — C3 Execution Tracker & Milestone Gates",
            "markdown": """# Phase C3 Execution Tracker & Milestone Gates

| Gate | Stage Description | Status | Verification Output |
| :--- | :--- | :---: | :--- |
| **C3.1-6** | Automated Scientific Data Integrity Gate | **PASS** | 7/7 tests passed (`test_c3_integrity.py`) |
| **C3.7** | Canonical Series Slot Mapping & Audit | **PASS** | 24,033 volumes joined, 0 anomalies (`c3_canonical_slot_mapping_audit.json`) |
| **C3.8** | 8-Study Overfit Engineering Sanity Test | **RUNNING** | Training in progress (`train_overfit_8study.py`) |
| **C3.9** | Single-A100 Sanity Run | **PLANNED** | Pending C3.8 PASS |
| **C3.10** | 2-GPU DDP Sanity Run | **PLANNED** | Pending C3.9 PASS |
| **C3.11** | Full 5-Fold Cross-Validation | **PLANNED** | Pending C3.10 PASS |
| **C3.12** | Locked Final Gold-58 Evaluation | **PLANNED** | Final evaluation once CV models frozen |
"""
        }
    ]

    req_id = 2
    for p in pages_to_create:
        print(f"[NOTION] Creating page: '{p['title']}'...")
        payload = {
            "parent": {"type": "data_source_id", "data_source_id": DATABASE_ID},
            "title": p["title"],
            "markdown": p["markdown"],
        }
        res = call_notion_write(proc, "create_page", payload, req_id)
        if res and res.get("ok"):
            print(f"  -> Created URL: {res['data'].get('url')}")
        else:
            print(f"  -> Failed: {res}")
        req_id += 1
        time.sleep(0.5)

    proc.terminate()
    print("[NOTION] Sync complete.")


if __name__ == "__main__":
    main()
