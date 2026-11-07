#!/usr/bin/env python3
import os
import sys
import json

source_path = os.environ.get("NOTION_SOURCE_PATH", "/tmp/notion_source.json")
if not os.path.exists(source_path):
    print(f"[INFO] Source file {source_path} not found. update_notion_project_phases.py is an audit helper.")
    sys.exit(0)

with open(source_path) as f:
    data = json.load(f)['data']

md = data['markdown']

old_sec = '### Phase C3 — Downstream 3D Vision Modeling'
idx = md.find(old_sec)
prefix = md[:idx]

new_sec = """### Phase C3 — Master 3D Vision Modeling, CV & Locked Gold-58 Evaluation (COMPLETE)
- **Execution Date**: 2026-09-12
- **Status**: **100% COMPLETE & VERIFIED**
- **Hardware**: 2 × NVIDIA A100 PCIe 80GB (rpgpu018) via PyTorch DDP
- **Architecture**: `MultiSeriesKneeModel` (Shared SmallResNet3D + Masked Pooling, 2,052,828 params)
- **Validation Gates Passed (All GREEN)**:
\t- `Gate 7`: Canonical 5-slot metadata mapping & collision audit (24,033/24,033 matched, PASS)
\t- `Gate 8`: 8-study overfit sanity test (Loss 0.7132 -> 0.4230, PASS)
\t- `Gate 9`: Single-A100 sanity run & checkpoint resume (PASS)
\t- `Gate 10`: 2-GPU DDP sanity & parameter synchronization (PASS)
\t- `Gate 11`: Production 5-Fold Stratified Cross-Validation (PASS)
\t- `Gate 12A`: Study-Level Out-of-Fold (OOF) Inference & Audit across 4,349 studies (PASS)
\t- `Gate 12B`: Locked Gold-58 Evaluation & Predetermined Rank Averaging Ensemble (PASS)
- **Hierarchical Performance Results**:
\t- **C3.11 Production CV (Development Mean Fold Val)**: `0.6333 ± 0.0326` Macro ROC-AUC
\t- **C3.12 Stage A OOF (Global Study-Level Concatenated, N=4,349)**: `0.6190` Macro ROC-AUC
\t- **C3.12 Stage B Locked Gold-58 (Predetermined Rank Ensemble, N=58)**: `0.5986` Macro ROC-AUC | `0.4526` Macro PR-AUC
- **Top Performing Targets on Gold-58**: Effusion (`0.7130`), Medial Meniscus (`0.6647`), Synovitis (`0.6583`), Fracture (`0.6319`), Baker's (`0.6105`), Lateral OA (`0.6064`)
- **Scientific Boundary**: Phase C3 lifecycle is complete. 0 Gold leakage verified. Phase C4 scaling is strictly awaiting review.
- **Consolidated Master Audit Documentation**: [Phase C3 — Master Multi-Series 3D Vision Modeling, Cross-Validation & Locked Gold-58 Evaluation Audit](https://app.notion.com/p/Phase-C3-Master-Multi-Series-3D-Vision-Modeling-Cross-Validation-Locked-Gold-58-Evaluation-Audi-3d9ad6e22d0081c5aa9dc0eaf87b19b2)."""

updated_md = prefix + new_sec

payload = {
    'operation': 'update_page_markdown',
    'payload': {
        'page_id': '3d7ad6e2-2d00-8090-baa0-c86f5b6ffc3e',
        'markdown': updated_md,
        'allow_deleting_content': True
    }
}

with open('/tmp/notion_update_project_phases.json', 'w') as f:
    json.dump(payload, f)

print('Updated markdown written cleanly, length:', len(updated_md))
