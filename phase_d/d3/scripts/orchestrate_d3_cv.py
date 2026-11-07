#!/usr/bin/env python3
"""
Phase D3 Master 5-Fold CV Orchestrator, OOF Aggregator & Dual-Evaluation Engine.
RSNA Knee Abnormality Detection

Orchestrates:
1. Production DDP training of DINOv2 ViT-S/14 on Phase D3 Remediated Labels across Folds 0..4 on 2 x A100 GPUs.
2. Global 4,349-study OOF aggregation, verification, and metric calculation.
3. Dual-ground-truth evaluation:
   - Evaluates OOF against Remediated D3 Labels (isolating the true clinical target performance)
   - Evaluates OOF against Legacy Phase B Production Labels (direct comparison with frozen D1.1/D1.2)
4. Emits complete structured metrics JSON and audit summaries.
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = os.environ.get("RSNA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase_d", "d1", "scripts"))
from d1_2_eval_utils import (
    roc_auc_score_np,
    pr_auc_score_np,
    evaluate_oof_df,
    PATHOLOGY_NAMES,
)

CONFIG_CV = os.path.join(PROJECT_ROOT, "phase_c/c3/configs/cv_splits.json")
D3_LABELS_CSV = os.path.join(PROJECT_ROOT, "phase_d/d3/labels/phase_d3_candidate_label_corpus.csv")
PROD_LABELS_CSV = os.path.join(PROJECT_ROOT, "labels/production/phase_b_step3_remediated_training_corpus.csv")
GOLD_CSV = os.path.join(PROJECT_ROOT, "csv/gold_studies.csv")

D3_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/scripts")
D3_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/checkpoints")
D3_VAL_PRED_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/predictions/val")
D3_PRED_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/predictions")
D3_REPORTS_DIR = os.path.join(PROJECT_ROOT, "phase_d/d3/reports")

D1_TEMPLATE_OOF_CSV = os.path.join(PROJECT_ROOT, "phase_d/d1/results/oof/d1_1_global_oof_predictions.csv")


def run_command(cmd: list, description: str):
    print(f"\n{'='*80}\n[ORCHESTRATOR D3] {description}\nCOMMAND: {' '.join(cmd)}\n{'='*80}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[ERROR] {description} FAILED with exit code {result.returncode} in {elapsed:.1f}s", flush=True)
        sys.exit(result.returncode)
    print(f"[SUCCESS] {description} completed in {elapsed:.1f}s", flush=True)


def train_all_folds():
    """Train Folds 0 through 4 sequentially using 2 x A100 DDP."""
    python_bin = sys.executable
    train_script = os.path.join(D3_SCRIPTS_DIR, "train_ddp_d3.py")

    for fold in range(5):
        ckpt_path = os.path.join(D3_CHECKPOINTS_DIR, f"fold_{fold}", "best_model.pt")
        val_pred_path = os.path.join(D3_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")

        if os.path.exists(ckpt_path) and os.path.exists(val_pred_path):
            print(f"[RESUME] Fold {fold} already completed. Checkpoint: {ckpt_path}", flush=True)
            continue

        cmd = [
            python_bin, "-m", "torch.distributed.run",
            "--nproc_per_node=2",
            f"--master_port={29520 + fold}",
            train_script,
            f"--fold={fold}",
            "--epochs=10",
            "--batch_size=2",
            "--grad_accum=2",
            "--backbone_lr=8e-6",
            "--head_lr=1e-3",
        ]
        run_command(cmd, f"Training Phase D3 DINOv2 Remediated Labels Fold {fold}")


def aggregate_d3_oof():
    """Aggregate 5-fold validation predictions into standardized global 4,349-study OOF format."""
    print("\n" + "=" * 80, flush=True)
    print("  AGGREGATING PHASE D3 5-FOLD OUT-OF-FOLD (OOF) PREDICTIONS", flush=True)
    print("=" * 80, flush=True)

    template_df = pd.read_csv(D1_TEMPLATE_OOF_CSV)
    uids_order = template_df["StudyInstanceUID"].tolist()

    val_dfs = []
    for fold in range(5):
        p_path = os.path.join(D3_VAL_PRED_DIR, f"fold_{fold}_val_predictions.csv")
        assert os.path.exists(p_path), f"Missing validation predictions for Fold {fold}: {p_path}"
        df_k = pd.read_csv(p_path)
        val_dfs.append(df_k)
        print(f"Fold {fold}: loaded {len(df_k)} studies", flush=True)

    raw_oof_df = pd.concat(val_dfs, axis=0).set_index("StudyInstanceUID").reindex(uids_order).reset_index()

    # Build standardized OOF dataframe
    d3_oof_df = template_df.copy()
    for t in PATHOLOGY_NAMES:
        d3_oof_df[f"pred_{t}"] = raw_oof_df[t].values

    oof_csv_path = os.path.join(D3_PRED_DIR, "d3_global_oof_predictions.csv")
    d3_oof_df.to_csv(oof_csv_path, index=False)
    print(f"[SAVED] D3 Global OOF predictions to: {oof_csv_path}", flush=True)

    # 1. Evaluate against Phase D3 Remediated Ground Truth
    print("\n--- Evaluating against D3 Remediated Labels ---", flush=True)
    d3_labels_df = pd.read_csv(D3_LABELS_CSV)
    d3_eval_df = d3_oof_df.copy()
    for t in PATHOLOGY_NAMES:
        sub_l = d3_labels_df[d3_labels_df["target"] == t].set_index("StudyInstanceUID").reindex(uids_order)
        d3_eval_df[f"true_{t}"] = sub_l["label_probability"].values
        d3_eval_df[f"weight_{t}"] = sub_l["sample_weight"].values

    d3_results = evaluate_oof_df(d3_eval_df, experiment_id="D3_REMEDIATED_LABELS_EVALUATION")

    # 2. Evaluate against Legacy Phase B Production Ground Truth
    print("\n--- Evaluating against Legacy Phase B Production Labels ---", flush=True)
    prod_results = evaluate_oof_df(d3_oof_df, experiment_id="D3_LEGACY_PRODUCTION_EVALUATION")

    combined_summary = {
        "experiment_id": "PHASE_D3_DINOV2_REMEDIATED_LABELS",
        "d3_remediated_evaluation": d3_results,
        "legacy_production_evaluation": prod_results,
        "delta_vs_d1_2_legacy": prod_results["global_macro_roc_auc_unweighted"] - 0.7592,
    }

    summary_json_path = os.path.join(D3_REPORTS_DIR, "d3_oof_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"\n[SAVED] D3 Evaluation Summary to: {summary_json_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  PHASE D3 FINAL 5-FOLD EVALUATION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Against D3 Remediated Ground Truth: Unw Macro AUC = {d3_results['global_macro_roc_auc_unweighted']:.4f} | Wtd AUC = {d3_results['global_macro_roc_auc_weighted']:.4f} | PR-AUC = {d3_results['global_macro_pr_auc']:.4f}")
    print(f"Against Legacy Production Labels:   Unw Macro AUC = {prod_results['global_macro_roc_auc_unweighted']:.4f} | Wtd AUC = {prod_results['global_macro_roc_auc_weighted']:.4f} | PR-AUC = {prod_results['global_macro_pr_auc']:.4f}")
    print(f"Δ vs Frozen D1.2-B2 (0.7592):       {combined_summary['delta_vs_d1_2_legacy']:+.4f}")
    print("=" * 80, flush=True)

    return d3_oof_df, combined_summary


def main():
    print("=" * 80, flush=True)
    print("  PHASE D3 MASTER WORKFLOW: DINOv2 + REMEDIATED LABELS 5-FOLD CV", flush=True)
    print("=" * 80, flush=True)
    train_all_folds()
    aggregate_d3_oof()
    print("\n[PHASE D3 COMPLETE] All training, OOF dual evaluation concluded.", flush=True)


if __name__ == "__main__":
    main()
